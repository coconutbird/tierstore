//! The tier router: mechanism, not policy.
//!
//! [`Router`] drives the sans-io [`ReadFlow`] from `tierstore-core` against
//! real tiers. All semantic choices (promotion, error fall-through, write
//! propagation, demotion) come from [`Policy`].
//!
//! The router implements the tier traits itself, so a router can be a tier
//! of another router — hierarchies compose.

use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use tierstore_core::{
    Displaced, OnReadError, OnWriteError, Policy, Probe, Promote, ReadFlow, ReadOutcome,
    ReadPolicy, ReadStep, Tier, TierRead, TierWrite, WriteMode,
};

use crate::error::{BoxError, RouterError, TierFailure};
use crate::report::{DeleteReport, KeyStatus, ReadReport};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Internal object-safe facade over a tier, with errors unified to
/// [`BoxError`]. Backends implement the generic capability traits; this
/// exists only so the router can hold heterogeneous tiers in one `Vec`.
trait DynTier<K, V>: Send + Sync {
    fn name(&self) -> &str;
    fn get<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<Option<V>, BoxError>>;
    fn exists<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<bool, BoxError>>;
    fn put(&self, key: K, value: V) -> BoxFuture<'_, Result<Displaced<K, V>, BoxError>>;
    fn delete<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<bool, BoxError>>;
    fn get_many<'a>(&'a self, keys: &'a [K]) -> BoxFuture<'a, Result<Vec<Option<V>>, BoxError>>;
    fn put_many(&self, entries: Vec<(K, V)>) -> BoxFuture<'_, Result<Displaced<K, V>, BoxError>>;
    fn delete_many<'a>(&'a self, keys: &'a [K]) -> BoxFuture<'a, Result<Vec<bool>, BoxError>>;
}

struct Adapter<T>(T);

impl<K, V, T> DynTier<K, V> for Adapter<T>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
    T: TierRead<Key = K, Value = V> + TierWrite + Send + Sync + 'static,
    T::Error: StdError + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.0.name()
    }

    fn get<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<Option<V>, BoxError>> {
        Box::pin(async move { self.0.get(key).await.map_err(Into::into) })
    }

    fn exists<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<bool, BoxError>> {
        Box::pin(async move { self.0.exists(key).await.map_err(Into::into) })
    }

    fn put(&self, key: K, value: V) -> BoxFuture<'_, Result<Displaced<K, V>, BoxError>> {
        Box::pin(async move { self.0.put(key, value).await.map_err(Into::into) })
    }

    fn delete<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<bool, BoxError>> {
        Box::pin(async move { self.0.delete(key).await.map_err(Into::into) })
    }

    fn get_many<'a>(&'a self, keys: &'a [K]) -> BoxFuture<'a, Result<Vec<Option<V>>, BoxError>> {
        Box::pin(async move { self.0.get_many(keys).await.map_err(Into::into) })
    }

    fn put_many(&self, entries: Vec<(K, V)>) -> BoxFuture<'_, Result<Displaced<K, V>, BoxError>> {
        Box::pin(async move { self.0.put_many(entries).await.map_err(Into::into) })
    }

    fn delete_many<'a>(&'a self, keys: &'a [K]) -> BoxFuture<'a, Result<Vec<bool>, BoxError>> {
        Box::pin(async move { self.0.delete_many(keys).await.map_err(Into::into) })
    }
}

/// Read-only facade: reads forward; the router never routes writes here (it
/// skips non-writable slots), so the write methods exist only to satisfy the
/// object trait and fail loudly if a bug ever reaches them.
struct ReadOnly<T>(T);

impl<K, V, T> DynTier<K, V> for ReadOnly<T>
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
    T: TierRead<Key = K, Value = V> + Send + Sync + 'static,
    T::Error: StdError + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.0.name()
    }

    fn get<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<Option<V>, BoxError>> {
        Box::pin(async move { self.0.get(key).await.map_err(Into::into) })
    }

    fn exists<'a>(&'a self, key: &'a K) -> BoxFuture<'a, Result<bool, BoxError>> {
        Box::pin(async move { self.0.exists(key).await.map_err(Into::into) })
    }

    fn get_many<'a>(&'a self, keys: &'a [K]) -> BoxFuture<'a, Result<Vec<Option<V>>, BoxError>> {
        Box::pin(async move { self.0.get_many(keys).await.map_err(Into::into) })
    }

    fn put(&self, _key: K, _value: V) -> BoxFuture<'_, Result<Displaced<K, V>, BoxError>> {
        Box::pin(async { Err("tier is read-only".into()) })
    }

    fn delete<'a>(&'a self, _key: &'a K) -> BoxFuture<'a, Result<bool, BoxError>> {
        Box::pin(async { Err("tier is read-only".into()) })
    }

    fn put_many(&self, _entries: Vec<(K, V)>) -> BoxFuture<'_, Result<Displaced<K, V>, BoxError>> {
        Box::pin(async { Err("tier is read-only".into()) })
    }

    fn delete_many<'a>(&'a self, _keys: &'a [K]) -> BoxFuture<'a, Result<Vec<bool>, BoxError>> {
        Box::pin(async { Err("tier is read-only".into()) })
    }
}

/// One routed tier plus its write capability and counters.
struct TierSlot<K, V> {
    tier: Box<dyn DynTier<K, V>>,
    writable: bool,
    counters: Counters,
}

#[derive(Debug, Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    errors: AtomicU64,
    puts: AtomicU64,
    deletes: AtomicU64,
}

impl Counters {
    fn bump(counter: &AtomicU64, amount: u64) {
        counter.fetch_add(amount, Ordering::Relaxed);
    }
}

/// Point-in-time per-tier operation counters from [`Router::stats`].
///
/// Semantics: `hits`/`misses` count successful read probes (including
/// existence checks), `puts` and `deletes` count successful write calls
/// (batches count per entry/key), and `errors` counts failed calls of any
/// kind (a failed batch counts once). Counters are `Relaxed` totals since
/// construction — cheap, not a consistent cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierStats {
    /// Diagnostic tier name.
    pub name: String,
    /// Whether the tier was added via `read_only_tier`.
    pub read_only: bool,
    /// Successful read probes that found the key.
    pub hits: u64,
    /// Successful read probes that confirmed absence.
    pub misses: u64,
    /// Failed operations of any kind.
    pub errors: u64,
    /// Entries successfully written (puts, promotions, demotions).
    pub puts: u64,
    /// Keys successfully deleted (including write-around invalidation).
    pub deletes: u64,
}

/// Routes reads and writes across an ordered stack of tiers.
///
/// Tier `0` is the topmost (fastest); reads probe downward. Behaviour is
/// governed entirely by [`Policy`]. Construct with [`Router::builder`].
///
/// # Example
///
/// ```
/// use tierstore::{MemoryTier, Router};
///
/// let router: Router<String, Vec<u8>> = Router::builder()
///     .tier(MemoryTier::unbounded()) // top: fastest
///     .tier(MemoryTier::unbounded()) // bottom: most durable
///     .build();
/// assert_eq!(router.tier_count(), 2);
/// ```
pub struct Router<K, V> {
    tiers: Vec<TierSlot<K, V>>,
    policy: Policy,
}

/// Builder for [`Router`]; add tiers top-down.
pub struct RouterBuilder<K, V> {
    tiers: Vec<TierSlot<K, V>>,
    policy: Policy,
}

fn slot_names<K, V>(slots: &[TierSlot<K, V>]) -> Vec<String> {
    slots
        .iter()
        .map(|slot| {
            if slot.writable {
                slot.tier.name().to_owned()
            } else {
                format!("{} (read-only)", slot.tier.name())
            }
        })
        .collect()
}

impl<K, V> fmt::Debug for Router<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Router")
            .field("tiers", &slot_names(&self.tiers))
            .field("policy", &self.policy)
            .finish()
    }
}

impl<K, V> fmt::Debug for RouterBuilder<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RouterBuilder")
            .field("tiers", &slot_names(&self.tiers))
            .field("policy", &self.policy)
            .finish()
    }
}

impl<K, V> Router<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Starts building a router with the default [`Policy`].
    #[must_use]
    pub fn builder() -> RouterBuilder<K, V> {
        RouterBuilder {
            tiers: Vec::new(),
            policy: Policy::default(),
        }
    }

    /// The router's routing policy.
    #[must_use]
    pub const fn policy(&self) -> Policy {
        self.policy
    }

    /// Number of tiers in the stack.
    #[must_use]
    pub const fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Point-in-time snapshot of per-tier operation counters, in tier order
    /// (0 is topmost). See [`TierStats`] for the counting semantics.
    #[must_use]
    pub fn stats(&self) -> Vec<TierStats> {
        self.tiers
            .iter()
            .map(|slot| TierStats {
                name: slot.tier.name().to_owned(),
                read_only: !slot.writable,
                hits: slot.counters.hits.load(Ordering::Relaxed),
                misses: slot.counters.misses.load(Ordering::Relaxed),
                errors: slot.counters.errors.load(Ordering::Relaxed),
                puts: slot.counters.puts.load(Ordering::Relaxed),
                deletes: slot.counters.deletes.load(Ordering::Relaxed),
            })
            .collect()
    }

    /// Pushes entries displaced from tier `from` down the hierarchy,
    /// cascading further displacements. Returns whatever fell off the bottom
    /// (i.e. was evicted from the store entirely).
    ///
    /// Demotion is best effort: entries destined for a tier that errors are
    /// dropped. For cache semantics that is a capacity loss, not data loss —
    /// but it is one reason an authoritative bottom tier should be reliable.
    async fn demote(&self, from: usize, entries: Displaced<K, V>) -> Displaced<K, V> {
        let mut current = entries;
        let mut target = from + 1;
        while target < self.tiers.len() && !current.is_empty() {
            // Read-only tiers cannot receive demotions: entries pass over
            // them to the next writable tier (or off the bottom).
            if !self.tiers[target].writable {
                target += 1;
                continue;
            }
            let mut next = Displaced::new();
            for (key, value) in current {
                match self.tiers[target].tier.put(key, value).await {
                    Ok(mut displaced) => {
                        Counters::bump(&self.tiers[target].counters.puts, 1);
                        next.append(&mut displaced);
                    }
                    Err(_) => Counters::bump(&self.tiers[target].counters.errors, 1),
                }
            }
            current = next;
            target += 1;
        }
        current
    }

    fn failure(&self, tier: usize, source: BoxError) -> TierFailure {
        TierFailure::new(tier, self.tiers[tier].tier.name(), source)
    }

    /// Best-effort batched promotion after a batched read: for every tier
    /// that produced hits, copy those hits into the tiers above it per the
    /// promotion policy, demoting whatever the copies displace.
    async fn promote_batch(&self, keys: &[K], statuses: &[KeyStatus<V>]) {
        if matches!(self.policy.read.promote, Promote::Never) {
            return;
        }
        let mut hits_by_tier: Vec<Vec<usize>> = vec![Vec::new(); self.tiers.len()];
        for (index, status) in statuses.iter().enumerate() {
            if let KeyStatus::Hit { tier, .. } = status {
                hits_by_tier[*tier].push(index);
            }
        }
        for (tier, hits) in hits_by_tier.iter().enumerate() {
            if tier == 0 || hits.is_empty() {
                continue;
            }
            let entries: Vec<(K, V)> = hits
                .iter()
                .filter_map(|&index| {
                    statuses[index]
                        .value()
                        .map(|value| (keys[index].clone(), value.clone()))
                })
                .collect();
            let targets = match self.policy.read.promote {
                Promote::TopOnly => 0..1,
                Promote::AllAbove => 0..tier,
                Promote::Never => return,
            };
            for target in targets {
                if !self.tiers[target].writable {
                    continue;
                }
                match self.tiers[target].tier.put_many(entries.clone()).await {
                    Ok(displaced) => {
                        Counters::bump(&self.tiers[target].counters.puts, entries.len() as u64);
                        if self.policy.demote_displaced && !displaced.is_empty() {
                            let _evicted = self.demote(target, displaced).await;
                        }
                    }
                    Err(_) => Counters::bump(&self.tiers[target].counters.errors, 1),
                }
            }
        }
    }

    /// Batched read with per-key outcomes: each lower tier is probed only
    /// with the keys still missing, hits record which tier served them, and
    /// keys left unresolved past a failing tier come back as
    /// [`KeyStatus::Inconclusive`] instead of masquerading as misses —
    /// partial success never discards resolved values.
    ///
    /// # Errors
    ///
    /// Only under [`OnReadError::FailFast`], where the first tier error
    /// aborts the whole batch (resolved values are discarded by design).
    pub async fn read_many(&self, keys: &[K]) -> Result<ReadReport<V>, RouterError> {
        let mut statuses: Vec<KeyStatus<V>> = vec![KeyStatus::Miss; keys.len()];
        let mut unresolved: Vec<usize> = (0..keys.len()).collect();
        let mut failures = Vec::new();
        for (tier_index, slot) in self.tiers.iter().enumerate() {
            if unresolved.is_empty() {
                break;
            }
            let subset: Vec<K> = unresolved
                .iter()
                .map(|&index| keys[index].clone())
                .collect();
            match slot.tier.get_many(&subset).await {
                Ok(found) => {
                    let found_count = found.iter().filter(|value| value.is_some()).count();
                    Counters::bump(&slot.counters.hits, found_count as u64);
                    Counters::bump(&slot.counters.misses, (subset.len() - found_count) as u64);
                    let mut still_missing = Vec::new();
                    for (&index, value) in unresolved.iter().zip(found) {
                        if let Some(value) = value {
                            statuses[index] = KeyStatus::Hit {
                                tier: tier_index,
                                value,
                            };
                        } else {
                            still_missing.push(index);
                        }
                    }
                    unresolved = still_missing;
                }
                Err(source) => {
                    Counters::bump(&slot.counters.errors, 1);
                    match self.policy.read.on_error {
                        OnReadError::FailFast => {
                            return Err(RouterError::Tier(self.failure(tier_index, source)));
                        }
                        OnReadError::FallThrough => {
                            failures.push(self.failure(tier_index, source));
                        }
                    }
                }
            }
        }
        if !failures.is_empty() {
            // Any key still unresolved was pending when a tier failed, so
            // its absence is unconfirmed.
            for &index in &unresolved {
                statuses[index] = KeyStatus::Inconclusive;
            }
        }
        self.promote_batch(keys, &statuses).await;
        Ok(ReadReport { statuses, failures })
    }

    /// Single-key read with tier provenance: [`Router::read_many`] for one
    /// key, unpacked. Useful when per-tier hit metrics matter for
    /// individual reads.
    ///
    /// # Errors
    ///
    /// Only under [`OnReadError::FailFast`], like `read_many`.
    pub async fn read_one(&self, key: &K) -> Result<(KeyStatus<V>, Vec<TierFailure>), RouterError> {
        let mut report = self.read_many(std::slice::from_ref(key)).await?;
        // One key in, one status out; the fallback is unreachable.
        let status = report.statuses.pop().unwrap_or(KeyStatus::Miss);
        Ok((status, report.failures))
    }

    /// Batched delete with per-key outcomes. Every *writable* tier is
    /// attempted for every key regardless of failures (skipping one
    /// guarantees resurrection); read-only tiers are untouched — their
    /// copies persist by nature. Check [`DeleteReport::is_complete`] before
    /// trusting the flags.
    pub async fn remove_many(&self, keys: &[K]) -> DeleteReport {
        let mut removed = vec![false; keys.len()];
        let mut failures = Vec::new();
        for (index, slot) in self.tiers.iter().enumerate() {
            if !slot.writable {
                continue;
            }
            match slot.tier.delete_many(keys).await {
                Ok(flags) => {
                    Counters::bump(&slot.counters.deletes, keys.len() as u64);
                    for (removed_flag, flag) in removed.iter_mut().zip(flags) {
                        *removed_flag |= flag;
                    }
                }
                Err(source) => {
                    Counters::bump(&slot.counters.errors, 1);
                    failures.push(TierFailure::new(index, slot.tier.name(), source));
                }
            }
        }
        DeleteReport { removed, failures }
    }
}

impl<K, V> Tier for Router<K, V> {
    type Key = K;
    type Value = V;
    type Error = RouterError;

    fn name(&self) -> &'static str {
        "router"
    }
}

impl<K, V> TierRead for Router<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Result<Option<V>, RouterError> {
        let mut flow = ReadFlow::new(self.tiers.len(), self.policy().read);
        let mut hit: Option<V> = None;
        let mut failures = Vec::new();
        loop {
            match flow.step() {
                ReadStep::Get { tier } => match self.tiers[tier].tier.get(key).await {
                    Ok(Some(value)) => {
                        Counters::bump(&self.tiers[tier].counters.hits, 1);
                        hit = Some(value);
                        flow.on_get(Probe::Hit);
                    }
                    Ok(None) => {
                        Counters::bump(&self.tiers[tier].counters.misses, 1);
                        flow.on_get(Probe::Miss);
                    }
                    Err(source) => {
                        Counters::bump(&self.tiers[tier].counters.errors, 1);
                        failures.push(self.failure(tier, source));
                        flow.on_get(Probe::Error);
                    }
                },
                ReadStep::Promote { tier } => {
                    if self.tiers[tier].writable
                        && let Some(value) = &hit
                    {
                        // Best effort: a failed promotion must not fail the
                        // read. Entries the promotion displaces roll over
                        // into the next tier down.
                        match self.tiers[tier].tier.put(key.clone(), value.clone()).await {
                            Ok(displaced) => {
                                Counters::bump(&self.tiers[tier].counters.puts, 1);
                                if self.policy.demote_displaced && !displaced.is_empty() {
                                    let _evicted = self.demote(tier, displaced).await;
                                }
                            }
                            Err(_) => Counters::bump(&self.tiers[tier].counters.errors, 1),
                        }
                    }
                    flow.on_promote();
                }
                ReadStep::Done(outcome) => {
                    return match outcome {
                        ReadOutcome::Hit { .. } => Ok(hit),
                        ReadOutcome::Miss => Ok(None),
                        ReadOutcome::Inconclusive => Err(RouterError::Inconclusive(failures)),
                        ReadOutcome::Failed { .. } => Err(failures.pop().map_or_else(
                            || RouterError::Inconclusive(Vec::new()),
                            RouterError::Tier,
                        )),
                    };
                }
            }
        }
    }

    async fn exists(&self, key: &K) -> Result<bool, RouterError> {
        // Existence checks never promote; reuse the read flow for probe
        // order and error classification only.
        let mut flow = ReadFlow::new(
            self.tiers.len(),
            ReadPolicy {
                promote: Promote::Never,
                on_error: self.policy.read.on_error,
            },
        );
        let mut failures = Vec::new();
        loop {
            match flow.step() {
                ReadStep::Get { tier } => match self.tiers[tier].tier.exists(key).await {
                    Ok(true) => {
                        Counters::bump(&self.tiers[tier].counters.hits, 1);
                        flow.on_get(Probe::Hit);
                    }
                    Ok(false) => {
                        Counters::bump(&self.tiers[tier].counters.misses, 1);
                        flow.on_get(Probe::Miss);
                    }
                    Err(source) => {
                        Counters::bump(&self.tiers[tier].counters.errors, 1);
                        failures.push(self.failure(tier, source));
                        flow.on_get(Probe::Error);
                    }
                },
                // Unreachable under Promote::Never; kept total for safety.
                ReadStep::Promote { .. } => flow.on_promote(),
                ReadStep::Done(outcome) => {
                    return match outcome {
                        ReadOutcome::Hit { .. } => Ok(true),
                        ReadOutcome::Miss => Ok(false),
                        ReadOutcome::Inconclusive => Err(RouterError::Inconclusive(failures)),
                        ReadOutcome::Failed { .. } => Err(failures.pop().map_or_else(
                            || RouterError::Inconclusive(Vec::new()),
                            RouterError::Tier,
                        )),
                    };
                }
            }
        }
    }

    /// Trait-level batched read: delegates to [`Router::read_many`] and
    /// degrades to the trait's whole-batch granularity (any inconclusive
    /// key makes the whole batch inconclusive). Callers holding a concrete
    /// `Router` should prefer `read_many` for per-key statuses.
    async fn get_many(&self, keys: &[K]) -> Result<Vec<Option<V>>, RouterError> {
        let report = self.read_many(keys).await?;
        let inconclusive = report
            .statuses
            .iter()
            .any(|status| matches!(status, KeyStatus::Inconclusive));
        if inconclusive {
            Err(RouterError::Inconclusive(report.failures))
        } else {
            Ok(report.into_values())
        }
    }
}

impl<K, V> TierWrite for Router<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn put(&self, key: K, value: V) -> Result<Displaced<K, V>, RouterError> {
        match self.policy.write {
            WriteMode::WriteThrough => {
                if !self.tiers.iter().any(|slot| slot.writable) {
                    return Err(RouterError::ReadOnly);
                }
                let mut evicted = Displaced::new();
                // Bottom-up: lower tiers accept the key before upper tiers
                // reference it, so an aborted write never leaves an upper
                // tier claiming a key its backing tiers rejected. Read-only
                // tiers are skipped: the router never writes them.
                for tier in (0..self.tiers.len()).rev() {
                    if !self.tiers[tier].writable {
                        continue;
                    }
                    match self.tiers[tier].tier.put(key.clone(), value.clone()).await {
                        Ok(displaced) => {
                            Counters::bump(&self.tiers[tier].counters.puts, 1);
                            if displaced.is_empty() {
                                continue;
                            }
                            if self.policy.demote_displaced {
                                evicted.extend(self.demote(tier, displaced).await);
                            } else {
                                evicted.extend(displaced);
                            }
                        }
                        Err(source) => {
                            Counters::bump(&self.tiers[tier].counters.errors, 1);
                            match self.policy.on_write_error {
                                OnWriteError::FailFast => {
                                    return Err(RouterError::Tier(self.failure(tier, source)));
                                }
                                // A failed fill is a capacity loss, not an
                                // operation failure; the error is in stats.
                                OnWriteError::BestEffort => {}
                            }
                        }
                    }
                }
                Ok(evicted)
            }
            WriteMode::WriteAround => {
                let Some(bottom) = self.tiers.iter().rposition(|slot| slot.writable) else {
                    return Err(RouterError::ReadOnly);
                };
                let displaced = match self.tiers[bottom].tier.put(key.clone(), value).await {
                    Ok(displaced) => {
                        Counters::bump(&self.tiers[bottom].counters.puts, 1);
                        displaced
                    }
                    Err(source) => {
                        Counters::bump(&self.tiers[bottom].counters.errors, 1);
                        return Err(RouterError::Tier(self.failure(bottom, source)));
                    }
                };
                // Upper writable copies are now stale and would shadow the
                // new value; they must be invalidated, and a failed
                // invalidation must surface (it means reads can return the
                // old value). Read-only uppers were never written by the
                // router, so there is nothing to invalidate there.
                let mut failures = Vec::new();
                for tier in 0..bottom {
                    if !self.tiers[tier].writable {
                        continue;
                    }
                    match self.tiers[tier].tier.delete(&key).await {
                        Ok(_) => Counters::bump(&self.tiers[tier].counters.deletes, 1),
                        Err(source) => {
                            Counters::bump(&self.tiers[tier].counters.errors, 1);
                            failures.push(self.failure(tier, source));
                        }
                    }
                }
                if failures.is_empty() {
                    Ok(displaced)
                } else {
                    Err(RouterError::Partial(failures))
                }
            }
        }
    }

    async fn delete(&self, key: &K) -> Result<bool, RouterError> {
        // Attempt every writable tier even after failures: leaving a copy in
        // a lower tier because an upper one errored would guarantee
        // resurrection. Read-only tiers are untouched — a key present there
        // will be served again once local copies are gone (resurrection by
        // design for an origin the router does not own).
        let mut existed = false;
        let mut failures = Vec::new();
        for (index, slot) in self.tiers.iter().enumerate() {
            if !slot.writable {
                continue;
            }
            match slot.tier.delete(key).await {
                Ok(present) => {
                    Counters::bump(&slot.counters.deletes, 1);
                    existed |= present;
                }
                Err(source) => {
                    Counters::bump(&slot.counters.errors, 1);
                    failures.push(TierFailure::new(index, slot.tier.name(), source));
                }
            }
        }
        if failures.is_empty() {
            Ok(existed)
        } else {
            Err(RouterError::Partial(failures))
        }
    }

    /// Batched write with the same propagation semantics as
    /// [`TierWrite::put`]: write-through goes bottom-up per tier,
    /// write-around writes the bottom and invalidates above.
    async fn put_many(&self, entries: Vec<(K, V)>) -> Result<Displaced<K, V>, RouterError> {
        match self.policy.write {
            WriteMode::WriteThrough => {
                if !self.tiers.iter().any(|slot| slot.writable) {
                    return Err(RouterError::ReadOnly);
                }
                let mut evicted = Displaced::new();
                for tier in (0..self.tiers.len()).rev() {
                    if !self.tiers[tier].writable {
                        continue;
                    }
                    match self.tiers[tier].tier.put_many(entries.clone()).await {
                        Ok(displaced) => {
                            Counters::bump(&self.tiers[tier].counters.puts, entries.len() as u64);
                            if displaced.is_empty() {
                                continue;
                            }
                            if self.policy.demote_displaced {
                                evicted.extend(self.demote(tier, displaced).await);
                            } else {
                                evicted.extend(displaced);
                            }
                        }
                        Err(source) => {
                            Counters::bump(&self.tiers[tier].counters.errors, 1);
                            match self.policy.on_write_error {
                                OnWriteError::FailFast => {
                                    return Err(RouterError::Tier(self.failure(tier, source)));
                                }
                                OnWriteError::BestEffort => {}
                            }
                        }
                    }
                }
                Ok(evicted)
            }
            WriteMode::WriteAround => {
                let Some(bottom) = self.tiers.iter().rposition(|slot| slot.writable) else {
                    return Err(RouterError::ReadOnly);
                };
                let displaced = match self.tiers[bottom].tier.put_many(entries.clone()).await {
                    Ok(displaced) => {
                        Counters::bump(&self.tiers[bottom].counters.puts, entries.len() as u64);
                        displaced
                    }
                    Err(source) => {
                        Counters::bump(&self.tiers[bottom].counters.errors, 1);
                        return Err(RouterError::Tier(self.failure(bottom, source)));
                    }
                };
                let keys: Vec<K> = entries.into_iter().map(|(key, _)| key).collect();
                let mut failures = Vec::new();
                for tier in 0..bottom {
                    if !self.tiers[tier].writable {
                        continue;
                    }
                    match self.tiers[tier].tier.delete_many(&keys).await {
                        Ok(_) => {
                            Counters::bump(&self.tiers[tier].counters.deletes, keys.len() as u64);
                        }
                        Err(source) => {
                            Counters::bump(&self.tiers[tier].counters.errors, 1);
                            failures.push(self.failure(tier, source));
                        }
                    }
                }
                if failures.is_empty() {
                    Ok(displaced)
                } else {
                    Err(RouterError::Partial(failures))
                }
            }
        }
    }

    /// Trait-level batched delete: delegates to [`Router::remove_many`] and
    /// degrades any failure to the trait's whole-batch [`RouterError::Partial`].
    /// Callers holding a concrete `Router` should prefer `remove_many` for
    /// per-key flags alongside the failures.
    async fn delete_many(&self, keys: &[K]) -> Result<Vec<bool>, RouterError> {
        let report = self.remove_many(keys).await;
        if report.is_complete() {
            Ok(report.removed)
        } else {
            Err(RouterError::Partial(report.failures))
        }
    }
}

impl<K, V> RouterBuilder<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Appends `tier` below all previously added tiers (first added is the
    /// topmost/fastest).
    #[must_use]
    pub fn tier<T>(mut self, tier: T) -> Self
    where
        T: TierRead<Key = K, Value = V> + TierWrite + Send + Sync + 'static,
        T::Error: StdError + Send + Sync + 'static,
    {
        self.tiers.push(TierSlot {
            tier: Box::new(Adapter(tier)),
            writable: true,
            counters: Counters::default(),
        });
        self
    }

    /// Appends a tier the router reads from but never writes to: writes,
    /// demotions, promotions, and deletes all skip it — the lane for an
    /// origin this process does not own (an object store, a replicated
    /// snapshot). Only [`TierRead`] is required, so fetch-only backends fit
    /// without stub write methods.
    ///
    /// Note the delete caveat: a key present in a read-only tier will be
    /// served again once local copies are gone — resurrection by design.
    #[must_use]
    pub fn read_only_tier<T>(mut self, tier: T) -> Self
    where
        T: TierRead<Key = K, Value = V> + Send + Sync + 'static,
        T::Error: StdError + Send + Sync + 'static,
    {
        self.tiers.push(TierSlot {
            tier: Box::new(ReadOnly(tier)),
            writable: false,
            counters: Counters::default(),
        });
        self
    }

    /// Replaces the routing policy (defaults to [`Policy::default`]).
    #[must_use]
    pub const fn policy(mut self, policy: Policy) -> Self {
        self.policy = policy;
        self
    }

    /// Finishes the router.
    ///
    /// # Panics
    ///
    /// Panics if no tiers were added; a router over zero tiers is a
    /// configuration bug, not a runtime condition.
    #[must_use]
    pub fn build(self) -> Router<K, V> {
        assert!(!self.tiers.is_empty(), "a router needs at least one tier");
        Router {
            tiers: self.tiers,
            policy: self.policy,
        }
    }
}
