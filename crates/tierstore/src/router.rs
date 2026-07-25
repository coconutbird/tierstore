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

use tierstore_core::{
    Displaced, OnReadError, Policy, Probe, Promote, ReadFlow, ReadOutcome, ReadPolicy, ReadStep,
    Tier, TierRead, TierWrite, WriteMode,
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
    tiers: Vec<Box<dyn DynTier<K, V>>>,
    policy: Policy,
}

/// Builder for [`Router`]; add tiers top-down.
pub struct RouterBuilder<K, V> {
    tiers: Vec<Box<dyn DynTier<K, V>>>,
    policy: Policy,
}

impl<K, V> fmt::Debug for Router<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.tiers.iter().map(|tier| tier.name()).collect();
        f.debug_struct("Router")
            .field("tiers", &names)
            .field("policy", &self.policy)
            .finish()
    }
}

impl<K, V> fmt::Debug for RouterBuilder<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.tiers.iter().map(|tier| tier.name()).collect();
        f.debug_struct("RouterBuilder")
            .field("tiers", &names)
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
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
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
            let mut next = Displaced::new();
            for (key, value) in current {
                if let Ok(mut displaced) = self.tiers[target].put(key, value).await {
                    next.append(&mut displaced);
                }
            }
            current = next;
            target += 1;
        }
        current
    }

    fn failure(&self, tier: usize, source: BoxError) -> TierFailure {
        TierFailure::new(tier, self.tiers[tier].name(), source)
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
                if let Ok(displaced) = self.tiers[target].put_many(entries.clone()).await
                    && self.policy.demote_displaced
                    && !displaced.is_empty()
                {
                    let _evicted = self.demote(target, displaced).await;
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
        for (tier_index, tier) in self.tiers.iter().enumerate() {
            if unresolved.is_empty() {
                break;
            }
            let subset: Vec<K> = unresolved
                .iter()
                .map(|&index| keys[index].clone())
                .collect();
            match tier.get_many(&subset).await {
                Ok(found) => {
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
                Err(source) => match self.policy.read.on_error {
                    OnReadError::FailFast => {
                        return Err(RouterError::Tier(self.failure(tier_index, source)));
                    }
                    OnReadError::FallThrough => {
                        failures.push(self.failure(tier_index, source));
                    }
                },
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

    /// Batched delete with per-key outcomes. Every tier is attempted for
    /// every key regardless of failures (skipping a tier guarantees
    /// resurrection); check [`DeleteReport::is_complete`] before trusting
    /// the flags.
    pub async fn remove_many(&self, keys: &[K]) -> DeleteReport {
        let mut removed = vec![false; keys.len()];
        let mut failures = Vec::new();
        for (index, tier) in self.tiers.iter().enumerate() {
            match tier.delete_many(keys).await {
                Ok(flags) => {
                    for (slot, flag) in removed.iter_mut().zip(flags) {
                        *slot |= flag;
                    }
                }
                Err(source) => failures.push(TierFailure::new(index, tier.name(), source)),
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
                ReadStep::Get { tier } => match self.tiers[tier].get(key).await {
                    Ok(Some(value)) => {
                        hit = Some(value);
                        flow.on_get(Probe::Hit);
                    }
                    Ok(None) => flow.on_get(Probe::Miss),
                    Err(source) => {
                        failures.push(self.failure(tier, source));
                        flow.on_get(Probe::Error);
                    }
                },
                ReadStep::Promote { tier } => {
                    if let Some(value) = &hit {
                        // Best effort: a failed promotion must not fail the
                        // read. Entries the promotion displaces roll over
                        // into the next tier down.
                        if let Ok(displaced) =
                            self.tiers[tier].put(key.clone(), value.clone()).await
                            && self.policy.demote_displaced
                            && !displaced.is_empty()
                        {
                            let _evicted = self.demote(tier, displaced).await;
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
                ReadStep::Get { tier } => match self.tiers[tier].exists(key).await {
                    Ok(true) => flow.on_get(Probe::Hit),
                    Ok(false) => flow.on_get(Probe::Miss),
                    Err(source) => {
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
                let mut evicted = Displaced::new();
                // Bottom-up: lower tiers accept the key before upper tiers
                // reference it, so an aborted write never leaves an upper
                // tier claiming a key its backing tiers rejected.
                for tier in (0..self.tiers.len()).rev() {
                    match self.tiers[tier].put(key.clone(), value.clone()).await {
                        Ok(displaced) => {
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
                            return Err(RouterError::Tier(self.failure(tier, source)));
                        }
                    }
                }
                Ok(evicted)
            }
            WriteMode::WriteAround => {
                let bottom = self.tiers.len() - 1;
                let displaced = match self.tiers[bottom].put(key.clone(), value).await {
                    Ok(displaced) => displaced,
                    Err(source) => return Err(RouterError::Tier(self.failure(bottom, source))),
                };
                // Upper copies are now stale and would shadow the new value;
                // they must be invalidated, and a failed invalidation must
                // surface (it means reads can return the old value).
                let mut failures = Vec::new();
                for tier in 0..bottom {
                    if let Err(source) = self.tiers[tier].delete(&key).await {
                        failures.push(self.failure(tier, source));
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
        // Attempt every tier even after failures: leaving a copy in a lower
        // tier because an upper one errored would guarantee resurrection.
        let mut existed = false;
        let mut failures = Vec::new();
        for (index, tier) in self.tiers.iter().enumerate() {
            match tier.delete(key).await {
                Ok(present) => existed |= present,
                Err(source) => {
                    failures.push(TierFailure::new(index, tier.name(), source));
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
                let mut evicted = Displaced::new();
                for tier in (0..self.tiers.len()).rev() {
                    match self.tiers[tier].put_many(entries.clone()).await {
                        Ok(displaced) => {
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
                            return Err(RouterError::Tier(self.failure(tier, source)));
                        }
                    }
                }
                Ok(evicted)
            }
            WriteMode::WriteAround => {
                let bottom = self.tiers.len() - 1;
                let displaced = match self.tiers[bottom].put_many(entries.clone()).await {
                    Ok(displaced) => displaced,
                    Err(source) => return Err(RouterError::Tier(self.failure(bottom, source))),
                };
                let keys: Vec<K> = entries.into_iter().map(|(key, _)| key).collect();
                let mut failures = Vec::new();
                for tier in 0..bottom {
                    if let Err(source) = self.tiers[tier].delete_many(&keys).await {
                        failures.push(self.failure(tier, source));
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
        self.tiers.push(Box::new(Adapter(tier)));
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
