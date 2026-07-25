//! A tiered read-through cache built **on top of** the generic router.
//!
//! This layer is deliberately thin: [`Router`](crate::Router) is the
//! mechanism, [`Policy`] the semantics, and the cache contributes
//! cache-appropriate defaults — exclusive promotion
//! ([`Promote::TopOnly`]) plus demotion-on-evict, i.e. the classic
//! *rollover* hierarchy where an entry evicted from hot rolls into warm
//! instead of vanishing.
//!
//! Stampede protection is built in: concurrent `get`s for the same key
//! coalesce into one fill (per-key single-flight, adopted from shardstore's
//! `ArtifactCache`). Cache-only concerns still to come: per-tier
//! TTL/staleness and negative caching.

use std::fmt;
use std::hash::Hash;

use tierstore_core::{
    Displaced, OnReadError, OnWriteError, Policy, Promote, ReadPolicy, TierRead, TierWrite,
    WriteMode,
};

use crate::error::RouterError;
use crate::report::{DeleteReport, ReadReport};
use crate::router::{Router, RouterBuilder};
use crate::single_flight::SingleFlight;

/// Read-through, write-through rollover cache over an ordered tier stack —
/// e.g. hot ([`MemoryTier`](crate::MemoryTier)) over warm
/// ([`DiskTier`](crate::DiskTier)) over cold (a remote store).
pub struct TieredCache<K, V> {
    router: Router<K, V>,
    gates: SingleFlight<K>,
    single_flight: bool,
}

/// Builder for [`TieredCache`]; add tiers top-down (hot first).
pub struct TieredCacheBuilder<K, V> {
    router: RouterBuilder<K, V>,
    single_flight: bool,
}

impl<K, V> fmt::Debug for TieredCache<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TieredCache")
            .field("router", &self.router)
            .field("single_flight", &self.single_flight)
            .finish_non_exhaustive()
    }
}

impl<K, V> fmt::Debug for TieredCacheBuilder<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TieredCacheBuilder")
            .field("router", &self.router)
            .field("single_flight", &self.single_flight)
            .finish()
    }
}

impl<K, V> TieredCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Starts building a cache with [`TieredCache::default_policy`].
    #[must_use]
    pub fn builder() -> TieredCacheBuilder<K, V> {
        TieredCacheBuilder {
            router: Router::builder().policy(Self::default_policy()),
            single_flight: true,
        }
    }

    /// The cache preset: exclusive promotion (each entry lives in roughly
    /// one cache tier, maximising combined capacity), rollover on eviction,
    /// fall-through on tier read errors, and best-effort write-through —
    /// a tier that rejects a fill is skipped, never blocking the others.
    #[must_use]
    pub const fn default_policy() -> Policy {
        Policy {
            read: ReadPolicy {
                promote: Promote::TopOnly,
                on_error: OnReadError::FallThrough,
            },
            write: WriteMode::WriteThrough,
            on_write_error: OnWriteError::BestEffort,
            demote_displaced: true,
        }
    }

    /// Reads through the hierarchy, promoting the hit per policy.
    ///
    /// With single-flight enabled (the default), concurrent `get`s for the
    /// same key coalesce: one caller performs the read-through and
    /// promotion, the rest wait and are then served from the tier the value
    /// was promoted into. This trades a brief same-key serialization (the
    /// gate is held for hot hits too) for never stampeding the cold tier;
    /// disable via [`TieredCacheBuilder::single_flight`] if same-key hot
    /// throughput matters more than fill deduplication.
    ///
    /// # Errors
    ///
    /// [`RouterError::Inconclusive`] when there was no hit and a tier
    /// failed (absence unconfirmed), or [`RouterError::Tier`] under
    /// fail-fast policies.
    pub async fn get(&self, key: &K) -> Result<Option<V>, RouterError> {
        if !self.single_flight {
            return self.router.get(key).await;
        }
        let gate = self.gates.acquire(key.clone()).await;
        let result = self.router.get(key).await;
        // Held through the fill on purpose: waiters must observe the
        // promoted value, not race the origin fetch.
        drop(gate);
        result
    }

    /// Checks the hierarchy for `key` without promoting.
    ///
    /// # Errors
    ///
    /// Same classification as [`TieredCache::get`].
    pub async fn contains(&self, key: &K) -> Result<bool, RouterError> {
        self.router.exists(key).await
    }

    /// Writes through the hierarchy. Returns entries this write pushed out
    /// of the store entirely (displaced off the bottommost tier).
    ///
    /// # Errors
    ///
    /// [`RouterError::Tier`] if a tier rejected the write (lower tiers keep
    /// what they already accepted), or [`RouterError::Partial`] for
    /// write-around invalidation failures.
    pub async fn put(&self, key: K, value: V) -> Result<Displaced<K, V>, RouterError> {
        self.router.put(key, value).await
    }

    /// Removes `key` from every tier.
    ///
    /// # Errors
    ///
    /// [`RouterError::Partial`] if any tier failed to delete — serious,
    /// because a surviving copy can resurrect the key.
    pub async fn invalidate(&self, key: &K) -> Result<bool, RouterError> {
        self.router.delete(key).await
    }

    /// Batched read-through with per-key outcomes: hits carry the tier that
    /// served them, misses are confirmed, and keys that could not be
    /// resolved past a failing tier are marked inconclusive — resolved
    /// values are never discarded because an unrelated key failed.
    ///
    /// Lower tiers are probed only with the keys still missing; hits are
    /// promoted per policy. Batches are **not** single-flighted (per-key
    /// gating across overlapping batches would need deadlock-free ordered
    /// acquisition; an open question).
    ///
    /// # Errors
    ///
    /// Only under fail-fast read policies (the cache default falls
    /// through).
    pub async fn get_many(&self, keys: &[K]) -> Result<ReadReport<V>, RouterError> {
        self.router.read_many(keys).await
    }

    /// Batched write-through; returns everything the batch pushed out of the
    /// store entirely.
    ///
    /// # Errors
    ///
    /// Same classification as [`TieredCache::put`].
    pub async fn put_many(&self, entries: Vec<(K, V)>) -> Result<Displaced<K, V>, RouterError> {
        self.router.put_many(entries).await
    }

    /// Batched invalidation with per-key outcomes: one "was it present
    /// anywhere" flag per key, plus any tier failures. A failure means the
    /// key may survive in that tier and resurrect later — check
    /// [`DeleteReport::is_complete`].
    pub async fn invalidate_many(&self, keys: &[K]) -> DeleteReport {
        self.router.remove_many(keys).await
    }

    /// Per-tier operation counters (hits, misses, errors, puts, deletes).
    #[must_use]
    pub fn stats(&self) -> Vec<crate::TierStats> {
        self.router.stats()
    }

    /// The underlying router, for anything the cache API does not expose.
    #[must_use]
    pub const fn router(&self) -> &Router<K, V> {
        &self.router
    }
}

impl<K, V> TieredCacheBuilder<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Appends `tier` below all previously added tiers (first added is the
    /// hottest).
    #[must_use]
    pub fn tier<T>(mut self, tier: T) -> Self
    where
        T: TierRead<Key = K, Value = V> + TierWrite + Send + Sync + 'static,
        T::Error: std::error::Error + Send + Sync + 'static,
    {
        self.router = self.router.tier(tier);
        self
    }

    /// Appends a read-only tier — the lane for an origin the cache fills
    /// from but never writes (an object store, a fetch-only service). Puts
    /// populate the writable cache layers only, and invalidation clears
    /// local copies while the origin re-serves the key afterwards.
    #[must_use]
    pub fn read_only_tier<T>(mut self, tier: T) -> Self
    where
        T: TierRead<Key = K, Value = V> + Send + Sync + 'static,
        T::Error: std::error::Error + Send + Sync + 'static,
    {
        self.router = self.router.read_only_tier(tier);
        self
    }

    /// Overrides the cache policy.
    #[must_use]
    pub fn policy(mut self, policy: Policy) -> Self {
        self.router = self.router.policy(policy);
        self
    }

    /// Enables or disables per-key single-flight on `get` (default:
    /// enabled). See [`TieredCache::get`] for the trade-off.
    #[must_use]
    pub const fn single_flight(mut self, enabled: bool) -> Self {
        self.single_flight = enabled;
        self
    }

    /// Finishes the cache.
    ///
    /// # Panics
    ///
    /// Panics if no tiers were added.
    #[must_use]
    pub fn build(self) -> TieredCache<K, V> {
        TieredCache {
            router: self.router.build(),
            gates: SingleFlight::new(),
            single_flight: self.single_flight,
        }
    }
}
