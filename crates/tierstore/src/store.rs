//! An authoritative tiered store built **on top of** the generic router —
//! the counterpart to [`TieredCache`](crate::TieredCache).
//!
//! Same mechanism, opposite stance: a cache optimises availability (fall
//! through failures, refill later), a store answers for the data itself.
//! Concretely:
//!
//! - **Reads fail fast.** A failing tier is an error, not a detour —
//!   integrity over availability.
//! - **Loss is loud.** A write that pushes entries off the *bottom* tier
//!   shrank the store; [`TieredStore::put`] returns them in
//!   [`StoreError::Evicted`] — the caller's last chance to save the data.
//!   (Bounded upper tiers are fine: demotion rolls their overflow down
//!   losslessly.)
//! - **Nothing moves behind your back.** No promotion by default; data
//!   lives where it was written until you say otherwise.
//! - **No single-flight.** Fill coalescing is a cache optimisation, not a
//!   storage semantic.
//!
//! # Cache over store
//!
//! `TieredStore` implements the tier traits itself, so the blessed layering
//! for authority-backed systems is to make the store the *bottom tier of a
//! [`TieredCache`](crate::TieredCache)*: cache tiers stay lenient and
//! availability-flavoured above, while reads that reach the store are
//! governed by its own (fail-fast) policy inside. Note the trait face is
//! *mechanical*: as a tier, `put` returns displaced entries per the
//! rollover contract (an enclosing router needs them); the loud
//! [`StoreError::Evicted`] stance lives on the inherent API for direct
//! users.

use std::error::Error as StdError;
use std::fmt;

use tierstore_core::{
    Displaced, OnReadError, Policy, Promote, ReadPolicy, Tier, TierRead, TierWrite, WriteMode,
};

use crate::error::RouterError;
use crate::report::{DeleteReport, ReadReport};
use crate::router::{Router, RouterBuilder};

/// Authoritative tiered store over an ordered tier stack.
///
/// # Example
///
/// ```
/// use tierstore::{MemoryTier, TieredStore};
///
/// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let store: TieredStore<String, String> = TieredStore::builder()
///     .tier(MemoryTier::unbounded())
///     .build();
/// store.put("k".to_owned(), "v".to_owned()).await?;
/// assert_eq!(store.get(&"k".to_owned()).await?, Some("v".to_owned()));
/// # Ok(()) }
/// ```
pub struct TieredStore<K, V> {
    router: Router<K, V>,
}

/// Builder for [`TieredStore`]; add tiers top-down (fastest first).
pub struct TieredStoreBuilder<K, V> {
    router: RouterBuilder<K, V>,
}

impl<K, V> fmt::Debug for TieredStore<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TieredStore")
            .field("router", &self.router)
            .finish()
    }
}

impl<K, V> fmt::Debug for TieredStoreBuilder<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TieredStoreBuilder")
            .field("router", &self.router)
            .finish()
    }
}

/// Error from a [`TieredStore`] operation.
#[derive(Debug)]
pub enum StoreError<K, V> {
    /// The underlying router failed (tier error, inconclusive read, partial
    /// delete).
    Router(RouterError),
    /// A write displaced these entries off the bottommost tier: for a store
    /// that is data loss, not eviction. They are returned so the caller can
    /// still persist them elsewhere.
    Evicted(Displaced<K, V>),
}

impl<K, V> From<RouterError> for StoreError<K, V> {
    fn from(error: RouterError) -> Self {
        Self::Router(error)
    }
}

impl<K, V> fmt::Display for StoreError<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Router(error) => write!(f, "{error}"),
            Self::Evicted(entries) => write!(
                f,
                "write displaced {} entr{} off the bottom tier — data loss for a store",
                entries.len(),
                if entries.len() == 1 { "y" } else { "ies" }
            ),
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> StdError for StoreError<K, V> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Router(error) => Some(error),
            Self::Evicted(_) => None,
        }
    }
}

impl<K, V> TieredStore<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Starts building a store with [`TieredStore::default_policy`].
    #[must_use]
    pub fn builder() -> TieredStoreBuilder<K, V> {
        TieredStoreBuilder {
            router: Router::builder().policy(Self::default_policy()),
        }
    }

    /// The store preset: fail-fast reads, no promotion, write-through, and
    /// demotion so bounded upper tiers overflow losslessly downward.
    #[must_use]
    pub const fn default_policy() -> Policy {
        Policy {
            read: ReadPolicy {
                promote: Promote::Never,
                on_error: OnReadError::FailFast,
            },
            write: WriteMode::WriteThrough,
            demote_displaced: true,
        }
    }

    /// Reads through the hierarchy. A miss is authoritative: `Ok(None)`
    /// means the store does not hold the key.
    ///
    /// # Errors
    ///
    /// [`StoreError::Router`] when a tier fails — under the fail-fast
    /// preset the read aborts rather than serving around a broken tier.
    pub async fn get(&self, key: &K) -> Result<Option<V>, StoreError<K, V>> {
        Ok(self.router.get(key).await?)
    }

    /// Batched read with per-key outcomes (hits carry tier provenance).
    ///
    /// # Errors
    ///
    /// Same classification as [`TieredStore::get`].
    pub async fn get_many(&self, keys: &[K]) -> Result<ReadReport<V>, StoreError<K, V>> {
        Ok(self.router.read_many(keys).await?)
    }

    /// Checks the hierarchy for `key`.
    ///
    /// # Errors
    ///
    /// Same classification as [`TieredStore::get`].
    pub async fn contains(&self, key: &K) -> Result<bool, StoreError<K, V>> {
        Ok(self.router.exists(key).await?)
    }

    /// Writes through every tier. Succeeds only if the data is fully
    /// retained: entries displaced off the bottom tier are returned as
    /// [`StoreError::Evicted`] instead of being silently dropped.
    ///
    /// # Errors
    ///
    /// [`StoreError::Router`] if a tier rejected the write,
    /// [`StoreError::Evicted`] if the hierarchy could not retain everything.
    pub async fn put(&self, key: K, value: V) -> Result<(), StoreError<K, V>> {
        let displaced = self.router.put(key, value).await?;
        if displaced.is_empty() {
            Ok(())
        } else {
            Err(StoreError::Evicted(displaced))
        }
    }

    /// Batched write-through with the same loss stance as
    /// [`TieredStore::put`].
    ///
    /// # Errors
    ///
    /// Same classification as [`TieredStore::put`].
    pub async fn put_many(&self, entries: Vec<(K, V)>) -> Result<(), StoreError<K, V>> {
        let displaced = self.router.put_many(entries).await?;
        if displaced.is_empty() {
            Ok(())
        } else {
            Err(StoreError::Evicted(displaced))
        }
    }

    /// Removes `key` from every tier, reporting whether it existed.
    ///
    /// # Errors
    ///
    /// [`StoreError::Router`] with [`RouterError::Partial`] if any tier
    /// failed to delete — a surviving copy could resurrect the key.
    pub async fn delete(&self, key: &K) -> Result<bool, StoreError<K, V>> {
        Ok(self.router.delete(key).await?)
    }

    /// Batched delete with per-key outcomes; failures are in the report.
    pub async fn delete_many(&self, keys: &[K]) -> DeleteReport {
        self.router.remove_many(keys).await
    }

    /// The underlying router, for anything the store API does not expose.
    #[must_use]
    pub const fn router(&self) -> &Router<K, V> {
        &self.router
    }
}

impl<K, V> Tier for TieredStore<K, V> {
    type Key = K;
    type Value = V;
    type Error = RouterError;

    fn name(&self) -> &'static str {
        "store"
    }
}

impl<K, V> TierRead for TieredStore<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Result<Option<V>, RouterError> {
        self.router.get(key).await
    }

    async fn exists(&self, key: &K) -> Result<bool, RouterError> {
        self.router.exists(key).await
    }

    async fn get_many(&self, keys: &[K]) -> Result<Vec<Option<V>>, RouterError> {
        self.router.get_many(keys).await
    }
}

impl<K, V> TierWrite for TieredStore<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn put(&self, key: K, value: V) -> Result<Displaced<K, V>, RouterError> {
        self.router.put(key, value).await
    }

    async fn delete(&self, key: &K) -> Result<bool, RouterError> {
        self.router.delete(key).await
    }

    async fn put_many(&self, entries: Vec<(K, V)>) -> Result<Displaced<K, V>, RouterError> {
        self.router.put_many(entries).await
    }

    async fn delete_many(&self, keys: &[K]) -> Result<Vec<bool>, RouterError> {
        self.router.delete_many(keys).await
    }
}

impl<K, V> TieredStoreBuilder<K, V>
where
    K: Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Appends `tier` below all previously added tiers (first added is the
    /// fastest).
    #[must_use]
    pub fn tier<T>(mut self, tier: T) -> Self
    where
        T: TierRead<Key = K, Value = V> + TierWrite + Send + Sync + 'static,
        T::Error: StdError + Send + Sync + 'static,
    {
        self.router = self.router.tier(tier);
        self
    }

    /// Overrides the store policy.
    #[must_use]
    pub fn policy(mut self, policy: Policy) -> Self {
        self.router = self.router.policy(policy);
        self
    }

    /// Finishes the store.
    ///
    /// # Panics
    ///
    /// Panics if no tiers were added.
    #[must_use]
    pub fn build(self) -> TieredStore<K, V> {
        TieredStore {
            router: self.router.build(),
        }
    }
}
