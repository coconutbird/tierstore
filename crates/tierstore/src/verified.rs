//! Trust-boundary verification middleware.
//!
//! Adopted from shardstore's store→node contract: bytes crossing from an
//! untrusted origin are verified **once at the boundary**, and a corrupt
//! value is a typed error, never a wrong answer later. Wrap the untrusted
//! tier (usually the remote/cold one) in [`VerifiedTier`]; a rejected value
//! surfaces as that tier *failing*, so the router's fall-through and
//! inconclusive-miss machinery applies unchanged — and since a failed
//! verification is never a hit, corrupt data is never returned *and never
//! promoted* into upper tiers.
//!
//! Writes and existence checks pass through unverified: verification guards
//! the direction in which trust is lost (reads from the origin), not data
//! this process produced.

use std::error::Error as StdError;
use std::fmt;

use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierWrite};

use crate::error::BoxError;

type Check<K, V> = dyn Fn(&K, &V) -> Result<(), BoxError> + Send + Sync;

/// Middleware tier that verifies every value served by its inner tier.
///
/// The check runs on each `get`/`get_many` hit before the value is released
/// to the caller. Keep it cheap (checksum, length, magic bytes) — it is on
/// the read path.
///
/// # Example
///
/// ```
/// use tierstore::{MemoryTier, VerifiedTier};
///
/// let origin: MemoryTier<String, Vec<u8>> = MemoryTier::unbounded();
/// let verified = VerifiedTier::new(origin, |_key, value| {
///     if value.is_empty() {
///         Err("empty value".into())
///     } else {
///         Ok(())
///     }
/// });
/// ```
pub struct VerifiedTier<T: Tier> {
    inner: T,
    name: String,
    check: Box<Check<T::Key, T::Value>>,
}

impl<T: Tier> VerifiedTier<T> {
    /// Wraps `inner` so every value it serves must pass `check`.
    pub fn new(
        inner: T,
        check: impl Fn(&T::Key, &T::Value) -> Result<(), BoxError> + Send + Sync + 'static,
    ) -> Self {
        let name = format!("verified({})", inner.name());
        Self {
            inner,
            name,
            check: Box::new(check),
        }
    }
}

impl<T: Tier> fmt::Debug for VerifiedTier<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifiedTier")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// Error from a [`VerifiedTier`]: the backend failed, or it answered with a
/// value that failed verification.
#[derive(Debug)]
pub enum VerifyError<E> {
    /// The inner tier itself failed.
    Backend(E),
    /// The inner tier answered, but the value was rejected by the check.
    Corrupt(BoxError),
}

impl<E: fmt::Display> fmt::Display for VerifyError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(f, "backend error: {error}"),
            Self::Corrupt(error) => write!(f, "verification rejected value: {error}"),
        }
    }
}

impl<E: StdError + 'static> StdError for VerifyError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Corrupt(error) => {
                let source: &(dyn StdError + 'static) = &**error;
                Some(source)
            }
        }
    }
}

impl<T: Tier> Tier for VerifiedTier<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = VerifyError<T::Error>;

    fn name(&self) -> &str {
        &self.name
    }
}

impl<T> TierRead for VerifiedTier<T>
where
    T: TierRead + Sync,
    T::Key: Sync,
    T::Value: Send,
{
    async fn get(&self, key: &T::Key) -> Result<Option<T::Value>, VerifyError<T::Error>> {
        match self.inner.get(key).await.map_err(VerifyError::Backend)? {
            Some(value) => {
                (self.check)(key, &value).map_err(VerifyError::Corrupt)?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    async fn exists(&self, key: &T::Key) -> Result<bool, VerifyError<T::Error>> {
        self.inner.exists(key).await.map_err(VerifyError::Backend)
    }

    /// Verifies every hit in the batch; one rejected value fails the batch
    /// at this tier (tier-level granularity, matching the trait contract).
    async fn get_many(
        &self,
        keys: &[T::Key],
    ) -> Result<Vec<Option<T::Value>>, VerifyError<T::Error>> {
        let values = self
            .inner
            .get_many(keys)
            .await
            .map_err(VerifyError::Backend)?;
        for (key, value) in keys.iter().zip(&values) {
            if let Some(value) = value {
                (self.check)(key, value).map_err(VerifyError::Corrupt)?;
            }
        }
        Ok(values)
    }
}

impl<T> TierWrite for VerifiedTier<T>
where
    T: TierWrite + Sync,
    T::Key: Send + Sync,
    T::Value: Send,
{
    async fn put(
        &self,
        key: T::Key,
        value: T::Value,
    ) -> Result<Displaced<T::Key, T::Value>, VerifyError<T::Error>> {
        self.inner
            .put(key, value)
            .await
            .map_err(VerifyError::Backend)
    }

    async fn delete(&self, key: &T::Key) -> Result<bool, VerifyError<T::Error>> {
        self.inner.delete(key).await.map_err(VerifyError::Backend)
    }

    async fn put_many(
        &self,
        entries: Vec<(T::Key, T::Value)>,
    ) -> Result<Displaced<T::Key, T::Value>, VerifyError<T::Error>> {
        self.inner
            .put_many(entries)
            .await
            .map_err(VerifyError::Backend)
    }

    async fn delete_many(&self, keys: &[T::Key]) -> Result<Vec<bool>, VerifyError<T::Error>> {
        self.inner
            .delete_many(keys)
            .await
            .map_err(VerifyError::Backend)
    }
}

impl<T> TierList for VerifiedTier<T>
where
    T: TierList + Sync,
    T::Cursor: Send,
{
    type Cursor = T::Cursor;

    async fn list(
        &self,
        cursor: Option<T::Cursor>,
        limit: usize,
    ) -> Result<Page<T::Key, T::Cursor>, VerifyError<T::Error>> {
        self.inner
            .list(cursor, limit)
            .await
            .map_err(VerifyError::Backend)
    }
}
