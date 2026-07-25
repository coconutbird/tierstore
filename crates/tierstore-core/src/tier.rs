//! Capability traits implemented by storage tiers.
//!
//! Capabilities are split into small traits so a backend only implements what
//! it supports. All methods take `&self`: tiers are meant to be shared across
//! tasks and use interior mutability (or be naturally stateless handles).
//!
//! Every returned future is required to be `Send`. That is a deliberate,
//! server-first contract — it lets routers box and drive tiers from
//! work-stealing executors. A `Local` (non-`Send`) variant is an open design
//! question.

use core::future::Future;
use core::ops::{Deref, Range};

#[cfg(feature = "alloc")]
use alloc::{sync::Arc, vec::Vec};

/// Entries displaced (evicted) by an insert, in eviction order.
///
/// Returned by [`TierWrite::put`] so that a router can *demote* them into the
/// next tier down instead of silently dropping them — the "rollover" in a
/// tiered rollover cache.
#[cfg(feature = "alloc")]
pub type Displaced<K, V> = Vec<(K, V)>;

/// Base trait naming the key, value, and error types shared by all
/// capabilities of a tier.
pub trait Tier {
    /// Key type this tier is addressed by.
    type Key;
    /// Value type this tier stores.
    type Value;
    /// Backend error type. Use [`core::convert::Infallible`] for tiers that
    /// cannot fail (e.g. plain in-memory maps).
    type Error;

    /// Short human-readable name for diagnostics and error reports.
    fn name(&self) -> &str
    where
        Self: Sized,
    {
        core::any::type_name::<Self>()
    }
}

/// Read capability: point lookups and existence checks.
pub trait TierRead: Tier {
    /// Looks up `key`.
    ///
    /// Absence is *data*, not an error: a missing key is `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns the backend error only when the lookup itself failed (I/O
    /// error, connection loss, …), i.e. when the tier cannot say whether the
    /// key exists.
    fn get(
        &self,
        key: &Self::Key,
    ) -> impl Future<Output = Result<Option<Self::Value>, Self::Error>> + Send;

    /// Reports whether `key` is present without necessarily retrieving it.
    ///
    /// This is separate from [`TierRead::get`] because many backends answer
    /// existence far more cheaply than retrieval (index probe, file metadata,
    /// `EXISTS` command). It must not be more stale than `get`.
    ///
    /// # Errors
    ///
    /// Returns the backend error when the check itself failed.
    fn exists(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send;

    /// Batched lookup: one slot per key, in the same order.
    ///
    /// Non-atomic: the default implementation loops over [`TierRead::get`]
    /// and aborts on the first backend error. Backends with a native batch
    /// primitive (`MGET`, `SELECT … IN`, a single lock pass) should override
    /// it.
    ///
    /// # Errors
    ///
    /// Returns the first backend error encountered; keys after it are not
    /// probed.
    #[cfg(feature = "alloc")]
    fn get_many(
        &self,
        keys: &[Self::Key],
    ) -> impl Future<Output = Result<Vec<Option<Self::Value>>, Self::Error>> + Send
    where
        Self: Sync,
        Self::Key: Sync,
        Self::Value: Send,
    {
        async move {
            let mut values = Vec::with_capacity(keys.len());
            for key in keys {
                values.push(self.get(key).await?);
            }
            Ok(values)
        }
    }
}

/// Zero-copy read capability: borrow the value in place instead of
/// materialising an owned copy.
///
/// The returned view is a guard: it keeps the entry pinned (typically by
/// holding the tier's lock) so eviction cannot invalidate the borrowed
/// bytes — that is the soundness contract implementors must uphold. Views
/// should therefore be short-lived; copy out for long processing, and note
/// that lock-backed views usually cannot be held across `await` points on
/// work-stealing executors.
///
/// This is a *direct-tier* capability. It does not flow through the
/// router: borrowed views cannot cross the router's type-erased tier
/// boundary without boxing, promotion/demotion inherently copy, and a
/// remote tier has no stable origin to borrow from. Reach for it on the
/// hot tier, where the memcpy actually shows up.
pub trait TierReadRef: Tier {
    /// Borrowed form of [`Tier::Value`] — e.g. `[u8]` when the value is
    /// `Vec<u8>`, or `Value` itself for in-memory stores.
    type Borrowed: ?Sized;

    /// Guard-like view pinning the entry while it exists.
    type ValueRef<'a>: Deref<Target = Self::Borrowed>
    where
        Self: 'a;

    /// Borrows `key`'s value in place.
    ///
    /// # Errors
    ///
    /// Returns the backend error when the lookup itself failed; absence is
    /// `Ok(None)`.
    fn get_ref<'s>(
        &'s self,
        key: &Self::Key,
    ) -> impl Future<Output = Result<Option<Self::ValueRef<'s>>, Self::Error>> + Send;
}

/// Ranged-read capability for byte-oriented tiers: serve a slice of a value
/// without materialising the whole thing.
///
/// This is the primitive for large-artifact stores (the `RandomRead` shape
/// from shardstore-style engines): an mmap tier serves a range as a
/// zero-copy slice, a file tier as one positional read, a remote tier as a
/// range request. The returned value is of the tier's normal `Value` type,
/// containing exactly the requested bytes.
///
/// Like [`TierReadRef`], this is a *direct-tier* capability — the router
/// does not route ranges. Chunk-granular caching through a router is the
/// same machinery with chunk keys (e.g. `(artifact, chunk_no)`), which
/// composes with promotion, rollover, and single-flight for free.
pub trait TierReadRange: Tier {
    /// Reads exactly `range` (absolute byte offsets) of `key`'s value.
    ///
    /// A missing key is `Ok(None)`. A range that does not lie fully within
    /// the value is a backend error, not a silent truncation.
    ///
    /// # Errors
    ///
    /// Returns the backend error when the read failed or the range is out
    /// of bounds.
    fn read_range(
        &self,
        key: &Self::Key,
        range: Range<u64>,
    ) -> impl Future<Output = Result<Option<Self::Value>, Self::Error>> + Send;
}

/// Write capability: inserts and deletes.
#[cfg(feature = "alloc")]
pub trait TierWrite: Tier {
    /// Inserts or replaces `key`.
    ///
    /// Returns the entries this insert displaced (evicted to make room), so a
    /// router can demote them down the hierarchy. Unbounded tiers and
    /// plain replacements return an empty list.
    ///
    /// # Errors
    ///
    /// Returns the backend error if the write failed; on error the tier's
    /// contents for `key` are unspecified (backends should strive for
    /// all-or-nothing writes).
    fn put(
        &self,
        key: Self::Key,
        value: Self::Value,
    ) -> impl Future<Output = Result<Displaced<Self::Key, Self::Value>, Self::Error>> + Send;

    /// Removes `key`, reporting whether it was present.
    ///
    /// # Errors
    ///
    /// Returns the backend error if the removal failed. Callers layering
    /// tiers must treat a failed delete as serious: a stale copy left in an
    /// upper tier can "resurrect" deleted data.
    fn delete(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send;

    /// Batched insert; aggregates displaced entries across the batch.
    ///
    /// Non-atomic: the default implementation loops over [`TierWrite::put`]
    /// and aborts on the first backend error — entries written before the
    /// failure stay written. Backends with a native batch write should
    /// override it.
    ///
    /// # Errors
    ///
    /// Returns the first backend error encountered.
    fn put_many(
        &self,
        entries: Vec<(Self::Key, Self::Value)>,
    ) -> impl Future<Output = Result<Displaced<Self::Key, Self::Value>, Self::Error>> + Send
    where
        Self: Sync,
        Self::Key: Send,
        Self::Value: Send,
    {
        async move {
            let mut displaced = Displaced::new();
            for (key, value) in entries {
                displaced.extend(self.put(key, value).await?);
            }
            Ok(displaced)
        }
    }

    /// Batched removal: one "was it present" flag per key, in the same
    /// order.
    ///
    /// Non-atomic; same contract as [`TierWrite::put_many`].
    ///
    /// # Errors
    ///
    /// Returns the first backend error encountered; keys after it are not
    /// touched.
    fn delete_many(
        &self,
        keys: &[Self::Key],
    ) -> impl Future<Output = Result<Vec<bool>, Self::Error>> + Send
    where
        Self: Sync,
        Self::Key: Sync,
    {
        async move {
            let mut removed = Vec::with_capacity(keys.len());
            for key in keys {
                removed.push(self.delete(key).await?);
            }
            Ok(removed)
        }
    }
}

/// One page of keys from [`TierList::list`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page<K, C> {
    /// Keys in this page, in the tier's listing order.
    pub keys: Vec<K>,
    /// Cursor for the next page, or `None` when the listing is exhausted.
    pub next: Option<C>,
}

/// Enumeration capability, paged by an opaque cursor.
///
/// Listing is paged because tiers can be arbitrarily large (a remote table);
/// an unbounded "give me every key" API is a footgun.
#[cfg(feature = "alloc")]
pub trait TierList: Tier {
    /// Opaque continuation token for paging.
    type Cursor;

    /// Returns up to `limit` keys, starting from `cursor` (or the beginning
    /// when `None`).
    ///
    /// Callers should pass `limit > 0`; a zero limit yields an empty page and
    /// need not make progress. Listings need not be stable under concurrent
    /// mutation.
    ///
    /// # Errors
    ///
    /// Returns the backend error if the listing failed.
    fn list(
        &self,
        cursor: Option<Self::Cursor>,
        limit: usize,
    ) -> impl Future<Output = Result<Page<Self::Key, Self::Cursor>, Self::Error>> + Send;
}

#[cfg(feature = "alloc")]
impl<T: Tier> Tier for Arc<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = T::Error;

    fn name(&self) -> &str {
        T::name(self)
    }
}

// The `T: Sync` bound exists so the batch methods can forward to `T`'s
// (possibly overridden) implementations instead of falling back to the
// looping defaults.
#[cfg(feature = "alloc")]
impl<T: TierRead + Sync> TierRead for Arc<T> {
    fn get(
        &self,
        key: &Self::Key,
    ) -> impl Future<Output = Result<Option<Self::Value>, Self::Error>> + Send {
        T::get(self, key)
    }

    fn exists(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        T::exists(self, key)
    }

    fn get_many(
        &self,
        keys: &[Self::Key],
    ) -> impl Future<Output = Result<Vec<Option<Self::Value>>, Self::Error>> + Send
    where
        Self: Sync,
        Self::Key: Sync,
        Self::Value: Send,
    {
        T::get_many(self, keys)
    }
}

#[cfg(feature = "alloc")]
impl<T: TierReadRef> TierReadRef for Arc<T> {
    type Borrowed = T::Borrowed;
    type ValueRef<'a>
        = T::ValueRef<'a>
    where
        Self: 'a;

    fn get_ref<'s>(
        &'s self,
        key: &Self::Key,
    ) -> impl Future<Output = Result<Option<Self::ValueRef<'s>>, Self::Error>> + Send {
        T::get_ref(self, key)
    }
}

#[cfg(feature = "alloc")]
impl<T: TierReadRange> TierReadRange for Arc<T> {
    fn read_range(
        &self,
        key: &Self::Key,
        range: Range<u64>,
    ) -> impl Future<Output = Result<Option<Self::Value>, Self::Error>> + Send {
        T::read_range(self, key, range)
    }
}

#[cfg(feature = "alloc")]
impl<T: TierWrite + Sync> TierWrite for Arc<T> {
    fn put(
        &self,
        key: Self::Key,
        value: Self::Value,
    ) -> impl Future<Output = Result<Displaced<Self::Key, Self::Value>, Self::Error>> + Send {
        T::put(self, key, value)
    }

    fn delete(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        T::delete(self, key)
    }

    fn put_many(
        &self,
        entries: Vec<(Self::Key, Self::Value)>,
    ) -> impl Future<Output = Result<Displaced<Self::Key, Self::Value>, Self::Error>> + Send
    where
        Self: Sync,
        Self::Key: Send,
        Self::Value: Send,
    {
        T::put_many(self, entries)
    }

    fn delete_many(
        &self,
        keys: &[Self::Key],
    ) -> impl Future<Output = Result<Vec<bool>, Self::Error>> + Send
    where
        Self: Sync,
        Self::Key: Sync,
    {
        T::delete_many(self, keys)
    }
}

#[cfg(feature = "alloc")]
impl<T: TierList> TierList for Arc<T> {
    type Cursor = T::Cursor;

    fn list(
        &self,
        cursor: Option<Self::Cursor>,
        limit: usize,
    ) -> impl Future<Output = Result<Page<Self::Key, Self::Cursor>, Self::Error>> + Send {
        T::list(self, cursor, limit)
    }
}
