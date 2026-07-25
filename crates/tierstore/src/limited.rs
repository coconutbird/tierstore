//! Concurrency-limiting middleware tier.
//!
//! Wrap a tier in [`LimitedTier`] to cap how many operations run against it
//! at once — the admission-control pattern in front of shardstore-style
//! origins. Two things this buys:
//!
//! - **Origin protection:** a remote store sees at most `limit` in-flight
//!   requests from this process, no matter how many callers fan in.
//! - **Transient-memory bounding:** fill memory is at most roughly
//!   `limit × value size` instead of `callers × value size`. (Same-key
//!   duplication is [`TieredCache`](crate::TieredCache)'s single-flight
//!   job; this bounds *distinct-key* fan-in.)
//!
//! The permit is held for the full operation, batches count as one
//! operation (they are one origin round-trip), and admission is unfair
//! (released waiters re-race) — adequate until proven otherwise.

use std::collections::VecDeque;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierWrite};

/// Middleware tier that admits at most `limit` concurrent operations to its
/// inner tier.
///
/// # Example
///
/// ```
/// use std::num::NonZeroUsize;
/// use tierstore::{LimitedTier, MemoryTier};
///
/// let origin: MemoryTier<String, Vec<u8>> = MemoryTier::unbounded();
/// let limited = LimitedTier::new(origin, NonZeroUsize::new(8).unwrap());
/// ```
pub struct LimitedTier<T> {
    inner: T,
    name: String,
    permits: Semaphore,
}

impl<T: Tier> LimitedTier<T> {
    /// Wraps `inner`, admitting at most `limit` concurrent operations.
    pub fn new(inner: T, limit: NonZeroUsize) -> Self {
        let name = format!("limited({})", inner.name());
        Self {
            inner,
            name,
            permits: Semaphore::new(limit.get()),
        }
    }
}

impl<T: Tier> std::fmt::Debug for LimitedTier<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LimitedTier")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

struct Semaphore {
    state: Mutex<SemState>,
}

struct SemState {
    available: usize,
    wakers: VecDeque<Waker>,
}

impl SemState {
    fn try_acquire(&mut self, waker: &Waker) -> bool {
        if self.available > 0 {
            self.available -= 1;
            true
        } else {
            if !self.wakers.iter().any(|w| w.will_wake(waker)) {
                self.wakers.push_back(waker.clone());
            }
            false
        }
    }

    /// Returns a permit and the wakers to notify; waking all lets waiters
    /// re-race, which cannot lose wakeups.
    fn release(&mut self) -> Vec<Waker> {
        self.available += 1;
        std::mem::take(&mut self.wakers).into()
    }
}

impl Semaphore {
    const fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(SemState {
                available: limit,
                wakers: VecDeque::new(),
            }),
        }
    }

    async fn acquire(&self) -> Permit<'_> {
        AcquirePermit { sem: self }.await;
        Permit { sem: self }
    }
}

struct AcquirePermit<'a> {
    sem: &'a Semaphore,
}

impl Future for AcquirePermit<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if lock(&self.sem.state).try_acquire(cx.waker()) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

struct Permit<'a> {
    sem: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        // Bind before waking so woken waiters do not contend on the state
        // lock we would otherwise still hold.
        let wakers = lock(&self.sem.state).release();
        for waker in wakers {
            waker.wake();
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<T: Tier> Tier for LimitedTier<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = T::Error;

    fn name(&self) -> &str {
        &self.name
    }
}

impl<T> TierRead for LimitedTier<T>
where
    T: TierRead + Sync,
    T::Key: Sync,
    T::Value: Send,
{
    async fn get(&self, key: &T::Key) -> Result<Option<T::Value>, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.get(key).await;
        drop(permit);
        result
    }

    async fn exists(&self, key: &T::Key) -> Result<bool, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.exists(key).await;
        drop(permit);
        result
    }

    async fn get_many(&self, keys: &[T::Key]) -> Result<Vec<Option<T::Value>>, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.get_many(keys).await;
        drop(permit);
        result
    }
}

impl<T> TierWrite for LimitedTier<T>
where
    T: TierWrite + Sync,
    T::Key: Send + Sync,
    T::Value: Send,
{
    async fn put(
        &self,
        key: T::Key,
        value: T::Value,
    ) -> Result<Displaced<T::Key, T::Value>, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.put(key, value).await;
        drop(permit);
        result
    }

    async fn delete(&self, key: &T::Key) -> Result<bool, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.delete(key).await;
        drop(permit);
        result
    }

    async fn put_many(
        &self,
        entries: Vec<(T::Key, T::Value)>,
    ) -> Result<Displaced<T::Key, T::Value>, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.put_many(entries).await;
        drop(permit);
        result
    }

    async fn delete_many(&self, keys: &[T::Key]) -> Result<Vec<bool>, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.delete_many(keys).await;
        drop(permit);
        result
    }
}

impl<T> TierList for LimitedTier<T>
where
    T: TierList + Sync,
    T::Cursor: Send,
{
    type Cursor = T::Cursor;

    async fn list(
        &self,
        cursor: Option<T::Cursor>,
        limit: usize,
    ) -> Result<Page<T::Key, T::Cursor>, T::Error> {
        let permit = self.permits.acquire().await;
        let result = self.inner.list(cursor, limit).await;
        drop(permit);
        result
    }
}
