//! Shared test helpers: a dependency-free executor and instrumented tiers.
#![allow(
    dead_code,
    unreachable_pub,
    reason = "shared across test binaries; not every test file uses every helper"
)]

use std::future::Future;
use std::io;
use std::marker::PhantomData;
use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use tierstore::{Displaced, Page, Tier, TierList, TierRead, TierWrite};

/// Minimal executor: every future in this test suite is either ready or
/// spin-poll-able, so a noop waker suffices.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// A tier whose every operation fails, for exercising error policies.
#[derive(Debug, Default)]
pub struct FailingTier<K, V>(PhantomData<fn() -> (K, V)>);

impl<K, V> Tier for FailingTier<K, V> {
    type Key = K;
    type Value = V;
    type Error = io::Error;

    fn name(&self) -> &'static str {
        "failing"
    }
}

impl<K: Sync, V: Send + Sync> TierRead for FailingTier<K, V> {
    async fn get(&self, _key: &K) -> io::Result<Option<V>> {
        Err(io::Error::other("failing tier: get"))
    }

    async fn exists(&self, _key: &K) -> io::Result<bool> {
        Err(io::Error::other("failing tier: exists"))
    }
}

impl<K: Send + Sync, V: Send + Sync> TierWrite for FailingTier<K, V> {
    async fn put(&self, _key: K, _value: V) -> io::Result<Displaced<K, V>> {
        Err(io::Error::other("failing tier: put"))
    }

    async fn delete(&self, _key: &K) -> io::Result<bool> {
        Err(io::Error::other("failing tier: delete"))
    }
}

/// Wraps a tier and counts `get` calls — stands in for an expensive remote
/// fetch. Also demonstrates middleware-style tier composition.
#[derive(Debug)]
pub struct CountingTier<T> {
    inner: T,
    gets: AtomicUsize,
}

impl<T> CountingTier<T> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            gets: AtomicUsize::new(0),
        }
    }

    /// Number of `get` calls that reached this tier.
    pub fn gets(&self) -> usize {
        self.gets.load(Ordering::Relaxed)
    }
}

impl<T: Tier> Tier for CountingTier<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = T::Error;

    fn name(&self) -> &'static str {
        "counting"
    }
}

impl<T: TierRead> TierRead for CountingTier<T> {
    fn get(
        &self,
        key: &Self::Key,
    ) -> impl Future<Output = Result<Option<Self::Value>, Self::Error>> + Send {
        self.gets.fetch_add(1, Ordering::Relaxed);
        self.inner.get(key)
    }

    fn exists(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        self.inner.exists(key)
    }
}

impl<T: TierWrite> TierWrite for CountingTier<T> {
    fn put(
        &self,
        key: Self::Key,
        value: Self::Value,
    ) -> impl Future<Output = Result<Displaced<Self::Key, Self::Value>, Self::Error>> + Send {
        self.inner.put(key, value)
    }

    fn delete(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        self.inner.delete(key)
    }
}

impl<T: TierList> TierList for CountingTier<T> {
    type Cursor = T::Cursor;

    fn list(
        &self,
        cursor: Option<Self::Cursor>,
        limit: usize,
    ) -> impl Future<Output = Result<Page<Self::Key, Self::Cursor>, Self::Error>> + Send {
        self.inner.list(cursor, limit)
    }
}

/// Wraps a tier and sleeps before every `get` (at call time, blocking the
/// calling thread), so tests running on separate OS threads can overlap
/// concurrent misses deterministically. Writes are not slowed.
#[derive(Debug)]
pub struct SlowTier<T> {
    inner: T,
    delay: Duration,
}

impl<T> SlowTier<T> {
    pub const fn new(inner: T, delay: Duration) -> Self {
        Self { inner, delay }
    }
}

impl<T: Tier> Tier for SlowTier<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = T::Error;

    fn name(&self) -> &'static str {
        "slow"
    }
}

impl<T: TierRead> TierRead for SlowTier<T> {
    fn get(
        &self,
        key: &Self::Key,
    ) -> impl Future<Output = Result<Option<Self::Value>, Self::Error>> + Send {
        std::thread::sleep(self.delay);
        self.inner.get(key)
    }

    fn exists(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        self.inner.exists(key)
    }
}

/// Wraps a tier and records the peak number of concurrent `get`s in flight,
/// so tests can prove admission limits.
#[derive(Debug)]
pub struct ConcurrencyProbe<T> {
    inner: T,
    current: AtomicUsize,
    peak: AtomicUsize,
}

impl<T> ConcurrencyProbe<T> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            current: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    /// Highest number of `get`s observed in flight simultaneously.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

impl<T: Tier> Tier for ConcurrencyProbe<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = T::Error;

    fn name(&self) -> &'static str {
        "probe"
    }
}

impl<T> TierRead for ConcurrencyProbe<T>
where
    T: TierRead + Sync,
    T::Key: Sync,
    T::Value: Send,
{
    async fn get(&self, key: &T::Key) -> Result<Option<T::Value>, T::Error> {
        let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        let result = self.inner.get(key).await;
        self.current.fetch_sub(1, Ordering::SeqCst);
        result
    }

    async fn exists(&self, key: &T::Key) -> Result<bool, T::Error> {
        self.inner.exists(key).await
    }
}

impl<T> TierWrite for ConcurrencyProbe<T>
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
        self.inner.put(key, value).await
    }

    async fn delete(&self, key: &T::Key) -> Result<bool, T::Error> {
        self.inner.delete(key).await
    }
}

impl<T: TierWrite> TierWrite for SlowTier<T> {
    fn put(
        &self,
        key: Self::Key,
        value: Self::Value,
    ) -> impl Future<Output = Result<Displaced<Self::Key, Self::Value>, Self::Error>> + Send {
        self.inner.put(key, value)
    }

    fn delete(&self, key: &Self::Key) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        self.inner.delete(key)
    }
}
