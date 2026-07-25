//! Blocking-work offload middleware.
//!
//! [`OffloadTier`] runs every operation of its inner tier on a dedicated
//! thread pool, so tiers that do blocking I/O inline (`DiskTier`,
//! mmap-backed tiers, any synchronous client) stop stalling the async
//! executor that drives the router. Dependency-free and executor-agnostic:
//! the pool is plain `std::thread` workers and the handoff is a tiny
//! waker-based oneshot.
//!
//! Intended for tiers whose futures complete without suspending (blocking
//! I/O wrapped in `async fn`) — a worker drives the inner future to
//! completion on the pool thread. A genuinely-async inner tier would
//! busy-spin a worker, and doesn't need offloading in the first place.
//!
//! # Example
//!
//! ```no_run
//! use std::num::NonZeroUsize;
//! use tierstore::{DiskTier, OffloadTier};
//!
//! # fn demo() -> std::io::Result<()> {
//! let warm = OffloadTier::new(
//!     DiskTier::open("/var/cache/myapp")?,
//!     NonZeroUsize::new(4).unwrap(),
//! );
//! # Ok(()) }
//! ```

use std::future::Future;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::{Pin, pin};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;

use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierReadRange, TierWrite};

/// Middleware tier that executes its inner tier's operations on a dedicated
/// pool of blocking worker threads.
pub struct OffloadTier<T> {
    inner: Arc<T>,
    name: String,
    pool: Pool,
}

impl<T: Tier> OffloadTier<T> {
    /// Wraps `inner`, running its operations on `threads` worker threads.
    pub fn new(inner: T, threads: NonZeroUsize) -> Self {
        let name = format!("offload({})", inner.name());
        Self {
            inner: Arc::new(inner),
            name,
            pool: Pool::spawn(threads.get()),
        }
    }
}

impl<T: Tier> std::fmt::Debug for OffloadTier<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OffloadTier")
            .field("name", &self.name)
            .field("threads", &self.pool.workers.len())
            .finish_non_exhaustive()
    }
}

type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    /// `Sender` is not `Sync`, so submissions go through a mutex; taking it
    /// on drop closes the channel and lets the workers exit.
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn spawn(threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = (0..threads)
            .map(|_| {
                let receiver = Arc::clone(&receiver);
                std::thread::spawn(move || {
                    loop {
                        // Bind so the receiver lock is not held while the
                        // job runs.
                        let job = lock(&receiver).recv();
                        match job {
                            Ok(job) => job(),
                            Err(_) => break,
                        }
                    }
                })
            })
            .collect();
        Self {
            sender: Mutex::new(Some(sender)),
            workers,
        }
    }

    /// Runs `work` on a pool thread; the returned future resolves when it
    /// finishes. A panic in `work` is resumed on the awaiting task.
    fn run<R, W>(&self, work: W) -> Oneshot<R>
    where
        R: Send + 'static,
        W: FnOnce() -> R + Send + 'static,
    {
        let state = Arc::new(Mutex::new(OneshotState::default()));
        let job_state = Arc::clone(&state);
        let job: Job = Box::new(move || {
            let outcome = catch_unwind(AssertUnwindSafe(work));
            let waker = {
                let mut shared = lock(&job_state);
                shared.outcome = Some(outcome);
                shared.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        });
        // The workers outlive every submission (they exit only when the
        // sender is dropped in `Pool::drop`), so this cannot fail while the
        // pool is alive.
        if let Some(sender) = lock(&self.sender).as_ref() {
            let _ = sender.send(job);
        }
        Oneshot { state }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        lock(&self.sender).take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct OneshotState<R> {
    outcome: Option<std::thread::Result<R>>,
    waker: Option<Waker>,
}

impl<R> Default for OneshotState<R> {
    fn default() -> Self {
        Self {
            outcome: None,
            waker: None,
        }
    }
}

impl<R> OneshotState<R> {
    /// Takes the finished outcome, or registers `waker` to be notified.
    fn take_or_register(&mut self, waker: &Waker) -> Option<std::thread::Result<R>> {
        let outcome = self.outcome.take();
        if outcome.is_none()
            && !self
                .waker
                .as_ref()
                .is_some_and(|current| current.will_wake(waker))
        {
            self.waker = Some(waker.clone());
        }
        outcome
    }
}

struct Oneshot<R> {
    state: Arc<Mutex<OneshotState<R>>>,
}

impl<R> Future for Oneshot<R> {
    type Output = R;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        // Bind so the state lock is released before the outcome is acted on
        // (a resumed panic must not poison the shared state).
        let outcome = lock(&self.state).take_or_register(cx.waker());
        match outcome {
            Some(Ok(value)) => Poll::Ready(value),
            Some(Err(panic)) => resume_unwind(panic),
            None => Poll::Pending,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Drives an inner-tier future to completion on the worker thread. The
/// inner tiers this middleware targets do their work inline, so the loop
/// effectively runs the body once.
fn drive<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

impl<T: Tier> Tier for OffloadTier<T> {
    type Key = T::Key;
    type Value = T::Value;
    type Error = T::Error;

    fn name(&self) -> &str {
        &self.name
    }
}

impl<T> TierRead for OffloadTier<T>
where
    T: TierRead + Send + Sync + 'static,
    T::Key: Clone + Send + Sync + 'static,
    T::Value: Send + 'static,
    T::Error: Send + 'static,
{
    async fn get(&self, key: &T::Key) -> Result<Option<T::Value>, T::Error> {
        let inner = Arc::clone(&self.inner);
        let key = key.clone();
        self.pool.run(move || drive(inner.get(&key))).await
    }

    async fn exists(&self, key: &T::Key) -> Result<bool, T::Error> {
        let inner = Arc::clone(&self.inner);
        let key = key.clone();
        self.pool.run(move || drive(inner.exists(&key))).await
    }

    async fn get_many(&self, keys: &[T::Key]) -> Result<Vec<Option<T::Value>>, T::Error> {
        let inner = Arc::clone(&self.inner);
        let keys = keys.to_vec();
        self.pool.run(move || drive(inner.get_many(&keys))).await
    }
}

impl<T> TierReadRange for OffloadTier<T>
where
    T: TierReadRange + Send + Sync + 'static,
    T::Key: Clone + Send + Sync + 'static,
    T::Value: Send + 'static,
    T::Error: Send + 'static,
{
    async fn read_range(
        &self,
        key: &T::Key,
        range: Range<u64>,
    ) -> Result<Option<T::Value>, T::Error> {
        let inner = Arc::clone(&self.inner);
        let key = key.clone();
        self.pool
            .run(move || drive(inner.read_range(&key, range)))
            .await
    }
}

impl<T> TierWrite for OffloadTier<T>
where
    T: TierWrite + Send + Sync + 'static,
    T::Key: Clone + Send + Sync + 'static,
    T::Value: Send + 'static,
    T::Error: Send + 'static,
{
    async fn put(
        &self,
        key: T::Key,
        value: T::Value,
    ) -> Result<Displaced<T::Key, T::Value>, T::Error> {
        let inner = Arc::clone(&self.inner);
        self.pool.run(move || drive(inner.put(key, value))).await
    }

    async fn delete(&self, key: &T::Key) -> Result<bool, T::Error> {
        let inner = Arc::clone(&self.inner);
        let key = key.clone();
        self.pool.run(move || drive(inner.delete(&key))).await
    }

    async fn put_many(
        &self,
        entries: Vec<(T::Key, T::Value)>,
    ) -> Result<Displaced<T::Key, T::Value>, T::Error> {
        let inner = Arc::clone(&self.inner);
        self.pool.run(move || drive(inner.put_many(entries))).await
    }

    async fn delete_many(&self, keys: &[T::Key]) -> Result<Vec<bool>, T::Error> {
        let inner = Arc::clone(&self.inner);
        let keys = keys.to_vec();
        self.pool.run(move || drive(inner.delete_many(&keys))).await
    }
}

impl<T> TierList for OffloadTier<T>
where
    T: TierList + Send + Sync + 'static,
    T::Key: Send + 'static,
    T::Cursor: Send + 'static,
    T::Error: Send + 'static,
{
    type Cursor = T::Cursor;

    async fn list(
        &self,
        cursor: Option<T::Cursor>,
        limit: usize,
    ) -> Result<Page<T::Key, T::Cursor>, T::Error> {
        let inner = Arc::clone(&self.inner);
        self.pool
            .run(move || drive(inner.list(cursor, limit)))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryTier;

    fn block_on<F: Future>(fut: F) -> F::Output {
        let mut fut = pin!(fut);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn offloaded_tier_round_trips() {
        let tier = OffloadTier::new(
            MemoryTier::unbounded(),
            NonZeroUsize::new(2).expect("nonzero"),
        );
        block_on(tier.put("k".to_owned(), 7_i32)).expect("put");
        assert_eq!(block_on(tier.get(&"k".to_owned())).expect("get"), Some(7));
        assert!(block_on(tier.exists(&"k".to_owned())).expect("exists"));
        assert!(block_on(tier.delete(&"k".to_owned())).expect("delete"));
        assert_eq!(block_on(tier.get(&"k".to_owned())).expect("get"), None);
    }

    #[test]
    fn offloaded_batches_and_listing_work() {
        let tier = OffloadTier::new(
            MemoryTier::unbounded(),
            NonZeroUsize::new(2).expect("nonzero"),
        );
        block_on(tier.put_many(vec![("a".to_owned(), 1_i32), ("b".to_owned(), 2)]))
            .expect("put_many");
        assert_eq!(
            block_on(tier.get_many(&["a".to_owned(), "b".to_owned(), "c".to_owned()]))
                .expect("get_many"),
            vec![Some(1), Some(2), None]
        );
        let page = block_on(tier.list(None, 10)).expect("list");
        assert_eq!(page.keys.len(), 2);
        assert_eq!(
            block_on(tier.delete_many(&["a".to_owned(), "c".to_owned()])).expect("delete_many"),
            vec![true, false]
        );
    }
}
