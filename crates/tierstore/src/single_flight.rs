//! Per-key single-flight gates: concurrent misses for the same key coalesce
//! into one fill instead of a thundering herd on the cold tier.
//!
//! Adopted from shardstore's `ArtifactCache` shape: a `std` lock guards the
//! *map* of gates, and only the per-key gate is held across the fill await.
//! Dependency-free and executor-agnostic — the gate is a tiny waker-queue
//! mutex rather than a `tokio` primitive. [`TieredCache`](crate::TieredCache)
//! uses it internally; it is exported for anyone building their own
//! coalescing layer over a router.

use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll, Waker};

/// Keyed single-flight: [`SingleFlight::acquire`] admits one holder per key
/// at a time; concurrent acquirers of the same key wait, then proceed one by
/// one. Distinct keys never contend.
///
/// # Example
///
/// ```
/// use tierstore::SingleFlight;
///
/// # async fn demo() {
/// let gates: SingleFlight<String> = SingleFlight::new();
/// let guard = gates.acquire("key".to_owned()).await;
/// // ...perform the expensive fill; concurrent acquirers of "key" wait...
/// drop(guard); // waiters proceed (and typically now hit the filled tier)
/// # }
/// ```
#[derive(Debug)]
pub struct SingleFlight<K> {
    map: Mutex<HashMap<K, Arc<Gate>>>,
}

#[derive(Debug, Default)]
struct Gate {
    state: Mutex<GateState>,
}

#[derive(Debug, Default)]
struct GateState {
    held: bool,
    wakers: Vec<Waker>,
}

impl GateState {
    /// Takes the gate if free; otherwise registers `waker` for the release.
    fn try_acquire(&mut self, waker: &Waker) -> bool {
        if self.held {
            if !self.wakers.iter().any(|w| w.will_wake(waker)) {
                self.wakers.push(waker.clone());
            }
            false
        } else {
            self.held = true;
            true
        }
    }

    /// Releases the gate, returning the wakers to notify. Waking all of
    /// them lets waiters re-race, which cannot lose wakeups.
    fn release(&mut self) -> Vec<Waker> {
        self.held = false;
        std::mem::take(&mut self.wakers)
    }
}

impl<K> SingleFlight<K> {
    /// An empty gate set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }
}

impl<K> Default for SingleFlight<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Clone> SingleFlight<K> {
    /// Acquires the gate for `key`, waiting until any current holder
    /// releases it. The returned guard releases on drop.
    pub async fn acquire(&self, key: K) -> SingleFlightGuard<'_, K> {
        let gate = {
            let mut map = lock(&self.map);
            Arc::clone(map.entry(key.clone()).or_default())
        };
        Acquire {
            gate: Arc::clone(&gate),
        }
        .await;
        SingleFlightGuard {
            gates: self,
            key,
            gate,
        }
    }
}

struct Acquire {
    gate: Arc<Gate>,
}

impl Future for Acquire {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if lock(&self.gate.state).try_acquire(cx.waker()) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Holds a per-key gate from a [`SingleFlight`]; dropping it releases the
/// key and wakes all waiters.
#[derive(Debug)]
pub struct SingleFlightGuard<'a, K: Eq + Hash + Clone> {
    gates: &'a SingleFlight<K>,
    key: K,
    gate: Arc<Gate>,
}

impl<K: Eq + Hash + Clone> Drop for SingleFlightGuard<'_, K> {
    fn drop(&mut self) {
        // Bind before waking: a `for` over the locked call would hold the
        // state lock through every `wake`, making woken waiters contend on
        // it immediately.
        let wakers = lock(&self.gate.state).release();
        for waker in wakers {
            waker.wake();
        }
        // Best-effort map cleanup: remove the entry when the map and this
        // guard appear to be the only holders. A racing acquire can briefly
        // end up on a detached gate; that only weakens coalescing for one
        // fill, never safety.
        let mut map = lock(&self.gates.map);
        if map
            .get(&self.key)
            .is_some_and(|g| Arc::ptr_eq(g, &self.gate) && Arc::strong_count(g) == 2)
        {
            map.remove(&self.key);
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::pin;

    fn block_on<F: Future>(fut: F) -> F::Output {
        let mut fut = pin!(fut);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = fut.as_mut().poll(&mut cx) {
                return value;
            }
        }
    }

    #[test]
    fn gate_reacquires_after_release_and_cleans_up() {
        let gates: SingleFlight<&str> = SingleFlight::new();
        drop(block_on(gates.acquire("k")));
        drop(block_on(gates.acquire("k")));
        assert!(
            lock(&gates.map).is_empty(),
            "released gates should be removed"
        );
    }

    #[test]
    fn distinct_keys_do_not_contend() {
        let gates: SingleFlight<&str> = SingleFlight::new();
        let a = block_on(gates.acquire("a"));
        // "b" must be acquirable while "a" is held.
        let b = block_on(gates.acquire("b"));
        drop(a);
        drop(b);
    }
}
