//! In-memory tier: the "hot" layer.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard, PoisonError};

use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierReadRef, TierWrite};

/// A shared in-memory map with optional entry-count and byte-budget bounds
/// and FIFO eviction.
///
/// Bounds compose with rollover: displaced entries are returned from
/// [`TierWrite::put`] so a router demotes them instead of dropping them.
/// With a byte budget, an entry heavier than the whole budget is displaced
/// *immediately* (including itself), which under a router means oversized
/// values roll straight through to the next tier down.
///
/// Eviction is FIFO (insertion order) for now — deliberately the simplest
/// policy that exercises displacement; pluggable strategies (LRU, LFU) are
/// an open design question.
///
/// This tier cannot fail: its error type is [`Infallible`].
pub struct MemoryTier<K, V> {
    max_entries: Option<NonZeroUsize>,
    bytes: Option<ByteBudget<K, V>>,
    inner: Mutex<Inner<K, V>>,
}

/// Weigher callback pricing an entry, typically by its byte size.
type Weigher<K, V> = dyn Fn(&K, &V) -> usize + Send + Sync;

struct ByteBudget<K, V> {
    budget: NonZeroUsize,
    weigh: Box<Weigher<K, V>>,
}

/// Snapshot of the tier's configured bounds, passed into the locked core.
#[derive(Clone, Copy)]
struct Limits {
    entries: Option<usize>,
    bytes: Option<usize>,
}

#[derive(Debug)]
struct Entry<V> {
    value: V,
    weight: usize,
}

#[derive(Debug)]
struct Inner<K, V> {
    map: HashMap<K, Entry<V>>,
    /// Insertion order, used both for FIFO eviction and stable listing.
    order: VecDeque<K>,
    /// Sum of entry weights (0 unless a byte budget is configured).
    total_weight: usize,
}

/// All mutation lives on `Inner`, which owns the "map, order, and weight
/// total stay in sync" invariant; the async tier methods lock, delegate,
/// and let the guard drop as a temporary.
impl<K, V> Inner<K, V>
where
    K: Eq + Hash + Clone,
{
    fn insert(&mut self, key: K, value: V, weight: usize, limits: Limits) -> Displaced<K, V> {
        if let Some(entry) = self.map.get_mut(&key) {
            // Replacement: position in `order` kept; weight adjusted, and a
            // heavier value can overflow the byte budget.
            let old = entry.weight;
            entry.value = value;
            entry.weight = weight;
            self.total_weight = self.total_weight.saturating_add(weight).saturating_sub(old);
            return self.evict_over(limits);
        }
        self.map.insert(key.clone(), Entry { value, weight });
        self.order.push_back(key);
        self.total_weight = self.total_weight.saturating_add(weight);
        self.evict_over(limits)
    }

    fn insert_batch(&mut self, entries: Vec<(K, V, usize)>, limits: Limits) -> Displaced<K, V> {
        let mut displaced = Displaced::new();
        for (key, value, weight) in entries {
            displaced.extend(self.insert(key, value, weight, limits));
        }
        displaced
    }

    /// FIFO-evicts until back under every configured bound. The newest
    /// entry is evicted last, so an entry that can never fit is displaced
    /// too — rollover then pushes it straight down a tier.
    fn evict_over(&mut self, limits: Limits) -> Displaced<K, V> {
        let mut displaced = Displaced::new();
        while self.over(limits) {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.map.remove(&oldest) {
                self.total_weight = self.total_weight.saturating_sub(entry.weight);
                displaced.push((oldest, entry.value));
            }
        }
        displaced
    }

    fn over(&self, limits: Limits) -> bool {
        limits.entries.is_some_and(|max| self.map.len() > max)
            || limits.bytes.is_some_and(|max| self.total_weight > max)
    }

    fn remove(&mut self, key: &K) -> bool {
        let Some(entry) = self.map.remove(key) else {
            return false;
        };
        self.total_weight = self.total_weight.saturating_sub(entry.weight);
        // O(n), acceptable for a hot tier of bounded size.
        self.order.retain(|k| k != key);
        true
    }

    fn page(&self, cursor: Option<usize>, limit: usize) -> Page<K, usize> {
        let offset = cursor.unwrap_or(0);
        let keys: Vec<K> = self
            .order
            .iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect();
        let end = offset.saturating_add(keys.len());
        let next = (limit > 0 && end < self.order.len()).then_some(end);
        Page { keys, next }
    }

    fn get_batch(&self, keys: &[K]) -> Vec<Option<V>>
    where
        V: Clone,
    {
        keys.iter()
            .map(|key| self.map.get(key).map(|entry| entry.value.clone()))
            .collect()
    }

    fn remove_batch(&mut self, keys: &[K]) -> Vec<bool> {
        keys.iter().map(|key| self.remove(key)).collect()
    }
}

impl<K, V> MemoryTier<K, V> {
    /// A tier that never evicts.
    #[must_use]
    pub fn unbounded() -> Self {
        Self {
            max_entries: None,
            bytes: None,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                total_weight: 0,
            }),
        }
    }

    /// A tier holding at most `capacity` entries; inserts beyond that
    /// displace the oldest entries.
    #[must_use]
    pub fn bounded(capacity: NonZeroUsize) -> Self {
        Self {
            max_entries: Some(capacity),
            ..Self::unbounded()
        }
    }

    /// A tier bounded by total weight: `weigh` prices each entry (typically
    /// its byte size), and inserts that push the total over `budget`
    /// displace the oldest entries until it fits again. An entry heavier
    /// than the whole budget is displaced immediately, itself included.
    ///
    /// # Example
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tierstore::MemoryTier;
    ///
    /// let budget = NonZeroUsize::new(64 * 1024).unwrap();
    /// let tier: MemoryTier<String, Vec<u8>> =
    ///     MemoryTier::bounded_bytes(budget, |key: &String, value: &Vec<u8>| {
    ///         key.len() + value.len()
    ///     });
    /// ```
    #[must_use]
    pub fn bounded_bytes(
        budget: NonZeroUsize,
        weigh: impl Fn(&K, &V) -> usize + Send + Sync + 'static,
    ) -> Self {
        Self {
            bytes: Some(ByteBudget {
                budget,
                weigh: Box::new(weigh),
            }),
            ..Self::unbounded()
        }
    }

    /// Current number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    /// Whether the tier is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current total weight (0 unless built with a byte budget).
    #[must_use]
    pub fn weight(&self) -> usize {
        self.lock().total_weight
    }

    fn weight_of(&self, key: &K, value: &V) -> usize {
        self.bytes.as_ref().map_or(0, |b| (b.weigh)(key, value))
    }

    fn limits(&self) -> Limits {
        Limits {
            entries: self.max_entries.map(NonZeroUsize::get),
            bytes: self.bytes.as_ref().map(|b| b.budget.get()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner<K, V>> {
        // A poisoned map is still structurally valid; recover instead of
        // propagating panics from unrelated threads.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<K, V> Default for MemoryTier<K, V> {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl<K, V> std::fmt::Debug for MemoryTier<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryTier")
            .field("max_entries", &self.max_entries)
            .field("byte_budget", &self.bytes.as_ref().map(|b| b.budget))
            .field("len", &self.lock().map.len())
            .finish_non_exhaustive()
    }
}

impl<K, V> Tier for MemoryTier<K, V> {
    type Key = K;
    type Value = V;
    type Error = Infallible;

    fn name(&self) -> &'static str {
        "memory"
    }
}

impl<K, V> TierRead for MemoryTier<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    async fn get(&self, key: &K) -> Result<Option<V>, Infallible> {
        Ok(self.lock().map.get(key).map(|entry| entry.value.clone()))
    }

    async fn exists(&self, key: &K) -> Result<bool, Infallible> {
        Ok(self.lock().map.contains_key(key))
    }

    /// One lock acquisition for the whole batch, instead of one per key.
    async fn get_many(&self, keys: &[K]) -> Result<Vec<Option<V>>, Infallible> {
        Ok(self.lock().get_batch(keys))
    }
}

/// Zero-copy view into a [`MemoryTier`] entry.
///
/// Holds the tier's lock for its lifetime, which is what makes the borrow
/// sound: no `put`/`delete` can displace the entry while the view exists.
/// Keep it short-lived — the tier is single-lock.
#[derive(Debug)]
pub struct MemoryRef<'a, K, V> {
    guard: MutexGuard<'a, Inner<K, V>>,
    key: K,
}

impl<K: Eq + Hash, V> std::ops::Deref for MemoryRef<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &V {
        // Present by construction, and the held guard prevents removal.
        &self
            .guard
            .map
            .get(&self.key)
            .expect("entry pinned by the held lock")
            .value
    }
}

impl<K, V> TierReadRef for MemoryTier<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    type Borrowed = V;
    type ValueRef<'a>
        = MemoryRef<'a, K, V>
    where
        Self: 'a;

    async fn get_ref<'s>(&'s self, key: &K) -> Result<Option<MemoryRef<'s, K, V>>, Infallible> {
        let guard = self.lock();
        if guard.map.contains_key(key) {
            Ok(Some(MemoryRef {
                guard,
                key: key.clone(),
            }))
        } else {
            Ok(None)
        }
    }
}

impl<K, V> TierWrite for MemoryTier<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    async fn put(&self, key: K, value: V) -> Result<Displaced<K, V>, Infallible> {
        let weight = self.weight_of(&key, &value);
        Ok(self.lock().insert(key, value, weight, self.limits()))
    }

    async fn delete(&self, key: &K) -> Result<bool, Infallible> {
        Ok(self.lock().remove(key))
    }

    /// One lock acquisition for the whole batch, instead of one per entry.
    async fn put_many(&self, entries: Vec<(K, V)>) -> Result<Displaced<K, V>, Infallible> {
        let weighted: Vec<(K, V, usize)> = entries
            .into_iter()
            .map(|(key, value)| {
                let weight = self.weight_of(&key, &value);
                (key, value, weight)
            })
            .collect();
        Ok(self.lock().insert_batch(weighted, self.limits()))
    }

    /// One lock acquisition for the whole batch, instead of one per key.
    async fn delete_many(&self, keys: &[K]) -> Result<Vec<bool>, Infallible> {
        Ok(self.lock().remove_batch(keys))
    }
}

impl<K, V> TierList for MemoryTier<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    type Cursor = usize;

    async fn list(
        &self,
        cursor: Option<usize>,
        limit: usize,
    ) -> Result<Page<K, usize>, Infallible> {
        Ok(self.lock().page(cursor, limit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<F: Future>(fut: F) -> F::Output {
        let mut fut = pin!(fut);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = fut.as_mut().poll(&mut cx) {
                return value;
            }
        }
    }

    fn cap(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("nonzero")
    }

    #[test]
    fn bounded_tier_displaces_oldest_first() {
        let tier = MemoryTier::bounded(cap(2));
        assert_eq!(block_on(tier.put("a", 1)).expect("infallible"), vec![]);
        assert_eq!(block_on(tier.put("b", 2)).expect("infallible"), vec![]);
        assert_eq!(
            block_on(tier.put("c", 3)).expect("infallible"),
            vec![("a", 1)]
        );
        assert_eq!(block_on(tier.get(&"a")).expect("infallible"), None);
        assert_eq!(block_on(tier.get(&"c")).expect("infallible"), Some(3));
    }

    #[test]
    fn replacement_does_not_evict() {
        let tier = MemoryTier::bounded(cap(2));
        block_on(tier.put("a", 1)).expect("infallible");
        block_on(tier.put("b", 2)).expect("infallible");
        assert_eq!(block_on(tier.put("a", 10)).expect("infallible"), vec![]);
        assert_eq!(block_on(tier.get(&"a")).expect("infallible"), Some(10));
        assert_eq!(tier.len(), 2);
    }

    #[test]
    fn byte_budget_displaces_by_weight() {
        let tier = MemoryTier::bounded_bytes(cap(10), |_key, value: &Vec<u8>| value.len());
        assert_eq!(
            block_on(tier.put("a", vec![0_u8; 4])).expect("infallible"),
            vec![]
        );
        assert_eq!(
            block_on(tier.put("b", vec![0_u8; 4])).expect("infallible"),
            vec![]
        );
        // 4 + 4 + 4 = 12 > 10: the oldest rolls out.
        assert_eq!(
            block_on(tier.put("c", vec![0_u8; 4])).expect("infallible"),
            vec![("a", vec![0_u8; 4])]
        );
        assert_eq!(tier.len(), 2);
        assert_eq!(tier.weight(), 8);
    }

    #[test]
    fn oversized_entry_rolls_straight_through() {
        let tier = MemoryTier::bounded_bytes(cap(10), |_key, value: &Vec<u8>| value.len());
        block_on(tier.put("small", vec![0_u8; 2])).expect("infallible");
        // 11 bytes can never fit a 10-byte budget: everything, including
        // the new entry itself, is displaced for demotion.
        let displaced = block_on(tier.put("huge", vec![0_u8; 11])).expect("infallible");
        assert_eq!(displaced.last().map(|(key, _)| *key), Some("huge"));
        assert!(tier.is_empty());
        assert_eq!(tier.weight(), 0);
    }

    #[test]
    fn replacement_reweighs_and_can_evict() {
        let tier = MemoryTier::bounded_bytes(cap(10), |_key, value: &Vec<u8>| value.len());
        block_on(tier.put("a", vec![0_u8; 4])).expect("infallible");
        block_on(tier.put("b", vec![0_u8; 4])).expect("infallible");
        // Replacing b with a heavier value overflows: the oldest rolls out.
        assert_eq!(
            block_on(tier.put("b", vec![0_u8; 7])).expect("infallible"),
            vec![("a", vec![0_u8; 4])]
        );
        assert_eq!(tier.weight(), 7);
    }

    #[test]
    fn get_ref_borrows_in_place() {
        let tier = MemoryTier::unbounded();
        block_on(tier.put("a", vec![1_u8, 2])).expect("infallible");
        assert!(
            block_on(tier.get_ref(&"missing"))
                .expect("infallible")
                .is_none()
        );
        // The view holds the tier's lock, so keep it a temporary: it drops
        // at the end of the statement, and a second tier call while it
        // lived would deadlock.
        assert_eq!(
            *block_on(tier.get_ref(&"a"))
                .expect("infallible")
                .expect("present"),
            vec![1_u8, 2]
        );
    }

    #[test]
    fn batched_ops_round_trip_under_one_lock() {
        let tier = MemoryTier::bounded(cap(2));
        let displaced =
            block_on(tier.put_many(vec![("a", 1), ("b", 2), ("c", 3)])).expect("infallible");
        assert_eq!(displaced, vec![("a", 1)]);
        assert_eq!(
            block_on(tier.get_many(&["b", "c", "a"])).expect("infallible"),
            vec![Some(2), Some(3), None]
        );
        assert_eq!(
            block_on(tier.delete_many(&["b", "missing"])).expect("infallible"),
            vec![true, false]
        );
    }

    #[test]
    fn listing_pages_in_insertion_order() {
        let tier = MemoryTier::unbounded();
        for key in ["a", "b", "c"] {
            block_on(tier.put(key, ())).expect("infallible");
        }
        let first = block_on(tier.list(None, 2)).expect("infallible");
        assert_eq!(first.keys, vec!["a", "b"]);
        let second = block_on(tier.list(first.next, 2)).expect("infallible");
        assert_eq!(second.keys, vec!["c"]);
        assert_eq!(second.next, None);
    }
}
