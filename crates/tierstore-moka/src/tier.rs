//! The [`MokaTier`] implementation — see the crate docs for the inclusive-only story.

use std::convert::Infallible;
use std::fmt;
use std::hash::Hash;

use tierstore_core::{Displaced, Tier, TierRead, TierWrite};

/// A [`moka::future::Cache`] behind the tier traits. See the [module
/// docs](crate) for the displacement caveat and when to prefer it.
pub struct MokaTier<K, V> {
    cache: moka::future::Cache<K, V>,
}

impl<K, V> fmt::Debug for MokaTier<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MokaTier")
            .field("entries", &self.cache.entry_count())
            .field("weighted_size", &self.cache.weighted_size())
            .finish_non_exhaustive()
    }
}

impl<K, V> MokaTier<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Wraps a preconfigured moka cache (build it with the re-exported
    /// [`moka`] builder for TTL/TTI, listeners, custom hashers, …).
    #[must_use]
    pub const fn new(cache: moka::future::Cache<K, V>) -> Self {
        Self { cache }
    }

    /// A tier holding at most `max_entries` entries.
    #[must_use]
    pub fn bounded(max_entries: u64) -> Self {
        Self::new(moka::future::Cache::new(max_entries))
    }

    /// A tier bounded by total weight: `weigh` prices each entry (typically
    /// its byte size) and moka keeps the weighted total under `budget`.
    #[must_use]
    pub fn bounded_weighted(
        budget: u64,
        weigh: impl Fn(&K, &V) -> u32 + Send + Sync + 'static,
    ) -> Self {
        Self::new(
            moka::future::Cache::builder()
                .max_capacity(budget)
                .weigher(weigh)
                .build(),
        )
    }

    /// The wrapped moka cache, for anything the tier traits do not expose
    /// (`run_pending_tasks`, size introspection, invalidation predicates).
    #[must_use]
    pub const fn inner(&self) -> &moka::future::Cache<K, V> {
        &self.cache
    }
}

impl<K, V> Tier for MokaTier<K, V> {
    type Key = K;
    type Value = V;
    type Error = Infallible;

    fn name(&self) -> &'static str {
        "moka"
    }
}

impl<K, V> TierRead for MokaTier<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Result<Option<V>, Infallible> {
        Ok(self.cache.get(key).await)
    }

    /// Existence checks do not refresh recency/frequency.
    async fn exists(&self, key: &K) -> Result<bool, Infallible> {
        Ok(self.cache.contains_key(key))
    }
}

impl<K, V> TierWrite for MokaTier<K, V>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Always reports an empty displacement list: moka evicts internally
    /// and asynchronously. Use this tier in inclusive hierarchies, where an
    /// unreported eviction loses nothing.
    async fn put(&self, key: K, value: V) -> Result<Displaced<K, V>, Infallible> {
        self.cache.insert(key, value).await;
        Ok(Displaced::new())
    }

    async fn delete(&self, key: &K) -> Result<bool, Infallible> {
        Ok(self.cache.remove(key).await.is_some())
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
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn round_trips_and_reports_no_displacement() {
        let tier: MokaTier<String, i32> = MokaTier::bounded(16);
        assert_eq!(
            block_on(tier.put("k".to_owned(), 7)).expect("infallible"),
            vec![],
            "moka evictions are internal; displacement is always empty"
        );
        assert_eq!(
            block_on(tier.get(&"k".to_owned())).expect("infallible"),
            Some(7)
        );
        assert!(block_on(tier.exists(&"k".to_owned())).expect("infallible"));
        assert!(block_on(tier.delete(&"k".to_owned())).expect("infallible"));
        assert_eq!(
            block_on(tier.get(&"k".to_owned())).expect("infallible"),
            None
        );
    }

    #[test]
    fn weigher_bounds_the_weighted_size() {
        let tier: MokaTier<String, Vec<u8>> =
            MokaTier::bounded_weighted(10, |_key: &String, value: &Vec<u8>| {
                u32::try_from(value.len()).unwrap_or(u32::MAX)
            });
        for name in ["a", "b", "c"] {
            block_on(tier.put(name.to_owned(), vec![0_u8; 6])).expect("infallible");
        }
        // moka maintenance is deferred; drive it, then the weighted total
        // must respect the budget.
        block_on(tier.inner().run_pending_tasks());
        assert!(
            tier.inner().weighted_size() <= 10,
            "weighted size {} must be within the budget",
            tier.inner().weighted_size()
        );
    }

    #[test]
    fn batch_defaults_loop_over_singular_ops() {
        let tier: MokaTier<String, i32> = MokaTier::bounded(16);
        block_on(tier.put_many(vec![("a".to_owned(), 1), ("b".to_owned(), 2)]))
            .expect("infallible");
        assert_eq!(
            block_on(tier.get_many(&["a".to_owned(), "b".to_owned(), "c".to_owned()]))
                .expect("infallible"),
            vec![Some(1), Some(2), None]
        );
        assert_eq!(
            block_on(tier.delete_many(&["a".to_owned(), "missing".to_owned()]))
                .expect("infallible"),
            vec![true, false]
        );
    }
}
