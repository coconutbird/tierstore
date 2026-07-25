//! The authoritative-store personality: fail-fast reads, loud data loss,
//! lossless rollover through bounded upper tiers.

mod common;

use std::num::NonZeroUsize;
use std::sync::Arc;

use common::{CountingTier, FailingTier, block_on};
use tierstore::{
    MemoryTier, RouterError, StoreError, TierRead, TierWrite, TieredCache, TieredStore,
};

const fn cap(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("nonzero capacity")
}

fn key(s: &str) -> String {
    s.to_owned()
}

#[test]
fn bounded_uppers_roll_over_losslessly() {
    let hot = Arc::new(MemoryTier::bounded(cap(1)));
    let warm = Arc::new(MemoryTier::unbounded());
    let store = TieredStore::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .build();

    // Two puts through a one-slot hot tier: the overflow demotes into warm
    // instead of being lost, so both writes succeed.
    block_on(store.put(key("a"), key("va"))).expect("put a");
    block_on(store.put(key("b"), key("vb"))).expect("put b");

    // No promotion: data is served from wherever it lives, and stays there.
    assert_eq!(
        block_on(store.get(&key("a"))).expect("get a"),
        Some(key("va"))
    );
    assert_eq!(
        block_on(store.get(&key("b"))).expect("get b"),
        Some(key("vb"))
    );
    assert_eq!(block_on(hot.get(&key("a"))).expect("hot peek"), None);
    // An absent key is an authoritative miss.
    assert_eq!(
        block_on(store.get(&key("absent"))).expect("get absent"),
        None
    );
}

#[test]
fn displacement_off_the_bottom_is_data_loss() {
    // A store whose *bottom* tier is bounded can lose data — that must be
    // an error carrying the evicted entries, never a silent shrink.
    let store: TieredStore<String, String> = TieredStore::builder()
        .tier(MemoryTier::bounded(cap(1)))
        .build();

    block_on(store.put(key("a"), key("va"))).expect("first put fits");
    match block_on(store.put(key("b"), key("vb"))) {
        Err(StoreError::Evicted(lost)) => {
            assert_eq!(lost, vec![(key("a"), key("va"))]);
        }
        other => panic!("expected data-loss error, got {other:?}"),
    }
}

#[test]
fn reads_fail_fast_instead_of_serving_around_failures() {
    // The same layout a cache would happily serve through: a failing tier
    // above a healthy one holding the value.
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let store = TieredStore::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .build();

    match block_on(store.get(&key("k"))) {
        Err(StoreError::Router(RouterError::Tier(failure))) => {
            assert_eq!(failure.tier(), 0);
        }
        other => panic!("a store must not serve around a broken tier, got {other:?}"),
    }
}

#[test]
fn cache_composes_over_store_as_its_bottom_tier() {
    // The blessed layering: cache tiers above, the authority below — the
    // store IS the cache's bottom tier.
    let backend = Arc::new(CountingTier::new(MemoryTier::unbounded()));
    let store = TieredStore::builder().tier(Arc::clone(&backend)).build();
    let hot = Arc::new(MemoryTier::bounded(cap(1)));
    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(store)
        .build();

    // Write-through reaches the authority first, then the cache layer.
    block_on(cache.put(key("a"), key("va"))).expect("put a");
    assert_eq!(
        block_on(backend.get(&key("a"))).expect("backend peek"),
        Some(key("va"))
    );

    // A hot hit never consults the authority.
    let peeks = backend.gets();
    assert_eq!(
        block_on(cache.get(&key("a"))).expect("get a"),
        Some(key("va"))
    );
    assert_eq!(backend.gets(), peeks);

    // Displacing `a` from the one-slot hot tier rolls it into the store
    // tier (already present there — lossless either way), and re-reading
    // it fills from the authority exactly once.
    block_on(cache.put(key("b"), key("vb"))).expect("put b");
    assert_eq!(block_on(hot.get(&key("a"))).expect("hot peek"), None);
    assert_eq!(
        block_on(cache.get(&key("a"))).expect("get a again"),
        Some(key("va"))
    );
    assert_eq!(backend.gets(), peeks + 1);
    assert_eq!(
        block_on(hot.get(&key("a"))).expect("hot peek"),
        Some(key("va")),
        "the fill must promote back into the cache layer"
    );
}

#[test]
fn partial_deletes_surface_loudly() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let store = TieredStore::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .build();

    match block_on(store.delete(&key("k"))) {
        Err(StoreError::Router(RouterError::Partial(failures))) => {
            assert_eq!(failures.len(), 1);
        }
        other => panic!("expected partial-delete error, got {other:?}"),
    }
    // The reachable tier was still cleaned up.
    assert_eq!(block_on(warm.get(&key("k"))).expect("warm peek"), None);
}
