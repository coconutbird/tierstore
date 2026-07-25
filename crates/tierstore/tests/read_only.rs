//! Read-only tiers: the lane for an origin the router reads from but never
//! writes — the docres/object-store shape.

mod common;

use std::sync::Arc;

use common::{ReadOnlySource, block_on};
use tierstore::{MemoryTier, Router, RouterError, TierRead, TierWrite, TieredCache};

fn key(s: &str) -> String {
    s.to_owned()
}

#[test]
fn cache_over_read_only_origin() {
    let hot = Arc::new(MemoryTier::unbounded());
    let origin = ReadOnlySource::new([(key("k"), key("v"))]);
    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .read_only_tier(origin)
        .build();

    // Read-through with promotion into the writable layer.
    assert_eq!(block_on(cache.get(&key("k"))).expect("get"), Some(key("v")));
    assert_eq!(block_on(hot.get(&key("k"))).expect("hot"), Some(key("v")));

    // Puts fill the cache layers only — the origin cannot be written, and
    // no write is ever attempted against it.
    block_on(cache.put(key("k2"), key("v2"))).expect("put");
    assert_eq!(
        block_on(cache.get(&key("k2"))).expect("get"),
        Some(key("v2"))
    );

    // Invalidation clears local copies; the origin re-serves the key
    // afterwards — resurrection by design for an origin the cache does not
    // own.
    assert!(block_on(cache.invalidate(&key("k"))).expect("invalidate"));
    assert_eq!(block_on(hot.get(&key("k"))).expect("hot"), None);
    assert_eq!(
        block_on(cache.get(&key("k"))).expect("get again"),
        Some(key("v"))
    );
}

#[test]
fn fully_read_only_router_rejects_writes() {
    let router: Router<String, String> = Router::builder()
        .read_only_tier(ReadOnlySource::new([(key("k"), key("v"))]))
        .build();

    assert_eq!(
        block_on(router.get(&key("k"))).expect("get"),
        Some(key("v"))
    );
    match block_on(router.put(key("x"), key("y"))) {
        Err(RouterError::ReadOnly) => {}
        other => panic!("expected RouterError::ReadOnly, got {other:?}"),
    }
    // Deletes are no-ops over read-only tiers: nothing deletable existed.
    assert!(!block_on(router.delete(&key("k"))).expect("delete"));
    // And the key is still served.
    assert_eq!(
        block_on(router.get(&key("k"))).expect("get again"),
        Some(key("v"))
    );
}
