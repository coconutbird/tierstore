//! Admission control: `LimitedTier` caps concurrent operations against the
//! tier it wraps, bounding both origin load and transient fill memory.

mod common;

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use common::{ConcurrencyProbe, SlowTier, block_on};
use tierstore::{LimitedTier, MemoryTier, Router, TierRead, TierWrite};

#[test]
fn limited_tier_caps_origin_concurrency() {
    // A slow origin instrumented to record peak concurrency.
    let origin = Arc::new(ConcurrencyProbe::new(SlowTier::new(
        MemoryTier::unbounded(),
        Duration::from_millis(20),
    )));
    for i in 0_i32..4 {
        block_on(origin.put(format!("k{i}"), i)).expect("seed origin");
    }

    let router: Arc<Router<String, i32>> = Arc::new(
        Router::builder()
            .tier(LimitedTier::new(
                Arc::clone(&origin),
                NonZeroUsize::new(1).expect("nonzero"),
            ))
            .build(),
    );

    // Four threads read four *distinct* keys — single-flight would not help
    // here; only admission control serializes the origin.
    let handles: Vec<_> = (0_i32..4)
        .map(|i| {
            let router = Arc::clone(&router);
            std::thread::spawn(move || block_on(router.get(&format!("k{i}"))).expect("get"))
        })
        .collect();
    for (i, handle) in handles.into_iter().enumerate() {
        let expected = i32::try_from(i).expect("small index");
        assert_eq!(handle.join().expect("reader thread"), Some(expected));
    }

    assert_eq!(
        origin.peak(),
        1,
        "limit=1 must fully serialize operations against the origin"
    );
}
