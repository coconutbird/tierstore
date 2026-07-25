//! Router semantics: promotion, rollover, error policies, composition.

mod common;

use std::num::NonZeroUsize;
use std::sync::Arc;

use common::{FailingTier, block_on};
use tierstore::{
    KeyStatus, MemoryTier, OnReadError, OnWriteError, Policy, Promote, ReadPolicy, Router,
    RouterError, TierRead, TierWrite, WriteMode,
};

const fn cap(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("nonzero capacity")
}

fn key(s: &str) -> String {
    s.to_owned()
}

#[test]
fn read_through_promotes_into_all_upper_tiers() {
    let hot = Arc::new(MemoryTier::unbounded());
    let warm = Arc::new(MemoryTier::unbounded());
    let cold = Arc::new(MemoryTier::unbounded());
    block_on(cold.put(key("user:1"), key("alice"))).expect("seed cold");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .tier(Arc::clone(&cold))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::AllAbove,
                on_error: OnReadError::FallThrough,
            },
            ..Policy::default()
        })
        .build();

    assert_eq!(
        block_on(router.get(&key("user:1"))).expect("routed get"),
        Some(key("alice"))
    );
    // Inclusive promotion: both upper tiers now hold the value.
    assert_eq!(
        block_on(hot.get(&key("user:1"))).expect("hot get"),
        Some(key("alice"))
    );
    assert_eq!(
        block_on(warm.get(&key("user:1"))).expect("warm get"),
        Some(key("alice"))
    );
}

#[test]
fn top_only_promotion_skips_middle_tiers() {
    let hot = Arc::new(MemoryTier::unbounded());
    let warm = Arc::new(MemoryTier::unbounded());
    let cold = Arc::new(MemoryTier::unbounded());
    block_on(cold.put(key("k"), 7_u32)).expect("seed cold");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .tier(Arc::clone(&cold))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::TopOnly,
                on_error: OnReadError::FallThrough,
            },
            ..Policy::default()
        })
        .build();

    assert_eq!(
        block_on(router.get(&key("k"))).expect("routed get"),
        Some(7)
    );
    assert_eq!(block_on(hot.get(&key("k"))).expect("hot get"), Some(7));
    assert_eq!(block_on(warm.get(&key("k"))).expect("warm get"), None);
}

#[test]
fn rollover_keeps_hot_evictions_reachable_via_warm() {
    let hot = Arc::new(MemoryTier::bounded(cap(1)));
    let warm = Arc::new(MemoryTier::unbounded());
    // `x` lives only in hot, `y` only in warm.
    block_on(hot.put(key("x"), key("vx"))).expect("seed hot");
    block_on(warm.put(key("y"), key("vy"))).expect("seed warm");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::TopOnly,
                on_error: OnReadError::FallThrough,
            },
            demote_displaced: true,
            ..Policy::default()
        })
        .build();

    // Reading `y` promotes it into hot (capacity 1), displacing `x`, which
    // must roll over into warm rather than vanish.
    assert_eq!(
        block_on(router.get(&key("y"))).expect("get y"),
        Some(key("vy"))
    );
    assert_eq!(
        block_on(hot.get(&key("y"))).expect("hot get"),
        Some(key("vy"))
    );
    assert_eq!(block_on(hot.get(&key("x"))).expect("hot get"), None);
    assert_eq!(
        block_on(warm.get(&key("x"))).expect("warm get"),
        Some(key("vx"))
    );
    // And the router still serves it.
    assert_eq!(
        block_on(router.get(&key("x"))).expect("get x"),
        Some(key("vx"))
    );
}

#[test]
fn displacement_off_the_bottom_is_returned_from_set() {
    let router: Router<String, String> =
        Router::builder().tier(MemoryTier::bounded(cap(1))).build();

    assert_eq!(
        block_on(router.put(key("a"), key("va"))).expect("put a"),
        vec![]
    );
    // `a` falls off the only (bottommost) tier: evicted from the store
    // entirely, and the caller is told so.
    assert_eq!(
        block_on(router.put(key("b"), key("vb"))).expect("put b"),
        vec![(key("a"), key("va"))]
    );
}

#[test]
fn fall_through_serves_from_below_a_failing_tier() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::Never,
                on_error: OnReadError::FallThrough,
            },
            ..Policy::default()
        })
        .build();

    assert_eq!(
        block_on(router.get(&key("k"))).expect("get"),
        Some(key("v"))
    );
}

#[test]
fn miss_past_a_failing_tier_is_inconclusive_not_none() {
    let router: Router<String, String> = Router::builder()
        .tier(FailingTier::default())
        .tier(MemoryTier::unbounded())
        .build();

    match block_on(router.get(&key("absent"))) {
        Err(RouterError::Inconclusive(failures)) => {
            assert_eq!(failures.len(), 1);
            assert_eq!(failures[0].tier(), 0);
            assert_eq!(failures[0].tier_name(), "failing");
        }
        other => panic!("expected inconclusive read, got {other:?}"),
    }
}

#[test]
fn fail_fast_surfaces_the_failing_tier() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::Never,
                on_error: OnReadError::FailFast,
            },
            ..Policy::default()
        })
        .build();

    match block_on(router.get(&key("k"))) {
        Err(RouterError::Tier(failure)) => assert_eq!(failure.tier(), 0),
        other => panic!("expected fail-fast tier error, got {other:?}"),
    }
}

#[test]
fn delete_attempts_every_tier_and_reports_partial_failure() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .build();

    match block_on(router.delete(&key("k"))) {
        Err(RouterError::Partial(failures)) => assert_eq!(failures.len(), 1),
        other => panic!("expected partial delete failure, got {other:?}"),
    }
    // The reachable tier was still cleaned up despite the failure above it.
    assert_eq!(block_on(warm.get(&key("k"))).expect("warm get"), None);
}

#[test]
fn delete_reports_whether_the_key_existed_anywhere() {
    let hot = Arc::new(MemoryTier::unbounded());
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .build();

    assert!(block_on(router.delete(&key("k"))).expect("delete"));
    assert!(!block_on(router.delete(&key("k"))).expect("second delete"));
}

#[test]
fn write_around_writes_bottom_and_invalidates_upper() {
    let hot = Arc::new(MemoryTier::unbounded());
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(hot.put(key("k"), key("stale"))).expect("seed hot");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .policy(Policy {
            write: WriteMode::WriteAround,
            ..Policy::default()
        })
        .build();

    block_on(router.put(key("k"), key("fresh"))).expect("write around");
    assert_eq!(
        block_on(warm.get(&key("k"))).expect("warm get"),
        Some(key("fresh"))
    );
    // The stale hot copy must be gone, or reads would shadow the new value.
    assert_eq!(block_on(hot.get(&key("k"))).expect("hot get"), None);
    assert_eq!(
        block_on(router.get(&key("k"))).expect("routed get"),
        Some(key("fresh"))
    );
}

#[test]
fn exists_falls_through_without_promoting() {
    let hot = Arc::new(MemoryTier::unbounded());
    let cold = Arc::new(MemoryTier::unbounded());
    block_on(cold.put(key("k"), key("v"))).expect("seed cold");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&cold))
        .build();

    assert!(block_on(router.exists(&key("k"))).expect("exists"));
    // Existence checks never promote.
    assert_eq!(block_on(hot.get(&key("k"))).expect("hot get"), None);
    assert!(!block_on(router.exists(&key("absent"))).expect("exists absent"));
}

#[test]
fn batched_get_gathers_across_tiers_and_promotes() {
    let hot = Arc::new(MemoryTier::unbounded());
    let warm = Arc::new(MemoryTier::unbounded());
    let cold = Arc::new(MemoryTier::unbounded());
    block_on(hot.put(key("a"), key("va"))).expect("seed hot");
    block_on(warm.put(key("b"), key("vb"))).expect("seed warm");
    block_on(cold.put(key("c"), key("vc"))).expect("seed cold");

    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .tier(Arc::clone(&cold))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::AllAbove,
                on_error: OnReadError::FallThrough,
            },
            ..Policy::default()
        })
        .build();

    let keys = [key("a"), key("b"), key("c"), key("nope")];
    assert_eq!(
        block_on(router.get_many(&keys)).expect("batched get"),
        vec![Some(key("va")), Some(key("vb")), Some(key("vc")), None]
    );
    // Hits found below were promoted into hot.
    assert_eq!(
        block_on(hot.get(&key("b"))).expect("hot get"),
        Some(key("vb"))
    );
    assert_eq!(
        block_on(hot.get(&key("c"))).expect("hot get"),
        Some(key("vc"))
    );
}

#[test]
fn batched_put_reports_bottom_displacement() {
    let router: Router<String, String> =
        Router::builder().tier(MemoryTier::bounded(cap(1))).build();

    let displaced = block_on(router.put_many(vec![(key("a"), key("va")), (key("b"), key("vb"))]))
        .expect("batched put");
    assert_eq!(displaced, vec![(key("a"), key("va"))]);
}

#[test]
fn batched_delete_attempts_every_tier_and_reports_partial() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .build();

    match block_on(router.delete_many(&[key("k")])) {
        Err(RouterError::Partial(failures)) => assert_eq!(failures.len(), 1),
        other => panic!("expected partial batched delete, got {other:?}"),
    }
    assert_eq!(block_on(warm.get(&key("k"))).expect("warm get"), None);
}

#[test]
fn batched_read_reports_per_key_statuses() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::Never,
                on_error: OnReadError::FallThrough,
            },
            ..Policy::default()
        })
        .build();

    let report = block_on(router.read_many(&[key("k"), key("absent")])).expect("read_many");
    assert!(!report.is_complete());
    assert_eq!(report.failures.len(), 1);
    // The resolved key survives as a hit with tier provenance…
    assert_eq!(
        report.statuses[0],
        KeyStatus::Hit {
            tier: 1,
            value: key("v")
        }
    );
    // …while the unresolved one is inconclusive, not a false miss.
    assert_eq!(report.statuses[1], KeyStatus::Inconclusive);
}

#[test]
fn batched_miss_past_failing_tier_is_inconclusive() {
    let warm = Arc::new(MemoryTier::unbounded());
    block_on(warm.put(key("k"), key("v"))).expect("seed warm");

    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .build();

    // "k" resolves below the failing tier, but "absent" cannot be confirmed
    // missing — whole-batch granularity makes the entire batch inconclusive.
    match block_on(router.get_many(&[key("k"), key("absent")])) {
        Err(RouterError::Inconclusive(failures)) => assert_eq!(failures.len(), 1),
        other => panic!("expected inconclusive batch, got {other:?}"),
    }
}

#[test]
fn best_effort_writes_skip_failing_tiers() {
    let warm = Arc::new(MemoryTier::unbounded());
    let router = Router::builder()
        .tier(FailingTier::default())
        .tier(Arc::clone(&warm))
        .policy(Policy {
            on_write_error: OnWriteError::BestEffort,
            ..Policy::default()
        })
        .build();

    // The failing tier is skipped; the healthy one is written; the caller
    // sees success (a failed fill is a capacity loss, not a failure).
    block_on(router.put(key("k"), key("v"))).expect("best-effort put");
    assert_eq!(block_on(warm.get(&key("k"))).expect("warm"), Some(key("v")));

    let stats = router.stats();
    assert_eq!(stats[0].errors, 1, "the skipped failure lands in stats");
    assert_eq!(stats[1].puts, 1);
}

#[test]
fn stats_count_per_tier_activity() {
    let hot = Arc::new(MemoryTier::unbounded());
    let cold = Arc::new(MemoryTier::unbounded());
    let router = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&cold))
        .build();

    block_on(router.put(key("k"), key("v"))).expect("put");
    assert_eq!(
        block_on(router.get(&key("k"))).expect("get"),
        Some(key("v"))
    );
    assert_eq!(block_on(router.get(&key("absent"))).expect("miss"), None);

    let stats = router.stats();
    assert_eq!(stats.len(), 2);
    assert_eq!(stats[0].name, "memory");
    assert!(!stats[0].read_only);
    // Write-through wrote both tiers.
    assert_eq!(stats[0].puts, 1);
    assert_eq!(stats[1].puts, 1);
    // The hit was served from hot; cold was never probed for it.
    assert_eq!(stats[0].hits, 1);
    assert_eq!(stats[1].hits, 0);
    // The confirmed miss probed both tiers.
    assert_eq!(stats[0].misses, 1);
    assert_eq!(stats[1].misses, 1);
    assert_eq!(stats[0].errors, 0);
}

#[test]
fn routers_compose_as_tiers_of_routers() {
    let hot = Arc::new(MemoryTier::unbounded());
    let warm = Arc::new(MemoryTier::unbounded());
    let cold = Arc::new(MemoryTier::unbounded());
    block_on(cold.put(key("k"), key("v"))).expect("seed cold");

    // The local hierarchy is itself a tier of the outer router.
    let local: Router<String, String> = Router::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .build();
    let outer = Router::builder()
        .tier(local)
        .tier(Arc::clone(&cold))
        .policy(Policy {
            read: ReadPolicy {
                promote: Promote::AllAbove,
                on_error: OnReadError::FallThrough,
            },
            ..Policy::default()
        })
        .build();

    assert_eq!(
        block_on(outer.get(&key("k"))).expect("outer get"),
        Some(key("v"))
    );
    // Promotion into the inner router wrote through its own hierarchy.
    assert_eq!(
        block_on(hot.get(&key("k"))).expect("hot get"),
        Some(key("v"))
    );
    assert_eq!(
        block_on(warm.get(&key("k"))).expect("warm get"),
        Some(key("v"))
    );
}
