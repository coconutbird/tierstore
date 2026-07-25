//! The flagship scenario: a hot/warm/cold rollover cache — memory over disk
//! over a mock remote store — built on the generic router.

mod common;

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use common::{CountingTier, SlowTier, block_on};
use tierstore::{
    DiskTier, KeyStatus, MemoryTier, RouterError, TierRead, TierWrite, TieredCache, VerifiedTier,
};

fn temp_root(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tierstore-test-{}-{name}", std::process::id()))
}

fn key(s: &str) -> String {
    s.to_owned()
}

fn val(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

#[test]
fn hot_warm_cold_rollover_cache() {
    let root = temp_root("cache");
    let _ = std::fs::remove_dir_all(&root);

    // Hot: two-entry in-memory tier. Warm: files on disk. Cold: a counting
    // wrapper standing in for a remote database.
    let hot = Arc::new(MemoryTier::bounded(
        NonZeroUsize::new(2).expect("nonzero capacity"),
    ));
    let warm = Arc::new(DiskTier::open(&root).expect("open warm tier"));
    let cold = Arc::new(CountingTier::new(MemoryTier::unbounded()));
    for name in ["ada", "grace", "barbara"] {
        block_on(cold.put(key(name), val(&format!("record for {name}")))).expect("seed cold");
    }

    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .tier(Arc::clone(&cold))
        .build();

    // First read fetches from the remote and promotes into hot (exclusive
    // promotion: warm stays untouched).
    assert_eq!(
        block_on(cache.get(&key("ada"))).expect("get ada"),
        Some(val("record for ada"))
    );
    assert_eq!(cold.gets(), 1);
    assert_eq!(block_on(warm.get(&key("ada"))).expect("warm peek"), None);

    // Second read is a pure hot hit: the remote is not consulted again.
    assert_eq!(
        block_on(cache.get(&key("ada"))).expect("get ada again"),
        Some(val("record for ada"))
    );
    assert_eq!(cold.gets(), 1);

    // Two more distinct reads overflow the two-entry hot tier: "ada" (the
    // oldest) is displaced and must roll over into warm, not vanish.
    assert_eq!(
        block_on(cache.get(&key("grace"))).expect("get grace"),
        Some(val("record for grace"))
    );
    assert_eq!(
        block_on(cache.get(&key("barbara"))).expect("get barbara"),
        Some(val("record for barbara"))
    );
    assert_eq!(cold.gets(), 3);
    assert_eq!(block_on(hot.get(&key("ada"))).expect("hot peek"), None);
    assert_eq!(
        block_on(warm.get(&key("ada"))).expect("warm peek"),
        Some(val("record for ada"))
    );

    // Reading the rolled-over key is served from disk — still no new remote
    // fetch — and it climbs back into hot.
    assert_eq!(
        block_on(cache.get(&key("ada"))).expect("get ada from warm"),
        Some(val("record for ada"))
    );
    assert_eq!(cold.gets(), 3);
    assert_eq!(
        block_on(hot.get(&key("ada"))).expect("hot peek"),
        Some(val("record for ada"))
    );

    // Write-through put lands everywhere down to the remote.
    block_on(cache.put(key("mary"), val("record for mary"))).expect("put mary");
    assert_eq!(
        block_on(cold.get(&key("mary"))).expect("cold peek"),
        Some(val("record for mary"))
    );

    // Invalidation removes the key from every tier, including the remote.
    assert!(block_on(cache.invalidate(&key("ada"))).expect("invalidate"));
    assert_eq!(
        block_on(cache.get(&key("ada"))).expect("get after invalidate"),
        None
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn batched_cache_round_trip() {
    let hot = Arc::new(MemoryTier::bounded(
        NonZeroUsize::new(2).expect("nonzero capacity"),
    ));
    let cold = Arc::new(MemoryTier::unbounded());
    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&cold))
        .build();

    block_on(cache.put_many(vec![
        (key("a"), val("1")),
        (key("b"), val("2")),
        (key("c"), val("3")),
    ]))
    .expect("batched put");

    let report = block_on(cache.get_many(&[key("a"), key("b"), key("nope")])).expect("batched get");
    assert!(report.is_complete());
    // A confirmed miss, not merely "no value".
    assert_eq!(report.statuses[2], KeyStatus::Miss);
    assert_eq!(
        report.into_values(),
        vec![Some(val("1")), Some(val("2")), None]
    );

    let report = block_on(cache.invalidate_many(&[key("a"), key("nope")]));
    assert!(report.is_complete());
    assert_eq!(report.removed, vec![true, false]);
    assert_eq!(
        block_on(cache.get(&key("a"))).expect("get after invalidate"),
        None
    );
}

#[test]
fn single_flight_coalesces_concurrent_misses() {
    let hot = Arc::new(MemoryTier::unbounded());
    // Cold origin: counted, and slow enough that all threads overlap on the
    // same miss.
    let cold = Arc::new(CountingTier::new(SlowTier::new(
        MemoryTier::unbounded(),
        Duration::from_millis(40),
    )));
    block_on(cold.put(key("k"), val("v"))).expect("seed cold");

    let cache = Arc::new(
        TieredCache::builder()
            .tier(Arc::clone(&hot))
            .tier(Arc::clone(&cold))
            .build(),
    );

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let cache = Arc::clone(&cache);
            std::thread::spawn(move || block_on(cache.get(&key("k"))).expect("cache get"))
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().expect("reader thread"), Some(val("v")));
    }

    // One leader fetched and promoted; the other three waited on the gate
    // and were then served from hot.
    assert_eq!(
        cold.gets(),
        1,
        "concurrent misses for one key must coalesce into a single cold fetch"
    );
}

#[test]
fn verification_rejects_corrupt_cold_values() {
    let hot = Arc::new(MemoryTier::unbounded());
    let origin = MemoryTier::unbounded();
    block_on(origin.put(key("good"), val("data"))).expect("seed");
    block_on(origin.put(key("bad"), b"corrupt payload".to_vec())).expect("seed");
    // Trust boundary: values served by the origin must pass the check once,
    // at the boundary — shardstore's store→node contract.
    let cold = VerifiedTier::new(origin, |_key: &String, value: &Vec<u8>| {
        if value.starts_with(b"corrupt") {
            Err("checksum mismatch".into())
        } else {
            Ok(())
        }
    });

    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(cold)
        .build();

    assert_eq!(
        block_on(cache.get(&key("good"))).expect("verified get"),
        Some(val("data"))
    );

    // The corrupt value surfaces as the verified tier *failing*: the read is
    // inconclusive, never a wrong answer.
    match block_on(cache.get(&key("bad"))) {
        Err(RouterError::Inconclusive(failures)) => {
            assert_eq!(failures.len(), 1);
            assert!(failures[0].tier_name().contains("verified"));
        }
        other => panic!("corrupt value must not be served, got {other:?}"),
    }
    // And it was never promoted into hot.
    assert_eq!(block_on(hot.get(&key("bad"))).expect("hot peek"), None);
}

#[test]
fn warm_tier_survives_reopening() {
    let root = temp_root("reopen");
    let _ = std::fs::remove_dir_all(&root);

    // Populate through one cache instance…
    {
        let warm = DiskTier::open(&root).expect("open warm tier");
        let cache: TieredCache<String, Vec<u8>> = TieredCache::builder()
            .tier(MemoryTier::unbounded())
            .tier(warm)
            .build();
        block_on(cache.put(key("persisted"), val("survives"))).expect("put");
    }

    // …and read it back through a fresh one: the warm tier is durable state,
    // which is the point of having it.
    let warm = DiskTier::open(&root).expect("reopen warm tier");
    let cache: TieredCache<String, Vec<u8>> = TieredCache::builder()
        .tier(MemoryTier::unbounded())
        .tier(warm)
        .build();
    assert_eq!(
        block_on(cache.get(&key("persisted"))).expect("get"),
        Some(val("survives"))
    );

    let _ = std::fs::remove_dir_all(&root);
}
