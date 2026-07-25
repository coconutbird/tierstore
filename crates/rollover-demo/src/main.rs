//! Proves rollover caching end-to-end on a real tier stack:
//! hot = memory-mapped file (2 slots), warm = disk, cold = simulated remote
//! (25ms per round-trip).
//!
//! Every step asserts; if this program prints its final line, the rollover
//! contract held. Run: `cargo run -p rollover-demo`

mod remote;
mod slot_mmap;

use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use remote::RemoteStore;
use slot_mmap::SlotMmapTier;
use tierstore::{DiskTier, TierRead, TierReadRef, TierWrite, TieredCache};

type Cache = TieredCache<String, Vec<u8>>;

const HOT_SLOTS: usize = 2;
const SLOT_SIZE: usize = 256;

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

fn record(name: &str) -> Vec<u8> {
    format!("record for {name}").into_bytes()
}

fn timed_get(cache: &Cache, key: &str) -> (Option<Vec<u8>>, Duration) {
    let start = Instant::now();
    let value = block_on(cache.get(&key.to_owned())).expect("cache get");
    (value, start.elapsed())
}

fn main() -> io::Result<()> {
    let dir = std::env::temp_dir().join(format!("tierstore-rollover-demo-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let remote = Arc::new(RemoteStore::new(Duration::from_millis(25)));
    for name in ["ada", "grace", "barbara"] {
        block_on(remote.put(name.to_owned(), record(name)))?;
    }

    println!(
        "hot  = mmap, {HOT_SLOTS} slots  ({})",
        dir.join("hot.mmap").display()
    );
    println!("warm = disk           ({})", dir.join("warm").display());
    println!("cold = simulated remote, 25ms per round-trip");
    println!();

    let hot = Arc::new(SlotMmapTier::open(
        &dir.join("hot.mmap"),
        HOT_SLOTS,
        SLOT_SIZE,
    )?);
    let warm = Arc::new(DiskTier::open(dir.join("warm"))?);
    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .tier(Arc::clone(&remote))
        .build();

    prove_rollover(&cache, &hot, &warm, &remote)?;

    // Write-through put: lands in every tier down to the remote, displacing
    // the oldest hot entry onto disk on the way.
    block_on(cache.put("diana".to_owned(), record("diana"))).expect("write-through put(diana)");
    assert!(
        block_on(remote.exists(&"diana".to_owned()))?,
        "write-through must reach the remote"
    );
    assert_eq!(
        block_on(warm.get(&"barbara".to_owned()))?,
        Some(record("barbara")),
        "the hot entry displaced by the put must roll over to disk"
    );
    println!("put(diana)                    write-through everywhere; barbara rolled to disk");

    hot.flush()?;
    drop(cache);
    drop(hot);
    println!();
    println!("-- simulated restart: mmap unmapped, cache rebuilt from the same files --");
    println!();

    prove_restart(&dir, &warm, &remote)?;

    println!();
    println!(
        "remote fetches for the whole run: {} — every other read was served by mmap or disk.",
        remote.fetches()
    );
    println!("rollover caching: proven.");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// Read-through, hot hits, and rollover on overflow.
fn prove_rollover(
    cache: &Cache,
    hot: &SlotMmapTier,
    warm: &DiskTier,
    remote: &RemoteStore,
) -> io::Result<()> {
    let (value, cold_latency) = timed_get(cache, "ada");
    assert_eq!(value, Some(record("ada")));
    assert_eq!(remote.fetches(), 1, "first read must fetch from the remote");
    println!("get(ada)       {cold_latency:>10.1?}  remote fetch #1, promoted into mmap");

    let (value, hot_latency) = timed_get(cache, "ada");
    assert_eq!(value, Some(record("ada")));
    assert_eq!(remote.fetches(), 1, "a hot hit must not fetch");
    assert!(
        hot_latency < cold_latency,
        "an mmap hit should beat the remote"
    );
    println!("get(ada)       {hot_latency:>10.1?}  mmap hit, no fetch");

    // A span, done soundly: borrow the bytes in place. The view holds the
    // tier's lock so eviction cannot invalidate it, and its address lands
    // inside the mapping itself — zero copy from origin.
    let mapping = hot.mapping_range();
    let view = block_on(hot.get_ref(&"ada".to_owned()))?.expect("ada must be in the mmap tier");
    assert_eq!(
        &*view,
        record("ada").as_slice(),
        "view must read the same bytes"
    );
    let address = view.as_ptr().addr();
    assert!(
        mapping.contains(&address),
        "view must point into the mapping — no copy was made"
    );
    drop(view);
    println!(
        "get_ref(ada)              zero-copy view at {address:#x}, inside the mapping {:#x}..{:#x}",
        mapping.start, mapping.end
    );

    timed_get(cache, "grace");
    timed_get(cache, "barbara");
    assert_eq!(remote.fetches(), 3);
    // Two slots, three promoted entries: ada (the oldest) must have been
    // displaced from the mmap and rolled over onto disk — not dropped.
    assert_eq!(
        block_on(hot.get(&"ada".to_owned()))?,
        None,
        "ada must be evicted from the mmap tier"
    );
    assert_eq!(
        block_on(warm.get(&"ada".to_owned()))?,
        Some(record("ada")),
        "ada must survive on disk"
    );
    println!("get(grace), get(barbara)      mmap overflow: ada rolled from mmap to disk");

    let (value, warm_latency) = timed_get(cache, "ada");
    assert_eq!(value, Some(record("ada")));
    assert_eq!(
        remote.fetches(),
        3,
        "a rolled-over read must be served from disk, not the remote"
    );
    assert_eq!(
        block_on(warm.get(&"grace".to_owned()))?,
        Some(record("grace")),
        "grace, displaced by ada's promotion, must roll to disk"
    );
    println!(
        "get(ada)       {warm_latency:>10.1?}  disk hit, promoted back to mmap; grace rolled to disk"
    );
    Ok(())
}

/// The mmap tier is file-backed: after a "restart" it serves its previous
/// contents without touching the remote.
fn prove_restart(dir: &Path, warm: &Arc<DiskTier>, remote: &Arc<RemoteStore>) -> io::Result<()> {
    let fetches_before = remote.fetches();

    // Reopen the same file: the index is rebuilt by scanning the mapping.
    let hot = Arc::new(SlotMmapTier::open(
        &dir.join("hot.mmap"),
        HOT_SLOTS,
        SLOT_SIZE,
    )?);
    assert_eq!(
        block_on(hot.get(&"ada".to_owned()))?,
        Some(record("ada")),
        "mmap must still hold ada after reopen"
    );
    assert_eq!(
        block_on(hot.get(&"diana".to_owned()))?,
        Some(record("diana")),
        "mmap must still hold diana after reopen"
    );

    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(warm))
        .tier(Arc::clone(remote))
        .build();

    let (value, hot_latency) = timed_get(&cache, "ada");
    assert_eq!(value, Some(record("ada")));
    let (value, _) = timed_get(&cache, "grace");
    assert_eq!(value, Some(record("grace")));
    assert_eq!(
        remote.fetches(),
        fetches_before,
        "post-restart reads must be served by mmap and disk only"
    );
    println!(
        "get(ada)       {hot_latency:>10.1?}  served from the reopened mmap — hot survived the restart"
    );
    println!("get(grace)                    served from disk — still no remote fetch");
    Ok(())
}
