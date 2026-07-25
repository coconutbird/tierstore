//! Hot/warm/cold walk-through: memory over disk over a mock remote store.
//!
//! Run with: `cargo run -p tierstore --example hot_warm_cold`

use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

use tierstore::{DiskTier, Displaced, MemoryTier, Tier, TierRead, TierWrite, TieredCache};

/// Mock "remote database": an in-memory map plus a fetch counter standing in
/// for the network round-trip we want to avoid.
struct RemoteDb {
    inner: MemoryTier<String, Vec<u8>>,
    fetches: AtomicUsize,
}

impl RemoteDb {
    fn new() -> Self {
        Self {
            inner: MemoryTier::unbounded(),
            fetches: AtomicUsize::new(0),
        }
    }

    fn fetches(&self) -> usize {
        self.fetches.load(Ordering::Relaxed)
    }
}

impl Tier for RemoteDb {
    type Key = String;
    type Value = Vec<u8>;
    type Error = std::convert::Infallible;

    fn name(&self) -> &'static str {
        "remote-db"
    }
}

impl TierRead for RemoteDb {
    fn get(
        &self,
        key: &String,
    ) -> impl Future<Output = Result<Option<Vec<u8>>, Self::Error>> + Send {
        self.fetches.fetch_add(1, Ordering::Relaxed);
        self.inner.get(key)
    }

    fn exists(&self, key: &String) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        self.inner.exists(key)
    }
}

impl TierWrite for RemoteDb {
    fn put(
        &self,
        key: String,
        value: Vec<u8>,
    ) -> impl Future<Output = Result<Displaced<String, Vec<u8>>, Self::Error>> + Send {
        self.inner.put(key, value)
    }

    fn delete(&self, key: &String) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        self.inner.delete(key)
    }
}

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

fn main() {
    let warm_root = std::env::temp_dir().join(format!("tierstore-example-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&warm_root);

    let hot = Arc::new(MemoryTier::bounded(NonZeroUsize::new(2).expect("nonzero")));
    let warm = Arc::new(DiskTier::open(&warm_root).expect("create warm dir"));
    let cold = Arc::new(RemoteDb::new());
    for name in ["ada", "grace", "barbara"] {
        block_on(cold.put(name.to_owned(), format!("record for {name}").into_bytes()))
            .expect("seed remote");
    }

    let cache = TieredCache::builder()
        .tier(Arc::clone(&hot))
        .tier(Arc::clone(&warm))
        .tier(Arc::clone(&cold))
        .build();

    println!(
        "hot = memory (capacity 2), warm = disk ({}), cold = mock remote db",
        warm_root.display()
    );
    println!();

    let get = |k: &str| {
        block_on(cache.get(&k.to_owned()))
            .expect("cache get")
            .is_some()
    };

    get("ada");
    println!(
        "get(ada)      -> promoted to hot          remote fetches = {}",
        cold.fetches()
    );
    get("ada");
    println!(
        "get(ada)      -> hot hit                  remote fetches = {}",
        cold.fetches()
    );
    get("grace");
    get("barbara");
    println!(
        "get(grace), get(barbara) fill hot         remote fetches = {}",
        cold.fetches()
    );

    let ada_in_hot = block_on(hot.get(&"ada".to_owned()))
        .expect("hot peek")
        .is_some();
    let ada_on_disk = block_on(warm.get(&"ada".to_owned()))
        .expect("warm peek")
        .is_some();
    println!("hot overflowed: ada in hot = {ada_in_hot}, ada rolled over to disk = {ada_on_disk}");

    get("ada");
    println!(
        "get(ada)      -> served from warm (disk)  remote fetches = {}",
        cold.fetches()
    );

    let _ = std::fs::remove_dir_all(&warm_root);
}
