//! Kernel-evictable, zero-copy warm tier for `tierstore`.
//!
//! [`MmapDiskTier`] stores one file per key and serves values as
//! [`Bytes`] backed directly by a read-only memory map
//! (`Bytes::from_owner`): reads copy nothing, clones are refcount bumps,
//! and RAM residency is the kernel's page cache — clean pages are evicted
//! under memory pressure and re-faulted from disk on the next touch. This
//! is the warm-tier model from shardstore's cache layer: bytes live on
//! local disk, not in anonymous RAM.
//!
//! # Immutability contract (what makes the mapping sound)
//!
//! Files under the tier's root are written exactly once via
//! temp-file-then-rename and are **never truncated or mutated in place**.
//! An overwrite swaps the directory entry to a fresh inode, so any inode
//! already mapped keeps its contents and length for as long as anything
//! references it. Two consequences:
//!
//! - **Snapshot semantics:** `Bytes` obtained before an overwrite continue
//!   to read the *old* value; reads after the overwrite see the new one.
//! - The tier must own its root directory: external processes truncating
//!   files in it would violate the mapping's safety contract.
//!
//! # Bounding disk
//!
//! [`MmapDiskTier::open_bounded`] caps the accounted bytes on disk: puts
//! that overflow the budget FIFO-evict the oldest entries and return them
//! as displaced — mapped from the evicted files themselves (an unlinked
//! inode stays readable while mapped), so rollover demotion is zero-copy.
//! Accounting is rebuilt from file sizes and mtimes on reopen.
//!
//! # Example
//!
//! ```no_run
//! use bytes::Bytes;
//! use tierstore_core::{TierRead, TierWrite};
//! use tierstore_mmap::MmapDiskTier;
//!
//! # async fn demo() -> std::io::Result<()> {
//! let tier = MmapDiskTier::open("/var/cache/myapp")?;
//! tier.put("key".to_owned(), Bytes::from_static(b"value")).await?;
//! // Served straight from the mapping: zero copy, kernel-evictable pages.
//! assert!(tier.get(&"key".to_owned()).await?.is_some());
//! # Ok(()) }
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::UNIX_EPOCH;

use bytes::Bytes;
use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierReadRange, TierWrite};

/// File-per-key tier serving mmap-backed, kernel-evictable [`Bytes`].
///
/// Keys are hex-encoded into file names (no path traversal, no
/// case-sensitivity hazards); values are raw bytes. See the [module
/// docs](self) for the immutability/snapshot contract and disk bounding.
pub struct MmapDiskTier {
    root: PathBuf,
    budget: Option<NonZeroU64>,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// Live mappings by key. A hit clones the `Bytes` (refcount bump into
    /// the same mapping). Entries are replaced on overwrite and removed on
    /// delete/eviction; outstanding clones keep their (old) inodes mapped.
    maps: HashMap<String, Bytes>,
    /// Accounted on-disk size per key.
    sizes: HashMap<String, u64>,
    /// FIFO eviction order (rebuilt from mtimes on reopen).
    order: VecDeque<String>,
    /// Sum of accounted sizes.
    total: u64,
}

impl fmt::Debug for MmapDiskTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.lock();
        f.debug_struct("MmapDiskTier")
            .field("root", &self.root)
            .field("budget", &self.budget)
            .field("entries", &inner.sizes.len())
            .field("disk_usage", &inner.total)
            .finish_non_exhaustive()
    }
}

impl MmapDiskTier {
    /// Opens (creating if needed) an unbounded tier: it never displaces.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the root cannot be created or scanned.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        Self::open_inner(root.into(), None)
    }

    /// Opens a byte-bounded tier: puts that push the accounted total over
    /// `budget` FIFO-evict the oldest entries and return them as displaced,
    /// mapped from the evicted files themselves — zero-copy demotion.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the root cannot be created or scanned.
    pub fn open_bounded(root: impl Into<PathBuf>, budget: NonZeroU64) -> io::Result<Self> {
        Self::open_inner(root.into(), Some(budget))
    }

    fn open_inner(root: PathBuf, budget: Option<NonZeroU64>) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        let inner = scan(&root)?;
        Ok(Self {
            root,
            budget,
            inner: Mutex::new(inner),
        })
    }

    /// The tier's root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Total accounted bytes currently on disk.
    #[must_use]
    pub fn disk_usage(&self) -> u64 {
        self.lock().total
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(hex_encode(key.as_bytes()))
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Maps `path` read-only as `Bytes`. `Ok(None)` when the file does not
    /// exist; empty files are served without mapping (zero-length maps are
    /// not portable).
    fn map_file(path: &Path) -> io::Result<Option<Bytes>> {
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if file.metadata()?.len() == 0 {
            return Ok(Some(Bytes::new()));
        }
        // SAFETY: files under the tier root are written once via
        // tmp+rename and never truncated in place — an overwrite swaps the
        // directory entry to a fresh inode, so an inode we map keeps its
        // length for the mapping's lifetime. External mutation of the root
        // violates the tier's documented ownership contract.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Some(Bytes::from_owner(mmap)))
    }
}

/// Rebuilds accounting from the directory: sizes from file lengths, FIFO
/// order approximated by mtime.
fn scan(root: &Path) -> io::Result<Inner> {
    let mut found = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(bytes) = hex_decode(name) else {
            continue;
        };
        let Ok(key) = String::from_utf8(bytes) else {
            continue;
        };
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue; // e.g. stray subdirectories from a foreign layout
        }
        let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
        found.push((key, metadata.len(), modified));
    }
    found.sort_by_key(|(_, _, modified)| *modified);
    let mut inner = Inner::default();
    for (key, len, _) in found {
        inner.total = inner.total.saturating_add(len);
        inner.sizes.insert(key.clone(), len);
        inner.order.push_back(key);
    }
    Ok(inner)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

fn hex_decode(name: &str) -> Option<Vec<u8>> {
    if !name.len().is_multiple_of(2) {
        return None;
    }
    (0..name.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(name.get(i..i + 2)?, 16).ok())
        .collect()
}

impl Tier for MmapDiskTier {
    type Key = String;
    type Value = Bytes;
    type Error = io::Error;

    fn name(&self) -> &'static str {
        "mmap-disk"
    }
}

impl TierRead for MmapDiskTier {
    async fn get(&self, key: &String) -> io::Result<Option<Bytes>> {
        // Bind before branching so the map lock is not held past the probe.
        let cached = self.lock().maps.get(key).cloned();
        if let Some(bytes) = cached {
            return Ok(Some(bytes));
        }
        // Map outside the lock; a racing get may map the same file twice,
        // which is benign (both mappings are valid, last insert wins).
        Ok(Self::map_file(&self.path_for(key))?.inspect(|bytes| {
            self.lock().maps.insert(key.clone(), bytes.clone());
        }))
    }

    async fn exists(&self, key: &String) -> io::Result<bool> {
        if self.lock().maps.contains_key(key) {
            return Ok(true);
        }
        self.path_for(key).try_exists()
    }
}

impl TierReadRange for MmapDiskTier {
    /// Zero-copy ranged read: a refcounted slice of the mapping itself.
    async fn read_range(&self, key: &String, range: Range<u64>) -> io::Result<Option<Bytes>> {
        let Some(bytes) = self.get(key).await? else {
            return Ok(None);
        };
        let bounds = usize::try_from(range.start)
            .ok()
            .zip(usize::try_from(range.end).ok());
        let Some((start, end)) = bounds else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "range too large",
            ));
        };
        if start > end || end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "range out of bounds",
            ));
        }
        Ok(Some(bytes.slice(start..end)))
    }
}

impl TierWrite for MmapDiskTier {
    /// Writes via temp-file-then-rename, then remaps: the cached value is
    /// file-backed, so the caller's (anonymous-RAM) input can be dropped
    /// and residency shifts to evictable page cache immediately. Under a
    /// budget, overflow FIFO-evicts oldest entries and returns them.
    async fn put(&self, key: String, value: Bytes) -> io::Result<Displaced<String, Bytes>> {
        let path = self.path_for(&key);
        // Hex names contain no `.`, so the temp name cannot collide with a
        // real entry (and `list` skips anything that fails hex decoding).
        let tmp = path.with_extension("tmp");
        let len = value.len() as u64;
        fs::write(&tmp, &value)?;
        fs::rename(&tmp, &path)?;
        drop(value);
        let mapped = Self::map_file(&path)?;

        let mut displaced = Displaced::new();
        let mut inner = self.lock();
        if let Some(bytes) = mapped {
            inner.maps.insert(key.clone(), bytes);
        }
        if let Some(old) = inner.sizes.insert(key.clone(), len) {
            inner.total = inner.total.saturating_sub(old);
        } else {
            inner.order.push_back(key);
        }
        inner.total = inner.total.saturating_add(len);
        if let Some(budget) = self.budget {
            while inner.total > budget.get() {
                let Some(oldest) = inner.order.pop_front() else {
                    break;
                };
                // Map before unlinking: the unlinked inode stays alive
                // while mapped, so the displaced value is served
                // zero-copy from the very file being evicted. The lock
                // stays held so a concurrent re-put cannot race the
                // unlink; a mapping failure drops that entry's value
                // (eviction proceeds) rather than corrupting accounting.
                let old_path = self.path_for(&oldest);
                let bytes = inner
                    .maps
                    .remove(&oldest)
                    .or_else(|| Self::map_file(&old_path).ok().flatten());
                if let Some(size) = inner.sizes.remove(&oldest) {
                    inner.total = inner.total.saturating_sub(size);
                }
                let _ = fs::remove_file(&old_path);
                if let Some(bytes) = bytes {
                    displaced.push((oldest, bytes));
                }
            }
        }
        // The lock spans accounting, eviction, and unlink on purpose (see
        // above); release it explicitly before returning.
        drop(inner);
        Ok(displaced)
    }

    async fn delete(&self, key: &String) -> io::Result<bool> {
        {
            let mut inner = self.lock();
            inner.maps.remove(key);
            if let Some(size) = inner.sizes.remove(key) {
                inner.total = inner.total.saturating_sub(size);
                inner.order.retain(|k| k != key);
            }
        }
        match fs::remove_file(self.path_for(key)) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }
}

impl TierList for MmapDiskTier {
    type Cursor = usize;

    async fn list(&self, cursor: Option<usize>, limit: usize) -> io::Result<Page<String, usize>> {
        let mut all = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.metadata()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(bytes) = hex_decode(name) else {
                continue;
            };
            let Ok(key) = String::from_utf8(bytes) else {
                continue;
            };
            all.push(key);
        }
        // Directory order is arbitrary; sort so paging is deterministic.
        all.sort_unstable();
        let offset = cursor.unwrap_or(0);
        let keys: Vec<String> = all.iter().skip(offset).take(limit).cloned().collect();
        let end = offset.saturating_add(keys.len());
        let next = (limit > 0 && end < all.len()).then_some(end);
        Ok(Page { keys, next })
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

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("tierstore-mmap-test-{}-{name}", std::process::id()))
    }

    fn key(s: &str) -> String {
        s.to_owned()
    }

    #[test]
    fn round_trips_and_persists_across_reopen() {
        let root = temp_root("roundtrip");
        let _ = fs::remove_dir_all(&root);
        {
            let tier = MmapDiskTier::open(&root).expect("open");
            block_on(tier.put(key("k"), Bytes::from_static(b"hello"))).expect("put");
            assert_eq!(
                block_on(tier.get(&key("k"))).expect("get"),
                Some(Bytes::from_static(b"hello"))
            );
            assert!(block_on(tier.exists(&key("k"))).expect("exists"));
        }
        // A fresh instance (empty mapping cache) maps the file on demand.
        let tier = MmapDiskTier::open(&root).expect("reopen");
        assert_eq!(
            block_on(tier.get(&key("k"))).expect("get after reopen"),
            Some(Bytes::from_static(b"hello"))
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn overwrite_gives_snapshot_semantics() {
        let root = temp_root("snapshot");
        let _ = fs::remove_dir_all(&root);
        let tier = MmapDiskTier::open(&root).expect("open");

        block_on(tier.put(key("k"), Bytes::from_static(b"old"))).expect("put old");
        let before = block_on(tier.get(&key("k")))
            .expect("get")
            .expect("present");
        block_on(tier.put(key("k"), Bytes::from_static(b"new"))).expect("put new");

        // The pre-overwrite handle still reads the old inode; new reads see
        // the new value.
        assert_eq!(before.as_ref(), b"old");
        assert_eq!(
            block_on(tier.get(&key("k"))).expect("get"),
            Some(Bytes::from_static(b"new"))
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn empty_values_and_deletes_work() {
        let root = temp_root("empty");
        let _ = fs::remove_dir_all(&root);
        let tier = MmapDiskTier::open(&root).expect("open");

        block_on(tier.put(key("empty"), Bytes::new())).expect("put empty");
        assert_eq!(
            block_on(tier.get(&key("empty"))).expect("get"),
            Some(Bytes::new())
        );
        assert!(block_on(tier.delete(&key("empty"))).expect("delete"));
        assert!(!block_on(tier.delete(&key("empty"))).expect("second delete"));
        assert_eq!(block_on(tier.get(&key("empty"))).expect("get"), None);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bounded_tier_evicts_fifo_with_zero_copy_displacement() {
        let root = temp_root("bounded");
        let _ = fs::remove_dir_all(&root);
        let tier =
            MmapDiskTier::open_bounded(&root, NonZeroU64::new(10).expect("nonzero")).expect("open");

        block_on(tier.put(key("a"), Bytes::from_static(b"aaaa"))).expect("put a");
        block_on(tier.put(key("b"), Bytes::from_static(b"bbbb"))).expect("put b");
        // 4 + 4 + 4 = 12 > 10: the oldest entry rolls out, its value served
        // from the (now unlinked) file it lived in.
        let displaced = block_on(tier.put(key("c"), Bytes::from_static(b"cccc"))).expect("put c");
        assert_eq!(displaced.len(), 1);
        assert_eq!(displaced[0].0, key("a"));
        assert_eq!(displaced[0].1.as_ref(), b"aaaa");
        assert_eq!(block_on(tier.get(&key("a"))).expect("get a"), None);
        assert_eq!(tier.disk_usage(), 8);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bounded_accounting_survives_reopen() {
        let root = temp_root("reopen-bounded");
        let _ = fs::remove_dir_all(&root);
        {
            let tier = MmapDiskTier::open(&root).expect("open");
            block_on(tier.put(key("x"), Bytes::from_static(b"xxxx"))).expect("put x");
            block_on(tier.put(key("y"), Bytes::from_static(b"yyyy"))).expect("put y");
        }
        let tier = MmapDiskTier::open_bounded(&root, NonZeroU64::new(10).expect("nonzero"))
            .expect("reopen");
        assert_eq!(tier.disk_usage(), 8, "accounting must rebuild from disk");
        // Overflow evicts entries whose accounting was rebuilt from disk.
        let displaced = block_on(tier.put(key("z"), Bytes::from_static(b"zzzz"))).expect("put z");
        assert_eq!(displaced.len(), 1);
        assert_eq!(tier.disk_usage(), 8);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ranged_reads_slice_the_mapping() {
        let root = temp_root("range");
        let _ = fs::remove_dir_all(&root);
        let tier = MmapDiskTier::open(&root).expect("open");
        block_on(tier.put(key("k"), Bytes::from_static(b"0123456789"))).expect("put");

        assert_eq!(
            block_on(tier.read_range(&key("k"), 2..5)).expect("range"),
            Some(Bytes::from_static(b"234"))
        );
        assert!(block_on(tier.read_range(&key("k"), 8..12)).is_err());
        assert_eq!(
            block_on(tier.read_range(&key("missing"), 0..1)).expect("missing"),
            None
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn listing_pages_sorted_keys() {
        let root = temp_root("list");
        let _ = fs::remove_dir_all(&root);
        let tier = MmapDiskTier::open(&root).expect("open");
        for name in ["b", "a", "c"] {
            block_on(tier.put(key(name), Bytes::from_static(b"x"))).expect("put");
        }
        let first = block_on(tier.list(None, 2)).expect("list");
        assert_eq!(first.keys, vec![key("a"), key("b")]);
        let second = block_on(tier.list(first.next, 2)).expect("list");
        assert_eq!(second.keys, vec![key("c")]);
        assert_eq!(second.next, None);
        let _ = fs::remove_dir_all(&root);
    }
}
