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
//! The tier is unbounded (it never displaces); bound the tiers above it and
//! let rollover land here.
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

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use bytes::Bytes;
use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierWrite};

/// File-per-key tier serving mmap-backed, kernel-evictable [`Bytes`].
///
/// Keys are hex-encoded into file names (no path traversal, no
/// case-sensitivity hazards); values are raw bytes. See the [module
/// docs](self) for the immutability/snapshot contract.
pub struct MmapDiskTier {
    root: PathBuf,
    /// Live mappings by key. A hit clones the `Bytes` (refcount bump into
    /// the same mapping). Entries are replaced on overwrite and removed on
    /// delete; outstanding clones keep their (old) inodes mapped.
    maps: Mutex<HashMap<String, Bytes>>,
}

impl fmt::Debug for MmapDiskTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapDiskTier")
            .field("root", &self.root)
            .field("mapped", &self.lock().len())
            .finish_non_exhaustive()
    }
}

impl MmapDiskTier {
    /// Opens (creating if needed) the tier's root directory.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the directory cannot be created.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            maps: Mutex::new(HashMap::new()),
        })
    }

    /// The tier's root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(hex_encode(key.as_bytes()))
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, Bytes>> {
        self.maps.lock().unwrap_or_else(PoisonError::into_inner)
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
        let cached = self.lock().get(key).cloned();
        if let Some(bytes) = cached {
            return Ok(Some(bytes));
        }
        // Map outside the lock; a racing get may map the same file twice,
        // which is benign (both mappings are valid, last insert wins).
        Ok(Self::map_file(&self.path_for(key))?.inspect(|bytes| {
            self.lock().insert(key.clone(), bytes.clone());
        }))
    }

    async fn exists(&self, key: &String) -> io::Result<bool> {
        if self.lock().contains_key(key) {
            return Ok(true);
        }
        self.path_for(key).try_exists()
    }
}

impl TierWrite for MmapDiskTier {
    /// Writes via temp-file-then-rename, then remaps: the cached value is
    /// file-backed, so the caller's (anonymous-RAM) input can be dropped
    /// and residency shifts to evictable page cache immediately.
    async fn put(&self, key: String, value: Bytes) -> io::Result<Displaced<String, Bytes>> {
        let path = self.path_for(&key);
        // Hex names contain no `.`, so the temp name cannot collide with a
        // real entry (and `list` skips anything that fails hex decoding).
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, &value)?;
        fs::rename(&tmp, &path)?;
        drop(value);
        if let Some(bytes) = Self::map_file(&path)? {
            self.lock().insert(key, bytes);
        }
        Ok(Displaced::new())
    }

    async fn delete(&self, key: &String) -> io::Result<bool> {
        self.lock().remove(key);
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
            let name = entry?.file_name();
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
