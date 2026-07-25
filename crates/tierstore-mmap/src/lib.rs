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

//! [`Bytes`]: bytes::Bytes

mod tier;

pub use tier::MmapDiskTier;
