//! moka-backed hot tier for `tierstore`.
//!
//! [`MokaTier`] puts a [`moka`] cache — sharded, mostly lock-free reads,
//! `TinyLFU` admission — behind the tier traits, as the recommended hot tier
//! for high-throughput deployments where `tierstore`'s zero-dependency
//! `MemoryTier` (a single-mutex reference implementation) would contend.
//!
//! # Displacement caveat (read this before choosing it)
//!
//! moka evicts internally and asynchronously: an insert never reports what
//! it displaced, so [`TierWrite::put`] on this tier always returns an empty
//! displacement list and **rollover demotion through it is a no-op**. That
//! makes it a first-class citizen for *inclusive* hierarchies (every fill
//! writes the lower tiers too, so an eviction loses nothing — the s3cache
//! shape) and the wrong tool for *exclusive* rollover hierarchies, where
//! evicted entries must demote to survive (use `MemoryTier` there).
//!
//! Reads refresh recency/frequency (that is moka's job); existence checks
//! do not. The wrapped cache is exposed via [`MokaTier::inner`], and the
//! [`moka`] crate is re-exported for configuring one from scratch
//! (TTL/TTI, listeners) before wrapping it with [`MokaTier::new`].
//!
//! # Example
//!
//! ```
//! use tierstore_moka::MokaTier;
//!
//! let hot: MokaTier<String, Vec<u8>> =
//!     MokaTier::bounded_weighted(64 * 1024 * 1024, |_key: &String, value: &Vec<u8>| {
//!         u32::try_from(value.len()).unwrap_or(u32::MAX)
//!     });
//! ```

//! [`TierWrite::put`]: tierstore_core::TierWrite::put

mod tier;

pub use moka;
pub use tier::MokaTier;
