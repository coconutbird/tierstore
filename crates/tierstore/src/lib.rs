//! Storage-tier router with batteries included.
//!
//! Layering, bottom to top:
//!
//! 1. [`tierstore_core`] — generic, `no_std`: capability traits
//!    ([`TierRead`], [`TierWrite`], [`TierList`]), routing [`Policy`], and
//!    the sans-io [`ReadFlow`] that makes routing decisions without I/O.
//! 2. [`Router`] — *mechanism*: drives the flow against real tiers, boxes
//!    heterogeneous backends behind one key/value type, unifies their errors.
//!    The router itself implements the tier traits, so routers nest inside
//!    routers (a "warm" tier can itself be a hierarchy).
//! 3. [`TieredCache`] and [`TieredStore`] — *semantics*: two presets built
//!    entirely on the router. The cache optimises availability (rollover,
//!    fall-through, single-flight); the store answers for the data
//!    (fail-fast reads, loud loss). The classic cache instantiation is hot
//!    ([`MemoryTier`]) over warm ([`DiskTier`]) over cold (your remote
//!    store).
//!
//! # Example
//!
//! ```
//! use std::num::NonZeroUsize;
//! use std::sync::Arc;
//! use tierstore::{MemoryTier, RouterError, TieredCache};
//!
//! # async fn demo() -> Result<(), RouterError> {
//! let hot = Arc::new(MemoryTier::bounded(NonZeroUsize::new(1024).unwrap()));
//! let cold = Arc::new(MemoryTier::unbounded()); // stand-in for your origin tier
//!
//! let cache: TieredCache<String, Vec<u8>> = TieredCache::builder()
//!     .tier(Arc::clone(&hot))
//!     .tier(Arc::clone(&cold))
//!     .build();
//!
//! cache.put("k".to_owned(), b"v".to_vec()).await?;
//! assert_eq!(cache.get(&"k".to_owned()).await?, Some(b"v".to_vec()));
//! # Ok(()) }
//! ```

mod cache;
mod disk;
mod error;
mod limited;
mod memory;
mod report;
mod router;
mod single_flight;
mod store;
mod verified;

pub use cache::{TieredCache, TieredCacheBuilder};
pub use disk::DiskTier;
pub use error::{BoxError, RouterError, TierFailure};
pub use limited::LimitedTier;
pub use memory::{MemoryRef, MemoryTier};
pub use report::{DeleteReport, KeyStatus, ReadReport};
pub use router::{Router, RouterBuilder};
pub use single_flight::{SingleFlight, SingleFlightGuard};
pub use store::{StoreError, TieredStore, TieredStoreBuilder};
pub use tierstore_core::{
    Displaced, OnReadError, Page, Policy, Probe, Promote, ReadFlow, ReadOutcome, ReadPolicy,
    ReadStep, Tier, TierList, TierRead, TierReadRef, TierWrite, WriteMode,
};
pub use verified::{VerifiedTier, VerifyError};
