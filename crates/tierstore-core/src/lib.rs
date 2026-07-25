//! Core abstractions for `tierstore`, a generic storage-tier router.
//!
//! A *tier* is any storage medium that can answer for keys: an in-memory map,
//! a directory on disk, a remote database. Tiers implement small capability
//! traits ([`TierRead`], and with the `alloc` feature [`tier::TierWrite`] /
//! [`tier::TierList`]) rather than one fat trait, so a backend only claims
//! what it can actually do.
//!
//! The routing *decisions* live here too, but as pure logic with zero I/O:
//! [`ReadFlow`] is a sans-io state machine that tells a driver which tier to
//! probe next, when to promote a hit upward, and how to classify the outcome
//! (including the honest [`ReadOutcome::Inconclusive`] case, where a miss
//! cannot be trusted because a tier failed). Drivers — like the `Router` in
//! the `tierstore` crate — execute those instructions against real tiers.
//!
//! This crate is `no_std` (tests aside). Without the `alloc` feature you keep
//! the read capability, all policy types, and the full read flow.
//!
//! # Example: driving the sans-io read flow
//!
//! The flow is pure logic — this example is a complete, runnable read over
//! three imaginary tiers where tier 1 holds the value:
//!
//! ```
//! use tierstore_core::{
//!     OnReadError, Probe, Promote, ReadFlow, ReadOutcome, ReadPolicy, ReadStep,
//! };
//!
//! let mut flow = ReadFlow::new(3, ReadPolicy {
//!     promote: Promote::AllAbove,
//!     on_error: OnReadError::FallThrough,
//! });
//!
//! assert_eq!(flow.step(), ReadStep::Get { tier: 0 });
//! flow.on_get(Probe::Miss);                 // tier 0: miss
//! assert_eq!(flow.step(), ReadStep::Get { tier: 1 });
//! flow.on_get(Probe::Hit);                  // tier 1: hit!
//! assert_eq!(flow.step(), ReadStep::Promote { tier: 0 });
//! flow.on_promote();                        // driver copies the hit upward
//! assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Hit { tier: 1 }));
//! ```

#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod policy;
pub mod read;
pub mod tier;

pub use policy::{OnReadError, OnWriteError, Policy, Promote, ReadPolicy, WriteMode};
pub use read::{Probe, ReadFlow, ReadOutcome, ReadStep};
#[cfg(feature = "alloc")]
pub use tier::{Displaced, Page, TierList, TierWrite};
pub use tier::{Tier, TierRead, TierReadRange, TierReadRef};
