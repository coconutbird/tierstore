//! Per-key outcome reports for batched router operations.
//!
//! A batch can partially succeed: some keys hit, some are confirmed absent,
//! and — when a tier fails mid-batch — some are simply unknown. Collapsing
//! that into a single `Result` either lies about the unknown keys or throws
//! away the resolved ones, so these report types keep one status per key
//! alongside whatever tier failures occurred.

use crate::error::TierFailure;

/// Outcome of one key in a batched read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyStatus<V> {
    /// The key was found.
    Hit {
        /// Tier that served it (0 is topmost).
        tier: usize,
        /// The value.
        value: V,
    },
    /// Every tier answered and none held the key: confirmed absent.
    Miss,
    /// The key went unresolved while at least one tier failed — absence is
    /// unconfirmed and must not be treated as a miss.
    Inconclusive,
}

impl<V> KeyStatus<V> {
    /// The value, if this key hit.
    #[must_use]
    pub const fn value(&self) -> Option<&V> {
        match self {
            Self::Hit { value, .. } => Some(value),
            Self::Miss | Self::Inconclusive => None,
        }
    }

    /// Consumes the status, returning the value if this key hit.
    #[must_use]
    pub fn into_value(self) -> Option<V> {
        match self {
            Self::Hit { value, .. } => Some(value),
            Self::Miss | Self::Inconclusive => None,
        }
    }
}

/// Result of a batched read: one status per key, in request order, plus the
/// tier failures encountered while probing.
#[derive(Debug)]
pub struct ReadReport<V> {
    /// One status per requested key, in request order.
    pub statuses: Vec<KeyStatus<V>>,
    /// Tiers that failed while probing. Empty means every status is fully
    /// trustworthy (all misses are confirmed).
    pub failures: Vec<TierFailure>,
}

impl<V> ReadReport<V> {
    /// Whether every tier answered — no status was degraded to
    /// [`KeyStatus::Inconclusive`].
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.failures.is_empty()
    }

    /// Collapses to plain values, discarding hit provenance and the
    /// miss/inconclusive distinction.
    #[must_use]
    pub fn into_values(self) -> Vec<Option<V>> {
        self.statuses
            .into_iter()
            .map(KeyStatus::into_value)
            .collect()
    }
}

/// Result of a batched delete: one "was it present" flag per key plus tier
/// failures.
///
/// Any failure here is serious: a copy surviving in an unreachable tier can
/// resurrect the key later.
#[derive(Debug)]
pub struct DeleteReport {
    /// Per key, in request order: whether any answering tier held it.
    pub removed: Vec<bool>,
    /// Tiers that failed; a key may still exist in those.
    pub failures: Vec<TierFailure>,
}

impl DeleteReport {
    /// Whether every tier processed every key.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.failures.is_empty()
    }
}
