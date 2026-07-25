//! Router-level errors.
//!
//! Each tier keeps its own error type; the router boxes them and records
//! *which* tier failed, because in a hierarchy the identity of the failing
//! layer is half the diagnosis.

use std::error::Error as StdError;
use std::fmt;

/// Boxed tier error as unified by the router.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// One tier's failure, with the tier's position and name attached.
#[derive(Debug)]
pub struct TierFailure {
    tier: usize,
    name: String,
    source: BoxError,
}

impl TierFailure {
    pub(crate) fn new(tier: usize, name: &str, source: BoxError) -> Self {
        Self {
            tier,
            name: name.to_owned(),
            source,
        }
    }

    /// Index of the failing tier (0 is topmost).
    #[must_use]
    pub const fn tier(&self) -> usize {
        self.tier
    }

    /// Diagnostic name of the failing tier.
    #[must_use]
    pub fn tier_name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for TierFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tier {} ({}) failed", self.tier, self.name)
    }
}

impl StdError for TierFailure {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        let source: &(dyn StdError + 'static) = &*self.source;
        Some(source)
    }
}

/// Error returned by [`Router`](crate::Router) operations.
///
/// The variants encode *how much* of the hierarchy the operation reached,
/// because partial cross-tier failures have consequences plain errors do not
/// (unconfirmed misses, stale copies, resurrected deletes).
#[derive(Debug)]
pub enum RouterError {
    /// A single tier failed and the operation was aborted (fail-fast reads,
    /// write-through aborts). Tiers not yet reached were left untouched.
    Tier(TierFailure),
    /// A fall-through read found no hit while at least one tier failed:
    /// absence is unconfirmed, so this is deliberately *not* `Ok(None)`.
    Inconclusive(Vec<TierFailure>),
    /// A multi-tier operation (delete, write-around invalidation) failed on
    /// some tiers after succeeding on others. Cross-tier state may now be
    /// inconsistent: stale upper copies can shadow new values or resurrect
    /// deleted keys.
    Partial(Vec<TierFailure>),
    /// A write was routed to a hierarchy with no writable tier (every tier
    /// was added via `read_only_tier`).
    ReadOnly,
}

impl RouterError {
    /// The individual tier failures behind this error.
    #[must_use]
    pub fn failures(&self) -> &[TierFailure] {
        match self {
            Self::Tier(failure) => std::slice::from_ref(failure),
            Self::Inconclusive(failures) | Self::Partial(failures) => failures,
            Self::ReadOnly => &[],
        }
    }
}

impl fmt::Display for RouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tier(failure) => write!(f, "{failure}"),
            Self::Inconclusive(failures) => write!(
                f,
                "read inconclusive: no hit and {} tier(s) failed",
                failures.len()
            ),
            Self::Partial(failures) => write!(
                f,
                "operation partially failed on {} tier(s); cross-tier state may be inconsistent",
                failures.len()
            ),
            Self::ReadOnly => write!(f, "write rejected: the router has no writable tier"),
        }
    }
}

impl StdError for RouterError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Tier(failure) => Some(failure),
            Self::Inconclusive(failures) | Self::Partial(failures) => {
                failures.first().map(|f| f as &(dyn StdError + 'static))
            }
            Self::ReadOnly => None,
        }
    }
}
