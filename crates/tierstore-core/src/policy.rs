//! Routing policy: the *semantics* of a tier hierarchy, kept separate from
//! the mechanism that executes them.
//!
//! The same router becomes an inclusive read-through cache, an exclusive
//! rollover cache, or a plain fallback chain purely by changing [`Policy`].

/// What to do with a value found below the top tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Promote {
    /// Leave the value where it was found. The hierarchy is a pure fallback
    /// chain — the neutral default; the cache/store presets pick their own.
    #[default]
    Never,
    /// Copy the hit into the topmost tier only. Combined with
    /// demotion-on-evict this yields an *exclusive* hierarchy: each entry
    /// lives in roughly one cache tier, maximising total capacity.
    TopOnly,
    /// Copy the hit into every tier above the one that answered, yielding an
    /// *inclusive* hierarchy: upper tiers duplicate lower ones.
    AllAbove,
}

/// How the read path treats a tier that *fails* (as opposed to one that
/// merely misses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnReadError {
    /// Skip the failing tier and keep probing downward. A miss after any
    /// failure is reported as inconclusive rather than as a confirmed miss,
    /// because the failed tier might have held the key.
    #[default]
    FallThrough,
    /// Abort the read on the first tier error.
    FailFast,
}

/// How write-through treats a tier that rejects a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnWriteError {
    /// Abort on the first tier error; lower tiers keep what they already
    /// accepted. The authoritative default.
    #[default]
    FailFast,
    /// Skip the failing tier and keep writing the rest — cache-fill
    /// semantics, where a failed fill is a capacity loss, not an operation
    /// failure. Errors still surface in per-tier stats. Applies to the
    /// write-through fan-out; a write-around bottom write always fails
    /// loudly (it is the only real write).
    BestEffort,
}

/// How writes propagate through the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WriteMode {
    /// Write every tier, bottom-up, so an upper tier never holds a key its
    /// lower tiers failed to accept. Write-back (dirty tracking + deferred
    /// flush) is deliberately out of scope for now.
    #[default]
    WriteThrough,
    /// Write only the bottommost tier and *invalidate* the key in the tiers
    /// above (a stale upper copy would otherwise shadow the new value).
    WriteAround,
}

/// Read-path policy: promotion strategy plus error handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadPolicy {
    /// What to do with hits found below the top tier.
    pub promote: Promote,
    /// How to treat failing tiers while probing.
    pub on_error: OnReadError,
}

/// Complete routing policy for a tier hierarchy.
///
/// The default is the *neutral fallback chain*: no promotion, no demotion,
/// write-through, fall-through on read errors — the router moves no data
/// around on its own. The semantic presets (`TieredCache`, `TieredStore` in
/// the `tierstore` crate) each pick their own flavour on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Policy {
    /// Read-path behaviour.
    pub read: ReadPolicy,
    /// Write-path behaviour.
    pub write: WriteMode,
    /// Tolerance for tiers that reject writes during write-through.
    pub on_write_error: OnWriteError,
    /// When an insert displaces entries from a tier, push them into the next
    /// tier down (cascading as needed) instead of dropping them. Entries
    /// displaced from the bottommost tier are evicted outright either way.
    pub demote_displaced: bool,
}
