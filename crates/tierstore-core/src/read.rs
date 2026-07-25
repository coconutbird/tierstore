//! The sans-io read flow.
//!
//! [`ReadFlow`] encodes every routing decision on the read path — probe
//! order, promotion, error classification — as a pure state machine over
//! *tier indices*. It performs no I/O and never touches keys or values; a
//! driver executes its instructions against real tiers and reports what
//! happened. This keeps the actual routing logic unit-testable without any
//! storage backend, executor, or even `alloc`.
//!
//! Driver loop:
//!
//! ```text
//! loop {
//!     match flow.step() {
//!         Get { tier }     => flow.on_get(probe result of tiers[tier].get(key)),
//!         Promote { tier } => { tiers[tier].put(key, hit value); flow.on_promote(); }
//!         Done(outcome)    => break outcome,
//!     }
//! }
//! ```
//!
//! Tier `0` is the topmost (fastest) tier; probing proceeds downward.

use crate::policy::{OnReadError, Promote, ReadPolicy};

/// The driver's next instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadStep {
    /// Call `get` on tier `tier` and report the result via
    /// [`ReadFlow::on_get`].
    Get {
        /// Tier to probe.
        tier: usize,
    },
    /// Insert the hit value into tier `tier` (best effort), then call
    /// [`ReadFlow::on_promote`]. A failed promotion must not fail the read.
    Promote {
        /// Tier to copy the hit into.
        tier: usize,
    },
    /// The read is finished.
    Done(ReadOutcome),
}

/// Final classification of a read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadOutcome {
    /// A tier answered with the value.
    Hit {
        /// Tier that answered.
        tier: usize,
    },
    /// Every tier answered and none held the key: a *confirmed* miss.
    Miss,
    /// No tier hit, but at least one failed while falling through — absence
    /// is unconfirmed and must not be treated as a trustworthy miss.
    Inconclusive,
    /// A tier failed under [`OnReadError::FailFast`].
    Failed {
        /// Tier that failed.
        tier: usize,
    },
}

/// What a probe of a single tier observed, as reported by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// The tier holds the key.
    Hit,
    /// The tier answered and does not hold the key.
    Miss,
    /// The tier failed to answer.
    Error,
}

#[derive(Debug, Clone, Copy)]
enum State {
    Probe { tier: usize },
    Promote { hit: usize, next: usize },
    Done(ReadOutcome),
}

/// Sans-io state machine for one read through a tier hierarchy.
///
/// See the [module docs](self) for the driver contract.
#[derive(Debug, Clone)]
pub struct ReadFlow {
    tiers: usize,
    policy: ReadPolicy,
    errors: usize,
    state: State,
}

impl ReadFlow {
    /// Starts a read over `tiers` tiers (index `0` is the topmost).
    ///
    /// A zero-tier hierarchy is vacuously a confirmed miss.
    #[must_use]
    pub const fn new(tiers: usize, policy: ReadPolicy) -> Self {
        let state = if tiers == 0 {
            State::Done(ReadOutcome::Miss)
        } else {
            State::Probe { tier: 0 }
        };
        Self {
            tiers,
            policy,
            errors: 0,
            state,
        }
    }

    /// The driver's current instruction. Idempotent; advance the flow with
    /// [`ReadFlow::on_get`] / [`ReadFlow::on_promote`].
    #[must_use]
    pub const fn step(&self) -> ReadStep {
        match self.state {
            State::Probe { tier } => ReadStep::Get { tier },
            State::Promote { next, .. } => ReadStep::Promote { tier: next },
            State::Done(outcome) => ReadStep::Done(outcome),
        }
    }

    /// Reports the result of the probe requested by [`ReadStep::Get`].
    ///
    /// # Panics
    ///
    /// Panics if the current step is not [`ReadStep::Get`]; that is a driver
    /// bug.
    pub fn on_get(&mut self, probe: Probe) {
        let State::Probe { tier } = self.state else {
            panic!("on_get called while the flow is not probing");
        };
        match probe {
            Probe::Hit => self.state = self.after_hit(tier),
            Probe::Miss => self.advance(tier),
            Probe::Error => match self.policy.on_error {
                OnReadError::FailFast => {
                    self.state = State::Done(ReadOutcome::Failed { tier });
                }
                OnReadError::FallThrough => {
                    self.errors += 1;
                    self.advance(tier);
                }
            },
        }
    }

    /// Reports that the promotion requested by [`ReadStep::Promote`] was
    /// attempted. Promotion is best effort, so there is nothing to report
    /// about its success.
    ///
    /// # Panics
    ///
    /// Panics if the current step is not [`ReadStep::Promote`]; that is a
    /// driver bug.
    pub fn on_promote(&mut self) {
        let State::Promote { hit, next } = self.state else {
            panic!("on_promote called while the flow is not promoting");
        };
        let done = match self.policy.promote {
            Promote::TopOnly => true,
            Promote::AllAbove => next + 1 == hit,
            // `after_hit` never enters the promote state under `Never`.
            Promote::Never => unreachable!("promotion state entered with Promote::Never"),
        };
        self.state = if done {
            State::Done(ReadOutcome::Hit { tier: hit })
        } else {
            State::Promote {
                hit,
                next: next + 1,
            }
        };
    }

    const fn advance(&mut self, tier: usize) {
        let next = tier + 1;
        self.state = if next < self.tiers {
            State::Probe { tier: next }
        } else if self.errors > 0 {
            State::Done(ReadOutcome::Inconclusive)
        } else {
            State::Done(ReadOutcome::Miss)
        };
    }

    const fn after_hit(&self, tier: usize) -> State {
        if tier == 0 || matches!(self.policy.promote, Promote::Never) {
            State::Done(ReadOutcome::Hit { tier })
        } else {
            // Fill top-first: the topmost tier serves the next read, so it
            // gets the value even if later promotions fail.
            State::Promote { hit: tier, next: 0 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn policy(promote: Promote, on_error: OnReadError) -> ReadPolicy {
        ReadPolicy { promote, on_error }
    }

    #[test]
    fn hit_on_top_tier_ends_immediately() {
        let mut flow = ReadFlow::new(3, policy(Promote::AllAbove, OnReadError::FallThrough));
        assert_eq!(flow.step(), ReadStep::Get { tier: 0 });
        flow.on_get(Probe::Hit);
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Hit { tier: 0 }));
    }

    #[test]
    fn hit_below_promotes_all_above_top_first() {
        let mut flow = ReadFlow::new(3, policy(Promote::AllAbove, OnReadError::FallThrough));
        flow.on_get(Probe::Miss);
        flow.on_get(Probe::Miss);
        assert_eq!(flow.step(), ReadStep::Get { tier: 2 });
        flow.on_get(Probe::Hit);
        assert_eq!(flow.step(), ReadStep::Promote { tier: 0 });
        flow.on_promote();
        assert_eq!(flow.step(), ReadStep::Promote { tier: 1 });
        flow.on_promote();
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Hit { tier: 2 }));
    }

    #[test]
    fn hit_below_promotes_top_only() {
        let mut flow = ReadFlow::new(3, policy(Promote::TopOnly, OnReadError::FallThrough));
        flow.on_get(Probe::Miss);
        flow.on_get(Probe::Miss);
        flow.on_get(Probe::Hit);
        assert_eq!(flow.step(), ReadStep::Promote { tier: 0 });
        flow.on_promote();
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Hit { tier: 2 }));
    }

    #[test]
    fn hit_below_never_promotes() {
        let mut flow = ReadFlow::new(2, policy(Promote::Never, OnReadError::FallThrough));
        flow.on_get(Probe::Miss);
        flow.on_get(Probe::Hit);
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Hit { tier: 1 }));
    }

    #[test]
    fn all_miss_is_a_confirmed_miss() {
        let mut flow = ReadFlow::new(2, policy(Promote::AllAbove, OnReadError::FallThrough));
        flow.on_get(Probe::Miss);
        flow.on_get(Probe::Miss);
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Miss));
    }

    #[test]
    fn error_then_hit_below_still_hits() {
        let mut flow = ReadFlow::new(2, policy(Promote::AllAbove, OnReadError::FallThrough));
        flow.on_get(Probe::Error);
        flow.on_get(Probe::Hit);
        assert_eq!(flow.step(), ReadStep::Promote { tier: 0 });
        flow.on_promote();
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Hit { tier: 1 }));
    }

    #[test]
    fn error_fall_through_with_miss_is_inconclusive() {
        let mut flow = ReadFlow::new(2, policy(Promote::AllAbove, OnReadError::FallThrough));
        flow.on_get(Probe::Error);
        flow.on_get(Probe::Miss);
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Inconclusive));
    }

    #[test]
    fn error_fail_fast_stops_at_failing_tier() {
        let mut flow = ReadFlow::new(3, policy(Promote::AllAbove, OnReadError::FailFast));
        flow.on_get(Probe::Miss);
        flow.on_get(Probe::Error);
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Failed { tier: 1 }));
    }

    #[test]
    fn zero_tiers_is_vacuously_a_miss() {
        let flow = ReadFlow::new(0, ReadPolicy::default());
        assert_eq!(flow.step(), ReadStep::Done(ReadOutcome::Miss));
    }

    #[test]
    #[should_panic(expected = "not probing")]
    fn on_get_outside_probe_panics() {
        let mut flow = ReadFlow::new(0, ReadPolicy::default());
        flow.on_get(Probe::Miss);
    }
}
