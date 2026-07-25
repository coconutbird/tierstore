//! Cross-node write sync for tierstore caches over the [groupnet] gossip
//! fabric.
//!
//! Each node publishes its recent writes as a compact ring inside one
//! versioned, gossiped group entry ([`WriteFeed`]); every peer turns entry
//! changes into typed events ([`PeerWrites`]): [`PeerWrite::Invalidate`]
//! for each new write, or [`PeerWrite::Resync`] when it provably missed
//! some (the peer's ring advanced past its cursor). The application applies
//! them — typically `cache.invalidate(&key)` per invalidation, and its own
//! flush/rebuild on a resync.
//!
//! # Semantics (read this once, rely on it forever)
//!
//! - **State-based, not a log.** The feed entry always carries the last N
//!   writes; gossip loss, event lag, and duplication are all safe because
//!   subscribers reconcile against the current entry and invalidation is
//!   idempotent. Missing writes are *detected*, never silently dropped:
//!   ring overflow past a slow peer degrades to an explicit `Resync`.
//! - **Eventual, bounded by the gossip cadence.** A peer observes a write
//!   after roughly one gossip round — this is cache coherence, not a
//!   consistency barrier. Systems needing read barriers layer them
//!   separately.
//! - **Keys travel by your codec.** Provide encode/decode closures (the
//!   `CodecTier` philosophy — no forced serde); a key that fails to decode
//!   is skipped, so keep codecs in lockstep across nodes.
//! - **History is not replayed.** A subscriber starts at each existing
//!   peer feed's current end (a fresh node has an empty cache; nothing to
//!   invalidate). Feeds appearing later replay their visible window — those
//!   writes are genuinely new.
//!
//! # Building consistency on top
//!
//! Each node's feed is totally ordered by sequence number, and that is
//! enough for **per-writer session consistency**: [`WriteFeed::publish`]
//! resolves to the write's sequence number; hand `(writer, seq)` to the
//! client as a token, and any node serving that client barriers with
//! [`FrontierView::reached`] before reading locally. The [`Frontier`] is
//! advanced by *your* apply loop after each invalidation lands, so
//! "reached" means applied, not merely delivered — a true read-your-writes
//! barrier.
//!
//! What this deliberately does not give you is linearizable multi-writer
//! ordering: that requires consensus, which groupnet excludes by design
//! (its coordinator is derived, not fenced). Pair with a consensus log
//! (e.g. openraft) if you need a total order across writers.
//!
//! # Example
//!
//! ```no_run
//! use groupnet::core::NodeId;
//! use groupnet::runtime::Node;
//! use groupnet::transport::mem::{MemTransport, Network};
//! use std::num::NonZeroUsize;
//! use tierstore_groupnet::{PeerWrite, PeerWrites, WriteFeed};
//!
//! # async fn demo(cache: tierstore::TieredCache<String, Vec<u8>>) {
//! let net = Network::new();
//! let me = NodeId::new("node-a");
//! let node = Node::builder(me.clone(), net.endpoint(me.clone())).spawn();
//! let group = node.join_group("cache");
//!
//! let feed = WriteFeed::new(
//!     group.clone(),
//!     NonZeroUsize::new(128).unwrap(),
//!     |key: &String| key.clone().into_bytes(),
//! );
//! let mut peers = PeerWrites::new(group, me, |bytes| {
//!     String::from_utf8(bytes.to_vec()).ok()
//! });
//! let (frontier, view) = tierstore_groupnet::Frontier::new();
//!
//! // After every local durable write (the seq is the client's RYW token):
//! let seq = feed.publish(&"user:1".to_owned()).await;
//!
//! // Apply peer writes, advancing the frontier only once applied:
//! tokio::spawn(async move {
//!     while let Some(event) = peers.next().await {
//!         match event {
//!             PeerWrite::Invalidate { peer, seq, key } => {
//!                 let _ = cache.invalidate(&key).await;
//!                 frontier.advance(&peer, seq);
//!             }
//!             PeerWrite::Resync { peer, applied_through } => {
//!                 // flush or rebuild local tiers, then:
//!                 frontier.advance(&peer, applied_through);
//!             }
//!         }
//!     }
//! });
//!
//! // Serving a client that carries a token (writer, seq): barrier first.
//! # let (writer, token_seq) = (NodeId::new("node-b"), 1);
//! if view.reached(&writer, token_seq).await {
//!     // local tiers now reflect that write — read locally
//! }
//! # }
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use groupnet::core::NodeId;
use groupnet::runtime::{Group, GroupEvent};
use tokio::sync::broadcast::error::RecvError;

/// The group entry key under which each node's write feed is gossiped.
const ENTRY_KEY: &str = "tierstore:writes";

/// Attempts before giving up on advertising a frame under inbox
/// backpressure (the ring keeps the write; the next publish re-carries it).
const PUBLISH_RETRIES: usize = 8;

type EncodeFn<K> = dyn Fn(&K) -> Vec<u8> + Send + Sync;
type DecodeFn<K> = dyn Fn(&[u8]) -> Option<K> + Send + Sync;

/// One peer-write notification from [`PeerWrites::next`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerWrite<K> {
    /// `peer` wrote `key`; local copies are stale and should be dropped.
    Invalidate {
        /// The node that performed the write.
        peer: NodeId,
        /// The write's sequence number in `peer`'s feed — advance the
        /// [`Frontier`] with it once the invalidation is applied.
        seq: u64,
        /// The written key.
        key: K,
    },
    /// `peer`'s feed advanced past this subscriber's cursor: some writes
    /// were provably missed. The application should flush or rebuild its
    /// local tiers, then advance the [`Frontier`] to `applied_through`.
    Resync {
        /// The node whose writes were missed.
        peer: NodeId,
        /// After flushing, every write of `peer` up to and including this
        /// sequence number is covered.
        applied_through: u64,
    },
}

/// The wire frame: `first_seq` plus the encoded keys of the last N writes,
/// sequential from `first_seq`.
struct Frame {
    first_seq: u64,
    keys: Vec<Vec<u8>>,
}

impl Frame {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + self.keys.iter().map(|k| 4 + k.len()).sum::<usize>());
        out.extend_from_slice(&self.first_seq.to_le_bytes());
        out.extend_from_slice(
            &u32::try_from(self.keys.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for key in &self.keys {
            out.extend_from_slice(&u32::try_from(key.len()).unwrap_or(u32::MAX).to_le_bytes());
            out.extend_from_slice(key);
        }
        out
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let first_seq = u64::from_le_bytes(bytes.get(0..8)?.try_into().ok()?);
        let count = u32::from_le_bytes(bytes.get(8..12)?.try_into().ok()?);
        let mut offset = 12_usize;
        let mut keys = Vec::with_capacity(usize::try_from(count).ok()?.min(4096));
        for _ in 0..count {
            let len = usize::try_from(u32::from_le_bytes(
                bytes.get(offset..offset + 4)?.try_into().ok()?,
            ))
            .ok()?;
            offset += 4;
            keys.push(bytes.get(offset..offset + len)?.to_vec());
            offset += len;
        }
        Some(Self { first_seq, keys })
    }

    const fn end(&self) -> u64 {
        self.first_seq + self.keys.len() as u64
    }
}

/// Ring of the last N encoded writes; all mutation keeps `first_seq` equal
/// to the sequence number of the front element.
struct Ring {
    first_seq: u64,
    keys: VecDeque<Vec<u8>>,
    capacity: usize,
}

impl Ring {
    fn push(&mut self, key: Vec<u8>) {
        self.keys.push_back(key);
        if self.keys.len() > self.capacity {
            self.keys.pop_front();
            self.first_seq += 1;
        }
    }

    fn frame(&self) -> Frame {
        Frame {
            first_seq: self.first_seq,
            keys: self.keys.iter().cloned().collect(),
        }
    }
}

/// Publisher half: advertises this node's writes to the group.
///
/// Call [`WriteFeed::publish`] after every local durable write. The feed is
/// best-effort under actor-inbox backpressure — a dropped advertisement is
/// re-carried by the next publish (the ring is state, not a log); call
/// [`WriteFeed::republish`] at quiescence points if the last write must be
/// advertised promptly.
pub struct WriteFeed<K> {
    group: Group,
    ring: Mutex<Ring>,
    encode: Box<EncodeFn<K>>,
}

impl<K> fmt::Debug for WriteFeed<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteFeed")
            .field("group", &self.group.id())
            .finish_non_exhaustive()
    }
}

impl<K> WriteFeed<K> {
    /// Creates a feed over `group`, remembering the last `capacity` writes.
    ///
    /// Size `capacity` for the write rate: peers that fall further behind
    /// than the ring holds receive a [`PeerWrite::Resync`] instead of the
    /// individual keys.
    pub fn new(
        group: Group,
        capacity: NonZeroUsize,
        encode: impl Fn(&K) -> Vec<u8> + Send + Sync + 'static,
    ) -> Self {
        Self {
            group,
            ring: Mutex::new(Ring {
                first_seq: 1,
                keys: VecDeque::new(),
                capacity: capacity.get(),
            }),
            encode: Box::new(encode),
        }
    }

    /// Records `key` as written and advertises the updated feed, resolving
    /// to the write's sequence number in this node's feed — the second half
    /// of a `(writer, seq)` read-your-writes token.
    ///
    /// The write is recorded in the ring synchronously (before the returned
    /// future is polled), so even a dropped future is re-carried by the
    /// next publish.
    pub fn publish(&self, key: &K) -> impl Future<Output = u64> + Send + '_ {
        let (seq, frame) = {
            let mut ring = self
                .ring
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let seq = ring.first_seq + ring.keys.len() as u64;
            ring.push((self.encode)(key));
            (seq, ring.frame().encode())
        };
        async move {
            self.advertise(frame).await;
            seq
        }
    }

    /// Re-advertises the current feed without recording a new write —
    /// useful at quiescence points after a `publish` hit backpressure.
    pub fn republish(&self) -> impl Future<Output = ()> + Send + '_ {
        let frame = {
            let ring = self
                .ring
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            ring.frame().encode()
        };
        self.advertise(frame)
    }

    async fn advertise(&self, frame: Vec<u8>) {
        for _ in 0..PUBLISH_RETRIES {
            if self.group.set_entry(ENTRY_KEY, frame.clone(), None).is_ok() {
                return;
            }
            // Inbox backpressure: yield and retry; on sustained pressure the
            // ring re-carries this write on the next publish.
            tokio::task::yield_now().await;
        }
    }
}

/// Subscriber half: turns peers' feed changes into [`PeerWrite`] events.
///
/// Drive it from a task: `while let Some(event) = peers.next().await { … }`.
/// Event-stream lag is handled internally by re-reading the always-current
/// entry snapshots — no write is ever silently skipped.
pub struct PeerWrites<K> {
    group: Group,
    me: NodeId,
    events: tokio::sync::broadcast::Receiver<GroupEvent>,
    /// Next unseen sequence number per peer feed.
    cursors: HashMap<NodeId, u64>,
    pending: VecDeque<PeerWrite<K>>,
    decode: Box<DecodeFn<K>>,
}

impl<K> fmt::Debug for PeerWrites<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PeerWrites")
            .field("group", &self.group.id())
            .field("me", &self.me)
            .field("peers", &self.cursors.len())
            .finish_non_exhaustive()
    }
}

impl<K> PeerWrites<K> {
    /// Subscribes to peer writes in `group`. `me` is this node's id (its
    /// own feed is ignored). Existing peer feeds start at their current
    /// end: history is not replayed.
    pub fn new(
        group: Group,
        me: NodeId,
        decode: impl Fn(&[u8]) -> Option<K> + Send + Sync + 'static,
    ) -> Self {
        let events = group.events();
        let mut cursors = HashMap::new();
        for (node, entries) in group.all_entries().iter() {
            if *node == me {
                continue;
            }
            if let Some(bytes) = entries.get(ENTRY_KEY)
                && let Some(frame) = Frame::decode(bytes)
            {
                cursors.insert(node.clone(), frame.end());
            }
        }
        Self {
            group,
            me,
            events,
            cursors,
            pending: VecDeque::new(),
            decode: Box::new(decode),
        }
    }

    /// The next peer write, or `None` once the group is gone.
    pub async fn next(&mut self) -> Option<PeerWrite<K>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }
            match self.events.recv().await {
                Ok(GroupEvent::NodeStateChanged { node, key })
                    if key == ENTRY_KEY && node != self.me =>
                {
                    self.scan(&node);
                }
                // Lag means missed edge triggers, never missed state: the
                // entry snapshots are current, so a full re-scan recovers.
                Err(RecvError::Lagged(_)) | Ok(GroupEvent::MembershipChanged) => self.scan_all(),
                Ok(_) => {}
                Err(RecvError::Closed) => return None,
            }
        }
    }

    fn scan_all(&mut self) {
        for node in self.group.members() {
            if node != self.me {
                self.scan(&node);
            }
        }
    }

    /// Reconciles one peer's feed against our cursor, queueing events.
    fn scan(&mut self, node: &NodeId) {
        let Some(bytes) = self.group.node_entry(node, ENTRY_KEY) else {
            return;
        };
        let Some(frame) = Frame::decode(&bytes) else {
            return;
        };
        let cursor = self.cursors.entry(node.clone()).or_insert(frame.first_seq);
        if *cursor < frame.first_seq {
            // The ring advanced past us: writes were provably missed.
            self.pending.push_back(PeerWrite::Resync {
                peer: node.clone(),
                applied_through: frame.first_seq.saturating_sub(1),
            });
            *cursor = frame.first_seq;
        }
        while *cursor < frame.end() {
            let Ok(index) = usize::try_from(*cursor - frame.first_seq) else {
                break;
            };
            if let Some(key) = (self.decode)(&frame.keys[index]) {
                self.pending.push_back(PeerWrite::Invalidate {
                    peer: node.clone(),
                    seq: *cursor,
                    key,
                });
            }
            *cursor += 1;
        }
    }
}

/// Applied-write watermarks per peer, advanced by the application's apply
/// loop — see [`Frontier`].
type Applied = HashMap<NodeId, u64>;

/// The writer half of the applied-write frontier.
///
/// The apply loop calls [`Frontier::advance`] after each peer write has
/// actually been applied (invalidation done, or resync flush finished).
/// Barriers on the matching [`FrontierView`] then mean *applied*, not
/// merely delivered.
#[derive(Debug)]
pub struct Frontier {
    tx: tokio::sync::watch::Sender<Applied>,
}

/// The reader half: cheap to clone, held wherever reads need a
/// read-your-writes barrier.
#[derive(Debug, Clone)]
pub struct FrontierView {
    rx: tokio::sync::watch::Receiver<Applied>,
}

impl Frontier {
    /// A fresh frontier (nothing applied) and its reader view.
    #[must_use]
    pub fn new() -> (Self, FrontierView) {
        let (tx, rx) = tokio::sync::watch::channel(Applied::new());
        (Self { tx }, FrontierView { rx })
    }

    /// Marks `peer`'s writes as applied through `seq` (monotonic: lower
    /// values are ignored).
    pub fn advance(&self, peer: &NodeId, seq: u64) {
        self.tx.send_modify(|applied| {
            let entry = applied.entry(peer.clone()).or_insert(0);
            if *entry < seq {
                *entry = seq;
            }
        });
    }
}

impl FrontierView {
    /// Waits until `peer`'s writes through `seq` have been applied locally.
    ///
    /// Returns `false` if the [`Frontier`] was dropped first (the apply
    /// loop is gone — do not serve reads assuming freshness). Combine with
    /// a caller-side timeout for bounded waiting.
    pub async fn reached(&self, peer: &NodeId, seq: u64) -> bool {
        let mut rx = self.rx.clone();
        rx.wait_for(|applied| applied.get(peer).is_some_and(|&s| s >= seq))
            .await
            .is_ok()
    }
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    #[test]
    fn frame_round_trips() {
        let frame = Frame {
            first_seq: 41,
            keys: vec![b"alpha".to_vec(), Vec::new(), b"c".to_vec()],
        };
        let decoded = Frame::decode(&frame.encode()).expect("decode");
        assert_eq!(decoded.first_seq, 41);
        assert_eq!(decoded.keys, frame.keys);
        assert_eq!(decoded.end(), 44);
    }

    #[test]
    fn truncated_frames_are_rejected() {
        let bytes = Frame {
            first_seq: 1,
            keys: vec![b"key".to_vec()],
        }
        .encode();
        for cut in 0..bytes.len() {
            assert!(Frame::decode(&bytes[..cut]).is_none(), "cut at {cut}");
        }
    }

    #[test]
    fn ring_overflow_advances_first_seq() {
        let mut ring = Ring {
            first_seq: 1,
            keys: VecDeque::new(),
            capacity: 2,
        };
        for key in [b"a".to_vec(), b"b".to_vec(), b"c".to_vec()] {
            ring.push(key);
        }
        let frame = ring.frame();
        assert_eq!(frame.first_seq, 2);
        assert_eq!(frame.keys, vec![b"b".to_vec(), b"c".to_vec()]);
    }
}
