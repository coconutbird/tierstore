//! Cross-node write sync for tierstore caches — the cache-flavored face of
//! [`groupnet_consistency`], where the machinery actually lives (so any
//! groupnet system, not just tierstore, can borrow it).
//!
//! The pattern for a tiered cache:
//!
//! - After every local durable write, [`WriteFeed::publish`] — the resolved
//!   sequence number is the client's `(writer, seq)` read-your-writes token.
//! - An apply loop turns [`PeerWrite::Wrote`] into `cache.invalidate(&key)`
//!   and [`PeerWrite::Gap`] into a coarse flush of the local tiers,
//!   advancing the [`Frontier`] only after each has actually been applied.
//! - A node serving a client that carries a token barriers with
//!   [`FrontierView::reached`] before reading locally — "reached" means
//!   applied, so the read is provably not stale for that writer.
//!
//! The semantics — state-based feeds, loss detected as explicit gaps (never
//! a silent skip), per-writer session consistency, consensus deliberately
//! out of scope — are documented once, in [`groupnet_consistency`]. Read
//! them there; rely on them here.
//!
//! # Example
//!
//! ```no_run
//! use groupnet::core::NodeId;
//! use groupnet::runtime::Node;
//! use groupnet::transport::mem::Network;
//! use std::num::NonZeroUsize;
//! use tierstore_groupnet::{Frontier, PeerWrite, PeerWrites, WriteFeed};
//!
//! # async fn demo(cache: std::sync::Arc<tierstore::TieredCache<String, Vec<u8>>>) {
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
//! let (frontier, view) = Frontier::new();
//!
//! // After every local durable write (the seq is the client's RYW token):
//! let seq = feed.publish(&"user:1".to_owned()).await;
//!
//! // Apply peer writes, advancing the frontier only once applied:
//! tokio::spawn(async move {
//!     while let Some(event) = peers.next().await {
//!         match event {
//!             PeerWrite::Wrote { peer, seq, key } => {
//!                 let _ = cache.invalidate(&key).await;
//!                 frontier.advance(&peer, seq);
//!             }
//!             PeerWrite::Gap {
//!                 peer,
//!                 missed_through,
//!             } => {
//!                 // flush or rebuild the local tiers, then:
//!                 frontier.advance(&peer, missed_through);
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

pub use groupnet_consistency::{Frontier, FrontierView, PeerWrite, PeerWrites, WriteFeed};
