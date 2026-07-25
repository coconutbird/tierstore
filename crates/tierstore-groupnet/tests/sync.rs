//! Two real groupnet nodes over the in-memory transport: writes published
//! on one node arrive as invalidations on the other, ring overflow degrades
//! to an explicit resync, and a node never reacts to its own writes.

use std::num::NonZeroUsize;
use std::time::Duration;

use groupnet::core::NodeId;
use groupnet::runtime::{Group, Node};
use groupnet::transport::mem::{MemTransport, Network};
use tierstore::{MemoryTier, TierRead, TieredCache};
use tierstore_groupnet::{Frontier, PeerWrite, PeerWrites, WriteFeed};

const GROUP: &str = "cache";

const fn cap(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("nonzero")
}

fn spawn_node(net: &Network, id: &str, peers: &[&str]) -> (NodeId, Node<MemTransport>, Group) {
    let me = NodeId::new(id);
    let mut builder = Node::builder(me.clone(), net.endpoint(me.clone()))
        .gossip_interval_ms(10)
        .anti_entropy_interval_ms(25);
    for peer in peers {
        builder = builder.seed(NodeId::new(*peer));
    }
    let node = builder.spawn();
    let group = node.join_group(GROUP);
    (me, node, group)
}

async fn converged(groups: &[&Group]) {
    for _ in 0..300 {
        if groups.iter().all(|g| g.members().len() == groups.len()) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("membership did not converge");
}

async fn next_event(peers: &mut PeerWrites<String>) -> PeerWrite<String> {
    tokio::time::timeout(Duration::from_secs(5), peers.next())
        .await
        .expect("timed out waiting for a peer write")
        .expect("event stream ended")
}

#[tokio::test]
async fn peer_writes_arrive_as_invalidations_and_apply_to_a_cache() {
    let net = Network::new();
    let (a_id, _a_node, a_group) = spawn_node(&net, "node-a", &["node-b"]);
    let (b_id, _b_node, b_group) = spawn_node(&net, "node-b", &["node-a"]);
    converged(&[&a_group, &b_group]).await;

    // Node B: a local cache holding a soon-stale copy, and a subscription.
    let hot = std::sync::Arc::new(MemoryTier::unbounded());
    let cache: TieredCache<String, Vec<u8>> = TieredCache::builder()
        .tier(std::sync::Arc::clone(&hot))
        .build();
    let _ = cache.put("user:1".to_owned(), b"stale".to_vec()).await;
    let mut peers = PeerWrites::new(b_group, b_id, |bytes| {
        String::from_utf8(bytes.to_vec()).ok()
    });

    // Node A publishes two writes; seqs are the read-your-writes tokens.
    let feed = WriteFeed::new(a_group, cap(128), |key: &String| key.clone().into_bytes());
    assert_eq!(feed.publish(&"user:1".to_owned()).await, 1);
    assert_eq!(feed.publish(&"user:2".to_owned()).await, 2);

    // B observes them in order and applies the invalidation.
    for (expected_seq, expected) in [(1, "user:1"), (2, "user:2")] {
        match next_event(&mut peers).await {
            PeerWrite::Invalidate { peer, seq, key } => {
                assert_eq!(peer, a_id);
                assert_eq!(seq, expected_seq);
                assert_eq!(key, expected);
                let _ = cache.invalidate(&key).await;
            }
            PeerWrite::Resync { .. } => panic!("no resync expected"),
        }
    }
    assert_eq!(
        hot.get(&"user:1".to_owned()).await.expect("hot peek"),
        None,
        "the peer's write must drop the stale local copy"
    );
}

#[tokio::test]
async fn ring_overflow_degrades_to_an_explicit_resync() {
    let net = Network::new();
    let (a_id, _a_node, a_group) = spawn_node(&net, "ov-a", &["ov-b"]);
    let (b_id, _b_node, b_group) = spawn_node(&net, "ov-b", &["ov-a"]);
    converged(&[&a_group, &b_group]).await;

    let mut peers = PeerWrites::new(b_group, b_id, |bytes| {
        String::from_utf8(bytes.to_vec()).ok()
    });
    // A tiny ring: two slots.
    let feed = WriteFeed::new(a_group, cap(2), |key: &String| key.clone().into_bytes());

    // B tracks the feed normally first (cursor lands at w1's end)…
    feed.publish(&"w1".to_owned()).await;
    match next_event(&mut peers).await {
        PeerWrite::Invalidate { key, .. } => assert_eq!(key, "w1"),
        PeerWrite::Resync { .. } => panic!("no resync yet"),
    }

    // …then A writes three more without B draining: w2 falls off the ring.
    for key in ["w2", "w3", "w4"] {
        feed.publish(&key.to_owned()).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await; // let gossip settle

    // B must learn it missed something — loudly — then catch the survivors.
    assert_eq!(
        next_event(&mut peers).await,
        PeerWrite::Resync {
            peer: a_id.clone(),
            applied_through: 2
        },
        "an overflowed ring must surface as a resync, never a silent skip"
    );
    assert_eq!(
        next_event(&mut peers).await,
        PeerWrite::Invalidate {
            peer: a_id.clone(),
            seq: 3,
            key: "w3".to_owned()
        }
    );
    assert_eq!(
        next_event(&mut peers).await,
        PeerWrite::Invalidate {
            peer: a_id,
            seq: 4,
            key: "w4".to_owned()
        }
    );
}

#[tokio::test]
async fn own_writes_are_ignored() {
    let net = Network::new();
    let (a_id, _a_node, a_group) = spawn_node(&net, "self-a", &["self-b"]);
    let (_b_id, _b_node, b_group) = spawn_node(&net, "self-b", &["self-a"]);
    converged(&[&a_group, &b_group]).await;

    // Feed and subscription on the SAME node.
    let feed = WriteFeed::new(a_group.clone(), cap(8), |key: &String| {
        key.clone().into_bytes()
    });
    let mut own = PeerWrites::new(a_group, a_id, |bytes| {
        String::from_utf8(bytes.to_vec()).ok()
    });
    feed.publish(&"local".to_owned()).await;

    // Nothing may arrive: a node does not invalidate itself.
    let quiet = tokio::time::timeout(Duration::from_millis(300), own.next()).await;
    assert!(quiet.is_err(), "own writes must not produce events");
}

#[tokio::test]
async fn read_your_writes_barrier_waits_for_the_applied_frontier() {
    let net = Network::new();
    let (a_id, _a_node, a_group) = spawn_node(&net, "ryw-a", &["ryw-b"]);
    let (b_id, _b_node, b_group) = spawn_node(&net, "ryw-b", &["ryw-a"]);
    converged(&[&a_group, &b_group]).await;

    // Node B: a cache holding a stale copy, an apply loop, and a frontier.
    let hot = std::sync::Arc::new(MemoryTier::unbounded());
    let cache: std::sync::Arc<TieredCache<String, Vec<u8>>> = std::sync::Arc::new(
        TieredCache::builder()
            .tier(std::sync::Arc::clone(&hot))
            .build(),
    );
    let _ = cache.put("user:1".to_owned(), b"stale".to_vec()).await;

    let mut peers = PeerWrites::new(b_group, b_id, |bytes| {
        String::from_utf8(bytes.to_vec()).ok()
    });
    let (frontier, view) = Frontier::new();
    let apply_cache = std::sync::Arc::clone(&cache);
    tokio::spawn(async move {
        while let Some(event) = peers.next().await {
            match event {
                PeerWrite::Invalidate { peer, seq, key } => {
                    let _ = apply_cache.invalidate(&key).await;
                    frontier.advance(&peer, seq);
                }
                PeerWrite::Resync {
                    peer,
                    applied_through,
                } => frontier.advance(&peer, applied_through),
            }
        }
    });

    // Node A writes; the returned seq is the client's token.
    let feed = WriteFeed::new(a_group, cap(64), |key: &String| key.clone().into_bytes());
    let token = feed.publish(&"user:1".to_owned()).await;

    // A client carrying (a, token) reads on B: the barrier resolves only
    // after the apply loop has actually invalidated — never a stale read.
    let reached = tokio::time::timeout(Duration::from_secs(5), view.reached(&a_id, token))
        .await
        .expect("barrier timed out");
    assert!(
        reached,
        "frontier must be reachable while the apply loop runs"
    );
    assert_eq!(
        hot.get(&"user:1".to_owned()).await.expect("hot peek"),
        None,
        "after the barrier, the stale copy is provably gone"
    );
}
