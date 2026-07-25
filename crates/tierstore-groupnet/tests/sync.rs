//! Two real groupnet nodes over the in-memory transport: writes published
//! on one node arrive as invalidations on the other, ring overflow degrades
//! to an explicit resync, and a node never reacts to its own writes.

use std::num::NonZeroUsize;
use std::time::Duration;

use groupnet::core::NodeId;
use groupnet::runtime::{Group, Node};
use groupnet::transport::mem::{MemTransport, Network};
use tierstore::{MemoryTier, TierRead, TieredCache};
use tierstore_groupnet::{PeerWrite, PeerWrites, WriteFeed};

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

    // Node A publishes two writes.
    let feed = WriteFeed::new(a_group, cap(128), |key: &String| key.clone().into_bytes());
    feed.publish(&"user:1".to_owned()).await;
    feed.publish(&"user:2".to_owned()).await;

    // B observes them in order and applies the invalidation.
    for expected in ["user:1", "user:2"] {
        match next_event(&mut peers).await {
            PeerWrite::Invalidate { peer, key } => {
                assert_eq!(peer, a_id);
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
        PeerWrite::Resync { peer: a_id.clone() },
        "an overflowed ring must surface as a resync, never a silent skip"
    );
    assert_eq!(
        next_event(&mut peers).await,
        PeerWrite::Invalidate {
            peer: a_id.clone(),
            key: "w3".to_owned()
        }
    );
    assert_eq!(
        next_event(&mut peers).await,
        PeerWrite::Invalidate {
            peer: a_id,
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
