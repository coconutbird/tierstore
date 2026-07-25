# tierstore

A generic **storage-tier router**: register N tiers (anything that can
`get`/`put`/`exists`/`list`), and route reads and writes through them with
explicit, pluggable policy. The first-class instantiation is a tiered
rollover cache — hot (memory) over warm (disk) over cold (remote fetch,
e.g. a database) — but the router doesn't know it's a cache; that's just one
policy.

## Layout

```
tierstore-core   no_std. Capability traits (TierRead / TierWrite / TierList /
                 TierReadRef), routing Policy, and ReadFlow — a sans-io state
                 machine that makes every read-path decision without I/O.
tierstore        std batteries, zero deps. Router (drives ReadFlow against
                 real tiers), MemoryTier (hot; entry- or byte-bounded),
                 DiskTier (warm), middleware tiers (VerifiedTier,
                 LimitedTier), SingleFlight, and two semantic presets built
                 entirely on the router: TieredCache (availability) and
                 TieredStore (authority).
tierstore-mmap   First adapter crate: MmapDiskTier, a kernel-evictable
                 zero-copy warm tier (file-per-key, served as mmap-backed
                 bytes::Bytes, snapshot-on-overwrite). The template for
                 backend adapters (redis, s3, postgres, …).
```

Mechanism and policy are separated on purpose: the router executes, `Policy`
decides. The same router is an inclusive read-through cache, an exclusive
rollover cache, or a plain fallback chain depending only on policy.

## Quick start

```rust
let cache = TieredCache::builder()
    .tier(Arc::clone(&hot))     // MemoryTier, bounded
    .tier(Arc::clone(&warm))    // DiskTier
    .tier(Arc::clone(&cold))    // your remote store
    .build();

let value = cache.get(&key).await?;   // read-through, promotes per policy
```

`cargo run -p tierstore --example hot_warm_cold` walks the whole story:
remote fetch → hot hit → hot overflow rolls entries onto disk → disk hit
without re-fetching.

`cargo run -p rollover-demo` is the assertion-gated proof on a *real* tier
stack — hot is a fixed-slot **memory-mapped file** (`memmap2`), warm is
disk, cold is a simulated remote with latency — and additionally proves the
mmap hot tier survives a process restart (index rebuilt from the mapping,
zero remote fetches). The `MmapTier` in that crate is the candidate to
graduate into a real tier once we're happy with it.

## Semantics worth knowing

- **Rollover.** `TierWrite::put` returns the entries it displaced; the
  router demotes them into the next tier down (cascading). Entries falling
  off the bottommost tier are returned to the caller — eviction is never
  silent.
- **Batching is first-class.** `get_many` / `put_many` / `delete_many` ship
  looping defaults so every tier supports them; backends override with real
  batch I/O (`MemoryTier` does one lock pass; a remote tier would use an
  `MGET`-style round-trip). The router probes each lower tier with only the
  still-missing keys.
- **Partial success has per-key statuses.** Batched router reads
  (`read_many`) and deletes (`remove_many`) return reports — each key is a
  `Hit` (with the tier that served it), a confirmed `Miss`, or
  `Inconclusive` when a failing tier left it unknown — with the tier
  failures alongside, so one bad key or tier never discards resolved
  values. The cache's `get_many` / `invalidate_many` return these reports.
- **Honest misses.** If a read falls through a *failing* tier and ends in a
  miss, you get `RouterError::Inconclusive`, not `Ok(None)` — the failed
  tier might have held the key.
- **Deletes are all-tier.** A delete attempts every tier even after
  failures, and partial failure is an error: a surviving upper copy would
  resurrect the key.
- **Write-around invalidates.** Writing only the bottom tier deletes the
  key from upper tiers, since a stale copy would shadow the new value.
- **Routers compose — and so do stores.** `Router` and `TieredStore` both
  implement the tier traits, so a "warm" tier can be a whole hierarchy, and
  the blessed layering for authority-backed systems is cache-over-store:
  `TieredCache [ hot, warm, TieredStore [ … ] ]`. Cache tiers stay lenient
  above; reads reaching the store are governed by its own fail-fast policy
  inside; writes hit the authority first.
- **Store vs cache is policy, plus a stance on loss.** `Policy::default()`
  is the neutral fallback chain (the router moves nothing on its own).
  `TieredCache` is the availability preset. `TieredStore` is the authority
  preset: fail-fast reads, strict deletes, and a write that pushes entries
  off the bottom tier returns them in `StoreError::Evicted` — data loss is
  an error carrying the data, never a silent shrink.
- **Stampede protection is built in.** `TieredCache` coalesces concurrent
  `get`s per key (single-flight: one caller fills, the rest wait and hit
  the promoted copy), dependency-free and executor-agnostic. Default on;
  `.single_flight(false)` opts out.
- **Trust boundaries are explicit.** Wrap an untrusted tier in
  `VerifiedTier` and every value it serves is checked once at the boundary;
  a rejected value is a *tier failure* (inconclusive read), never data, and
  is never promoted upward. Both patterns are adopted from shardstore's
  cache layer.
- **Zero-copy reads are a capability.** `TierReadRef::get_ref` returns a
  guard-held view that derefs to the value *in place* (the demo's
  `MmapTier` serves views pointing directly into the mapping). Direct-tier
  only: views can't cross the router's type-erased boundary, and
  promotion/demotion inherently copy (see open question 12).

## Memory story

- **Bound the hot set:** `MemoryTier::bounded` (entry count) or
  `MemoryTier::bounded_bytes(budget, weigher)` (byte budget). Overflow rolls
  down; an entry heavier than the whole budget rolls straight through to
  the next tier instead of thrashing the hot set.
- **Bound transit:** values move between tiers as owned `V` clones, so pick
  a cheap-clone `V` for large values — `bytes::Bytes` turns every boundary
  clone into a refcount bump, and the router is already generic over it.
- **Bound residency:** `tierstore-mmap`'s `MmapDiskTier` serves values as
  mmap-backed `Bytes`: zero-copy reads, snapshot-on-overwrite, and RAM
  residency managed by the kernel's page cache (evictable under pressure) —
  shardstore's warm-tier model.
- **Bound concurrency:** wrap an origin in `LimitedTier` to cap in-flight
  operations against it; transient fill memory becomes ~`limit × value
  size` instead of `callers × value size`. Single-flight already dedupes
  same-key fills; this bounds distinct-key fan-in.

## Open questions (deliberately unresolved)

1. `Send` futures are part of the trait contract (server-first). Is a
   non-`Send` "local" variant worth the surface?
2. Read-only tiers: cold stores you never write back to. Currently every
   routed tier must implement `TierWrite`.
3. `TierList` for the router itself (cross-tier cursor unification, dedup).
4. Write-back mode (dirty tracking + flush) — v2 at the earliest.
5. Single-flight granularity: the cache gates whole `get`s per key, so
   same-key *hot* hits serialize briefly too; a probe-then-gate refinement
   and deadlock-free batched-`get` coalescing are open.
6. Per-tier TTL / staleness, negative caching.
7. Demotion churn when a lower tier is smaller than the one above it.
8. Typed keys/values vs bytes: `DiskTier` is concrete (`String`/`Vec<u8>`);
   a codec adapter tier would bridge typed hierarchies onto byte stores.
9. Eviction *order* in `MemoryTier` is FIFO (bounds are entry- and
   byte-aware now); pluggable ordering (LRU, LFU) would live behind another
   small trait.
10. Static (generic tuple) tier composition to avoid boxing on the hot path.
11. The batched *trait* methods stay lowest-common-denominator
    (`Result<Vec<…>, _>`), so a nested router driven through the tier traits
    degrades its per-key statuses to whole-batch errors; richer trait-level
    batch contracts and a sans-io *batched* read flow are open questions.
12. Zero-copy through the *router*: `TierReadRef` views are direct-tier
    (they hold the tier's lock). The refcounted-value path is shipped
    (`tierstore-mmap` + `V = Bytes` make boundary clones free); router-level
    borrowed views (boxed guards, top-tier fast path) remain open.

## Prior art

[foyer](https://github.com/foyer-rs/foyer) is the mature memory+disk hybrid
cache in Rust — if you want a fast two-tier cache product, use it. tierstore
is the *abstraction* instead: N pluggable tiers behind small traits, policy
separated from mechanism, `no_std` core, composable routers.

## Toolchain

Rust **edition 2024**, tracking stable (no pinned MSRV). `tierstore-core`
and `tierstore` are
dependency-free (dev-deps included) and `unsafe`-free (`forbid`);
`tierstore-mmap` carries the one documented `unsafe` block that memory
mapping requires, plus `memmap2` and `bytes`. CI gates: rustfmt, clippy
`pedantic` + `nursery` at `-D warnings`, all tests including doctests, a
no-`alloc` core build, docs with `-D warnings`, and the assertion-gated
rollover demo.

## Status

Early but real: the mechanism, both semantic presets, and the memory story
are implemented and tested end to end. Version 0.1.x — expect API movement
along the open questions above.

## License

MIT. See [LICENSE](LICENSE).
