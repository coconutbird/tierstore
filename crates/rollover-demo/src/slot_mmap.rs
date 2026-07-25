//! A fixed-capacity, file-backed, memory-mapped hot tier.
//!
//! The file is divided into `slots` fixed-size slots, each holding one
//! `used | key_len | val_len | key | value` record. An in-memory index maps
//! keys to slots and is rebuilt by scanning the mapping on open — so a
//! reopened tier (e.g. after a process restart) still serves everything it
//! held. When all slots are used, an insert FIFO-evicts the oldest entry
//! and returns it as displaced, which is exactly the rollover contract the
//! router demotes on.
//!
//! Demo-quality on purpose: one big lock, linear slot scan on open, no
//! checksums. It exists to prove the tier contract against a real mmap, not
//! to be the production tier yet.
//!
//! Not to be confused with the library's `tierstore-mmap::MmapDiskTier`,
//! which is the opposite design point: immutable file-per-key storage with
//! snapshot-on-overwrite, serving zero-copy `Bytes`. This one is a
//! *mutable, fixed-capacity slot store* whose point is displacement.

use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};

use memmap2::MmapMut;
use tierstore::{Displaced, Tier, TierRead, TierReadRef, TierWrite};

/// `used(1) | key_len(2, LE) | val_len(4, LE)`
const HEADER: usize = 7;

pub struct SlotMmapTier {
    inner: Mutex<Inner>,
}

struct Inner {
    mmap: MmapMut,
    slot_size: usize,
    index: HashMap<String, usize>,
    /// Insertion order for FIFO eviction.
    order: VecDeque<String>,
    free: Vec<usize>,
}

impl std::fmt::Debug for SlotMmapTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("SlotMmapTier")
            .field("slot_size", &inner.slot_size)
            .field("len", &inner.index.len())
            .finish_non_exhaustive()
    }
}

impl SlotMmapTier {
    /// Opens (creating or reusing) the backing file and rebuilds the index
    /// from whatever records it already holds.
    pub fn open(path: &Path, slots: usize, slot_size: usize) -> io::Result<Self> {
        assert!(
            slot_size > HEADER,
            "slot_size must exceed the record header"
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = (slots * slot_size) as u64;
        if file.metadata()?.len() != len {
            file.set_len(len)?;
        }
        // SAFETY: we hold this file open for the lifetime of the mapping and
        // nothing else truncates or remaps it while the tier is alive (demo
        // assumption; a production tier would take a file lock).
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let mut index = HashMap::new();
        let mut order = VecDeque::new();
        let mut free = Vec::new();
        for slot in 0..slots {
            if let Some((key, _)) = read_slot(&mmap, slot_size, slot) {
                index.insert(key.clone(), slot);
                order.push_back(key);
            } else {
                free.push(slot);
            }
        }
        Ok(Self {
            inner: Mutex::new(Inner {
                mmap,
                slot_size,
                index,
                order,
                free,
            }),
        })
    }

    /// Flushes dirty pages to the backing file (durability point).
    pub fn flush(&self) -> io::Result<()> {
        self.lock().mmap.flush()
    }

    /// Address range of the mapping, for proving that zero-copy views point
    /// into it. Do not call while holding a view (single lock).
    pub fn mapping_range(&self) -> std::ops::Range<usize> {
        let inner = self.lock();
        let base = inner.mmap.as_ptr().addr();
        base..base + inner.mmap.len()
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Absolute `(start, len)` of a used slot's value bytes within the mapping.
fn value_range(mmap: &[u8], slot_size: usize, slot: usize) -> Option<(usize, usize)> {
    let base = slot * slot_size;
    let header = &mmap[base..base + HEADER];
    if header[0] != 1 {
        return None;
    }
    let key_len = usize::from(u16::from_le_bytes([header[1], header[2]]));
    let val_len = usize::try_from(u32::from_le_bytes([
        header[3], header[4], header[5], header[6],
    ]))
    .ok()?;
    let start = base + HEADER + key_len;
    if start + val_len > base + slot_size {
        return None; // corrupt slot: treat as free
    }
    Some((start, val_len))
}

fn read_slot(mmap: &[u8], slot_size: usize, slot: usize) -> Option<(String, Vec<u8>)> {
    let (start, len) = value_range(mmap, slot_size, slot)?;
    let key_start = slot * slot_size + HEADER;
    let key = String::from_utf8(mmap[key_start..start].to_vec()).ok()?;
    let value = mmap[start..start + len].to_vec();
    Some((key, value))
}

impl Inner {
    fn write_slot(&mut self, slot: usize, key: &str, value: &[u8]) -> io::Result<()> {
        let key_len = u16::try_from(key.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "key too long"))?;
        let val_len = u32::try_from(value.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "value too long"))?;
        let base = slot * self.slot_size;
        self.mmap[base] = 1;
        self.mmap[base + 1..base + 3].copy_from_slice(&key_len.to_le_bytes());
        self.mmap[base + 3..base + HEADER].copy_from_slice(&val_len.to_le_bytes());
        let key_start = base + HEADER;
        self.mmap[key_start..key_start + key.len()].copy_from_slice(key.as_bytes());
        let val_start = key_start + key.len();
        self.mmap[val_start..val_start + value.len()].copy_from_slice(value);
        Ok(())
    }

    fn fits(&self, key: &str, value: &[u8]) -> io::Result<()> {
        let needed = HEADER + key.len() + value.len();
        if needed > self.slot_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("entry needs {needed} bytes, slot holds {}", self.slot_size),
            ));
        }
        Ok(())
    }

    fn fetch(&self, key: &str) -> Option<Vec<u8>> {
        self.index
            .get(key)
            .and_then(|&slot| read_slot(&self.mmap, self.slot_size, slot))
            .map(|(_, value)| value)
    }

    fn insert(&mut self, key: String, value: &[u8]) -> io::Result<Displaced<String, Vec<u8>>> {
        // Validate before evicting anyone: an oversized entry must not cost
        // an existing one its slot.
        self.fits(&key, value)?;

        if let Some(&slot) = self.index.get(&key) {
            self.write_slot(slot, &key, value)?;
            return Ok(Displaced::new());
        }

        let mut displaced = Displaced::new();
        let slot = if let Some(slot) = self.free.pop() {
            slot
        } else {
            // Full: FIFO-evict the oldest entry and hand it back for
            // demotion — the rollover contract.
            let Some(oldest) = self.order.pop_front() else {
                return Err(io::Error::other("mmap tier has zero slots"));
            };
            let Some(slot) = self.index.remove(&oldest) else {
                return Err(io::Error::other("index and order out of sync"));
            };
            if let Some(entry) = read_slot(&self.mmap, self.slot_size, slot) {
                displaced.push(entry);
            }
            slot
        };
        self.write_slot(slot, &key, value)?;
        self.index.insert(key.clone(), slot);
        self.order.push_back(key);
        Ok(displaced)
    }

    fn remove(&mut self, key: &str) -> bool {
        let Some(slot) = self.index.remove(key) else {
            return false;
        };
        let base = slot * self.slot_size;
        self.mmap[base] = 0;
        self.free.push(slot);
        self.order.retain(|k| k.as_str() != key);
        true
    }
}

impl Tier for SlotMmapTier {
    type Key = String;
    type Value = Vec<u8>;
    type Error = io::Error;

    fn name(&self) -> &'static str {
        "mmap"
    }
}

impl TierRead for SlotMmapTier {
    async fn get(&self, key: &String) -> io::Result<Option<Vec<u8>>> {
        Ok(self.lock().fetch(key))
    }

    async fn exists(&self, key: &String) -> io::Result<bool> {
        Ok(self.lock().index.contains_key(key))
    }
}

/// Zero-copy view into an [`SlotMmapTier`] slot: derefs to bytes inside the
/// mapping itself. Holds the tier's lock, so the slot cannot be evicted or
/// overwritten while the view exists — keep it short-lived.
pub struct MmapRef<'a> {
    guard: MutexGuard<'a, Inner>,
    start: usize,
    len: usize,
}

impl std::fmt::Debug for MmapRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapRef")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl std::ops::Deref for MmapRef<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.guard.mmap[self.start..self.start + self.len]
    }
}

impl TierReadRef for SlotMmapTier {
    type Borrowed = [u8];
    type ValueRef<'a>
        = MmapRef<'a>
    where
        Self: 'a;

    async fn get_ref<'s>(&'s self, key: &String) -> io::Result<Option<MmapRef<'s>>> {
        let guard = self.lock();
        let Some(&slot) = guard.index.get(key) else {
            return Ok(None);
        };
        let Some((start, len)) = value_range(&guard.mmap, guard.slot_size, slot) else {
            return Ok(None);
        };
        Ok(Some(MmapRef { guard, start, len }))
    }
}

impl TierWrite for SlotMmapTier {
    async fn put(&self, key: String, value: Vec<u8>) -> io::Result<Displaced<String, Vec<u8>>> {
        self.lock().insert(key, &value)
    }

    async fn delete(&self, key: &String) -> io::Result<bool> {
        Ok(self.lock().remove(key))
    }
}
