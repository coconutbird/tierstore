//! On-disk tier: the "warm" layer.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::ops::Range;
use std::path::PathBuf;

use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierReadRange, TierWrite};

/// File-per-key tier under a root directory.
///
/// Keys (`String`) are hex-encoded into file names, which sidesteps path
/// traversal and case-insensitive-filesystem hazards; values are raw bytes
/// (`Vec<u8>`). Writes go through a temp file + rename so a crash never
/// leaves a torn value. The tier is unbounded: it never displaces.
///
/// Prototype notes: I/O is synchronous `std::fs` inside async methods, which
/// blocks the calling task — a production version would use `spawn_blocking`
/// or a real async file API. Typed keys/values belong to a future codec
/// adapter; this tier stays deliberately concrete.
///
/// # Example
///
/// ```no_run
/// use tierstore::{DiskTier, TierRead, TierWrite};
///
/// # async fn demo() -> std::io::Result<()> {
/// let tier = DiskTier::open("/var/cache/myapp")?;
/// tier.put("key".to_owned(), b"value".to_vec()).await?;
/// assert!(tier.get(&"key".to_owned()).await?.is_some());
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct DiskTier {
    root: PathBuf,
}

impl DiskTier {
    /// Opens (creating if needed) the tier's root directory.
    ///
    /// # Errors
    ///
    /// Returns the I/O error if the directory cannot be created.
    pub fn open(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The tier's root directory.
    #[must_use]
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.root.join(hex_encode(key.as_bytes()))
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

fn hex_decode(name: &str) -> Option<Vec<u8>> {
    if !name.len().is_multiple_of(2) {
        return None;
    }
    (0..name.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(name.get(i..i + 2)?, 16).ok())
        .collect()
}

impl Tier for DiskTier {
    type Key = String;
    type Value = Vec<u8>;
    type Error = io::Error;

    fn name(&self) -> &'static str {
        "disk"
    }
}

impl TierRead for DiskTier {
    async fn get(&self, key: &String) -> io::Result<Option<Vec<u8>>> {
        match fs::read(self.path_for(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    async fn exists(&self, key: &String) -> io::Result<bool> {
        self.path_for(key).try_exists()
    }
}

impl TierReadRange for DiskTier {
    /// One positional read of exactly the requested bytes — the whole value
    /// is never materialised.
    async fn read_range(&self, key: &String, range: Range<u64>) -> io::Result<Option<Vec<u8>>> {
        use std::io::{Read as _, Seek as _, SeekFrom};
        if range.start > range.end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "inverted range",
            ));
        }
        let mut file = match fs::File::open(self.path_for(key)) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let len = usize::try_from(range.end - range.start)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "range too large"))?;
        file.seek(SeekFrom::Start(range.start))?;
        let mut buf = vec![0_u8; len];
        file.read_exact(&mut buf)?;
        Ok(Some(buf))
    }
}

impl TierWrite for DiskTier {
    async fn put(&self, key: String, value: Vec<u8>) -> io::Result<Displaced<String, Vec<u8>>> {
        let path = self.path_for(&key);
        // Hex names contain no `.`, so the temp name cannot collide with a
        // real entry (and `list` skips anything that fails hex decoding).
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, &value)?;
        fs::rename(&tmp, &path)?;
        Ok(Displaced::new())
    }

    async fn delete(&self, key: &String) -> io::Result<bool> {
        match fs::remove_file(self.path_for(key)) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }
}

impl TierList for DiskTier {
    type Cursor = usize;

    async fn list(&self, cursor: Option<usize>, limit: usize) -> io::Result<Page<String, usize>> {
        let mut all = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let name = entry?.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(bytes) = hex_decode(name) else {
                continue;
            };
            let Ok(key) = String::from_utf8(bytes) else {
                continue;
            };
            all.push(key);
        }
        // Directory order is arbitrary; sort so paging is deterministic.
        all.sort_unstable();
        let offset = cursor.unwrap_or(0);
        let keys: Vec<String> = all.iter().skip(offset).take(limit).cloned().collect();
        let end = offset.saturating_add(keys.len());
        let next = (limit > 0 && end < all.len()).then_some(end);
        Ok(Page { keys, next })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<F: Future>(fut: F) -> F::Output {
        let mut fut = pin!(fut);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = fut.as_mut().poll(&mut cx) {
                return value;
            }
        }
    }

    #[test]
    fn ranged_reads_are_exact() {
        let root =
            std::env::temp_dir().join(format!("tierstore-disk-range-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let tier = DiskTier::open(&root).expect("open");
        block_on(tier.put("k".to_owned(), (0_u8..100).collect())).expect("put");

        assert_eq!(
            block_on(tier.read_range(&"k".to_owned(), 10..14)).expect("range"),
            Some(vec![10, 11, 12, 13])
        );
        assert!(
            block_on(tier.read_range(&"k".to_owned(), 90..110)).is_err(),
            "out-of-bounds must error, not truncate"
        );
        assert_eq!(
            block_on(tier.read_range(&"missing".to_owned(), 0..1)).expect("missing"),
            None
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn hex_round_trips() {
        let key = "user:1/profile — naïve";
        let encoded = hex_encode(key.as_bytes());
        assert!(encoded.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            hex_decode(&encoded).and_then(|b| String::from_utf8(b).ok()),
            Some(key.to_owned())
        );
    }

    #[test]
    fn tmp_names_are_not_decodable_keys() {
        assert_eq!(hex_decode("6162.tmp"), None);
    }
}
