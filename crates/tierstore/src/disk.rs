//! On-disk tier: the "warm" layer.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::PathBuf;

use tierstore_core::{Displaced, Page, Tier, TierList, TierRead, TierWrite};

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
