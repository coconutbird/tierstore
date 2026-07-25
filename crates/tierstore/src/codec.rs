//! Typed-values-over-byte-store codec middleware.
//!
//! Byte-oriented tiers (`DiskTier`, mmap tiers, object stores) speak
//! `String → bytes`; applications speak typed keys and values.
//! [`CodecTier`] bridges the two: a one-way key mapping into the byte
//! tier's key space (hashes are fine — it is never reversed), and a value
//! codec that **embeds the original key in the encoding**. That embedded
//! key is what lets rollover work through the boundary: entries the byte
//! tier displaces are decoded back into fully typed `(key, value)` pairs,
//! so a router can demote them onward.
//!
//! Codec failures are typed errors ([`CodecError::Codec`]), which the
//! router treats as tier failures — a corrupt or oversized entry falls
//! through / skips per policy instead of corrupting the data path. Use the
//! cache preset's best-effort writes if an encode rejection (e.g. an
//! oversize gate) should skip this tier rather than fail the fill.
//!
//! `TierList` is deliberately not forwarded: listing would yield inner
//! (hashed) keys that cannot be reversed.

use std::error::Error as StdError;
use std::fmt;

use tierstore_core::{Displaced, Tier, TierRead, TierWrite};

use crate::error::BoxError;

type KeyFn<K, IK> = dyn Fn(&K) -> IK + Send + Sync;
type EncodeFn<K, V, IV> = dyn Fn(&K, &V) -> Result<IV, BoxError> + Send + Sync;
type DecodeFn<K, V, IV> = dyn Fn(IV) -> Result<(K, V), BoxError> + Send + Sync;

/// Middleware tier presenting typed keys/values over a byte-oriented inner
/// tier.
///
/// The key-embedding contract: `encode` receives the original key and must
/// embed it in the encoded bytes; `decode` returns it. That is what lets
/// entries displaced by the byte tier climb back out fully typed for
/// demotion.
pub struct CodecTier<T: Tier, K, V> {
    inner: T,
    name: String,
    to_key: Box<KeyFn<K, T::Key>>,
    encode: Box<EncodeFn<K, V, T::Value>>,
    decode: Box<DecodeFn<K, V, T::Value>>,
}

impl<T: Tier, K, V> CodecTier<T, K, V> {
    /// Wraps `inner` with a key mapping and a value codec.
    ///
    /// `decode` must return the *original* key that `encode` embedded, so
    /// displaced entries can climb back out of the byte tier typed.
    pub fn new(
        inner: T,
        to_key: impl Fn(&K) -> T::Key + Send + Sync + 'static,
        encode: impl Fn(&K, &V) -> Result<T::Value, BoxError> + Send + Sync + 'static,
        decode: impl Fn(T::Value) -> Result<(K, V), BoxError> + Send + Sync + 'static,
    ) -> Self {
        let name = format!("codec({})", inner.name());
        Self {
            inner,
            name,
            to_key: Box::new(to_key),
            encode: Box::new(encode),
            decode: Box::new(decode),
        }
    }
}

impl<T: Tier, K, V> fmt::Debug for CodecTier<T, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodecTier")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// Error from a [`CodecTier`]: the backend failed, or a value failed to
/// encode/decode.
#[derive(Debug)]
pub enum CodecError<E> {
    /// The inner tier itself failed.
    Backend(E),
    /// Encoding or decoding a value failed (corrupt entry, oversize
    /// rejection, schema drift).
    Codec(BoxError),
}

impl<E: fmt::Display> fmt::Display for CodecError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => write!(f, "backend error: {error}"),
            Self::Codec(error) => write!(f, "codec error: {error}"),
        }
    }
}

impl<E: StdError + 'static> StdError for CodecError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Codec(error) => {
                let source: &(dyn StdError + 'static) = &**error;
                Some(source)
            }
        }
    }
}

impl<T: Tier, K, V> Tier for CodecTier<T, K, V> {
    type Key = K;
    type Value = V;
    type Error = CodecError<T::Error>;

    fn name(&self) -> &str {
        &self.name
    }
}

impl<T, K, V> TierRead for CodecTier<T, K, V>
where
    T: TierRead + Sync,
    T::Key: Send + Sync,
    T::Value: Send,
    K: Send + Sync,
    V: Send,
{
    async fn get(&self, key: &K) -> Result<Option<V>, CodecError<T::Error>> {
        let inner_key = (self.to_key)(key);
        match self
            .inner
            .get(&inner_key)
            .await
            .map_err(CodecError::Backend)?
        {
            None => Ok(None),
            Some(encoded) => {
                let (_key, value) = (self.decode)(encoded).map_err(CodecError::Codec)?;
                Ok(Some(value))
            }
        }
    }

    async fn exists(&self, key: &K) -> Result<bool, CodecError<T::Error>> {
        let inner_key = (self.to_key)(key);
        self.inner
            .exists(&inner_key)
            .await
            .map_err(CodecError::Backend)
    }

    async fn get_many(&self, keys: &[K]) -> Result<Vec<Option<V>>, CodecError<T::Error>> {
        let inner_keys: Vec<T::Key> = keys.iter().map(|key| (self.to_key)(key)).collect();
        let found = self
            .inner
            .get_many(&inner_keys)
            .await
            .map_err(CodecError::Backend)?;
        let mut values = Vec::with_capacity(found.len());
        for encoded in found {
            values.push(match encoded {
                None => None,
                Some(encoded) => {
                    let (_key, value) = (self.decode)(encoded).map_err(CodecError::Codec)?;
                    Some(value)
                }
            });
        }
        Ok(values)
    }
}

impl<T, K, V> TierWrite for CodecTier<T, K, V>
where
    T: TierWrite + Sync,
    T::Key: Send + Sync,
    T::Value: Send,
    K: Send + Sync,
    V: Send,
{
    async fn put(&self, key: K, value: V) -> Result<Displaced<K, V>, CodecError<T::Error>> {
        let encoded = (self.encode)(&key, &value).map_err(CodecError::Codec)?;
        let inner_key = (self.to_key)(&key);
        let displaced = self
            .inner
            .put(inner_key, encoded)
            .await
            .map_err(CodecError::Backend)?;
        // Displaced entries come back inner-typed; the embedded original
        // key lets them climb back out fully typed. An entry that fails to
        // decode is dropped — it was being evicted anyway.
        Ok(displaced
            .into_iter()
            .filter_map(|(_inner_key, encoded)| (self.decode)(encoded).ok())
            .collect())
    }

    async fn delete(&self, key: &K) -> Result<bool, CodecError<T::Error>> {
        let inner_key = (self.to_key)(key);
        self.inner
            .delete(&inner_key)
            .await
            .map_err(CodecError::Backend)
    }

    async fn put_many(
        &self,
        entries: Vec<(K, V)>,
    ) -> Result<Displaced<K, V>, CodecError<T::Error>> {
        let mut encoded_entries = Vec::with_capacity(entries.len());
        for (key, value) in &entries {
            let encoded = (self.encode)(key, value).map_err(CodecError::Codec)?;
            encoded_entries.push(((self.to_key)(key), encoded));
        }
        let displaced = self
            .inner
            .put_many(encoded_entries)
            .await
            .map_err(CodecError::Backend)?;
        Ok(displaced
            .into_iter()
            .filter_map(|(_inner_key, encoded)| (self.decode)(encoded).ok())
            .collect())
    }

    async fn delete_many(&self, keys: &[K]) -> Result<Vec<bool>, CodecError<T::Error>> {
        let inner_keys: Vec<T::Key> = keys.iter().map(|key| (self.to_key)(key)).collect();
        self.inner
            .delete_many(&inner_keys)
            .await
            .map_err(CodecError::Backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryTier, Router};
    use std::future::Future;
    use std::num::NonZeroUsize;
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

    /// Typed value with a hand-rolled `key\n<id>\n<body>` byte encoding.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Doc {
        id: u32,
        body: String,
    }

    fn typed_over_bytes(
        inner: MemoryTier<String, Vec<u8>>,
    ) -> CodecTier<MemoryTier<String, Vec<u8>>, String, Doc> {
        CodecTier::new(
            inner,
            |key: &String| format!("hashed:{key}"),
            |key: &String, doc: &Doc| Ok(format!("{key}\n{}\n{}", doc.id, doc.body).into_bytes()),
            |bytes: Vec<u8>| {
                let text = String::from_utf8(bytes).map_err(|e| -> BoxError { e.into() })?;
                let mut parts = text.splitn(3, '\n');
                let (Some(key), Some(id), Some(body)) = (parts.next(), parts.next(), parts.next())
                else {
                    return Err("malformed record".into());
                };
                let id = id
                    .parse()
                    .map_err(|e: std::num::ParseIntError| -> BoxError { e.into() })?;
                Ok((
                    key.to_owned(),
                    Doc {
                        id,
                        body: body.to_owned(),
                    },
                ))
            },
        )
    }

    #[test]
    fn typed_round_trip_over_a_byte_tier() {
        let tier = typed_over_bytes(MemoryTier::unbounded());
        let doc = Doc {
            id: 7,
            body: "hello".to_owned(),
        };
        block_on(tier.put("k".to_owned(), doc.clone())).expect("put");
        assert_eq!(block_on(tier.get(&"k".to_owned())).expect("get"), Some(doc));
        assert!(block_on(tier.exists(&"k".to_owned())).expect("exists"));
        assert!(block_on(tier.delete(&"k".to_owned())).expect("delete"));
        assert_eq!(block_on(tier.get(&"k".to_owned())).expect("get"), None);
    }

    #[test]
    fn displaced_entries_climb_back_out_typed() {
        // A bounded byte tier under the codec: displacement crosses the
        // codec boundary and must come back with the ORIGINAL typed key.
        let byte_tier = MemoryTier::bounded(NonZeroUsize::new(1).expect("nonzero"));
        let router: Router<String, Doc> =
            Router::builder().tier(typed_over_bytes(byte_tier)).build();

        let first = Doc {
            id: 1,
            body: "one".to_owned(),
        };
        block_on(router.put("a".to_owned(), first.clone())).expect("put a");
        let displaced = block_on(router.put(
            "b".to_owned(),
            Doc {
                id: 2,
                body: "two".to_owned(),
            },
        ))
        .expect("put b");
        assert_eq!(displaced, vec![("a".to_owned(), first)]);
    }
}
