//! Tokenizer and vocabulary access via libllama.
//!
//! [`Tokenizer`] borrows a [`Model`] and exposes
//! upstream's `llama_tokenize` / `llama_detokenize` plus the per-token
//! vocabulary getters (`llama_vocab_get_text`, `..._score`, `..._attr`,
//! `..._is_eog`, `..._is_control`) and the special-token ids behind a
//! safe API. Native `llama_token` is `i32`; ForgeCore uses
//! [`TokenId`] (`u32`) everywhere and converts
//! explicitly at the boundary.
//!
//! Safety invariants (audited against upstream `v0.5.0`
//! `src/llama-vocab.cpp`; see the phase-1 report §7):
//!
//! * Every per-token getter indexes the native id table without a
//!   bounds check (`vector::at` throws across the C boundary, which
//!   terminates the process; `is_control` uses unchecked
//!   `operator[]`, which is UB out of bounds). Every caller-supplied
//!   id is therefore validated against the vocabulary size before any
//!   native call — including `is_eog`, which would be safe alone, to
//!   keep one uniform rule.
//! * Every special-token getter and `llama_tokenize` assert the vocab
//!   type is not `NONE` (`GGML_ASSERT` is always active, even in
//!   release). [`Tokenizer::open`] reads the type first —
//!   `llama_vocab_type` itself is a trivial assert-free accessor —
//!   and refuses `NONE` vocabs, so no later call can hit those
//!   asserts.
//! * Upstream guards every implicit BOS/EOS/SEP insert during
//!   `add_special` encodes with `GGML_ASSERT`. [`Tokenizer::encode`]
//!   pre-checks the required ids are present
//!   (`check_add_special`) and returns [`Error`] instead of
//!   aborting.
//! * The borrowed [`Model`] keeps the native
//!   model — and its immutable vocabulary — alive for the whole
//!   `Tokenizer` lifetime, so the cached metadata and the raw vocab
//!   pointer stay valid.

use crate::batch::TokenId;
use crate::error::{Error, Result};
use crate::model::{u32_from_upstream, Model};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;

/// Vocabulary (tokenizer) kind, mirroring `enum llama_vocab_type`.
///
/// There is no `None` variant: [`Tokenizer::open`] refuses `NONE`
/// vocabs, since every native getter asserts against them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabType {
    Spm,
    Bpe,
    Wpm,
    Ugm,
    Rwkv,
    Plamo2,
    Test,
    /// Upstream reported a kind this ForgeCore does not know yet.
    Unknown(i32),
}

impl VocabType {
    /// Map a native `llama_vocab_type` discriminant; `None` for `NONE`.
    fn from_llama(id: c_int) -> Option<Self> {
        use forge_sys::vocab_type as native;
        match id {
            native::NONE => None,
            native::SPM => Some(Self::Spm),
            native::BPE => Some(Self::Bpe),
            native::WPM => Some(Self::Wpm),
            native::UGM => Some(Self::Ugm),
            native::RWKV => Some(Self::Rwkv),
            native::PLAMO2 => Some(Self::Plamo2),
            native::TEST => Some(Self::Test),
            other => Some(Self::Unknown(other)),
        }
    }
}

/// Per-token attribute bitmask, mirroring `enum llama_token_attr`.
///
/// Tokens usually carry exactly one of [`UNKNOWN`](Self::UNKNOWN),
/// [`UNUSED`](Self::UNUSED), [`NORMAL`](Self::NORMAL),
/// [`CONTROL`](Self::CONTROL), [`USER_DEFINED`](Self::USER_DEFINED) or
/// [`BYTE`](Self::BYTE); the remaining flags combine with those.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenAttr(u32);

impl TokenAttr {
    pub const UNDEFINED: Self = Self(forge_sys::token_attr::UNDEFINED as u32);
    pub const UNKNOWN: Self = Self(forge_sys::token_attr::UNKNOWN as u32);
    pub const UNUSED: Self = Self(forge_sys::token_attr::UNUSED as u32);
    pub const NORMAL: Self = Self(forge_sys::token_attr::NORMAL as u32);
    pub const CONTROL: Self = Self(forge_sys::token_attr::CONTROL as u32);
    pub const USER_DEFINED: Self = Self(forge_sys::token_attr::USER_DEFINED as u32);
    pub const BYTE: Self = Self(forge_sys::token_attr::BYTE as u32);
    pub const NORMALIZED: Self = Self(forge_sys::token_attr::NORMALIZED as u32);
    pub const LSTRIP: Self = Self(forge_sys::token_attr::LSTRIP as u32);
    pub const RSTRIP: Self = Self(forge_sys::token_attr::RSTRIP as u32);
    pub const SINGLE_WORD: Self = Self(forge_sys::token_attr::SINGLE_WORD as u32);

    /// Raw bitmask as reported by upstream.
    pub fn bits(self) -> u32 {
        self.0
    }

    /// Whether all of `flag`'s bits are set.
    pub fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }

    /// Whether no bits are set (`UNDEFINED`).
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Special-token ids; `None` where the model defines none
/// (native `LLAMA_TOKEN_NULL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpecialTokens {
    pub bos: Option<TokenId>,
    pub eos: Option<TokenId>,
    pub eot: Option<TokenId>,
    pub sep: Option<TokenId>,
    pub nl: Option<TokenId>,
    pub pad: Option<TokenId>,
    pub mask: Option<TokenId>,
}

/// Text-to-token options. `#[non_exhaustive]` so later flags can be
/// added without breaking callers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Prepend/append BOS/EOS as the model configures
    /// ([`Tokenizer::adds_bos`] / [`adds_eos`](Tokenizer::adds_eos)).
    /// Upstream aborts when a required id is missing; ForgeCore
    /// returns [`Error`] instead (see `check_add_special`).
    pub add_special: bool,
    /// Parse special/control token spellings in the input (`"<s>"`
    /// becomes the BOS id) instead of encoding them as plaintext.
    /// Never inserts a leading space.
    pub parse_special: bool,
}

impl Default for EncodeOptions {
    /// Upstream `common` convention: specials added, not parsed.
    fn default() -> Self {
        Self {
            add_special: true,
            parse_special: false,
        }
    }
}

/// Token-to-text options. `#[non_exhaustive]` so later flags can be
/// added without breaking callers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Strip a leading BOS / trailing EOS when the model configures
    /// them ([`Tokenizer::adds_bos`] /
    /// [`adds_eos`](Tokenizer::adds_eos)).
    pub remove_special: bool,
    /// Render special/control tokens as their piece text. When
    /// false they are skipped: decoding `[BOS, tok, EOS]` yields
    /// just `tok`.
    pub unparse_special: bool,
}

impl Default for DecodeOptions {
    /// Plain pieces: no edge-stripping, control pieces skipped.
    fn default() -> Self {
        Self {
            remove_special: false,
            unparse_special: false,
        }
    }
}

/// A model's tokenizer, borrowing the [`Model`].
///
/// Cheap to open (one metadata round-trip, then cached); `!Send +
/// !Sync` like all ForgeCore handles.
pub struct Tokenizer<'model> {
    model: &'model Model,
    vocab: *const forge_sys::llama_vocab,
    n_vocab: u32,
    vocab_type: VocabType,
    add_bos: bool,
    add_eos: bool,
    special: SpecialTokens,
}

impl<'model> Tokenizer<'model> {
    /// Borrow `model`'s tokenizer and snapshot its immutable metadata.
    ///
    /// Fails when the model has no vocabulary or its vocab type is
    /// `NONE` (such models have no tokenizer upstream can run).
    pub fn open(model: &'model Model) -> Result<Self> {
        // SAFETY: raw is a live model; NULL vocab is checked.
        let vocab = unsafe { forge_sys::llama_model_get_vocab(model.raw()) };
        if vocab.is_null() {
            return Err(Error::model("model has no vocabulary"));
        }
        // SAFETY: vocab is non-NULL; `llama_vocab_type` is a trivial
        // assert-free accessor, valid for every vocab type including
        // NONE (which is rejected right here, before any asserting
        // getter runs).
        let vocab_type = unsafe {
            VocabType::from_llama(forge_sys::llama_vocab_type(vocab))
                .ok_or_else(|| Error::tokenizer("model has no tokenizer (vocabulary type NONE)"))?
        };
        // SAFETY: vocab is a live non-NONE vocab; pure metadata
        // getters on immutable post-load state.
        let n_vocab = unsafe {
            u32_from_upstream("vocabulary size", forge_sys::llama_vocab_n_tokens(vocab))
        }?;
        let (add_bos, add_eos) = unsafe {
            (
                forge_sys::llama_vocab_get_add_bos(vocab),
                forge_sys::llama_vocab_get_add_eos(vocab),
            )
        };
        let special = SpecialTokens {
            bos: special_from_native("BOS", unsafe { forge_sys::llama_vocab_bos(vocab) }, n_vocab)?,
            eos: special_from_native("EOS", unsafe { forge_sys::llama_vocab_eos(vocab) }, n_vocab)?,
            eot: special_from_native("EOT", unsafe { forge_sys::llama_vocab_eot(vocab) }, n_vocab)?,
            sep: special_from_native("SEP", unsafe { forge_sys::llama_vocab_sep(vocab) }, n_vocab)?,
            nl: special_from_native("NL", unsafe { forge_sys::llama_vocab_nl(vocab) }, n_vocab)?,
            pad: special_from_native("PAD", unsafe { forge_sys::llama_vocab_pad(vocab) }, n_vocab)?,
            mask: special_from_native(
                "MASK",
                unsafe { forge_sys::llama_vocab_mask(vocab) },
                n_vocab,
            )?,
        };
        Ok(Self {
            model,
            vocab,
            n_vocab,
            vocab_type,
            add_bos,
            add_eos,
            special,
        })
    }

    /// Vocabulary size (token count).
    pub fn n_vocab(&self) -> u32 {
        self.n_vocab
    }

    /// Which tokenizer kind this vocabulary uses.
    pub fn vocab_type(&self) -> VocabType {
        self.vocab_type
    }

    /// Whether the model configures a leading BOS on `add_special`
    /// encodes (and strips one on `remove_special` decodes).
    pub fn adds_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether the model configures a trailing EOS on `add_special`
    /// encodes (and strips one on `remove_special` decodes).
    pub fn adds_eos(&self) -> bool {
        self.add_eos
    }

    /// Special-token ids defined by the model.
    pub fn special_tokens(&self) -> SpecialTokens {
        self.special
    }

    /// Piece text of one token (owned copy; upstream's buffer is never
    /// exposed). Non-UTF-8 bytes become U+FFFD, since byte-level
    /// vocabularies legitimately hold arbitrary bytes.
    pub fn token_text(&self, id: TokenId) -> Result<String> {
        let native = checked_token_id(id, self.n_vocab)?;
        // SAFETY: id < n_vocab, so the native table lookup is in
        // bounds; the returned C string is borrowed only for the
        // copy. NULL is checked defensively (upstream never returns
        // it for a valid id).
        let text = unsafe { forge_sys::llama_vocab_get_text(self.vocab, native) };
        if text.is_null() {
            return Err(Error::tokenizer(format!("no text for token id {id}")));
        }
        Ok(unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned())
    }

    /// Vocabulary score of one token (merge/logit bias; 0.0 when the
    /// model stores none).
    pub fn token_score(&self, id: TokenId) -> Result<f32> {
        let native = checked_token_id(id, self.n_vocab)?;
        // SAFETY: id < n_vocab, so the native table lookup is in bounds.
        Ok(unsafe { forge_sys::llama_vocab_get_score(self.vocab, native) })
    }

    /// Attribute bitmask of one token.
    pub fn token_attr(&self, id: TokenId) -> Result<TokenAttr> {
        let native = checked_token_id(id, self.n_vocab)?;
        // SAFETY: id < n_vocab, so the native table lookup is in bounds.
        Ok(TokenAttr(
            unsafe { forge_sys::llama_vocab_get_attr(self.vocab, native) } as u32,
        ))
    }

    /// Whether the token ends generation (EOS, EOT, ...).
    pub fn is_eog(&self, id: TokenId) -> Result<bool> {
        let native = checked_token_id(id, self.n_vocab)?;
        // SAFETY: id < n_vocab. (`is_eog` alone would be bounds-safe
        // upstream, but every per-token entry point validates, with
        // no exceptions.)
        Ok(unsafe { forge_sys::llama_vocab_is_eog(self.vocab, native) })
    }

    /// Whether the token is a control (non-renderable) token.
    pub fn is_control(&self, id: TokenId) -> Result<bool> {
        let native = checked_token_id(id, self.n_vocab)?;
        // SAFETY: id < n_vocab; `is_control` uses unchecked
        // `operator[]` upstream, so this check averts UB.
        Ok(unsafe { forge_sys::llama_vocab_is_control(self.vocab, native) })
    }

    /// Encode `text` to token ids.
    ///
    /// `add_special` inserts BOS/EOS exactly as the model configures
    /// (see [`adds_bos`](Self::adds_bos)); the default
    /// [`EncodeOptions`] follow the upstream `common` convention
    /// (specials added, not parsed). Returned ids are always valid
    /// for this vocabulary.
    pub fn encode(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        // Fast path: with no specials requested, every upstream
        // tokenizer only iterates the (empty) fragment list for empty
        // input, so the result is empty without calling native code.
        if text.is_empty() && !options.add_special {
            return Ok(Vec::new());
        }
        let text_len = c_int::try_from(text.len())
            .map_err(|_| Error::tokenizer(format!("text too long: {} bytes", text.len())))?;
        if options.add_special {
            check_add_special(self.vocab_type, self.add_bos, self.add_eos, &self.special)?;
        }
        // Sizing call (NULL/0, as upstream's own callers do), then a
        // fill call; native tokenization is deterministic for fixed
        // input, so one fill always fits.
        let mut tokens: Vec<c_int> = Vec::new();
        loop {
            let out = if tokens.is_empty() {
                ptr::null_mut()
            } else {
                tokens.as_mut_ptr()
            };
            // SAFETY: out points to tokens.len() live `c_int`s (or is
            // NULL with capacity 0 for the sizing call); text points
            // to text_len live bytes (length-delimited upstream, so
            // interior NULs are data, not terminators); the vocab is
            // live via the model borrow.
            let ret = unsafe {
                forge_sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast::<c_char>(),
                    text_len,
                    out,
                    tokens.len() as c_int,
                    options.add_special,
                    options.parse_special,
                )
            };
            match capacity_step(ret)? {
                Capacity::Ready(count) => {
                    tokens.truncate(count);
                    break tokens
                        .into_iter()
                        .map(|id| {
                            TokenId::try_from(id).map_err(|_| {
                                Error::tokenizer(format!(
                                    "native tokenizer returned invalid id {id}"
                                ))
                            })
                        })
                        .collect();
                }
                Capacity::Grow(need) => tokens.resize(need, 0),
            }
        }
    }

    /// Decode token ids to text (inverse of [`encode`](Self::encode)).
    ///
    /// Bytes come back verbatim as upstream renders them: pieces
    /// concatenate with no separator, so an SPM round-trip yields
    /// the input with `' '` mapped to `'▁'` (SPM represents spaces
    /// as U+2581, including the conventional leading one the
    /// encoder emits). Every id must be valid for this vocabulary; fails on
    /// the first out-of-range id without calling native code. Also
    /// fails when the bytes are not valid UTF-8 (decoding a bare
    /// prefix of a multi-byte sequence, ...); byte-exact callers are
    /// future work.
    pub fn decode(&self, tokens: &[TokenId], options: &DecodeOptions) -> Result<String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }
        let mut native = Vec::with_capacity(tokens.len());
        for id in tokens {
            native.push(checked_token_id(*id, self.n_vocab)?);
        }
        let n_tokens = c_int::try_from(native.len())
            .map_err(|_| Error::tokenizer(format!("too many tokens: {}", native.len())))?;
        // Sizing call, then a fill call (same protocol as encode).
        let mut text: Vec<u8> = Vec::new();
        loop {
            let out = if text.is_empty() {
                ptr::null_mut()
            } else {
                text.as_mut_ptr().cast::<c_char>()
            };
            let out_max = c_int::try_from(text.len()).map_err(|_| {
                Error::tokenizer(format!("decoded text too long: {} bytes", text.len()))
            })?;
            // SAFETY: native holds n_tokens live `c_int`s, all
            // validated in range; out points to text.len() live bytes
            // (or is NULL with capacity 0); upstream writes at most
            // out_max bytes and no NUL terminator.
            let ret = unsafe {
                forge_sys::llama_detokenize(
                    self.vocab,
                    native.as_ptr(),
                    n_tokens,
                    out,
                    out_max,
                    options.remove_special,
                    options.unparse_special,
                )
            };
            match capacity_step(ret)? {
                Capacity::Ready(count) => {
                    text.truncate(count);
                    break bytes_to_text(text);
                }
                Capacity::Grow(need) => text.resize(need, 0),
            }
        }
    }
}

impl std::fmt::Debug for Tokenizer<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tokenizer")
            .field("model", &self.model)
            .field("n_vocab", &self.n_vocab)
            .field("vocab_type", &self.vocab_type)
            .field("adds_bos", &self.add_bos)
            .field("adds_eos", &self.add_eos)
            .field("special_tokens", &self.special)
            .finish_non_exhaustive()
    }
}

/// One step of the native capacity protocol: `llama_tokenize` and
/// `llama_detokenize` return the output size on success and
/// `-(required capacity)` when the buffer is too small.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Capacity {
    /// Output complete with exactly this many items.
    Ready(usize),
    /// Output truncated; retry with (at least) this capacity.
    Grow(usize),
}

/// Interpret one native capacity-protocol return value.
fn capacity_step(ret: c_int) -> Result<Capacity> {
    if ret == c_int::MIN {
        return Err(Error::tokenizer(
            "native tokenizer reports integer overflow (output exceeds i32::MAX)",
        ));
    }
    if ret < 0 {
        // `ret > c_int::MIN`, so the negation cannot overflow; the
        // result fits `c_int`, hence `usize`.
        Ok(Capacity::Grow((-ret) as usize))
    } else {
        Ok(Capacity::Ready(ret as usize))
    }
}

/// Validate a caller-supplied token id against the vocabulary size
/// and convert it to the native `llama_token` (`i32`).
fn checked_token_id(id: TokenId, n_vocab: u32) -> Result<c_int> {
    if id < n_vocab {
        // `n_vocab` itself came from a non-negative `c_int`, so every
        // smaller `u32` fits.
        Ok(id as c_int)
    } else {
        Err(Error::tokenizer(format!(
            "token id {id} out of range (vocab size {n_vocab})"
        )))
    }
}

/// Interpret one native special-token id: `LLAMA_TOKEN_NULL` means
/// "absent"; anything else must be a valid id for this vocabulary.
fn special_from_native(name: &str, id: c_int, n_vocab: u32) -> Result<Option<TokenId>> {
    if id == forge_sys::LLAMA_TOKEN_NULL {
        return Ok(None);
    }
    if id < 0 || id as u32 >= n_vocab {
        return Err(Error::tokenizer(format!(
            "model reports invalid special token {name}: {id} (vocab size {n_vocab})"
        )));
    }
    Ok(Some(id as u32))
}

/// Refuse `add_special` encodes the native tokenizer would abort on.
///
/// Upstream guards every implicit BOS/EOS/SEP insert with
/// `GGML_ASSERT` (always active, even in release): SPM/BPE/UGM assert
/// the id is present whenever the matching model flag is set, while
/// WPM asserts BOS *and* SEP unconditionally. RWKV, PLAMO2 and TEST
/// ignore `add_special`, so they are always safe; unknown future
/// types are refused rather than risk an abort.
fn check_add_special(
    vocab_type: VocabType,
    add_bos: bool,
    add_eos: bool,
    special: &SpecialTokens,
) -> Result<()> {
    match vocab_type {
        VocabType::Spm | VocabType::Bpe | VocabType::Ugm => {
            if add_bos && special.bos.is_none() {
                return Err(Error::tokenizer(
                    "cannot add special tokens: model configures BOS but defines no BOS token",
                ));
            }
            if add_eos && special.eos.is_none() {
                return Err(Error::tokenizer(
                    "cannot add special tokens: model configures EOS but defines no EOS token",
                ));
            }
        }
        VocabType::Wpm => {
            if special.bos.is_none() {
                return Err(Error::tokenizer(
                    "cannot add special tokens: WPM model defines no BOS token",
                ));
            }
            if special.sep.is_none() {
                return Err(Error::tokenizer(
                    "cannot add special tokens: WPM model defines no SEP token",
                ));
            }
        }
        VocabType::Rwkv | VocabType::Plamo2 | VocabType::Test => {}
        VocabType::Unknown(id) => {
            return Err(Error::unsupported(format!(
                "add_special for unknown vocab type {id}"
            )));
        }
    }
    Ok(())
}

/// Convert decoded bytes to text, rejecting invalid UTF-8.
fn bytes_to_text(bytes: Vec<u8>) -> Result<String> {
    let len = bytes.len();
    String::from_utf8(bytes)
        .map_err(|_| Error::tokenizer(format!("decoded text is not valid UTF-8 ({len} bytes)")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specials_with(
        bos: Option<TokenId>,
        eos: Option<TokenId>,
        sep: Option<TokenId>,
    ) -> SpecialTokens {
        SpecialTokens {
            bos,
            eos,
            eot: None,
            sep,
            nl: None,
            pad: None,
            mask: None,
        }
    }

    #[test]
    fn vocab_type_from_llama_ids() {
        use forge_sys::vocab_type as native;
        assert_eq!(VocabType::from_llama(native::NONE), None);
        assert_eq!(VocabType::from_llama(native::SPM), Some(VocabType::Spm));
        assert_eq!(VocabType::from_llama(native::BPE), Some(VocabType::Bpe));
        assert_eq!(VocabType::from_llama(native::WPM), Some(VocabType::Wpm));
        assert_eq!(VocabType::from_llama(native::UGM), Some(VocabType::Ugm));
        assert_eq!(VocabType::from_llama(native::RWKV), Some(VocabType::Rwkv));
        assert_eq!(
            VocabType::from_llama(native::PLAMO2),
            Some(VocabType::Plamo2)
        );
        assert_eq!(VocabType::from_llama(native::TEST), Some(VocabType::Test));
        assert_eq!(VocabType::from_llama(1234), Some(VocabType::Unknown(1234)));
    }

    #[test]
    fn token_attr_flags_match_upstream_bits() {
        assert_eq!(TokenAttr::UNDEFINED.bits(), 0);
        assert_eq!(TokenAttr::UNKNOWN.bits(), 1);
        assert_eq!(TokenAttr::UNUSED.bits(), 2);
        assert_eq!(TokenAttr::NORMAL.bits(), 4);
        assert_eq!(TokenAttr::CONTROL.bits(), 8);
        assert_eq!(TokenAttr::USER_DEFINED.bits(), 16);
        assert_eq!(TokenAttr::BYTE.bits(), 32);
        assert_eq!(TokenAttr::NORMALIZED.bits(), 64);
        assert_eq!(TokenAttr::LSTRIP.bits(), 128);
        assert_eq!(TokenAttr::RSTRIP.bits(), 256);
        assert_eq!(TokenAttr::SINGLE_WORD.bits(), 512);
        assert!(TokenAttr::UNDEFINED.is_empty());
        assert!(!TokenAttr::NORMAL.is_empty());
        assert!(
            TokenAttr(TokenAttr::NORMAL.bits() | TokenAttr::LSTRIP.bits())
                .contains(TokenAttr::NORMAL)
        );
        assert!(!TokenAttr::NORMAL.contains(TokenAttr::CONTROL));
    }

    #[test]
    fn capacity_step_maps_the_native_protocol() {
        assert_eq!(capacity_step(0), Ok(Capacity::Ready(0)));
        assert_eq!(capacity_step(5), Ok(Capacity::Ready(5)));
        assert_eq!(capacity_step(-7), Ok(Capacity::Grow(7)));
        assert_eq!(
            capacity_step(c_int::MIN + 1),
            Ok(Capacity::Grow(c_int::MAX as usize))
        );
        let overflow = capacity_step(c_int::MIN).expect_err("INT32_MIN must fail");
        assert!(
            overflow.to_string().contains("overflow"),
            "overflow names itself: {overflow}"
        );
    }

    #[test]
    fn checked_token_id_enforces_vocab_bounds() {
        assert_eq!(checked_token_id(0, 32), Ok(0));
        assert_eq!(checked_token_id(31, 32), Ok(31));
        let edge = checked_token_id(32, 32).expect_err("id == n_vocab must fail");
        assert!(
            edge.to_string().contains("32"),
            "error names the bad id: {edge}"
        );
        checked_token_id(TokenId::MAX, 32).expect_err("u32::MAX must fail");
    }

    #[test]
    fn special_from_native_maps_null_and_rejects_garbage() {
        assert_eq!(
            special_from_native("BOS", forge_sys::LLAMA_TOKEN_NULL, 32),
            Ok(None)
        );
        assert_eq!(special_from_native("BOS", 1, 32), Ok(Some(1)));
        assert_eq!(special_from_native("BOS", 31, 32), Ok(Some(31)));
        special_from_native("BOS", -2, 32).expect_err("negative garbage must fail");
        special_from_native("BOS", 32, 32).expect_err("id == n_vocab must fail");
    }

    #[test]
    fn add_special_guard_covers_every_vocab_type() {
        let full = specials_with(Some(1), Some(2), Some(3));
        let no_bos = specials_with(None, Some(2), Some(3));
        let no_eos = specials_with(Some(1), None, Some(3));
        let no_sep = specials_with(Some(1), Some(2), None);
        // SPM/BPE/UGM follow the model flags.
        for vocab in [VocabType::Spm, VocabType::Bpe, VocabType::Ugm] {
            check_add_special(vocab, true, true, &full).expect("full passes");
            check_add_special(vocab, false, false, &no_bos).expect("flags off passes");
            check_add_special(vocab, true, false, &no_bos).expect_err("missing BOS fails");
            check_add_special(vocab, false, true, &no_eos).expect_err("missing EOS fails");
        }
        // WPM needs BOS and SEP regardless of the flags.
        check_add_special(VocabType::Wpm, false, false, &full).expect("WPM full passes");
        check_add_special(VocabType::Wpm, false, false, &no_bos).expect_err("WPM needs BOS");
        check_add_special(VocabType::Wpm, false, false, &no_sep).expect_err("WPM needs SEP");
        // Types that ignore add_special always pass.
        for vocab in [VocabType::Rwkv, VocabType::Plamo2, VocabType::Test] {
            check_add_special(vocab, true, true, &no_bos).expect("ignored passes");
        }
        // Unknown future types are refused, never risked.
        let unknown = check_add_special(VocabType::Unknown(99), true, true, &full)
            .expect_err("unknown type must fail");
        assert!(
            unknown.to_string().starts_with("unsupported"),
            "unknown type is unsupported: {unknown}"
        );
    }

    #[test]
    fn bytes_to_text_rejects_invalid_utf8() {
        assert_eq!(bytes_to_text(b"tok5".to_vec()), Ok("tok5".to_string()));
        assert_eq!(bytes_to_text(Vec::new()), Ok(String::new()));
        let bad = bytes_to_text(vec![0xff]).expect_err("0xFF must fail");
        assert!(
            bad.to_string().contains("UTF-8"),
            "error names the encoding: {bad}"
        );
    }

    #[test]
    fn option_defaults_follow_upstream_common() {
        assert_eq!(
            EncodeOptions::default(),
            EncodeOptions {
                add_special: true,
                parse_special: false,
            }
        );
        assert_eq!(
            DecodeOptions::default(),
            DecodeOptions {
                remove_special: false,
                unparse_special: false,
            }
        );
    }
}
