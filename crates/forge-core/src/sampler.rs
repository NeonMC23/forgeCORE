//! Native sampler chains over logits.
//!
//! [`SamplerChain`] owns one native `llama_sampler` chain
//! (greedy/dist/top-k/top-p/min-p/temp elements) and applies it to
//! candidate logits via [`sample`](SamplerChain::sample), returning a
//! [`TokenId`]. [`SampleConfig`] is the explicit,
//! validated configuration that builds the canonical chain shape
//! (filters → temperature → selector).
//!
//! Composition with the execution path:
//!
//! ```text
//! Context::logits() -> Logits -> logits.values() -> SamplerChain::sample()
//!     -> TokenId -> BatchBuilder::push / Tokenizer::decode
//! ```
//!
//! The sampler borrows the logits slice only for the call (native
//! needs `{id, logit}` pairs, so one format-conversion copy into a
//! caller-owned candidate array is required and documented, not
//! avoided). Sampled ids index the input slice, so they are always
//! valid for the vocabulary the logits came from.
//!
//! Statefulness and seeds: a chain holding `dist` is stateful — its
//! `mt19937` advances once per [`sample`](SamplerChain::sample).
//! [`reset`](SamplerChain::reset) replays determinism by re-seeding
//! from the *original* seed: explicit seeds reproduce the stream,
//! while random-seeded chains re-randomize. `None` seed means random
//! (native `LLAMA_DEFAULT_SEED`); `Some(u32::MAX)` spells the same
//! thing explicitly. [`seed`](SamplerChain::seed) reports the
//! resolved seed in use, or `None` when the chain holds no seeded
//! sampler.
//!
//! Every chain that should produce a token must end with a selector
//! (`greedy` or `dist`); anything else (empty chain, filters only)
//! selects nothing, which [`sample`](SamplerChain::sample) reports
//! as [`Error`] — never an abort.
//!
//! Out of scope (assessed against the pinned native API, not bound):
//! context-bound `llama_sampler_sample` (bypasses owned `Logits`),
//! penalties/repetition (policy, not primitive), grammar/mixture
//! samplers (mirostat, DRY, XTC, typical, top-n-sigma, temp-ext,
//! adaptive-p, infill, logit-bias — later passes per the roadmap),
//! custom `llama_sampler_i` vtables, and backend (graph) samplers.
//! Logits are expected finite (real decode output always is);
//! non-finite input is garbage-in-garbage-out, exactly as upstream.

use crate::batch::TokenId;
use crate::error::{Error, Result};
use std::marker::PhantomData;
use std::os::raw::c_int;
use std::rc::Rc;

/// Minimum candidates every truncation filter keeps (native
/// `min_keep`). Fixed at the upstream `common` default so chains
/// stay total: a filter can never produce an empty candidate set.
const MIN_KEEP: usize = 1;

/// Explicit sampling configuration; validated purely, built into a
/// chain by [`build_chain`](SampleConfig::build_chain).
/// `#[non_exhaustive]` so later filters can be added without
/// breaking callers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleConfig {
    /// Temperature divider. `0.0` selects greedy mode (argmax tail,
    /// fully deterministic, `seed` ignored); otherwise a temp
    /// element plus a `dist` tail. Must be finite and non-negative.
    pub temperature: f32,
    /// Keep the top-`k` logits (`None` = no filter). `Some(0)` and
    /// values above `i32::MAX` are invalid; `k` at or above the
    /// candidate count keeps everything.
    pub top_k: Option<u32>,
    /// Nucleus mass in `(0, 1]` (`None` = no filter). `1.0` keeps
    /// everything; `0.0`, negatives, values above `1.0`, and
    /// non-finite values are invalid.
    pub top_p: Option<f32>,
    /// Minimum per-token probability mass relative to the max, in
    /// `(0, 1]` (`None` = no filter). Same validity rules as
    /// `top_p`.
    pub min_p: Option<f32>,
    /// RNG seed for the `dist` tail. `None` (or `Some(u32::MAX)`)
    /// means random; any other value reproduces its stream.
    pub seed: Option<u32>,
}

impl Default for SampleConfig {
    /// Pure greedy: deterministic, no RNG, no filters. (No
    /// application policy — the identity of sampling.)
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_p: None,
            seed: None,
        }
    }
}

impl SampleConfig {
    /// Check every field; [`build_chain`](SampleConfig::build_chain)
    /// and the `SamplerChain::push_*` methods enforce the same
    /// rules element by element.
    pub fn validate(&self) -> Result<()> {
        validate_temp(self.temperature)?;
        if let Some(k) = self.top_k {
            validate_top_k(k)?;
        }
        if let Some(p) = self.top_p {
            validate_prob(p, "top_p")?;
        }
        if let Some(p) = self.min_p {
            validate_prob(p, "min_p")?;
        }
        Ok(())
    }

    /// Build the canonical chain: `top_k → top_p → min_p`, then a
    /// `greedy` tail for `temperature == 0.0`, else a `temp` element
    /// plus a `dist` tail. (`seed` is ignored in greedy mode — there
    /// is no RNG to seed.)
    pub fn build_chain(&self) -> Result<SamplerChain> {
        self.validate()?;
        let mut chain = SamplerChain::new();
        if let Some(k) = self.top_k {
            chain.push_top_k(k)?;
        }
        if let Some(p) = self.top_p {
            chain.push_top_p(p)?;
        }
        if let Some(p) = self.min_p {
            chain.push_min_p(p)?;
        }
        if self.temperature == 0.0 {
            chain.push_greedy();
        } else {
            chain.push_temp(self.temperature)?;
            chain.push_dist(self.seed);
        }
        Ok(chain)
    }
}

/// An owned native sampler chain.
///
/// Owns exactly one `llama_sampler` chain and frees it (plus every
/// added element) on drop. `!Send + !Sync` like all ForgeCore
/// handles. A chain is decoupled from any [`Context`](crate::context::Context):
/// it samples caller-supplied logits, so it borrows nothing.
pub struct SamplerChain {
    raw: *mut forge_sys::llama_sampler,
    // Ownership marker only: keeps the chain on its thread.
    marker: PhantomData<Rc<()>>,
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful chain_init, is freed
        // exactly once, and owns every element added to it (native
        // frees elements with the chain).
        unsafe { forge_sys::llama_sampler_free(self.raw) };
    }
}

impl SamplerChain {
    /// Create an empty chain. Only fails via native OOM throw
    /// (process abort, like a Rust allocation failure).
    pub fn new() -> Self {
        // SAFETY: default params are valid by construction; the
        // constructor only fails via C++ OOM throw, never NULL.
        let raw = unsafe {
            forge_sys::llama_sampler_chain_init(forge_sys::llama_sampler_chain_default_params())
        };
        Self {
            raw,
            marker: PhantomData,
        }
    }

    /// Append a greedy (argmax, first-max-wins) selector.
    pub fn push_greedy(&mut self) {
        // SAFETY: raw is a live chain; the constructor only fails via
        // OOM throw; the chain takes ownership of the element.
        unsafe {
            forge_sys::llama_sampler_chain_add(self.raw, forge_sys::llama_sampler_init_greedy());
        }
    }

    /// Append a distribution sampler. `None` seeds randomly;
    /// `Some(s)` reproduces stream `s` (including across
    /// [`reset`](Self::reset)).
    pub fn push_dist(&mut self, seed: Option<u32>) {
        // SAFETY: raw is a live chain; the constructor only fails via
        // OOM throw; the chain takes ownership of the element.
        unsafe {
            forge_sys::llama_sampler_chain_add(
                self.raw,
                forge_sys::llama_sampler_init_dist(seed_or_random(seed)),
            );
        }
    }

    /// Append a top-k filter (`k >= 1`, `k <= i32::MAX`).
    pub fn push_top_k(&mut self, k: u32) -> Result<()> {
        let k = validate_top_k(k)?;
        // SAFETY: raw is a live chain; the constructor only fails via
        // OOM throw; the chain takes ownership of the element.
        unsafe {
            forge_sys::llama_sampler_chain_add(self.raw, forge_sys::llama_sampler_init_top_k(k));
        }
        Ok(())
    }

    /// Append a nucleus (top-p) filter (`p` in `(0, 1]`, finite).
    pub fn push_top_p(&mut self, p: f32) -> Result<()> {
        let p = validate_prob(p, "top_p")?;
        // SAFETY: raw is a live chain; the constructor only fails via
        // OOM throw; the chain takes ownership of the element.
        unsafe {
            forge_sys::llama_sampler_chain_add(
                self.raw,
                forge_sys::llama_sampler_init_top_p(p, MIN_KEEP),
            );
        }
        Ok(())
    }

    /// Append a min-p filter (`p` in `(0, 1]`, finite).
    pub fn push_min_p(&mut self, p: f32) -> Result<()> {
        let p = validate_prob(p, "min_p")?;
        // SAFETY: raw is a live chain; the constructor only fails via
        // OOM throw; the chain takes ownership of the element.
        unsafe {
            forge_sys::llama_sampler_chain_add(
                self.raw,
                forge_sys::llama_sampler_init_min_p(p, MIN_KEEP),
            );
        }
        Ok(())
    }

    /// Append a temperature divider (finite, non-negative; `0.0`
    /// keeps only the argmax, matching native semantics).
    pub fn push_temp(&mut self, temp: f32) -> Result<()> {
        let temp = validate_temp(temp)?;
        // SAFETY: raw is a live chain; the constructor only fails via
        // OOM throw; the chain takes ownership of the element.
        unsafe {
            forge_sys::llama_sampler_chain_add(self.raw, forge_sys::llama_sampler_init_temp(temp));
        }
        Ok(())
    }

    /// Number of elements in the chain.
    pub fn len(&self) -> usize {
        // SAFETY: raw is a live chain; the count is non-negative.
        let n = unsafe { forge_sys::llama_sampler_chain_n(self.raw) };
        n as usize
    }

    /// Whether the chain holds no elements (such a chain selects
    /// nothing — see [`sample`](Self::sample)).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove the element at `index`, freeing it. Out-of-range
    /// indices are [`Error`], never native UB (native reports NULL,
    /// which is checked).
    pub fn remove(&mut self, index: usize) -> Result<()> {
        let index = c_int::try_from(index)
            .map_err(|_| Error::sample(format!("chain index {index} out of range")))?;
        // SAFETY: raw is a live chain; NULL (out of range) is
        // checked; a removed element is owned by the caller, so it
        // is freed exactly once here.
        unsafe {
            let removed = forge_sys::llama_sampler_chain_remove(self.raw, index);
            if removed.is_null() {
                return Err(Error::sample(format!(
                    "chain index {index} out of range (length {})",
                    self.len()
                )));
            }
            forge_sys::llama_sampler_free(removed);
        }
        Ok(())
    }

    /// Resolved RNG seed in use, or `None` when the chain holds no
    /// seeded sampler. Random-seeded chains report their resolved
    /// (unpredictable) seed.
    pub fn seed(&self) -> Option<u32> {
        // SAFETY: raw is a live chain; pure metadata read.
        let seed = unsafe { forge_sys::llama_sampler_get_seed(self.raw) };
        (seed != forge_sys::LLAMA_DEFAULT_SEED).then_some(seed)
    }

    /// Reset element state: `dist` re-seeds from its original seed,
    /// so explicit-seed chains replay their stream from the start.
    /// (Random-seeded chains re-randomize — there is no fixed stream
    /// to restore.) All other bound elements are stateless.
    pub fn reset(&mut self) {
        // SAFETY: raw is a live chain; stateless elements ignore it.
        unsafe { forge_sys::llama_sampler_reset(self.raw) };
    }

    /// Accept `token` as sampled (the second half of the native
    /// sample/accept protocol). A no-op for every bound element in
    /// the CPU path; exposed so multi-step callers follow the
    /// protocol future stateful samplers will require.
    /// [`sample`](Self::sample) already accepts internally.
    pub fn accept(&mut self, token: TokenId) -> Result<()> {
        let token = c_int::try_from(token)
            .map_err(|_| Error::sample(format!("token id {token} too large")))?;
        // SAFETY: raw is a live chain; the token fits `llama_token`.
        unsafe { forge_sys::llama_sampler_accept(self.raw, token) };
        Ok(())
    }

    /// Sample one token id from `logits` (index `i` becomes candidate
    /// id `i`, so pass `Logits::values()` for model output).
    ///
    /// The chain applies in order over a caller-owned candidate
    /// array (one `{id, logit}` conversion copy, required by the
    /// native layout), the selected id is read back and accepted
    /// internally, and its [`TokenId`] is
    /// returned. Empty input and chains that select nothing (empty
    /// chain, filters without a selector) are [`Error`].
    pub fn sample(&mut self, logits: &[f32]) -> Result<TokenId> {
        if logits.is_empty() {
            return Err(Error::sample("cannot sample from empty logits"));
        }
        if logits.len() > c_int::MAX as usize {
            return Err(Error::sample(format!(
                "logits too large: {} values",
                logits.len()
            )));
        }
        let mut candidates: Vec<forge_sys::llama_token_data> = logits
            .iter()
            .enumerate()
            .map(|(index, &logit)| forge_sys::llama_token_data {
                // `index < logits.len() <= c_int::MAX`, so this fits.
                id: index as c_int,
                logit,
                p: 0.0,
            })
            .collect();
        let mut array = forge_sys::llama_token_data_array {
            data: candidates.as_mut_ptr(),
            size: candidates.len(),
            selected: -1,
            sorted: false,
        };
        // SAFETY: raw is a live chain; the array borrows
        // `candidates`, which outlives the call; native shrinks
        // `size` and sets `selected` in place. Both are validated
        // before any indexing (mirroring native's own assert as an
        // error instead of an abort).
        unsafe { forge_sys::llama_sampler_apply(self.raw, &mut array) };
        if array.selected < 0 || array.selected as usize >= candidates.len() {
            return Err(Error::sample(
                "chain selected nothing (empty chain or no greedy/dist selector?)",
            ));
        }
        // `selected` is a validated in-range index; its `id` came
        // from the same range, so the narrowing cannot fail — it is
        // still total rather than truncating.
        let id = candidates[array.selected as usize].id;
        let token = TokenId::try_from(id).map_err(|_| Error::sample(format!("invalid id {id}")))?;
        // SAFETY: raw is a live chain; `id` fits `llama_token`.
        unsafe { forge_sys::llama_sampler_accept(self.raw, id) };
        Ok(token)
    }
}

impl Default for SamplerChain {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SamplerChain {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SamplerChain")
            .field("len", &self.len())
            .field("seed", &self.seed())
            .finish_non_exhaustive()
    }
}

/// Map an optional seed to the native value (`None` = random).
fn seed_or_random(seed: Option<u32>) -> u32 {
    seed.unwrap_or(forge_sys::LLAMA_DEFAULT_SEED)
}

/// Validate a top-k parameter, returning the native value.
fn validate_top_k(k: u32) -> Result<c_int> {
    if k == 0 {
        return Err(Error::sample("top_k must be at least 1"));
    }
    c_int::try_from(k).map_err(|_| Error::sample(format!("top_k {k} too large")))
}

/// Validate a nucleus/min-p mass in `(0, 1]`.
fn validate_prob(p: f32, name: &str) -> Result<f32> {
    if !p.is_finite() {
        return Err(Error::sample(format!("{name} must be finite, got {p}")));
    }
    if p <= 0.0 || p > 1.0 {
        return Err(Error::sample(format!("{name} must be in (0, 1], got {p}")));
    }
    Ok(p)
}

/// Validate a temperature divider (finite, non-negative; `0.0` —
/// and `-0.0`, which compares equal — is greedy mode).
fn validate_temp(temp: f32) -> Result<f32> {
    if !temp.is_finite() {
        return Err(Error::sample(format!(
            "temperature must be finite, got {temp}"
        )));
    }
    if temp < 0.0 {
        return Err(Error::sample(format!(
            "temperature must be non-negative, got {temp}"
        )));
    }
    Ok(temp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with(
        temperature: f32,
        top_k: Option<u32>,
        top_p: Option<f32>,
        min_p: Option<f32>,
    ) -> SampleConfig {
        SampleConfig {
            temperature,
            top_k,
            top_p,
            min_p,
            seed: None,
        }
    }

    #[test]
    fn config_default_is_greedy() {
        assert_eq!(
            SampleConfig::default(),
            SampleConfig {
                temperature: 0.0,
                top_k: None,
                top_p: None,
                min_p: None,
                seed: None,
            }
        );
    }

    #[test]
    fn seed_none_maps_to_random_sentinel() {
        assert_eq!(seed_or_random(None), forge_sys::LLAMA_DEFAULT_SEED);
        assert_eq!(seed_or_random(Some(42)), 42);
        assert_eq!(seed_or_random(Some(u32::MAX)), u32::MAX);
    }

    #[test]
    fn temperature_validation() {
        for temp in [0.0, -0.0, 0.5, 1.0, 2.0] {
            assert!(validate_temp(temp).is_ok(), "temp {temp}");
        }
        for temp in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, -0.5] {
            let error = validate_temp(temp).expect_err("must fail");
            assert!(error.to_string().starts_with("sample error"), "{error}");
        }
    }

    #[test]
    fn top_k_validation() {
        assert_eq!(validate_top_k(1), Ok(1));
        assert_eq!(validate_top_k(40), Ok(40));
        assert_eq!(
            validate_top_k(u32::try_from(c_int::MAX).unwrap()),
            Ok(c_int::MAX)
        );
        validate_top_k(0).expect_err("k=0 must fail");
        validate_top_k(u32::MAX).expect_err("k > i32::MAX must fail");
    }

    #[test]
    fn probability_validation() {
        for name in ["top_p", "min_p"] {
            for p in [f32::MIN_POSITIVE, 0.5, 1.0] {
                assert!(validate_prob(p, name).is_ok(), "{name} {p}");
            }
            for p in [0.0, -0.5, 1.5, f32::NAN, f32::INFINITY] {
                let error = validate_prob(p, name).expect_err("must fail");
                assert!(error.to_string().starts_with("sample error"), "{error}");
                assert!(error.to_string().contains(name), "{error}");
            }
        }
    }

    #[test]
    fn config_validate_covers_every_field() {
        config_with(0.8, Some(40), Some(0.9), Some(0.1))
            .validate()
            .expect("valid");
        config_with(f32::NAN, None, None, None)
            .validate()
            .expect_err("bad temp");
        config_with(0.8, Some(0), None, None)
            .validate()
            .expect_err("bad top_k");
        config_with(0.8, None, Some(0.0), None)
            .validate()
            .expect_err("bad top_p");
        config_with(0.8, None, None, Some(2.0))
            .validate()
            .expect_err("bad min_p");
    }
}
