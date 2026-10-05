//! Context-owned memory (KV cache) sequence operations.
//!
//! [`Memory`] is a borrow of the native memory object owned by a
//! [`Context`]: it performs sequence surgery
//! (`clear`, `remove_range`, `copy_seq`, `keep_seq`, position shifts)
//! and reports per-sequence position bounds, but owns nothing and frees
//! nothing. Obtain it via
//! [`Context::memory`](crate::context::Context::memory), which returns
//! [`None`] for model architectures that have no memory object at all
//! (BERT-family encoders — verified in the native memory factory).
//!
//! Sequence-id bounds differ per operation because the native memory
//! implementations disagree (verified against every implementation at
//! the pin — plain, recurrent, hybrid, iswa, msa, dsa, dsa-iswa, and
//! dsv4 caches):
//!
//! * The *general* bound is `n_seq_max` on split caches and
//!   `llama_max_parallel_sequences` (256) on unified caches. It
//!   applies to [`Memory::remove_range`], [`Memory::copy_seq`],
//!   [`Memory::shift_positions`], [`Memory::scale_positions`],
//!   [`Memory::pos_min`]/[`Memory::pos_max`], and sequence-state
//!   export. Out-of-range ids abort upstream (verified SIGABRT), so
//!   they are refused here with [`Error::invalid`].
//! * [`Memory::keep_seq`] and sequence-state restore additionally
//!   require `seq < n_seq_max` even on unified caches: the DSV4 cache
//!   asserts that tighter bound (and the recurrent cache does once
//!   rollback snapshots are enabled). ForgeCore cannot detect which
//!   implementation backs a context — no such getter exists at the
//!   pin — so the tighter bound applies uniformly.
//!
//! One residual cannot be validated away: on DeepSeek-V4-architecture
//! models with unified KV, `remove_range`/`copy_seq` with
//! `seq >= n_seq_max` abort inside the compressed-state helpers. The
//! general bound stays 256 there because every other implementation
//! handles those ids safely and refusing them would strand
//! legitimately decoded sequences; RAMforge's slot discipline (`seq <
//! n_seq_max`) never reaches the residual. See the P5 report.
//!
//! Position shifts ([`Memory::shift_positions`]) are refused on
//! multi-position (MROPE/IMROPE) models — upstream aborts there — and
//! guarded against `i32` overflow of the observed position range; a
//! sequence that reports empty is a no-op without calling native
//! code. [`Memory::scale_positions`] requires a divisor `>= 1`
//! (division by zero was verified SIGFPE). Ranges are `[start, end)`
//! with `end == None` meaning infinity; inverted ranges are refused
//! rather than silently treated as empty.
//!
//! Like all ForgeCore handles, `Memory` is `!Send + !Sync` and exposes
//! no raw pointers.

use crate::batch::SeqId;
use crate::context::Context;
use crate::error::{Error, Result};
use std::marker::PhantomData;
use std::os::raw::c_int;

/// Largest accepted absolute position shift.
///
/// Native positions are `i32`; shifts are additionally guarded against
/// the observed per-sequence range (see [`Memory::shift_positions`).
/// A billion positions exceeds any realistic context by orders of
/// magnitude, so this backstop only bites on caller bugs.
const MAX_SHIFT: i32 = (1 << 30) - 1;

/// Borrowed handle to a context's native memory object.
///
/// Created by [`Context::memory`](crate::context::Context::memory);
/// carries the validation facts (`n_seq_max`, unified mode and bound,
/// shift support) cached at context creation so every operation can be
/// checked before reaching native code.
pub struct Memory<'a> {
    raw: forge_sys::llama_memory_t,
    n_seq_max: u32,
    unified_limit: u32,
    kv_unified: bool,
    shift_allowed: bool,
    // Exclusive borrow of the owning context: the native object stays
    // alive exactly while this handle does, and no decode or state op
    // can interleave with a sequence op.
    _borrow: PhantomData<&'a mut Context>,
}

impl<'a> Memory<'a> {
    pub(crate) fn borrow(
        raw: forge_sys::llama_memory_t,
        n_seq_max: u32,
        unified_limit: u32,
        kv_unified: bool,
        shift_allowed: bool,
    ) -> Self {
        Self {
            raw,
            n_seq_max,
            unified_limit,
            kv_unified,
            shift_allowed,
            _borrow: PhantomData,
        }
    }

    /// Exclusive upper bound for sequence ids on the general
    /// operations (`remove_range`, `copy_seq`, shifts, position
    /// queries, sequence-state export): `n_seq_max` on split caches,
    /// the native parallel-sequence limit on unified caches.
    /// [`Memory::keep_seq`] and sequence-state restore accept only
    /// `n_seq_max` (see the module docs).
    pub fn seq_limit(&self) -> u32 {
        if self.kv_unified {
            self.unified_limit
        } else {
            self.n_seq_max
        }
    }

    /// Validate a caller sequence id against `bound`, converting to
    /// the native width. Every native implementation asserts (aborts)
    /// on out-of-range ids for at least some operation, so the check
    /// lives here, not upstream.
    fn check_seq(&self, seq: SeqId, bound: u32, op: &str) -> Result<c_int> {
        if seq >= bound {
            return Err(Error::invalid(format!(
                "memory {op}: seq id {seq} out of range (limit {bound})"
            )));
        }
        c_int::try_from(seq)
            .map_err(|_| Error::invalid(format!("memory {op}: seq id {seq} too large")))
    }

    /// Validate a `[start, end)` range and convert to native widths
    /// (`end == None` becomes -1, upstream infinity). Inverted ranges
    /// are refused: upstream would silently no-op, which masks caller
    /// bugs.
    fn check_range(start: u32, end: Option<u32>, op: &str) -> Result<(c_int, c_int)> {
        if let Some(end) = end {
            if start > end {
                return Err(Error::invalid(format!(
                    "memory {op}: inverted range [{start}, {end})"
                )));
            }
        }
        let p0 = c_int::try_from(start)
            .map_err(|_| Error::invalid(format!("memory {op}: start {start} too large")))?;
        let p1 = match end {
            None => -1,
            Some(end) => c_int::try_from(end)
                .map_err(|_| Error::invalid(format!("memory {op}: end {end} too large")))?,
        };
        Ok((p0, p1))
    }

    /// Clear all sequences. `data == true` additionally zeroes the
    /// data buffers (slower; leaves no stale rows behind), matching
    /// the native flag exactly.
    pub fn clear(&mut self, data: bool) {
        // SAFETY: raw is a live memory object borrowed from a live
        // context; the native call null-checks, holds no preconditions
        // callers can violate, and cannot fail.
        unsafe { forge_sys::llama_memory_clear(self.raw, data) };
    }

    /// Remove positions `[start, end)` of `seq` (`end == None` removes
    /// to infinity). Removing a whole sequence (`start == 0`, `end ==
    /// None`) always succeeds; partial removal can be refused by the
    /// native implementation (recurrent models cannot erase the tail
    /// of their state; DSV4 supports only full-tail removal), which
    /// surfaces as [`Error::memory`].
    pub fn remove_range(&mut self, seq: SeqId, start: u32, end: Option<u32>) -> Result<()> {
        let id = self.check_seq(seq, self.seq_limit(), "remove_range")?;
        let (p0, p1) = Self::check_range(start, end, "remove_range")?;
        // SAFETY: raw is live; seq/range pre-validated (the native
        // range assert cannot fire); `false` (native refusal) is
        // mapped, never ignored.
        let ok = unsafe { forge_sys::llama_memory_seq_rm(self.raw, id, p0, p1) };
        if ok {
            Ok(())
        } else {
            Err(Error::memory(format!(
                "remove_range [{start}, {}) of seq {seq} refused by native memory",
                end.map_or("inf".to_string(), |e| e.to_string()),
            )))
        }
    }

    /// Copy the whole of sequence `src` onto `dst`, replacing `dst`'s
    /// previous contents. Partial copies are not exposed: the DSV4
    /// cache aborts on them and split caches abort on partial
    /// cross-stream copies (both verified), so only full copies have
    /// a uniform contract. Copying a sequence onto itself is a no-op.
    pub fn copy_seq(&mut self, src: SeqId, dst: SeqId) -> Result<()> {
        let limit = self.seq_limit();
        let s = self.check_seq(src, limit, "copy_seq")?;
        let d = self.check_seq(dst, limit, "copy_seq")?;
        if src == dst {
            return Ok(());
        }
        // SAFETY: raw is live; both ids pre-validated; (0, -1) is the
        // full range on every implementation (DSV4's full-copy-only
        // assert and the cross-stream full-buffer assert both hold).
        unsafe { forge_sys::llama_memory_seq_cp(self.raw, s, d, 0, -1) };
        Ok(())
    }

    /// Remove every sequence except `seq`. Requires `seq < n_seq_max`
    /// even on unified caches (see the module docs).
    pub fn keep_seq(&mut self, seq: SeqId) -> Result<()> {
        let id = self.check_seq(seq, self.n_seq_max, "keep_seq")?;
        // SAFETY: raw is live; the id satisfies the tightest bound
        // any implementation asserts.
        unsafe { forge_sys::llama_memory_seq_keep(self.raw, id) };
        Ok(())
    }

    /// Add `delta` to positions `[start, end)` of `seq` (`end == None`
    /// shifts to infinity). Cells shifted below zero are freed by
    /// KV-style caches (native behavior); recurrent caches keep the
    /// shifted tail position.
    ///
    /// Refusals (all [`Error::invalid`], none reaching native code):
    /// out-of-range `seq`, inverted/unrepresentable range, shifts on
    /// multi-position (MROPE/IMROPE) models (upstream aborts), shifts
    /// larger than ±(`2^30 - 1`), shifts that would overflow `i32` for
    /// the observed position range, and shifts on an inconsistently
    /// reported range. A sequence that reports empty is a no-op.
    pub fn shift_positions(
        &mut self,
        seq: SeqId,
        start: u32,
        end: Option<u32>,
        delta: i32,
    ) -> Result<()> {
        let id = self.check_seq(seq, self.seq_limit(), "shift_positions")?;
        let (p0, p1) = Self::check_range(start, end, "shift_positions")?;
        if delta == 0 {
            return Ok(());
        }
        if !(-MAX_SHIFT..=MAX_SHIFT).contains(&delta) {
            return Err(Error::invalid(format!(
                "memory shift_positions: delta {delta} exceeds ±{MAX_SHIFT}"
            )));
        }
        if !self.shift_allowed {
            return Err(Error::invalid(
                "memory shift_positions: position shifts are not supported \
                 on multi-position (MROPE/IMROPE) models",
            ));
        }
        let (lo, hi) = self.observed_range(id)?;
        if lo == -1 && hi == -1 {
            // Observed empty on every reporting cache: native would
            // touch no cells, so skip the call entirely. (SWA-family
            // caches can hold history evicted from the reporting
            // window; such unobservable cells intentionally do not
            // shift — see the P5 report.)
            return Ok(());
        }
        if lo > hi {
            // Attested on no implementation reachable through symmetric
            // ops; the range cannot be guarded, so refuse.
            return Err(Error::invalid(format!(
                "memory shift_positions: seq {seq} reports inconsistent \
                 positions [{lo}, {hi}]; refusing shift"
            )));
        }
        if lo.checked_add(delta).is_none() || hi.checked_add(delta).is_none() {
            return Err(Error::invalid(format!(
                "memory shift_positions: delta {delta} would overflow i32 \
                 positions [{lo}, {hi}] of seq {seq}"
            )));
        }
        // SAFETY: raw is live; seq/range pre-validated; the model is
        // single-position (native shift assert holds); delta is
        // range-checked against every observable cell, so the native
        // addition cannot overflow.
        unsafe { forge_sys::llama_memory_seq_add(self.raw, id, p0, p1, delta) };
        Ok(())
    }

    /// Integer-divide positions `[start, end)` of `seq` by `divisor`
    /// (`end == None` scales to infinity). `divisor == 1` is a no-op.
    /// The divisor must be `>= 1`: upstream divides natively, and
    /// division by zero was verified SIGFPE. Same model gate as
    /// [`Memory::shift_positions`].
    pub fn scale_positions(
        &mut self,
        seq: SeqId,
        start: u32,
        end: Option<u32>,
        divisor: u32,
    ) -> Result<()> {
        let id = self.check_seq(seq, self.seq_limit(), "scale_positions")?;
        let (p0, p1) = Self::check_range(start, end, "scale_positions")?;
        if divisor == 0 {
            return Err(Error::invalid(
                "memory scale_positions: divisor must be at least 1",
            ));
        }
        let d = c_int::try_from(divisor).map_err(|_| {
            Error::invalid(format!(
                "memory scale_positions: divisor {divisor} too large"
            ))
        })?;
        if divisor == 1 {
            return Ok(());
        }
        if !self.shift_allowed {
            return Err(Error::invalid(
                "memory scale_positions: position scaling is not supported \
                 on multi-position (MROPE/IMROPE) models",
            ));
        }
        // SAFETY: raw is live; seq/range pre-validated; the model is
        // single-position (native assert holds); d >= 1, which rules
        // out both division by zero and INT_MIN / -1, so the native
        // division is defined for every i32 position.
        unsafe { forge_sys::llama_memory_seq_div(self.raw, id, p0, p1, d) };
        Ok(())
    }

    /// Smallest position present for `seq`, or [`None`] when the
    /// sequence holds no cells. Passthrough of the native query,
    /// including per-implementation reporting quirks (DSV4 reports the
    /// current boundary for both bounds).
    pub fn pos_min(&self, seq: SeqId) -> Result<Option<u32>> {
        let id = self.check_seq(seq, self.seq_limit(), "pos_min")?;
        // SAFETY: raw is live; the id is pre-validated; the native
        // getter is const (pure read) and returns -1 or a valid
        // position.
        let raw = unsafe { forge_sys::llama_memory_seq_pos_min(self.raw, id) };
        Ok(u32::try_from(raw).ok())
    }

    /// Largest position present for `seq`, or [`None`] when the
    /// sequence holds no cells. See [`Memory::pos_min`].
    pub fn pos_max(&self, seq: SeqId) -> Result<Option<u32>> {
        let id = self.check_seq(seq, self.seq_limit(), "pos_max")?;
        // SAFETY: as in `pos_min`.
        let raw = unsafe { forge_sys::llama_memory_seq_pos_max(self.raw, id) };
        Ok(u32::try_from(raw).ok())
    }

    /// Whether the native memory supports K-shift updates (device-side
    /// position shifting during decode). Informational: `false` on
    /// Step-3.5 and multi-position models, `true` otherwise. This does
    /// not gate [`Memory::shift_positions`] (which applies eagerly);
    /// the shift gate is the rope family instead.
    pub fn can_shift(&self) -> bool {
        // SAFETY: raw is live; the native getter is const and infallible.
        unsafe { forge_sys::llama_memory_can_shift(self.raw) }
    }

    /// Raw observed `[min, max]` bounds for the shift guard. Queries
    /// both native getters on the already-validated native id.
    fn observed_range(&self, id: c_int) -> Result<(c_int, c_int)> {
        // SAFETY: raw is live; the id was validated by the caller; both
        // getters are const (pure reads).
        let lo = unsafe { forge_sys::llama_memory_seq_pos_min(self.raw, id) };
        let hi = unsafe { forge_sys::llama_memory_seq_pos_max(self.raw, id) };
        Ok((lo, hi))
    }
}

impl std::fmt::Debug for Memory<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("n_seq_max", &self.n_seq_max)
            .field("seq_limit", &self.seq_limit())
            .finish_non_exhaustive()
    }
}

/// Owned whole-context state snapshot.
///
/// The bytes are exactly what the native serializer emits: a
/// model-architecture tag plus the live KV cells with per-layer type
/// headers (verified by probe — no logits, no embeddings). Import with
/// [`Context::import_state`](crate::context::Context::import_state);
/// the bytes are only valid for the same model architecture and
/// compatible cache geometry, which native checks structurally (no
/// checksums: flips inside KV data rows are accepted, flips in
/// headers are rejected).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct State(Vec<u8>);

/// Owned single-sequence state snapshot.
///
/// Same representation as [`State`] but scoped to one sequence (plus a
/// magic header). Export with
/// [`Context::export_seq_state`](crate::context::Context::export_seq_state),
/// restore with
/// [`Context::import_seq_state`](crate::context::Context::import_seq_state).
/// The distinct type keeps whole and per-sequence snapshots from being
/// mixed up at import.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeqState(Vec<u8>);

/// Shared accessors for the owned state snapshots.
macro_rules! state_bytes {
    ($name:ident) => {
        impl $name {
            /// Wrap caller-owned bytes as a snapshot. No validation
            /// happens here: import validates structurally and rejects
            /// corrupt, truncated, or mismatched bytes.
            pub fn from_bytes(bytes: Vec<u8>) -> Self {
                Self(bytes)
            }

            /// Snapshot bytes for storage or transport.
            pub fn as_bytes(&self) -> &[u8] {
                &self.0
            }

            /// Snapshot length in bytes.
            pub fn len(&self) -> usize {
                self.0.len()
            }

            /// Whether the snapshot holds no bytes (never produced by
            /// export; always rejected at import).
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Unwrap back into the owned byte vector.
            pub fn into_bytes(self) -> Vec<u8> {
                self.0
            }

            pub(crate) fn from_exported(bytes: Vec<u8>) -> Self {
                Self(bytes)
            }
        }
    };
}

state_bytes!(State);
state_bytes!(SeqState);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_shift_backstop_is_near_half_i32() {
        assert_eq!(MAX_SHIFT, 1_073_741_823);
        assert_eq!(MAX_SHIFT, (1 << 30) - 1);
    }

    #[test]
    fn state_snapshot_accessors_roundtrip() {
        let state = State::from_bytes(vec![1, 2, 3]);
        assert_eq!(state.as_bytes(), &[1, 2, 3]);
        assert_eq!(state.len(), 3);
        assert!(!state.is_empty());
        assert_eq!(state.clone().into_bytes(), vec![1, 2, 3]);
        assert!(State::from_bytes(vec![]).is_empty());
        assert!(State::default().is_empty());

        let seq = SeqState::from_bytes(vec![9]);
        assert_eq!(seq.as_bytes(), &[9]);
        assert_eq!(seq.len(), 1);
        assert!(!seq.is_empty());
        assert_eq!(seq.into_bytes(), vec![9]);
    }
}
