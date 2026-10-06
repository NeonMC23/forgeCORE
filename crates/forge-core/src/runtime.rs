//! Eager tensor ops over [`Backend`](crate::backend::Backend) execution.
//!
//! Every op builds a tiny ggml graph, computes it immediately on the
//! tensors' backend, and returns the fresh result [`Tensor`]. Inputs
//! must live on the same backend; there is no silent fallback.
//!
//! The dtype/shape/stride preconditions below mirror the native
//! asserts and dispatch tables exactly (see the Phase-6 report for
//! the per-op audit): each rule names the abort, overrun, or silent
//! misread it prevents.

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{new_exec_ctx, Tensor};
use std::ptr;
use std::rc::Rc;

/// RoPE rotation ordering. Only the two modes `ggml_rope_ext`
/// supports are exposed: the header directs MROPE/VISION callers to
/// `ggml_rope_multi` (explicit sections), which P6 does not bind —
/// `rope_ext` zeroes the sections, and the compute kernel aborts
/// (`GGML_ASSERT(sections[0] > 0 || ...)`) whenever the MROPE bit is
/// set, so the missing modes are unreachable by construction rather
/// than by a runtime check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeMode {
    /// Standard interleaved RoPE (`GGML_ROPE_TYPE_NORMAL` = 0).
    Normal,
    /// NeoX-style half rotation (`GGML_ROPE_TYPE_NEOX` = 2).
    NeoX,
}

impl RopeMode {
    fn ggml_mode(self) -> std::os::raw::c_int {
        match self {
            RopeMode::Normal => forge_sys::rope_type::NORM,
            RopeMode::NeoX => forge_sys::rope_type::NEOX,
        }
    }
}

/// RoPE parameters (mirrors `ggml_rope_ext`, minus the mode-sectored
/// variants). `c` (frequency factors) is not bound in P6 — pass
/// `None` semantics by omitting it: the binding always passes NULL.
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    /// Rotary dimensions (even, `2 <= n_dims <= a.ne0`).
    pub n_dims: usize,
    /// Rotation ordering.
    pub mode: RopeMode,
    /// Original context length for YaRN scaling (>= 1).
    pub n_ctx_orig: usize,
    /// Rotary base frequency (typically 10000.0).
    pub freq_base: f32,
    /// Frequency scale multiplier.
    pub freq_scale: f32,
    /// YaRN extrapolation factor.
    pub ext_factor: f32,
    /// YaRN attention factor.
    pub attn_factor: f32,
    /// YaRN fast beta.
    pub beta_fast: f32,
    /// YaRN slow beta.
    pub beta_slow: f32,
}

impl Default for RopeParams {
    /// Plain RoPE over the full head dim: NeoX ordering,
    /// `freq_base = 10000`, unit scales, zero YaRN corrections.
    /// `n_dims`/`n_ctx_orig` still need setting per call.
    fn default() -> Self {
        Self {
            n_dims: 0,
            mode: RopeMode::NeoX,
            n_ctx_orig: 1,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// Same-backend check via allocation-owner identity.
fn check_same_backend(inputs: &[&Tensor], op: &str) -> Result<()> {
    let first = inputs[0].backend_inner();
    for other in &inputs[1..] {
        if !Rc::ptr_eq(first, other.backend_inner()) {
            return Err(Error::backend(format!(
                "{op} needs tensors on one backend"
            )));
        }
    }
    Ok(())
}

/// Binary elementwise precondition: one backend, F32 pair, identical
/// shapes, `a` contiguous. `b` may be strided (the CPU kernels read
/// it through strides); `a` must be contiguous because the row loop
/// advances it by packed rows (`ggml_are_same_shape` + the kernel's
/// `nb00 == sizeof(float)` requirement).
fn check_pair(a: &Tensor, b: &Tensor, op: &str) -> Result<()> {
    check_same_backend(&[a, b], op)?;
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(Error::unsupported(format!(
            "{op} needs F32 inputs, got {} and {}",
            a.dtype().name(),
            b.dtype().name()
        )));
    }
    if a.shape() != b.shape() {
        return Err(Error::invalid(format!(
            "{op} needs identical shapes, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    if !a.is_contiguous() {
        return Err(Error::invalid(format!(
            "{op} needs a contiguous first input (use `cont` first)"
        )));
    }
    Ok(())
}

/// Unary precondition: F32 + contiguous.
fn check_unary(a: &Tensor, op: &str) -> Result<()> {
    if a.dtype() != DType::F32 {
        return Err(Error::unsupported(format!(
            "{op} needs an F32 input, got {}",
            a.dtype().name()
        )));
    }
    if !a.is_contiguous() {
        return Err(Error::invalid(format!(
            "{op} needs a contiguous input (use `cont` first)"
        )));
    }
    Ok(())
}

/// Norm precondition: unary + non-negative epsilon (`eps < 0` and NaN
/// trip the native `eps >= 0` assert; +inf is arithmetic-sound).
fn check_norm(a: &Tensor, eps: f32, op: &str) -> Result<()> {
    check_unary(a, op)?;
    if !(eps >= 0.0) {
        return Err(Error::invalid(format!("{op} needs eps >= 0, got {eps}")));
    }
    Ok(())
}

/// Run one scratch graph (op node already built in `ctx`) on
/// `backend`, then wrap the result. `inputs` feeds the graph-size
/// bound. NULL op nodes (context OOM) unwind the context.
pub(crate) fn finish(
    op: &str,
    backend: &Backend,
    ctx: *mut forge_sys::ggml_context,
    graph_cap: usize,
    raw: *mut forge_sys::ggml_tensor,
    dtype: DType,
    shape: Vec<usize>,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    if raw.is_null() {
        // SAFETY: ctx is live and uniquely owned here.
        unsafe { forge_sys::ggml_free(ctx) };
        return Err(Error::backend(format!("{op}: ggml node creation failed")));
    }
    // SAFETY: backend/ctx are live; the graph helpers only assert
    // liveness; compute status is mapped below.
    unsafe {
        let graph = forge_sys::ggml_new_graph_custom(ctx, graph_cap, false);
        if graph.is_null() {
            forge_sys::ggml_free(ctx);
            return Err(Error::backend(format!("{op}: ggml graph creation failed")));
        }
        forge_sys::ggml_build_forward_expand(graph, raw);
        let buffer = forge_sys::ggml_backend_alloc_ctx_tensors(ctx, backend.raw());
        if buffer.is_null() {
            forge_sys::ggml_free(ctx);
            return Err(Error::backend(format!("{op}: backend buffer allocation failed")));
        }
        let status = forge_sys::ggml_backend_graph_compute(backend.raw(), graph);
        if status != forge_sys::status::SUCCESS {
            forge_sys::ggml_backend_buffer_free(buffer);
            forge_sys::ggml_free(ctx);
            return Err(Error::backend(format!(
                "{op} failed: {}",
                Backend::status_message(status)
            )));
        }
        let graph_size = Tensor::child_graph_size(inputs)?;
        Ok(Tensor::wrap_computed(
            backend, ctx, raw, buffer, dtype, shape, graph_size,
        ))
    }
}

/// Elementwise `a + b`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_pair(a, b, "add")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    // SAFETY: ctx/inputs live; preconditions mirror the asserts.
    unsafe {
        let raw = forge_sys::ggml_add(ctx, a.raw(), b.raw());
        finish("add", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a, b])
    }
}

/// Elementwise `a - b`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_pair(a, b, "sub")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_sub(ctx, a.raw(), b.raw());
        finish("sub", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a, b])
    }
}

/// Elementwise `a * b`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_pair(a, b, "mul")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_mul(ctx, a.raw(), b.raw());
        finish("mul", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a, b])
    }
}

/// Elementwise `a / b`.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_pair(a, b, "div")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_div(ctx, a.raw(), b.raw());
        finish("div", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a, b])
    }
}

/// Matrix product over the first two dims with row-broadcast over the
/// rest: `a [K, M, ...] * b [K, N, ...] -> [M, N, ...]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_same_backend(&[a, b], "matmul")?;
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(Error::unsupported(format!(
            "matmul needs F32 inputs, got {} and {}",
            a.dtype().name(),
            b.dtype().name()
        )));
    }
    let ashape = a.shape();
    let bshape = b.shape();
    if ashape.len() != 2 || bshape.len() != 2 {
        return Err(Error::invalid(format!(
            "matmul needs 2-D inputs, got rank {} and {}",
            ashape.len(),
            bshape.len()
        )));
    }
    if ashape[0] != bshape[0] {
        return Err(Error::invalid(format!(
            "matmul inner mismatch: {} vs {}",
            ashape[0], bshape[0]
        )));
    }
    if !a.is_contiguous() {
        // The `mul_mat` kernel indexes A by packed rows
        // (`nb00 == sizeof(float)` reads); a strided A silently
        // misreads. (P5 needed no such check: its tensors were
        // always contiguous.)
        return Err(Error::invalid(
            "matmul needs a contiguous first input (use `cont` first)",
        ));
    }
    if Tensor::is_transposed_native(a.raw()) {
        // `is_transposed(a)` takes the kernel's vector/dot path with
        // B-row assumptions the strided layout does not meet; refuse
        // instead of silently computing the wrong product.
        return Err(Error::invalid(
            "matmul refuses a transposed first input (use `cont` first)",
        ));
    }

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    let out_shape = vec![ashape[1], bshape[1]];
    unsafe {
        let raw = forge_sys::ggml_mul_mat(ctx, a.raw(), b.raw());
       
        finish("matmul", &backend, ctx, graph_cap, raw, DType::F32, out_shape, &[a, b])
    }
}

/// SiLU activation, `x * sigmoid(x)`.
pub fn silu(a: &Tensor) -> Result<Tensor> {
    check_unary(a, "silu")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(1)?;
    unsafe {
        let raw = forge_sys::ggml_silu(ctx, a.raw());
        finish("silu", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}

/// Elementwise square.
pub fn sqr(a: &Tensor) -> Result<Tensor> {
    check_unary(a, "sqr")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(1)?;
    unsafe {
        let raw = forge_sys::ggml_sqr(ctx, a.raw());
        finish("sqr", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}

/// Elementwise square root.
pub fn sqrt(a: &Tensor) -> Result<Tensor> {
    check_unary(a, "sqrt")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(1)?;
    unsafe {
        let raw = forge_sys::ggml_sqrt(ctx, a.raw());
        finish("sqrt", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}

/// Elementwise `a * s`.
pub fn scale(a: &Tensor, s: f32) -> Result<Tensor> {
    check_unary(a, "scale")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(1)?;
    unsafe {
        let raw = forge_sys::ggml_scale(ctx, a.raw(), s);
        finish("scale", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}
/// RMS normalization over dim 0 (per column), `x / rms(x, eps)`.
pub fn rms_norm(a: &Tensor, eps: f32) -> Result<Tensor> {
    check_norm(a, eps, "rms_norm")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(1)?;
    unsafe {
        let raw = forge_sys::ggml_rms_norm(ctx, a.raw(), eps);
        finish("rms_norm", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}

/// Mean-centered normalization over dim 0 (per column).
pub fn norm(a: &Tensor, eps: f32) -> Result<Tensor> {
    check_norm(a, eps, "norm")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(1)?;
    unsafe {
        let raw = forge_sys::ggml_norm(ctx, a.raw(), eps);
        finish("norm", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}

/// Softmax over dim 0 (per column).
pub fn soft_max(a: &Tensor) -> Result<Tensor> {
    check_unary(a, "soft_max")?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(2)?;
    unsafe {
        let raw = forge_sys::ggml_soft_max(ctx, a.raw());
        finish("soft_max", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a])
    }
}

/// Masked softmax over dim 0: `softmax((a + slope * mask) * scale)`
/// with optional ALiBi `max_bias` (0 disables it).
///
/// `mask` must be F32 or F16 (the kernel reads F16 masks through a
/// dedicated path and any other dtype as F32 — silently wrong), must
/// be contiguous (rows are read packed), and must cover `a`:
/// `mask.ne0 >= a.ne0` and `mask.ne1 >= a.ne1` (the row loop indexes
/// the mask directly in dim 1 and reads full rows in dim 0 — short
/// masks over-read with no assert). Higher mask dims broadcast by
/// modulo. `scale`/`max_bias` are pure arithmetic (any `f32`).
pub fn soft_max_ext(
    a: &Tensor,
    mask: &Tensor,
    scale: f32,
    max_bias: f32,
) -> Result<Tensor> {
    check_same_backend(&[a, mask], "soft_max_ext")?;
    check_unary(a, "soft_max_ext")?;
    if mask.dtype() != DType::F32 && mask.dtype() != DType::F16 {
        return Err(Error::unsupported(format!(
            "soft_max_ext needs an F32 or F16 mask, got {}",
            mask.dtype().name()
        )));
    }
    if !mask.is_contiguous() {
        return Err(Error::invalid(
            "soft_max_ext needs a contiguous mask (use `cont` first)",
        ));
    }
    let extent = |shape: &[usize], dim: usize| shape.get(dim).copied().unwrap_or(1);
    let (ane0, ane1, ane2, ane3) = (
        extent(a.shape(), 0),
        extent(a.shape(), 1),
        extent(a.shape(), 2),
        extent(a.shape(), 3),
    );
    let (mne0, mne1, mne2, mne3) = (
        extent(mask.shape(), 0),
        extent(mask.shape(), 1),
        extent(mask.shape(), 2),
        extent(mask.shape(), 3),
    );
    if mne0 != ane0 || mne1 < ane1 || ane2 % mne2 != 0 || ane3 % mne3 != 0 {
        return Err(Error::invalid(format!(
            "soft_max_ext mask [{mne0}, {mne1}, {mne2}, {mne3}] does not fit input [{ane0}, {ane1}, {ane2}, {ane3}] (need equal ne0, covering ne1, dividing ne2/ne3)"
        )));
    }

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_soft_max_ext(ctx, a.raw(), mask.raw(), scale, max_bias);
        finish("soft_max_ext", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a, mask])
    }
}
/// Rotary position embeddings over dim 0 (`ggml_rope_ext`).
///
/// `a` is `[ne0, n_heads, n_tokens]` (rank 3, F32, contiguous,
/// even `ne0`); `positions` is a contiguous I32 vector of exactly
/// `n_tokens` ids. `params` carries `n_dims` (even,
/// `2 <= n_dims <= ne0`), the mode, and the frequency/YaRN scalars
/// (pure arithmetic — any `f32` is sound).
///
/// The even-`ne0` rule is a hardening over native: the remainder
/// loop copies trailing channels in pairs, so an odd `ne0` reads and
/// writes one element past each row (verified in source; see the
/// Phase-6 report). The frequency-factors input `c` is always NULL
/// in P6.
pub fn rope(
    a: &Tensor,
    positions: &Tensor,
    params: &RopeParams,
) -> Result<Tensor> {
    check_same_backend(&[a, positions], "rope")?;
    check_unary(a, "rope")?;
    if a.shape().len() != 3 {
        return Err(Error::invalid(format!(
            "rope needs a rank-3 input [ne0, heads, tokens], got {:?}",
            a.shape()
        )));
    }
    let ne0 = a.shape()[0];
    if ne0 % 2 != 0 {
        return Err(Error::invalid(format!(
            "rope needs an even dim 0 (odd widths overrun the row), got {ne0}"
        )));
    }
    if positions.dtype() != DType::I32 {
        return Err(Error::unsupported(format!(
            "rope needs I32 positions, got {}",
            positions.dtype().name()
        )));
    }
    if positions.shape().len() != 1 {
        return Err(Error::invalid(format!(
            "rope needs a position vector, got shape {:?}",
            positions.shape()
        )));
    }
    if !positions.is_contiguous() {
        return Err(Error::invalid(
            "rope needs contiguous positions (use `cont` first)",
        ));
    }
    let n_tokens = a.shape()[2];
    if positions.shape()[0] != n_tokens {
        return Err(Error::invalid(format!(
            "rope needs {n_tokens} positions (a.ne2), got {}",
            positions.shape()[0]
        )));
    }
    if params.n_dims < 2 || params.n_dims % 2 != 0 || params.n_dims > ne0 {
        return Err(Error::invalid(format!(
            "rope needs an even n_dims in 2..={ne0}, got {}",
            params.n_dims
        )));
    }
    if params.n_ctx_orig < 1 {
        return Err(Error::invalid(format!(
            "rope needs n_ctx_orig >= 1, got {}",
            params.n_ctx_orig
        )));
    }
    let n_dims = std::os::raw::c_int::try_from(params.n_dims)
        .map_err(|_| Error::invalid("rope n_dims exceeds i32 range"))?;
    let n_ctx_orig = std::os::raw::c_int::try_from(params.n_ctx_orig)
        .map_err(|_| Error::invalid("rope n_ctx_orig exceeds i32 range"))?;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_rope_ext(
            ctx,
            a.raw(),
            positions.raw(),
            ptr::null_mut(),
            n_dims,
            params.mode.ggml_mode(),
            n_ctx_orig,
            params.freq_base,
            params.freq_scale,
            params.ext_factor,
            params.attn_factor,
            params.beta_fast,
            params.beta_slow,
        );
        finish("rope", &backend, ctx, graph_cap, raw, DType::F32, a.shape().to_vec(), &[a, positions])
    }
}

/// Whether the `get_rows` kernel implements `dtype` tables
/// (everything except the aborting Q8_K/I8/I16/F64; shared by
/// [`get_rows`] and op probes).
pub(crate) fn get_rows_table_supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::F16
            | DType::BF16
            | DType::I32
            | DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Q8_0
            | DType::Q8_1
            | DType::Q2_K
            | DType::Q3_K
            | DType::Q4_K
            | DType::Q5_K
            | DType::Q6_K
    )
}

/// Gather rows: `table [d, nrows, ...]` × I32 `indices [k, ...]` →
/// `[d, k, ...]` (same dtype for I32 tables, else F32).
///
/// The table must be contiguous and its dtype must be one the kernel
/// implements (F32/F16/BF16/I32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K —
/// Q8_K/I8/I16/F64 abort). Indices must be contiguous I32 with
/// `indices.ne1 == table.ne2`, `indices.ne2 == table.ne3`,
/// `indices.ne3 == 1` (the constructor asserts), and every index is
/// range-checked against the row count (out-of-range indices
/// over-read with no assert).
pub fn get_rows(table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    check_same_backend(&[table, indices], "get_rows")?;
    if !get_rows_table_supported(table.dtype()) {
        return Err(Error::unsupported(format!(
            "get_rows does not support {} tables",
            table.dtype().name()
        )));
    }
    if !table.is_contiguous() {
        return Err(Error::invalid(
            "get_rows needs a contiguous table (use `cont` first)",
        ));
    }
    if indices.dtype() != DType::I32 {
        return Err(Error::unsupported(format!(
            "get_rows needs I32 indices, got {}",
            indices.dtype().name()
        )));
    }
    if !indices.is_contiguous() {
        return Err(Error::invalid(
            "get_rows needs contiguous indices (use `cont` first)",
        ));
    }
    let extent = |shape: &[usize], dim: usize| shape.get(dim).copied().unwrap_or(1);
    if extent(indices.shape(), 1) != extent(table.shape(), 2)
        || extent(indices.shape(), 2) != extent(table.shape(), 3)
        || extent(indices.shape(), 3) != 1
    {
        return Err(Error::invalid(format!(
            "get_rows needs indices [k, t2, t3] with t2 == table.ne2 ({}) and t3 == table.ne3 ({}), got {:?}",
            extent(table.shape(), 2),
            extent(table.shape(), 3),
            indices.shape()
        )));
    }
    let nrows = extent(table.shape(), 1);
    for (pos, &row) in indices.to_vec_i32()?.iter().enumerate() {
        if row < 0 || row as usize >= nrows {
            return Err(Error::invalid(format!(
                "get_rows index {row} at position {pos} is outside [0, {nrows})"
            )));
        }
    }
    let out_dtype = if table.dtype() == DType::I32 { DType::I32 } else { DType::F32 };
    let out_shape = vec![
        extent(table.shape(), 0),
        extent(indices.shape(), 0),
        extent(indices.shape(), 1),
        extent(indices.shape(), 2),
    ];

    let backend = Backend::from_inner(Rc::clone(table.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_get_rows(ctx, table.raw(), indices.raw());
        finish("get_rows", &backend, ctx, graph_cap, raw, out_dtype, out_shape, &[table, indices])
    }
}

/// Concatenate `a` and `b` along `dim` (shapes must match on every
/// other axis; quantized inputs must both be contiguous — the
/// strided path asserts packed rows per block size).
pub fn concat(a: &Tensor, b: &Tensor, dim: usize) -> Result<Tensor> {
    check_same_backend(&[a, b], "concat")?;
    if dim >= crate::tensor::MAX_DIMS {
        return Err(Error::invalid(format!(
            "concat dim must be < {}, got {dim}",
            crate::tensor::MAX_DIMS
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(Error::invalid(format!(
            "concat needs matching dtypes, got {} and {}",
            a.dtype().name(),
            b.dtype().name()
        )));
    }
    let extent = |shape: &[usize], d: usize| shape.get(d).copied().unwrap_or(1);
    for d in 0..crate::tensor::MAX_DIMS {
        if d != dim && extent(a.shape(), d) != extent(b.shape(), d) {
            return Err(Error::invalid(format!(
                "concat needs matching extents off-axis (axis {d}: {} vs {})",
                extent(a.shape(), d),
                extent(b.shape(), d)
            )));
        }
    }
    if a.dtype().is_quantized() && (!a.is_contiguous() || !b.is_contiguous()) {
        return Err(Error::invalid(format!(
            "concat needs contiguous {} inputs (use `cont` first)",
            a.dtype().name()
        )));
    }
    let joined = extent(a.shape(), dim)
        .checked_add(extent(b.shape(), dim))
        .ok_or_else(|| Error::invalid("concat extent overflow"))?;
    if i64::try_from(joined).is_err() {
        // The native constructor adds int64 extents; reject the sum
        // before it can overflow there.
        return Err(Error::invalid("concat extent exceeds i64 range"));
    }
    let mut out_shape = vec![
        extent(a.shape(), 0),
        extent(a.shape(), 1),
        extent(a.shape(), 2),
        extent(a.shape(), 3),
    ];
    out_shape[dim] = joined;

    let backend = Backend::from_inner(Rc::clone(a.backend_inner()));
    let (ctx, graph_cap) = new_exec_ctx(3)?;
    unsafe {
        let raw = forge_sys::ggml_concat(
            ctx,
            a.raw(),
            b.raw(),
            dim as std::os::raw::c_int,
        );
        finish("concat", &backend, ctx, graph_cap, raw, a.dtype(), out_shape, &[a, b])
    }
}
