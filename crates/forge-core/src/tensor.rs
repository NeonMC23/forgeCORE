//! Owned ggml tensor with its storage.
//!
//! [`Tensor`] owns everything ggml needs for one tensor: a metadata
//! context, the tensor itself, and a share of the backend buffer
//! holding its data. Shapes use ggml `ne` order (extent of dimension
//! 0 first, up to 4 dimensions), normalized by stripping trailing
//! extents of 1 (so `[m, n, 1]` reports as `[m, n]`, matching
//! `ggml_n_dims`); data layout is exactly what ggml uses, with no
//! hidden transposes.
//!
//! Zero-copy shape ops ([`Tensor::view_2d`], [`Tensor::reshape`],
//! [`Tensor::transpose`], [`Tensor::permute`]) share the parent's
//! buffer allocation through reference counting: dropping every
//! tensor sharing an allocation frees the buffer, and no view can
//! outlive its storage. Compute ops ([`add`](crate::add),
//! [`matmul`](crate::matmul), ...) execute eagerly and return fresh
//! computed tensors with their own allocations.

use crate::backend::{Backend, BackendInner};
use crate::dtype::DType;
use crate::error::{Error, Result};
use std::ffi::{c_void, CStr, CString};
use std::ptr;
use std::rc::Rc;

/// Maximum tensor rank (ggml supports up to 4 dimensions).
pub const MAX_DIMS: usize = 4;

/// Shared backend-buffer allocation behind one or more [`Tensor`]s.
///
/// Views, reshapes, transposes, and permutes alias their parent's
/// storage instead of copying it; the buffer is freed when the last
/// tensor holding a share is dropped. Contexts are shared through
/// [`History`] for the same reason: a result's `src` chain points
/// into its inputs' contexts, so contexts are only freed with the
/// last downstream tensor.
struct Allocation {
    buffer: *mut forge_sys::ggml_backend_buffer,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: buffer came from a successful allocation call, is
        // freed exactly once (Drop on the shared owner), and every
        // tensor pointing data into it is gone (each holds an Rc share).
        unsafe { forge_sys::ggml_backend_buffer_free(self.buffer) };
    }
}

/// Shared owner of one ggml metadata context (tensor structs live in
/// the context pool). Freed when the last share drops.
struct CtxAlloc {
    ctx: *mut forge_sys::ggml_context,
}

impl Drop for CtxAlloc {
    fn drop(&mut self) {
        // SAFETY: ctx came from a successful context call and is
        // freed exactly once (Drop on the shared owner); every tensor
        // struct carved from it is gone (each live downstream tensor
        // holds a share through its History).
        unsafe { forge_sys::ggml_free(self.ctx) };
    }
}

/// One node of a tensor's forward tape: this tensor's own storage
/// plus shares of every ancestor's (op inputs, view parents).
///
/// ggml graph expansion (`ggml_build_forward_expand`) walks the whole
/// ancestor DAG through raw `src`/`view_src` pointers, and graph
/// (re)compute reads ancestor buffers — so a result must keep its
/// inputs' contexts AND buffers alive even after the input `Tensor`s
/// drop. Without this, using an op result after its statement-scoped
/// inputs dropped is heap use-after-free in safe code (found by the
/// `cpu_exec_proof` attention test: `munmap_chunk(): invalid
/// pointer`). Retention is DAG-shared (one node per tensor, parents
/// linked, never flattened), so a chain of N tensors holds N nodes,
/// not N-squared. Dropping a result releases everything it alone
/// retained.
struct History {
    alloc: Rc<Allocation>,
    // Retention-only: never read except by Drop (which frees the
    // context and releases the ancestors). The shares exist so
    // downstream ggml nodes keep pointing at live storage.
    #[allow(dead_code)]
    ctx: Rc<CtxAlloc>,
    #[allow(dead_code)]
    parents: Vec<Rc<History>>,
}

/// Allocate a metadata-only ggml context sized for `n_tensors`
/// tensors plus one scratch graph of `graph_cap` nodes/leafs.
/// Returns the context and the graph capacity to pass to
/// `ggml_new_graph_custom`.
///
/// Eager ops expand the op node with `ggml_build_forward_expand`,
/// which walks the whole ancestor DAG (recomputing it), so the cap
/// must cover every reachable node and leaf — not just the op node.
/// Callers pass `1 + inputs' graph bounds` (see
/// [`Tensor::child_graph_size`]); a fixed cap would abort on deep
/// chains once either array overflowed.
pub(crate) fn new_exec_ctx(
    n_tensors: usize,
    graph_cap: usize,
) -> Result<(*mut forge_sys::ggml_context, usize)> {
    if graph_cap < 1 {
        return Err(Error::invalid("exec graph capacity must be at least 1"));
    }
    if graph_cap > u32::MAX as usize {
        // The native size polynomial (`nodes + leafs + hash(2*cap) + ...`)
        // is wrap-free below 2^32 slots (every term stays under 2^38
        // bytes); above that the context could be undersized while the
        // node arrays stay huge — a heap overrun. No real graph comes
        // close (2^32 nodes need 128 GiB for the arrays alone).
        return Err(Error::invalid("exec graph capacity exceeds u32 range"));
    }
    // SAFETY: no arguments / pure size computation.
    let per_tensor = unsafe { forge_sys::ggml_tensor_overhead() };
    let tensors = per_tensor
        .checked_mul(n_tensors.saturating_add(2))
        .ok_or_else(|| Error::invalid("tensor context size overflow"))?;
    let graph = unsafe { forge_sys::ggml_graph_overhead_custom(graph_cap, false) };
    let mem_size = tensors
        .checked_add(graph)
        .ok_or_else(|| Error::invalid("exec context size overflow"))?;
    let params = forge_sys::ggml_init_params {
        mem_size,
        mem_buffer: ptr::null_mut(),
        no_alloc: true,
    };
    // SAFETY: params are valid (owned pool, NULL buffer = allocate
    // internally); NULL return (OOM) is checked.
    let ctx = unsafe { forge_sys::ggml_init(params) };
    if ctx.is_null() {
        return Err(Error::backend("ggml context allocation failed"));
    }
    Ok((ctx, graph_cap))
}

/// Allocate a metadata-only ggml context sized for `n_tensors` tensors.
pub(crate) fn new_ctx(n_tensors: usize) -> Result<*mut forge_sys::ggml_context> {
    // SAFETY: no arguments.
    let per_tensor = unsafe { forge_sys::ggml_tensor_overhead() };
    let mem_size = per_tensor
        .checked_mul(n_tensors.saturating_add(2))
        .ok_or_else(|| Error::invalid("tensor context size overflow"))?;
    let params = forge_sys::ggml_init_params {
        mem_size,
        mem_buffer: ptr::null_mut(),
        no_alloc: true,
    };
    // SAFETY: params are valid (owned pool, NULL buffer = allocate
    // internally); NULL return (OOM) is checked by the caller.
    let ctx = unsafe { forge_sys::ggml_init(params) };
    if ctx.is_null() {
        return Err(Error::backend("ggml context allocation failed"));
    }
    Ok(ctx)
}

/// Strip trailing extents of 1 (ggml ignores them: `ggml_n_dims`
/// counts only extents above 1, minimum rank 1).
fn normalize_shape(shape: &[usize]) -> Vec<usize> {
    let mut end = shape.len();
    while end > 1 && shape[end - 1] == 1 {
        end -= 1;
    }
    shape[..end].to_vec()
}

/// Validate a Rust shape into ggml `ne` extents plus the element count.
///
/// Besides rank/extent/overflow checks, quantized dtypes require
/// dim-0 block divisibility: ggml never asserts it at creation, but a
/// ragged dim 0 silently corrupts strides (integer-division
/// truncation), so malformed quant shapes are refused here.
fn checked_ne(dtype: DType, shape: &[usize]) -> Result<(Vec<i64>, usize)> {
    if shape.is_empty() || shape.len() > MAX_DIMS {
        return Err(Error::invalid(format!(
            "rank must be 1..={MAX_DIMS}, got {}",
            shape.len()
        )));
    }
    let mut ne = Vec::with_capacity(shape.len());
    let mut nelements = 1usize;
    for (axis, &extent) in shape.iter().enumerate() {
        if extent == 0 {
            return Err(Error::invalid(format!("axis {axis} has extent 0")));
        }
        let extent_i64 =
            i64::try_from(extent).map_err(|_| Error::invalid("extent exceeds i64 range"))?;
        nelements = nelements
            .checked_mul(extent)
            .ok_or_else(|| Error::invalid("element count overflow"))?;
        ne.push(extent_i64);
    }
    if dtype.is_quantized() && !shape[0].is_multiple_of(dtype.block_len()) {
        return Err(Error::invalid(format!(
            "dim-0 extent {} is not a multiple of the {} block length {}",
            shape[0],
            dtype.name(),
            dtype.block_len()
        )));
    }
    Ok((ne, nelements))
}

/// Extent of `dim`, treating missing trailing dims as 1.
fn extent_of(shape: &[usize], dim: usize) -> usize {
    shape.get(dim).copied().unwrap_or(1)
}

/// Contiguous byte size of `ne` extents for `dtype`, mirroring
/// `ggml_new_tensor_impl` (`row_size(ne0) * ne1 * ...`). Returns
/// `None` on overflow or (quantized) block raggedness.
fn contiguous_nbytes(dtype: DType, ne: &[i64]) -> Option<usize> {
    if dtype.is_quantized() && ne[0] % dtype.block_len() as i64 != 0 {
        return None;
    }
    // SAFETY: dtype is valid and (for quant types) dim 0 is
    // block-divisible, which is exactly what `ggml_row_size`
    // debug-asserts.
    let mut size = unsafe { forge_sys::ggml_row_size(dtype.ggml_type(), ne[0]) };
    for &e in &ne[1..] {
        size = size.checked_mul(usize::try_from(e).ok()?)?;
    }
    Some(size)
}

/// An owned tensor living on one [`Backend`].
///
/// A tensor retains its whole ggml history (see the private `History`
/// tape): op results keep their inputs' storage alive, views keep
/// their parent.
/// Dropping an input or a parent never dangles a result or a view;
/// memory releases when the last downstream tensor drops.
pub struct Tensor {
    // Field order is load-bearing: struct fields drop in declaration
    // order, so `history` (every buffer owner, own and ancestral) is
    // declared before `backend` to guarantee buffers free while the
    // backend is still alive. (Buffer freeing on some backends needs
    // a live backend; freeing the backend first would be use-after-free.)
    history: Rc<History>,
    backend: Rc<BackendInner>,
    raw: *mut forge_sys::ggml_tensor,
    dtype: DType,
    shape: Vec<usize>,
    /// Upper bound on reachable graph nodes + leafs (1 + the inputs'
    /// bounds). [`Graph`](crate::Graph) uses it to refuse expansions
    /// that could overflow the native graph (a release abort).
    graph_size: usize,
}

impl Tensor {
    /// Upper bound helper: 1 + the inputs' bounds, or an error on
    /// (practically unreachable) overflow.
    pub(crate) fn child_graph_size(inputs: &[&Tensor]) -> Result<usize> {
        let mut size = 1usize;
        for input in inputs {
            size = size
                .checked_add(input.graph_size)
                .ok_or_else(|| Error::invalid("graph size bound overflow"))?;
        }
        Ok(size)
    }

    /// Consistency cross-checks against native metadata. The Rust-side
    /// shape/dtype copies are trusted by the API, so every
    /// constructor asserts (debug) that ggml agrees on element count,
    /// rank, and contiguity of fresh tensors.
    fn debug_cross_check(&self, expect_contiguous: bool) {
        debug_assert_eq!(
            // SAFETY: raw is a live tensor.
            unsafe { forge_sys::ggml_nelements(self.raw) },
            self.nelements() as i64,
            "native element count matches shape"
        );
        debug_assert_eq!(
            // SAFETY: raw is a live tensor.
            unsafe { forge_sys::ggml_n_dims(self.raw) },
            self.shape.len() as std::os::raw::c_int,
            "native rank matches normalized shape"
        );
        if expect_contiguous {
            debug_assert!(
                // SAFETY: raw is a live tensor.
                unsafe { forge_sys::ggml_is_contiguous(self.raw) },
                "fresh tensor is contiguous"
            );
        }
    }

    /// Create an uninitialized tensor of `dtype`/`shape` on `backend`.
    ///
    /// Contents are undefined until written by an op, an upload, or a
    /// fill. Quantized dtypes require a block-divisible dim 0.
    pub fn empty(backend: &Backend, dtype: DType, shape: &[usize]) -> Result<Self> {
        let shape = normalize_shape(shape);
        let (ne, _) = checked_ne(dtype, &shape)?;
        let ctx = new_ctx(1)?;
        // SAFETY: ctx is live; ne points to shape.len() valid i64s;
        // dtype/rank are validated; NULL returns unwind the context.
        unsafe {
            let raw = forge_sys::ggml_new_tensor(
                ctx,
                dtype.ggml_type(),
                i32::try_from(ne.len()).map_err(|_| Error::invalid("rank exceeds i32 range"))?,
                ne.as_ptr(),
            );
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("ggml tensor creation failed"));
            }
            let buffer = forge_sys::ggml_backend_alloc_ctx_tensors(ctx, backend.raw());
            if buffer.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("backend buffer allocation failed"));
            }
            let tensor = Self {
                history: Rc::new(History {
                    alloc: Rc::new(Allocation { buffer }),
                    ctx: Rc::new(CtxAlloc { ctx }),
                    parents: Vec::new(),
                }),
                backend: Rc::clone(backend.inner()),
                raw,
                dtype,
                shape,
                graph_size: 1,
            };
            tensor.debug_cross_check(true);
            Ok(tensor)
        }
    }

    /// Create an F32 tensor on `backend` and upload row bytes from `data`.
    ///
    /// `shape` is ggml `ne` order; `data.len()` must equal the element
    /// count and is uploaded verbatim.
    pub fn from_f32(backend: &Backend, shape: &[usize], data: &[f32]) -> Result<Self> {
        let tensor = Self::empty(backend, DType::F32, shape)?;
        tensor.upload_f32(data)?;
        Ok(tensor)
    }

    /// Create an I32 tensor on `backend` and upload `data` verbatim.
    ///
    /// I32 tensors carry row indices
    /// ([`get_rows`](crate::get_rows)) and RoPE positions
    /// ([`rope`](crate::rope)).
    pub fn from_i32(backend: &Backend, shape: &[usize], data: &[i32]) -> Result<Self> {
        let tensor = Self::empty(backend, DType::I32, shape)?;
        tensor.upload_i32(data)?;
        Ok(tensor)
    }

    /// Create a tensor of any dtype and upload raw `bytes` verbatim.
    ///
    /// `bytes.len()` must equal the native byte size exactly; for
    /// quantized dtypes the caller supplies valid blocks (e.g. via a
    /// [`Tensor::cast`] from F32, or zeroed blocks, which decode to
    /// finite zeros for every mapped quant type with a dequantizer —
    /// Q8_1/Q8_K have none and are rejected from dequantizing ops).
    pub fn from_bytes(
        backend: &Backend,
        dtype: DType,
        shape: &[usize],
        bytes: &[u8],
    ) -> Result<Self> {
        let tensor = Self::empty(backend, dtype, shape)?;
        tensor.upload_bytes(bytes)?;
        Ok(tensor)
    }

    /// Wrap a tensor produced by a ggml op plus its owned allocations.
    /// The result retains `inputs`' storage (see [`History`]).
    // Private constructor: eight heterogeneous, all-used parameters; a
    // bundle struct would add indirection at every op call site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn wrap_computed(
        backend: &Backend,
        ctx: *mut forge_sys::ggml_context,
        raw: *mut forge_sys::ggml_tensor,
        buffer: *mut forge_sys::ggml_backend_buffer,
        dtype: DType,
        shape: Vec<usize>,
        graph_size: usize,
        inputs: &[&Tensor],
    ) -> Self {
        let tensor = Self {
            history: Rc::new(History {
                alloc: Rc::new(Allocation { buffer }),
                ctx: Rc::new(CtxAlloc { ctx }),
                parents: inputs.iter().map(|t| Rc::clone(&t.history)).collect(),
            }),
            backend: Rc::clone(backend.inner()),
            raw,
            dtype,
            shape: normalize_shape(&shape),
            graph_size,
        };
        tensor.debug_cross_check(true);
        tensor
    }

    /// Wrap a zero-copy shape result (view/reshape/transpose/permute)
    /// sharing `parent`'s allocation and retaining its history. The
    /// new tensor keeps its own metadata context; `expect_contiguous`
    /// selects the cross-check.
    fn wrap_aliased(
        parent: &Tensor,
        ctx: *mut forge_sys::ggml_context,
        raw: *mut forge_sys::ggml_tensor,
        shape: Vec<usize>,
        expect_contiguous: bool,
    ) -> Result<Self> {
        let tensor = Self {
            history: Rc::new(History {
                alloc: Rc::clone(&parent.history.alloc),
                ctx: Rc::new(CtxAlloc { ctx }),
                parents: vec![Rc::clone(&parent.history)],
            }),
            backend: Rc::clone(&parent.backend),
            raw,
            dtype: parent.dtype,
            shape: normalize_shape(&shape),
            graph_size: Self::child_graph_size(&[parent])?,
        };
        tensor.debug_cross_check(expect_contiguous);
        // The alias points into the parent's live buffer (shared Rc)
        // and keeps the parent's history alive (shared History, so
        // dropping the parent never dangles `view_src`), and its bytes
        // are inside the parent's footprint (validated by the caller),
        // so the data pointer is sound.
        Ok(tensor)
    }

    /// Upload F32 row data into a contiguous F32 tensor (packed
    /// element order; a contiguous view writes through to the parent
    /// region it covers).
    pub fn upload_f32(&self, data: &[f32]) -> Result<()> {
        if self.dtype != DType::F32 {
            return Err(Error::unsupported(format!(
                "F32 upload needs an F32 tensor, got {}",
                self.dtype.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "F32 upload needs a contiguous tensor (use `cont` first)",
            ));
        }
        if self.nelements() != data.len() {
            return Err(Error::invalid(format!(
                "data has {} elements but shape {:?} needs {}",
                data.len(),
                self.shape,
                self.nelements()
            )));
        }
        let nbytes = self.nbytes();
        if nbytes != std::mem::size_of_val(data) {
            return Err(Error::backend("ggml byte size disagrees with F32 layout"));
        }
        // SAFETY: tensor is allocated with nbytes of live backend
        // storage; data points to nbytes of valid host memory;
        // offset + size == nbytes is in bounds.
        unsafe {
            forge_sys::ggml_backend_tensor_set(self.raw, data.as_ptr().cast::<c_void>(), 0, nbytes);
        }
        Ok(())
    }

    /// Upload I32 row data into this tensor.
    pub fn upload_i32(&self, data: &[i32]) -> Result<()> {
        if self.dtype != DType::I32 {
            return Err(Error::unsupported(format!(
                "I32 upload needs an I32 tensor, got {}",
                self.dtype.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "I32 upload needs a contiguous tensor (use `cont` first)",
            ));
        }
        if self.nelements() != data.len() {
            return Err(Error::invalid(format!(
                "data has {} elements but shape {:?} needs {}",
                data.len(),
                self.shape,
                self.nelements()
            )));
        }
        let nbytes = self.nbytes();
        if nbytes != std::mem::size_of_val(data) {
            return Err(Error::backend("ggml byte size disagrees with I32 layout"));
        }
        // SAFETY: as for `upload_f32`.
        unsafe {
            forge_sys::ggml_backend_tensor_set(self.raw, data.as_ptr().cast::<c_void>(), 0, nbytes);
        }
        Ok(())
    }

    /// Upload raw bytes into this tensor's full footprint (any dtype,
    /// any strides; `bytes.len()` must equal [`nbytes`](Self::nbytes)).
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<()> {
        let nbytes = self.nbytes();
        if nbytes != bytes.len() {
            return Err(Error::invalid(format!(
                "bytes has {} bytes but the tensor footprint is {nbytes}",
                bytes.len()
            )));
        }
        // SAFETY: as for `upload_f32`; views resolve to the parent
        // buffer natively and the footprint stays in bounds.
        unsafe {
            forge_sys::ggml_backend_tensor_set(
                self.raw,
                bytes.as_ptr().cast::<c_void>(),
                0,
                nbytes,
            );
        }
        Ok(())
    }

    /// Download an F32 tensor to the host.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Error::unsupported(format!(
                "host download only supports F32, got {}",
                self.dtype.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "F32 download needs a contiguous tensor (use `cont` first)",
            ));
        }
        let mut out = vec![0.0f32; self.nelements()];
        let nbytes = self.nbytes();
        if nbytes != std::mem::size_of_val(out.as_slice()) {
            return Err(Error::backend("ggml byte size disagrees with F32 layout"));
        }
        // SAFETY: tensor is F32 with nbytes of live backend storage;
        // out points to nbytes of valid host memory.
        unsafe {
            forge_sys::ggml_backend_tensor_get(
                self.raw,
                out.as_mut_ptr().cast::<c_void>(),
                0,
                nbytes,
            );
        }
        Ok(out)
    }

    /// Download an I32 tensor to the host.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        if self.dtype != DType::I32 {
            return Err(Error::unsupported(format!(
                "I32 download needs an I32 tensor, got {}",
                self.dtype.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "I32 download needs a contiguous tensor (use `cont` first)",
            ));
        }
        let mut out = vec![0i32; self.nelements()];
        let nbytes = self.nbytes();
        if nbytes != std::mem::size_of_val(out.as_slice()) {
            return Err(Error::backend("ggml byte size disagrees with I32 layout"));
        }
        // SAFETY: as for `to_vec_f32`.
        unsafe {
            forge_sys::ggml_backend_tensor_get(
                self.raw,
                out.as_mut_ptr().cast::<c_void>(),
                0,
                nbytes,
            );
        }
        Ok(out)
    }

    /// Download this tensor's full footprint as raw bytes (any dtype,
    /// any strides).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let nbytes = self.nbytes();
        let mut out = vec![0u8; nbytes];
        // SAFETY: out holds exactly nbytes of live host memory;
        // offset + size == nbytes is in bounds.
        unsafe {
            forge_sys::ggml_backend_tensor_get(
                self.raw,
                out.as_mut_ptr().cast::<c_void>(),
                0,
                nbytes,
            );
        }
        Ok(out)
    }

    /// Fill every element of a contiguous F32 tensor with `value`
    /// (native `ggml_set_f32`, in place).
    pub fn fill_f32(&self, value: f32) -> Result<()> {
        if self.dtype != DType::F32 {
            return Err(Error::unsupported(format!(
                "F32 fill needs an F32 tensor, got {}",
                self.dtype.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "F32 fill needs a contiguous tensor (use `cont` first)",
            ));
        }
        // SAFETY: tensor is allocated F32 + contiguous, which
        // satisfies the implementation (packed dim 0, correct row
        // strides) and avoids its quant-dtype abort.
        unsafe {
            forge_sys::ggml_set_f32(self.raw, value);
        }
        Ok(())
    }

    /// Fill this tensor's full footprint with `byte` (any dtype, any
    /// strides; views fill the parent region they cover).
    ///
    /// Implemented through `ggml_backend_tensor_set` staging rather
    /// than `ggml_backend_tensor_memset`: memset support is a
    /// per-buffer interface pointer that aborts when unimplemented and
    /// cannot be queried, while set/get is implemented by every
    /// backend at the pin.
    pub fn fill_bytes(&self, byte: u8) -> Result<()> {
        let bytes = vec![byte; self.nbytes()];
        self.upload_bytes(&bytes)
    }

    /// Copy this tensor into `dst` (native `ggml_backend_tensor_copy`;
    /// works across backends, staging through the host when needed).
    ///
    /// Both tensors must share dtype, shape, and contiguity, and
    /// neither may be a view: the native copy asserts identical
    /// layout (same type/ne/strides — implied by the Rust checks) and
    /// reads the *source* buffer without resolving views. ForgeCore
    /// views carry a NULL buffer (no `view_init` runs on the
    /// alias-only path), which trips the native NULL assert
    /// (`ggml-backend.cpp`, `buffer_get_type`) — a NULL dereference
    /// under NDEBUG. Destination views happen to survive (the set
    /// path resolves them onto the parent buffer) but are refused
    /// symmetrically rather than relying on that path.
    pub fn copy_into(&self, dst: &Tensor) -> Result<()> {
        if self.dtype != dst.dtype {
            return Err(Error::invalid(format!(
                "copy needs matching dtypes, got {} and {}",
                self.dtype.name(),
                dst.dtype.name()
            )));
        }
        if self.shape != dst.shape {
            return Err(Error::invalid(format!(
                "copy needs matching shapes, got {:?} and {:?}",
                self.shape, dst.shape
            )));
        }
        if !self.is_contiguous() || !dst.is_contiguous() {
            return Err(Error::invalid(
                "copy needs contiguous tensors on both ends (use `cont` first)",
            ));
        }
        if self.is_view() || dst.is_view() {
            return Err(Error::invalid(
                "copy refuses views (native copy asserts on a view's NULL source buffer)",
            ));
        }
        // SAFETY: same dtype + same shape + both contiguous implies
        // identical native layout (strides derive deterministically),
        // both buffers are live (non-views are allocated), and
        // self-copy is a natively handled no-op.
        unsafe {
            forge_sys::ggml_backend_tensor_copy(self.raw, dst.raw);
        }
        Ok(())
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shape in ggml `ne` order (trailing extents of 1 stripped).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total element count.
    pub fn nelements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Storage footprint in bytes, as reported by ggml (stride-aware:
    /// strided views report their whole span, not packed size).
    pub fn nbytes(&self) -> usize {
        // SAFETY: raw is a live tensor; takes no aliased pointers.
        unsafe { forge_sys::ggml_nbytes(self.raw) }
    }

    /// Whether the tensor is packed with no gaps or permutation
    /// (native `ggml_is_contiguous`).
    pub fn is_contiguous(&self) -> bool {
        // SAFETY: raw is a live tensor.
        unsafe { forge_sys::ggml_is_contiguous(self.raw) }
    }

    /// Whether the tensor aliases another tensor's storage (native
    /// `ggml_is_view`). Views share their parent's allocation: they
    /// stay valid as long as any Rust handle to the allocation lives,
    /// regardless of drop order.
    pub fn is_view(&self) -> bool {
        // SAFETY: raw is a live tensor.
        unsafe { forge_sys::ggml_is_view(self.raw) }
    }

    /// Tensor name (`ggml_get_name`; empty unless [`set_name`](Self::set_name)
    /// was called — ggml suffixes op results, e.g. `"x (reshaped)"`,
    /// and names anonymous graph tensors `leaf_N`/`node_N` when they
    /// are first added to a [`Graph`](crate::graph::Graph)).
    pub fn name(&self) -> String {
        // SAFETY: raw is live; the name array is always
        // NUL-terminated (zero-initialized, truncation-safe writes).
        unsafe {
            CStr::from_ptr(forge_sys::ggml_get_name(self.raw))
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Set the tensor name (NUL bytes rejected; longer than 63
    /// bytes truncated at a UTF-8 boundary — ggml names hold
    /// `GGML_MAX_NAME` = 64 bytes including the NUL).
    pub fn set_name(&self, name: &str) -> Result<()> {
        if name.as_bytes().contains(&0) {
            return Err(Error::invalid("tensor name contains a NUL byte"));
        }
        let mut end = name.len().min(63);
        while end > 0 && !name.is_char_boundary(end) {
            end -= 1;
        }
        let truncated = &name[..end];
        let cname =
            CString::new(truncated).map_err(|_| Error::invalid("tensor name is not C-safe"))?;
        // SAFETY: raw is live; cname is a valid NUL-terminated
        // string; the copy is truncation-safe.
        unsafe {
            forge_sys::ggml_set_name(self.raw, cname.as_ptr());
        }
        Ok(())
    }

    /// Short description of the producing op (`ggml_op_desc`, e.g.
    /// `"ADD"`, `"MUL_MAT"`, `"NONE"` for leaves).
    pub fn op_name(&self) -> String {
        // SAFETY: raw is live; op_desc returns a static string.
        unsafe {
            CStr::from_ptr(forge_sys::ggml_op_desc(self.raw))
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Raw tensor pointer (live while `self` lives).
    pub(crate) fn raw(&self) -> *mut forge_sys::ggml_tensor {
        self.raw
    }

    /// Shared backend owner (identity checks).
    pub(crate) fn backend_inner(&self) -> &Rc<BackendInner> {
        &self.backend
    }

    /// Graph-size bound of this tensor (capacity planning).
    pub(crate) fn graph_size(&self) -> usize {
        self.graph_size
    }

    /// Shared view-argument validation: extents are positive i64s,
    /// `offset` is block-aligned, the view's packed-form size plus
    /// `offset` fits the parent footprint (the native
    /// `view_tensor_impl` assert), and the strided footprint itself
    /// fits (soundness for reads through custom strides).
    fn check_view_args(
        &self,
        op: &str,
        ne: &[usize],
        nb: &[usize],
        offset: usize,
    ) -> Result<Vec<i64>> {
        let mut ne_i64 = Vec::with_capacity(ne.len());
        for (axis, &extent) in ne.iter().enumerate() {
            if extent == 0 {
                return Err(Error::invalid(format!("{op}: axis {axis} has extent 0")));
            }
            ne_i64.push(
                i64::try_from(extent)
                    .map_err(|_| Error::invalid(format!("{op}: extent exceeds i64")))?,
            );
        }
        if self.dtype.is_quantized() && !ne[0].is_multiple_of(self.dtype.block_len()) {
            return Err(Error::invalid(format!(
                "{op}: dim-0 extent {} is not a multiple of the {} block length {}",
                ne[0],
                self.dtype.name(),
                self.dtype.block_len()
            )));
        }
        let type_size = self.dtype.type_size();
        if !offset.is_multiple_of(type_size) {
            // No native assert covers this; a misaligned offset would
            // make kernels dereference misaligned element pointers
            // (undefined behavior on strict-alignment targets).
            return Err(Error::invalid(format!(
                "{op}: offset {offset} is not a multiple of the element size {type_size}"
            )));
        }
        for (axis, &stride) in nb.iter().enumerate() {
            if stride % type_size != 0 {
                return Err(Error::invalid(format!(
                    "{op}: stride {stride} on axis {} is not a multiple of the element size {type_size}",
                    axis + 1
                )));
            }
        }
        // Native assert form: packed-form size + offset <= parent
        // footprint.
        let packed = contiguous_nbytes(self.dtype, &ne_i64)
            .ok_or_else(|| Error::invalid(format!("{op}: packed view size overflow")))?;
        let parent_bytes = self.nbytes();
        let packed_end = packed
            .checked_add(offset)
            .ok_or_else(|| Error::invalid(format!("{op}: offset overflow")))?;
        if packed_end > parent_bytes {
            return Err(Error::invalid(format!(
                "{op}: view (packed {packed} + offset {offset}) exceeds the parent footprint {parent_bytes}"
            )));
        }
        // Strided footprint: last byte touched through custom
        // strides must be inside the parent.
        let mut span = forge_sys_row_size(self.dtype, ne_i64[0]);
        for (dim, &stride) in nb.iter().enumerate() {
            let steps = ne[dim + 1].saturating_sub(1);
            span = span
                .checked_add(
                    stride.checked_mul(steps).ok_or_else(|| {
                        Error::invalid(format!("{op}: strided footprint overflow"))
                    })?,
                )
                .ok_or_else(|| Error::invalid(format!("{op}: strided footprint overflow")))?;
        }
        let span_end = span
            .checked_add(offset)
            .ok_or_else(|| Error::invalid(format!("{op}: offset overflow")))?;
        if span_end > parent_bytes {
            return Err(Error::invalid(format!(
                "{op}: strided footprint ({span} + offset {offset}) exceeds the parent footprint {parent_bytes}"
            )));
        }
        Ok(ne_i64)
    }

    /// 1-D view: `ne0` elements starting at byte `offset`.
    pub fn view_1d(&self, ne0: usize, offset: usize) -> Result<Self> {
        let ne = self.check_view_args("view_1d", &[ne0], &[], offset)?;
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = forge_sys::ggml_view_1d(ctx, self.raw, ne[0], offset);
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("view_1d: ggml view creation failed"));
            }
            Self::wrap_aliased(self, ctx, raw, vec![ne0], false)
        }
    }

    /// 2-D view with byte row stride `nb1` and byte `offset`.
    pub fn view_2d(&self, ne: [usize; 2], nb1: usize, offset: usize) -> Result<Self> {
        let ne_i64 = self.check_view_args("view_2d", &ne, &[nb1], offset)?;
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = forge_sys::ggml_view_2d(ctx, self.raw, ne_i64[0], ne_i64[1], nb1, offset);
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("view_2d: ggml view creation failed"));
            }
            Self::wrap_aliased(self, ctx, raw, ne.to_vec(), false)
        }
    }
    /// 3-D view with byte strides `nb1`/`nb2` and byte `offset`.
    pub fn view_3d(&self, ne: [usize; 3], nb: [usize; 2], offset: usize) -> Result<Self> {
        let ne_i64 = self.check_view_args("view_3d", &ne, &nb, offset)?;
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = forge_sys::ggml_view_3d(
                ctx, self.raw, ne_i64[0], ne_i64[1], ne_i64[2], nb[0], nb[1], offset,
            );
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("view_3d: ggml view creation failed"));
            }
            Self::wrap_aliased(self, ctx, raw, ne.to_vec(), false)
        }
    }

    /// 4-D view with byte strides `nb1`/`nb2`/`nb3` and byte `offset`.
    pub fn view_4d(&self, ne: [usize; 4], nb: [usize; 3], offset: usize) -> Result<Self> {
        let ne_i64 = self.check_view_args("view_4d", &ne, &nb, offset)?;
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = forge_sys::ggml_view_4d(
                ctx, self.raw, ne_i64[0], ne_i64[1], ne_i64[2], ne_i64[3], nb[0], nb[1], nb[2],
                offset,
            );
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("view_4d: ggml view creation failed"));
            }
            Self::wrap_aliased(self, ctx, raw, ne.to_vec(), false)
        }
    }

    /// Zero-copy reshape to `shape` (same element count; ggml asserts
    /// `is_contiguous`, so strided tensors must pass through
    /// [`cont`](Self::cont) first).
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let shape = normalize_shape(shape);
        let (ne, nelements) = checked_ne(self.dtype, &shape)?;
        if nelements != self.nelements() {
            return Err(Error::invalid(format!(
                "reshape needs {} elements, got shape {:?} ({} elements)",
                self.nelements(),
                shape,
                nelements
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "reshape needs a contiguous tensor (use `cont` first)",
            ));
        }
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = match ne.len() {
                1 => forge_sys::ggml_reshape_1d(ctx, self.raw, ne[0]),
                2 => forge_sys::ggml_reshape_2d(ctx, self.raw, ne[0], ne[1]),
                3 => forge_sys::ggml_reshape_3d(ctx, self.raw, ne[0], ne[1], ne[2]),
                _ => forge_sys::ggml_reshape_4d(ctx, self.raw, ne[0], ne[1], ne[2], ne[3]),
            };
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("reshape: ggml reshape failed"));
            }
            // A contiguous reshape of a contiguous parent keeps
            // packed strides (verified: the constructor copies the
            // packed-stride pattern for matching footprints).
            Self::wrap_aliased(self, ctx, raw, shape, true)
        }
    }

    /// Zero-copy transpose of dims 0 and 1.
    ///
    /// The native constructor reuses the view path, which asserts the
    /// packed-form size fits the parent footprint — true for ordinary
    /// tensors but false for overlapping views (footprint smaller
    /// than packed size), hence the explicit check.
    pub fn transpose(&self) -> Result<Self> {
        self.check_alias_footprint("transpose")?;
        let e = [
            extent_of(&self.shape, 0),
            extent_of(&self.shape, 1),
            extent_of(&self.shape, 2),
            extent_of(&self.shape, 3),
        ];
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = forge_sys::ggml_transpose(ctx, self.raw);
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("transpose: ggml transpose failed"));
            }
            Self::wrap_aliased(self, ctx, raw, vec![e[1], e[0], e[2], e[3]], false)
        }
    }

    /// Zero-copy axis permutation (`axes` maps new axis -> old axis;
    /// must list each of 0..4 exactly once).
    pub fn permute(&self, axes: [usize; 4]) -> Result<Self> {
        let mut seen = [false; 4];
        for &axis in &axes {
            if axis >= MAX_DIMS || seen[axis] {
                return Err(Error::invalid(format!(
                    "permute needs each of 0..4 exactly once, got {axes:?}"
                )));
            }
            seen[axis] = true;
        }
        self.check_alias_footprint("permute")?;
        let ctx = new_ctx(1)?;
        unsafe {
            let raw = forge_sys::ggml_permute(
                ctx,
                self.raw,
                axes[0] as std::os::raw::c_int,
                axes[1] as std::os::raw::c_int,
                axes[2] as std::os::raw::c_int,
                axes[3] as std::os::raw::c_int,
            );
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("permute: ggml permute failed"));
            }
            let shape = axes.map(|axis| extent_of(&self.shape, axis)).to_vec();
            Self::wrap_aliased(self, ctx, raw, shape, false)
        }
    }

    /// Packed-form footprint check for [`transpose`](Self::transpose)
    /// / [`permute`](Self::permute): the native view path asserts
    /// `packed_size <= parent_nbytes`, which overlapping views can
    /// violate.
    fn check_alias_footprint(&self, op: &str) -> Result<()> {
        let mut ne = Vec::with_capacity(MAX_DIMS);
        for dim in 0..MAX_DIMS {
            ne.push(
                i64::try_from(extent_of(&self.shape, dim))
                    .map_err(|_| Error::invalid(format!("{op}: extent exceeds i64 range")))?,
            );
        }
        let packed = contiguous_nbytes(self.dtype, &ne)
            .ok_or_else(|| Error::invalid(format!("{op}: packed size overflow")))?;
        if packed > self.nbytes() {
            return Err(Error::invalid(format!(
                "{op}: packed size {packed} exceeds the tensor footprint {} (overlapping view)",
                self.nbytes()
            )));
        }
        Ok(())
    }
    /// Fresh contiguous copy (native `ggml_cont`, executed eagerly).
    ///
    /// Quantized tensors must already be contiguous: the strided copy
    /// path sizes rows as `ne0 * block_bytes`, which over-counts a
    /// block row by the block factor and overruns both ends.
    /// Non-quantized tensors may have any strides (the strided path
    /// copies element-wise, correctly but slowly).
    pub fn cont(&self) -> Result<Self> {
        if self.dtype.is_quantized() && !self.is_contiguous() {
            return Err(Error::invalid(format!(
                "cont needs a contiguous {} tensor (strided quant copy is not implemented)",
                self.dtype.name()
            )));
        }
        let backend = Backend::from_inner(Rc::clone(&self.backend));
        let graph_size = Tensor::child_graph_size(&[self])?;
        let (ctx, graph_cap) = new_exec_ctx(1, graph_size)?;
        unsafe {
            let raw = forge_sys::ggml_cont(ctx, self.raw);
            crate::runtime::finish(
                "cont",
                &backend,
                ctx,
                graph_cap,
                raw,
                self.dtype,
                self.shape.clone(),
                graph_size,
                &[self],
            )
        }
    }

    /// Eager dtype conversion (native `ggml_cast`).
    ///
    /// Supported pairs (an audited strict subset of the CPU
    /// kernel's conversions — native additionally quantizes F16/BF16
    /// into Q*, which stays rejected as unaudited; rejected pairs
    /// abort natively, or crash on a NULL dequantizer row):
    /// same-type, F32 to F16/BF16/I32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, F16
    /// to F32/BF16, BF16 to F32/F16, I32 to F32, and
    /// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K to F32.
    /// Q8_1 and Q8_K have no dequantizer row (`to_float` is NULL in
    /// the type table — converting out of them would call a NULL
    /// pointer), and K-quant super-blocks are not quantizable into
    /// (their hierarchical layout needs a deeper audit than the flat
    /// one-scale-per-block family got). The input must be contiguous:
    /// strided sources abort the quantizing path (native contiguity
    /// assert, verified in source), and strided quant sources
    /// silently misread on the dequantizing path (probe-measured
    /// max|diff| of ~4 against the logical transpose). F32 to quant
    /// additionally needs a block-divisible dim
    /// 0 (the quantizer integer-divides the row into blocks). F32 to
    /// I32 truncates toward zero.
    pub fn cast(&self, dtype: DType) -> Result<Self> {
        let from = self.dtype;
        if !cast_pair_supported(from, dtype) {
            return Err(Error::unsupported(format!(
                "cast from {} to {} is not implemented",
                from.name(),
                dtype.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid(
                "cast needs a contiguous input (use `cont` first)",
            ));
        }
        if from == DType::F32
            && dtype.is_quantized()
            && !self.shape[0].is_multiple_of(dtype.block_len())
        {
            return Err(Error::invalid(format!(
                "cast to {} needs dim 0 divisible by {}, got {}",
                dtype.name(),
                dtype.block_len(),
                self.shape[0]
            )));
        }
        let backend = Backend::from_inner(Rc::clone(&self.backend));
        let graph_size = Tensor::child_graph_size(&[self])?;
        let (ctx, graph_cap) = new_exec_ctx(1, graph_size)?;
        unsafe {
            let raw = forge_sys::ggml_cast(ctx, self.raw, dtype.ggml_type());
            crate::runtime::finish(
                "cast",
                &backend,
                ctx,
                graph_cap,
                raw,
                dtype,
                self.shape.clone(),
                graph_size,
                &[self],
            )
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("op", &self.op_name())
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}

/// Whether `ggml_cast` implements `from -> to` (the CPU kernel's
/// conversion table; shared by [`Tensor::cast`] and op probes).
///
/// Same-type pairs copy bytes (always sound). The float/int pairs
/// mirror an audited subset of the kernel's option table (native
/// also quantizes F16/BF16 into Q*; excluded as unaudited).
/// `F32 -> Q*` covers the
/// flat-block family only (`from_float` present, one scale per
/// block); K-quant super-blocks need a deeper audit and `Q8_1` has
/// no dequantizer, so quantizing into it would be a one-way trap.
/// `Q* -> F32` covers every mapped quant type *with* a dequantizer —
/// `Q8_1`/`Q8_K` have no `to_float` row (verified in the type table)
/// and would call a NULL function pointer, so they are excluded.
pub(crate) fn cast_pair_supported(from: DType, to: DType) -> bool {
    from == to
        || matches!(
            (from, to),
            (DType::F32, DType::F16)
                | (DType::F32, DType::BF16)
                | (DType::F32, DType::I32)
                | (DType::F32, DType::Q4_0)
                | (DType::F32, DType::Q4_1)
                | (DType::F32, DType::Q5_0)
                | (DType::F32, DType::Q5_1)
                | (DType::F32, DType::Q8_0)
                | (DType::F16, DType::F32)
                | (DType::F16, DType::BF16)
                | (DType::BF16, DType::F32)
                | (DType::BF16, DType::F16)
                | (DType::I32, DType::F32)
        )
        || (to == DType::F32
            && matches!(
                from,
                DType::Q4_0
                    | DType::Q4_1
                    | DType::Q5_0
                    | DType::Q5_1
                    | DType::Q8_0
                    | DType::Q2_K
                    | DType::Q3_K
                    | DType::Q4_K
                    | DType::Q5_K
                    | DType::Q6_K
            ))
}

/// Row byte size for `ne0` elements (`ggml_row_size`; the caller
/// guarantees block divisibility for quantized dtypes).
fn forge_sys_row_size(dtype: DType, ne0: i64) -> usize {
    // SAFETY: dtype is valid and dim 0 is block-divisible, which is
    // exactly what `ggml_row_size` debug-asserts.
    unsafe { forge_sys::ggml_row_size(dtype.ggml_type(), ne0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_trailing_ones() {
        assert_eq!(normalize_shape(&[2, 3, 1, 1]), vec![2, 3]);
        assert_eq!(normalize_shape(&[5]), vec![5]);
        assert_eq!(normalize_shape(&[1, 1, 1]), vec![1]);
        assert_eq!(normalize_shape(&[1, 4]), vec![1, 4]);
    }

    #[test]
    fn checked_ne_rejects_bad_shapes() {
        assert!(checked_ne(DType::F32, &[]).is_err());
        assert!(checked_ne(DType::F32, &[1, 2, 3, 4, 5]).is_err());
        assert!(checked_ne(DType::F32, &[4, 0]).is_err());
        assert!(checked_ne(DType::F32, &[usize::MAX, usize::MAX]).is_err());
        let (ne, n) = checked_ne(DType::F32, &[4, 3]).unwrap();
        assert_eq!((ne, n), (vec![4, 3], 12));
    }

    #[test]
    fn checked_ne_enforces_quant_blocks() {
        assert!(checked_ne(DType::Q4_0, &[30, 4]).is_err());
        assert!(checked_ne(DType::Q4_0, &[32, 4]).is_ok());
        assert!(checked_ne(DType::Q8_0, &[31]).is_err());
        assert!(checked_ne(DType::F32, &[7]).is_ok());
    }
}
