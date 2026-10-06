//! Multi-output ggml graphs and CPU execution plans.
//!
//! [`Graph`] collects output [`Tensor`](crate::tensor::Tensor)s into
//! one `ggml_cgraph` so shared subexpressions compute once; [`Plan`]
//! compiles a graph for repeated execution on the CPU backend.
//!
//! Lifetimes and ownership:
//! - Output tensors are borrowed (`Graph<'t>`): the caller keeps them
//!   (and their buffers) alive while the graph — and any plan built
//!   from it — is used. Dropping an output while its graph lives is
//!   a compile error.
//! - The graph holds a clone of the outputs' backend handle, so the
//!   backend cannot be freed first either, and every output must live
//!   on that one backend (mixed-backend graphs are refused —
//!   `ggml_build_forward_expand` has no cross-backend path).
//! - Plans borrow their graph (`Plan<'a>`); the graph context (node
//!   arrays) therefore outlives every plan even though the native
//!   CPU plan copies the node list.
//! - Plans are CPU-only: at the pin, every non-CPU backend leaves the
//!   plan interface NULL (verified in source), and plan creation
//!   asserts it is set — calling it elsewhere would abort. The
//!   CPU gate makes that abort unreachable.
//!
//! Capacity: the native graph overflows its node/leaf arrays with a
//! release abort, and leaf counts have no public getter, so each
//! [`Tensor`](crate::tensor::Tensor) carries an upper bound
//! (`1 + inputs' bounds`, covering both new nodes and new leafs) and
//! [`Graph::add_output`] refuses expansions that could overflow. The
//! bound is conservative (shared subexpressions counted twice),
//! never unsound.

use crate::backend::{Backend, BackendInner};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
use std::rc::Rc;

/// Default graph capacity (nodes + leafs bound) when the caller does
/// not size the graph explicitly.
pub const DEFAULT_GRAPH_CAPACITY: usize = 2048;

/// Owned snapshot of one computed graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInfo {
    /// Node name (ggml assigns `"<input> (op)"`-style names unless
    /// [`Tensor::set_name`](crate::tensor::Tensor::set_name) overrode them).
    pub name: String,
    /// Producing op (`ggml_op_desc`, e.g. `"ADD"`, `"MUL_MAT"`).
    pub op: String,
    /// Element count.
    pub nelements: usize,
    /// Storage footprint in bytes.
    pub nbytes: usize,
    /// Rank (normalized shape length).
    pub n_dims: usize,
    /// Whether the node output is packed contiguous.
    pub is_contiguous: bool,
    /// Whether the node aliases another tensor's storage.
    pub is_view: bool,
}

/// Snapshot a live graph node into owned [`NodeInfo`].
///
/// # Safety
///
/// The caller guarantees `node` points to a live tensor.
unsafe fn snapshot_node(node: *mut forge_sys::ggml_tensor) -> NodeInfo {
    // SAFETY: node is live per the caller contract; every getter
    // only reads metadata; the name/op strings are NUL-terminated
    // (ggml zero-initializes names and writes them truncation-safe).
    unsafe {
        NodeInfo {
            name: CStr::from_ptr(forge_sys::ggml_get_name(node))
                .to_string_lossy()
                .into_owned(),
            op: CStr::from_ptr(forge_sys::ggml_op_desc(node))
                .to_string_lossy()
                .into_owned(),
            nelements: usize::try_from(forge_sys::ggml_nelements(node)).unwrap_or(0),
            nbytes: forge_sys::ggml_nbytes(node),
            n_dims: usize::try_from(forge_sys::ggml_n_dims(node)).unwrap_or(0),
            is_contiguous: forge_sys::ggml_is_contiguous(node),
            is_view: forge_sys::ggml_is_view(node),
        }
    }
}

/// A multi-output computation graph borrowing its output tensors.
pub struct Graph<'t> {
    ctx: *mut forge_sys::ggml_context,
    raw: *mut forge_sys::ggml_cgraph,
    cap: usize,
    /// Conservative upper bound on the native leaf count (grows by
    /// each output's bound; native leafs have no public getter).
    leaf_high: usize,
    /// The one backend every output must live on.
    backend: Option<Rc<BackendInner>>,
    _tensors: PhantomData<&'t Tensor>,
}

impl<'t> Graph<'t> {
    /// Empty graph with [`DEFAULT_GRAPH_CAPACITY`] node+leaf capacity.
    pub fn new() -> Result<Self> {
        Self::with_capacity(DEFAULT_GRAPH_CAPACITY)
    }

    /// Empty graph with `capacity` node+leaf slots. The context is
    /// sized with `ggml_graph_overhead_custom` exactly: an undersized
    /// context makes `ggml_new_graph_custom` dereference NULL (a
    /// release SEGV / debug abort), so the size is never guessed.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        if capacity < 1 {
            return Err(Error::invalid("graph capacity must be at least 1"));
        }
        if capacity > u32::MAX as usize {
            // The native size polynomial is wrap-free below 2^32 slots;
            // above that the context could be undersized while the node
            // arrays stay huge — a heap overrun (see `new_exec_ctx`).
            return Err(Error::invalid("graph capacity exceeds u32 range"));
        }
        // SAFETY: pure size computation.
        let mem_size = unsafe { forge_sys::ggml_graph_overhead_custom(capacity, false) };
        let params = forge_sys::ggml_init_params {
            mem_size,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        // SAFETY: params valid (owned pool, exact size); NULL
        // returns unwind the context.
        unsafe {
            let ctx = forge_sys::ggml_init(params);
            if ctx.is_null() {
                return Err(Error::backend("ggml graph context allocation failed"));
            }
            let raw = forge_sys::ggml_new_graph_custom(ctx, capacity, false);
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("ggml graph creation failed"));
            }
            Ok(Self {
                ctx,
                raw,
                cap: capacity,
                leaf_high: 0,
                backend: None,
                _tensors: PhantomData,
            })
        }
    }

    /// Node+leaf capacity.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Number of compute nodes currently in the graph (native
    /// `ggml_graph_n_nodes`; total).
    pub fn n_nodes(&self) -> usize {
        // SAFETY: raw is a live graph.
        let n = unsafe { forge_sys::ggml_graph_n_nodes(self.raw) };
        usize::try_from(n).unwrap_or(0)
    }

    /// Snapshot compute node `index` (both ends bounds-checked:
    /// `ggml_graph_node` asserts only the lower bound, so an upper
    /// overrun would silently misread).
    pub fn node(&self, index: usize) -> Result<NodeInfo> {
        if index >= self.n_nodes() {
            return Err(Error::invalid(format!(
                "node index {index} is outside the graph ({} nodes)",
                self.n_nodes()
            )));
        }
        let index_i32 = std::os::raw::c_int::try_from(index)
            .map_err(|_| Error::invalid("node index exceeds i32 range"))?;
        // SAFETY: index is bounds-checked, so the node is live for
        // the graph's lifetime.
        unsafe {
            let node = forge_sys::ggml_graph_node(self.raw, index_i32);
            Ok(snapshot_node(node))
        }
    }

    /// Find a graph tensor by exact name (`ggml_graph_get_tensor`;
    /// NULL = absent). Names never contain NUL bytes, so a NUL in
    /// `name` simply matches nothing.
    pub fn find(&self, name: &str) -> Option<NodeInfo> {
        let cname = CString::new(name).ok()?;
        // SAFETY: raw is a live graph; cname is NUL-terminated;
        // NULL (absent) is checked, so a found node is live.
        unsafe {
            let node = forge_sys::ggml_graph_get_tensor(self.raw, cname.as_ptr());
            if node.is_null() {
                return None;
            }
            Some(snapshot_node(node))
        }
    }

    /// Add `output` (and its reachable inputs) to the graph.
    ///
    /// Refuses mixed-backend outputs and expansions that could
    /// overflow the node or leaf arrays (a release abort natively):
    /// both `n_nodes + output.bound` and `leaf_high + output.bound`
    /// must fit. Adding the same tensor twice adds nothing the second
    /// time (the native visited set persists across calls); the bound
    /// accounts each call separately, so this stays conservative.
    /// Anonymous tensors are named `leaf_N`/`node_N` on first add
    /// (native `ggml_format_name`), which is visible through
    /// [`Tensor::name`](crate::tensor::Tensor::name) afterwards.
    pub fn add_output(&mut self, output: &'t Tensor) -> Result<()> {
        match &self.backend {
            None => {
                self.backend = Some(Rc::clone(output.backend_inner()));
            }
            Some(backend) => {
                if !Rc::ptr_eq(backend, output.backend_inner()) {
                    return Err(Error::backend(
                        "graph outputs must live on one backend",
                    ));
                }
            }
        }
        let bound = output.graph_size();
        let nodes_end = self
            .n_nodes()
            .checked_add(bound)
            .ok_or_else(|| Error::invalid("graph node bound overflow"))?;
        let leafs_end = self
            .leaf_high
            .checked_add(bound)
            .ok_or_else(|| Error::invalid("graph leaf bound overflow"))?;
        if nodes_end > self.cap || leafs_end > self.cap {
            return Err(Error::invalid(format!(
                "graph capacity {} cannot take output with bound {bound} ({} nodes, leaf bound {})",
                self.cap,
                self.n_nodes(),
                self.leaf_high
            )));
        }
        // SAFETY: the output's tensors and buffers are borrowed live
        // ('t), the backend matches, and both array bounds fit, so
        // the expansion cannot overflow.
        unsafe {
            forge_sys::ggml_build_forward_expand(self.raw, output.raw());
        }
        self.leaf_high = leafs_end;
        Ok(())
    }

    /// Compile this graph into a reusable CPU [`Plan`].
    ///
    /// `backend` must be the CPU backend the outputs live on (plans
    /// exist only for CPU; the graph's backend must match — a plan
    /// cannot migrate a graph across backends).
    pub fn plan(&self, backend: &Backend) -> Result<Plan<'_>> {
        if !backend.is_cpu() {
            return Err(Error::unsupported(
                "graph plans need the CPU backend (only the CPU backend implements plans)",
            ));
        }
        match &self.backend {
            Some(graph_backend) => {
                if !Rc::ptr_eq(graph_backend, backend.inner()) {
                    return Err(Error::backend(
                        "plan backend differs from the graph backend",
                    ));
                }
            }
            None => {
                return Err(Error::invalid(
                    "cannot plan an empty graph (add an output first)",
                ));
            }
        }
        // SAFETY: CPU backend (plan interface present — verified in
        // source for CPU, and `is_cpu` gates the rest), live graph;
        // NULL (OOM) is checked. The plan borrows the graph ('_) and
        // the graph borrows the tensors ('t: '_), so every tensor
        // and buffer outlives the plan.
        unsafe {
            let raw = forge_sys::ggml_backend_graph_plan_create(backend.raw(), self.raw);
            if raw.is_null() {
                return Err(Error::backend("graph plan creation failed"));
            }
            Ok(Plan {
                backend: backend.clone(),
                raw,
                _graph: PhantomData,
            })
        }
    }

    /// Compute the graph once on `backend` (must be the outputs'
    /// backend; CPU and GPU alike — single-shot execution needs no
    /// plan interface).
    pub fn compute(&self, backend: &Backend) -> Result<()> {
        match &self.backend {
            Some(graph_backend) => {
                if !Rc::ptr_eq(graph_backend, backend.inner()) {
                    return Err(Error::backend(
                        "compute backend differs from the graph backend",
                    ));
                }
            }
            None => {
                return Err(Error::invalid(
                    "cannot compute an empty graph (add an output first)",
                ));
            }
        }
        // SAFETY: backend matches the graph's, tensors and buffers
        // are borrowed live; the status code is mapped below.
        unsafe {
            let status = forge_sys::ggml_backend_graph_compute(backend.raw(), self.raw);
            if status != forge_sys::status::SUCCESS {
                return Err(Error::backend(format!(
                    "graph compute failed: {}",
                    Backend::status_message(status)
                )));
            }
            Ok(())
        }
    }
}

impl<'t> Drop for Graph<'t> {
    fn drop(&mut self) {
        // SAFETY: ctx is owned by this graph and freed exactly once;
        // output tensors live in their own contexts ('t outlives the
        // graph, so they are still valid — and untouched — here).
        unsafe {
            forge_sys::ggml_free(self.ctx);
        }
    }
}

impl std::fmt::Debug for Graph<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Graph")
            .field("n_nodes", &self.n_nodes())
            .field("capacity", &self.cap)
            .finish_non_exhaustive()
    }
}

/// A compiled CPU execution plan borrowing its [`Graph`].
pub struct Plan<'a> {
    backend: Backend,
    raw: *mut c_void,
    _graph: PhantomData<&'a Graph<'a>>,
}

impl Plan<'_> {
    /// Execute the plan (results land in the graph's output tensors;
    /// re-upload inputs and call again to re-run).
    pub fn compute(&self) -> Result<()> {
        // SAFETY: the plan was created for this backend and borrows
        // a live graph whose tensors and buffers are live; the
        // status code is mapped below.
        unsafe {
            let status = forge_sys::ggml_backend_graph_plan_compute(self.backend.raw(), self.raw);
            if status != forge_sys::status::SUCCESS {
                return Err(Error::backend(format!(
                    "graph plan compute failed: {}",
                    Backend::status_message(status)
                )));
            }
            Ok(())
        }
    }
}

impl Drop for Plan<'_> {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful plan creation and is
        // freed exactly once, on the backend that created it.
        unsafe {
            forge_sys::ggml_backend_graph_plan_free(self.backend.raw(), self.raw);
        }
    }
}

impl std::fmt::Debug for Plan<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Plan").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_floor() {
        assert!(Graph::with_capacity(0).is_err());
        // Lifetimes: the graph borrows nothing yet, so any scope works.
        let graph = Graph::with_capacity(8).unwrap();
        assert_eq!(graph.capacity(), 8);
        assert_eq!(graph.n_nodes(), 0);
    }
}
