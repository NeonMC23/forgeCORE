//! Safe execution contexts over `llama_context`.
//!
//! A [`Context`] owns one native context plus a share of its [`Model`],
//! so the model always outlives every context built from it. Build a
//! [`Batch`], [`decode`](Context::decode) it, read
//! [`Logits`] (or [`Embeddings`] from an embeddings context).
//!
//! [`ContextOptions`] is `#[non_exhaustive]` and exposes the low-level
//! context/batch controls: sizing (`n_ctx`, `n_batch`, `n_ubatch`,
//! `n_seq_max`, output caps), threading, KV cache dtypes,
//! embeddings/pooling/attention selection, and the advisory
//! `offload_kqv`/`op_offload` flags. Sequence surgery lives on the
//! borrowed [`Memory`] handle and byte-oriented
//! snapshots on the `export/import_state` methods. Every
//! numeric sanitize upstream performs (context padding, batch
//! clamping, pooling/attention resolution) is queried back after `open`
//! and reported honestly — [`Context`] never trusts the request.
//!
//! [`decode`](Context::decode) refuses up front what upstream cannot
//! take safely: empty batches, batches larger than the effective
//! `n_batch` (upstream aborts), batches larger than the effective
//! `n_ubatch` under non-causal attention (upstream aborts), and
//! sequence ids outside the effective `n_seq_max` (unless unified KV
//! defers sequence validation to upstream — except on DeepSeek-V4,
//! whose memory keeps per-sequence streams even when unified, so the
//! `n_seq_max` bound holds there too). Token ranges,
//! position consecutiveness, and KV capacity stay upstream's domain and
//! map to [`Error::decode`] — nothing is hidden or retried.
//!
//! Like all ForgeCore handles, `Context` is `!Send + !Sync` and frees
//! its native context on drop.

use crate::batch::{Batch, SeqId};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::memory::{Memory, SeqState, State};
use crate::model::{Model, ModelInner, PageSample, PageSampleStatus};
use crate::residency::ContextLifetimeState;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::rc::Rc;

/// Pooled-embedding mode (upstream `llama_pooling_type`).
///
/// `Unspecified` (the default) resolves to the model default, or
/// `None` when the model has none. Pooling only produces output on
/// embeddings contexts; on other contexts it is accepted and ignored.
/// `Rank` attaches a classifier head whose per-sequence rows hold
/// [`Model::n_cls_out`](crate::model::Model::n_cls_out) floats instead
/// of [`Model::n_embd_out`](crate::model::Model::n_embd_out).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PoolingType {
    /// Resolve from the model (or `None`). The default.
    #[default]
    Unspecified,
    /// No pooling: per-token outputs only.
    None,
    /// Mean-pool token rows per sequence.
    Mean,
    /// First-token (CLS) row per sequence.
    Cls,
    /// Last-token row per sequence.
    Last,
    /// Classifier-head ranks per sequence.
    Rank,
}

impl PoolingType {
    fn as_native(self) -> c_int {
        match self {
            Self::Unspecified => forge_sys::pooling_type::UNSPECIFIED,
            Self::None => forge_sys::pooling_type::NONE,
            Self::Mean => forge_sys::pooling_type::MEAN,
            Self::Cls => forge_sys::pooling_type::CLS,
            Self::Last => forge_sys::pooling_type::LAST,
            Self::Rank => forge_sys::pooling_type::RANK,
        }
    }

    /// Map a resolved upstream value back. Unknown values (only
    /// reachable via a corrupt model default) behave like `None`:
    /// upstream's output switch matches no arm and extracts nothing.
    fn from_native(value: c_int) -> Self {
        match value {
            forge_sys::pooling_type::MEAN => Self::Mean,
            forge_sys::pooling_type::CLS => Self::Cls,
            forge_sys::pooling_type::LAST => Self::Last,
            forge_sys::pooling_type::RANK => Self::Rank,
            _ => Self::None,
        }
    }
}

/// Attention masking (upstream `llama_attention_type`).
///
/// `Unspecified` (the default) resolves to the model default (causal
/// unless the model says otherwise). `NonCausal` additionally requires
/// every decoded batch to fit in one micro-batch (`n_tokens <=
/// n_ubatch`); upstream aborts otherwise, so [`decode`](Context::decode)
/// enforces it up front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AttentionType {
    /// Resolve from the model. The default.
    #[default]
    Unspecified,
    /// Causal (autoregressive) masking.
    Causal,
    /// Non-causal (bidirectional) masking.
    NonCausal,
}

impl AttentionType {
    fn as_native(self) -> c_int {
        match self {
            Self::Unspecified => forge_sys::attention_type::UNSPECIFIED,
            Self::Causal => forge_sys::attention_type::CAUSAL,
            Self::NonCausal => forge_sys::attention_type::NON_CAUSAL,
        }
    }
}

/// Flash-attention selection (upstream `llama_flash_attn_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FlashAttnType {
    /// Upstream default: enabled except where unsupported.
    #[default]
    Auto,
    /// Never use flash attention.
    Disabled,
    /// Always use flash attention (required for tensor-split models
    /// and quantized V caches; upstream fails context creation
    /// otherwise).
    Enabled,
}

impl FlashAttnType {
    fn as_native(self) -> c_int {
        match self {
            Self::Auto => forge_sys::flash_attn_type::AUTO,
            Self::Disabled => forge_sys::flash_attn_type::DISABLED,
            Self::Enabled => forge_sys::flash_attn_type::ENABLED,
        }
    }
}

/// Context creation options. `#[non_exhaustive]` so future phases can
/// add fields without breaking callers.
///
/// Every field defaults to the pre-P4 behavior: causal decoder
/// context, upstream batch sizing and thread lockstep. Requested
/// values are sanitized by upstream at init (context padding, batch
/// clamping, pooling/attention resolution); the effective values are
/// queried back and exposed on [`Context`].
///
/// Deliberately absent (kept at upstream defaults): RoPE/YaRN tuning,
/// recurrent rollback snapshots, MTP context type, SWA sizing and
/// full-cache selection, performance counters (P7), callbacks, and
/// backend sampler chains. See the P4/P5 engineering reports for the
/// per-field rationale.
/// Opt-in scheduler eval tracing. Enabling this makes the pinned scheduler
/// compute observed nodes in groups and synchronize after each group; it is
/// diagnostic instrumentation, not a residency hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextTraceOptions {
    pub max_events: usize,
}

impl Default for ContextTraceOptions {
    fn default() -> Self {
        Self { max_events: 4096 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracePhase {
    /// Scheduler asks whether this node should be observed.
    Requested,
    /// The observed node group has been computed and the backend synchronized.
    CompletedSynchronized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceCompletion {
    NotCompleted,
    BackendSynchronized,
}

/// One bounded graph-trace event. Names/op/dependencies are copied; no native
/// pointers are retained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEvent {
    pub execution_id: u64,
    pub sequence: u64,
    /// Transient native tensor identity for this graph; not durable beyond the
    /// trace.
    pub node_id: u64,
    pub node_name: Option<String>,
    pub op_name: Option<String>,
    pub bytes: u64,
    pub dependencies: Vec<String>,
    /// Transient native identities of source tensors.
    pub dependency_ids: Vec<u64>,
    pub phase: TracePhase,
    pub completion: TraceCompletion,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TraceSummary {
    pub executions: u64,
    pub requested_events: u64,
    pub completed_events: u64,
    pub dropped_events: u64,
    pub max_events: usize,
    pub retained_events: usize,
}

/// Maximum number of CPU node-dispatch events retained in one context.
pub const MAX_NODE_DISPATCH_EVENTS: usize = 65_536;
pub const MAX_NODE_DISPATCH_SAMPLE_PAGES: u64 = 4096;
const NODE_NAME_CAPACITY: usize = 128;
const OP_NAME_CAPACITY: usize = 64;
const SOURCE_NAME_CAPACITY: usize = 128;
const MAX_DISPATCH_SOURCES: usize = 8;

/// Opt-in CPU node-dispatch observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDispatchOptions {
    /// Maximum retained events. Must be in `1..=MAX_NODE_DISPATCH_EVENTS`.
    pub max_events: usize,
    /// Maximum pages sampled per declared source tensor. Zero disables
    /// pre-dispatch residency sampling.
    pub sample_pages: u64,
}

impl Default for NodeDispatchOptions {
    fn default() -> Self {
        Self {
            max_events: 4096,
            sample_pages: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeDispatchBackend {
    Cpu,
}

/// One CPU graph-node dispatch-start observation.
///
/// This is emitted before workers enter the node kernel. It does not prove
/// that every declared source was read, that a page fault occurred, that I/O
/// happened, or that the node completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDispatchEvent {
    pub model_instance_id: u64,
    pub context_id: u64,
    pub execution_sequence: u64,
    pub event_sequence: u64,
    pub node_index: u32,
    pub node_count: u32,
    pub n_threads: u32,
    pub backend: NodeDispatchBackend,
    pub node_name: Option<String>,
    pub node_name_truncated: bool,
    pub op_name: Option<String>,
    pub op_name_truncated: bool,
    /// Declared source tensor names, in native source order, truncated to
    /// [`MAX_DISPATCH_SOURCES`] entries.
    pub source_names: Vec<String>,
    /// Bounded per-source residency observations made before dispatch.
    pub source_samples: Vec<NodeDispatchSource>,
    pub sources_truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDispatchSource {
    pub name: String,
    pub sample: Option<PageSample>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NodeDispatchSummary {
    pub executions: u64,
    pub events: u64,
    pub dropped_events: u64,
    pub retained_events: usize,
    pub max_events: usize,
}

#[derive(Clone, Copy)]
struct RawDispatchSource {
    name: [u8; SOURCE_NAME_CAPACITY],
    len: u16,
    truncated: bool,
    sample: PageSample,
}

impl Default for RawDispatchSource {
    fn default() -> Self {
        Self {
            name: [0; SOURCE_NAME_CAPACITY],
            len: 0,
            truncated: false,
            sample: PageSample::default(),
        }
    }
}

#[derive(Clone, Copy)]
struct RawDispatchEvent {
    model_instance_id: u64,
    context_id: u64,
    execution_sequence: u64,
    event_sequence: u64,
    node_index: u32,
    node_count: u32,
    n_threads: u32,
    node_name: [u8; NODE_NAME_CAPACITY],
    node_name_len: u16,
    node_name_truncated: bool,
    op_name: [u8; OP_NAME_CAPACITY],
    op_name_len: u16,
    op_name_truncated: bool,
    sources: [RawDispatchSource; MAX_DISPATCH_SOURCES],
    source_count: u8,
    sources_truncated: bool,
}

impl Default for RawDispatchEvent {
    fn default() -> Self {
        Self {
            model_instance_id: 0,
            context_id: 0,
            execution_sequence: 0,
            event_sequence: 0,
            node_index: 0,
            node_count: 0,
            n_threads: 0,
            node_name: [0; NODE_NAME_CAPACITY],
            node_name_len: 0,
            node_name_truncated: false,
            op_name: [0; OP_NAME_CAPACITY],
            op_name_len: 0,
            op_name_truncated: false,
            sources: [RawDispatchSource::default(); MAX_DISPATCH_SOURCES],
            source_count: 0,
            sources_truncated: false,
        }
    }
}

struct DispatchBuffer {
    max_events: usize,
    next_event_sequence: u64,
    execution_sequence: u64,
    total_events: u64,
    dropped_events: u64,
    events: VecDeque<RawDispatchEvent>,
}

struct DispatchState {
    buffer: RefCell<DispatchBuffer>,
    model_instance_id: u64,
    context_id: u64,
    model: *const forge_sys::llama_model,
    sample_pages: u64,
}

impl DispatchState {
    fn new(
        model_instance_id: u64,
        context_id: u64,
        model: *const forge_sys::llama_model,
        sample_pages: u64,
        max_events: usize,
    ) -> Self {
        Self {
            buffer: RefCell::new(DispatchBuffer {
                max_events,
                next_event_sequence: 0,
                execution_sequence: 0,
                total_events: 0,
                dropped_events: 0,
                events: VecDeque::with_capacity(max_events),
            }),
            model_instance_id,
            context_id,
            model,
            sample_pages,
        }
    }

    fn begin_execution(&self) {
        if let Ok(mut buffer) = self.buffer.try_borrow_mut() {
            buffer.execution_sequence = buffer.execution_sequence.saturating_add(1);
        }
    }

    fn record(&self, node: *mut forge_sys::ggml_tensor, index: i32, count: i32, threads: i32) {
        let Ok(mut buffer) = self.buffer.try_borrow_mut() else {
            return;
        };
        let mut event = RawDispatchEvent {
            model_instance_id: self.model_instance_id,
            context_id: self.context_id,
            execution_sequence: buffer.execution_sequence,
            event_sequence: buffer.next_event_sequence,
            node_index: u32::try_from(index).unwrap_or(0),
            node_count: u32::try_from(count).unwrap_or(0),
            n_threads: u32::try_from(threads).unwrap_or(0),
            ..RawDispatchEvent::default()
        };
        if !node.is_null() {
            // SAFETY: the scheduler owns the node for this callback. All
            // pointers are copied immediately; no native pointer is retained.
            let (len, truncated) =
                unsafe { copy_c_name(forge_sys::ggml_get_name(node), &mut event.node_name) };
            event.node_name_len = len;
            event.node_name_truncated = truncated;
            let (len, truncated) =
                unsafe { copy_c_name(forge_sys::llama_tensor_op_name(node), &mut event.op_name) };
            event.op_name_len = len;
            event.op_name_truncated = truncated;
            let source_count = unsafe { forge_sys::llama_tensor_src_count(node) };
            let source_total = usize::try_from(source_count).unwrap_or(0);
            let stored = source_total.min(MAX_DISPATCH_SOURCES);
            event.source_count = stored as u8;
            event.sources_truncated = source_total > stored;
            for source_index in 0..stored {
                let index = i32::try_from(source_index).unwrap_or(0);
                let (len, truncated) = unsafe {
                    copy_c_name(
                        forge_sys::llama_tensor_src_name(node, index),
                        &mut event.sources[source_index].name,
                    )
                };
                event.sources[source_index].len = len;
                event.sources[source_index].truncated = truncated;
                if self.sample_pages > 0 && !self.model.is_null() {
                    let mut sample = forge_sys::llama_page_sample_result::default();
                    let name_ptr = unsafe { forge_sys::llama_tensor_src_name(node, index) };
                    if !name_ptr.is_null() {
                        let rc = unsafe {
                            forge_sys::llama_model_sample_tensor_pages(
                                self.model,
                                name_ptr,
                                self.model_instance_id,
                                self.sample_pages,
                                &mut sample,
                            )
                        };
                        event.sources[source_index].sample = page_sample_from_raw(rc, sample);
                    }
                }
            }
        }
        buffer.next_event_sequence = buffer.next_event_sequence.saturating_add(1);
        buffer.total_events = buffer.total_events.saturating_add(1);
        if buffer.events.len() >= buffer.max_events {
            buffer.events.pop_front();
            buffer.dropped_events = buffer.dropped_events.saturating_add(1);
        }
        buffer.events.push_back(event);
    }

    fn events(&self) -> Vec<NodeDispatchEvent> {
        let Ok(buffer) = self.buffer.try_borrow() else {
            return Vec::new();
        };
        buffer.events.iter().map(raw_to_event).collect()
    }

    fn take_events(&self) -> Vec<NodeDispatchEvent> {
        let Ok(mut buffer) = self.buffer.try_borrow_mut() else {
            return Vec::new();
        };
        buffer
            .events
            .drain(..)
            .map(|raw| raw_to_event(&raw))
            .collect()
    }

    fn clear(&self) {
        if let Ok(mut buffer) = self.buffer.try_borrow_mut() {
            buffer.events.clear();
            buffer.dropped_events = 0;
        }
    }

    fn summary(&self) -> NodeDispatchSummary {
        let Ok(buffer) = self.buffer.try_borrow() else {
            return NodeDispatchSummary::default();
        };
        NodeDispatchSummary {
            executions: buffer.execution_sequence,
            events: buffer.total_events,
            dropped_events: buffer.dropped_events,
            retained_events: buffer.events.len(),
            max_events: buffer.max_events,
        }
    }
}

unsafe fn copy_c_name(pointer: *const c_char, destination: &mut [u8]) -> (u16, bool) {
    if pointer.is_null() {
        return (0, false);
    }
    // SAFETY: native names are NUL-terminated for the callback duration.
    let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes();
    let copied = bytes.len().min(destination.len());
    destination[..copied].copy_from_slice(&bytes[..copied]);
    (copied as u16, bytes.len() > copied)
}

fn text_from_raw(bytes: &[u8], len: u16) -> Option<String> {
    let len = (len as usize).min(bytes.len());
    (len > 0).then(|| String::from_utf8_lossy(&bytes[..len]).into_owned())
}

fn page_sample_from_raw(rc: i32, raw: forge_sys::llama_page_sample_result) -> PageSample {
    let status = if rc != 0 {
        PageSampleStatus::Error
    } else {
        match raw.error {
            0 if raw.truncated != 0 || raw.unknown_pages > 0 => PageSampleStatus::Partial,
            0 => PageSampleStatus::Complete,
            95 => PageSampleStatus::Unsupported,
            116 => PageSampleStatus::Stale,
            2 => PageSampleStatus::Missing,
            22 => PageSampleStatus::Error,
            _ => PageSampleStatus::Unknown,
        }
    };
    PageSample {
        requested_pages: raw.requested_pages,
        sampled_pages: raw.sampled_pages,
        resident_pages: raw.resident_pages,
        nonresident_pages: raw.nonresident_pages,
        unknown_pages: raw.unknown_pages,
        page_size: raw.page_size,
        max_pages: raw.max_pages,
        mapping_id: raw.mapping_id,
        mapping_generation: raw.mapping_generation,
        truncated: raw.truncated != 0,
        error: raw.error,
        status,
    }
}

fn raw_to_event(raw: &RawDispatchEvent) -> NodeDispatchEvent {
    let mut source_names = Vec::with_capacity(raw.source_count as usize);
    let mut source_samples = Vec::with_capacity(raw.source_count as usize);
    for source in raw.sources.iter().take(raw.source_count as usize) {
        if let Some(name) = text_from_raw(&source.name, source.len) {
            source_names.push(name.clone());
            source_samples.push(NodeDispatchSource {
                name,
                sample: (source.sample.status != PageSampleStatus::Unknown
                    || source.sample.requested_pages > 0
                    || source.sample.error != 0)
                    .then_some(source.sample),
            });
        }
    }
    NodeDispatchEvent {
        model_instance_id: raw.model_instance_id,
        context_id: raw.context_id,
        execution_sequence: raw.execution_sequence,
        event_sequence: raw.event_sequence,
        node_index: raw.node_index,
        node_count: raw.node_count,
        n_threads: raw.n_threads,
        backend: NodeDispatchBackend::Cpu,
        node_name: text_from_raw(&raw.node_name, raw.node_name_len),
        node_name_truncated: raw.node_name_truncated,
        op_name: text_from_raw(&raw.op_name, raw.op_name_len),
        op_name_truncated: raw.op_name_truncated,
        source_names,
        source_samples,
        sources_truncated: raw.sources_truncated,
    }
}

unsafe extern "C" fn dispatch_callback(
    node: *mut forge_sys::ggml_tensor,
    node_index: c_int,
    node_count: c_int,
    n_threads: c_int,
    user_data: *mut c_void,
) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: user_data is the Box<DispatchState> owned by the live Context.
    // The callback performs only bounded copies and a bounded push.
    let state = unsafe { &*(user_data as *const DispatchState) };
    state.record(node, node_index, node_count, n_threads);
}

struct TraceBuffer {
    max_events: usize,
    execution_id: u64,
    sequence: u64,
    requested_events: u64,
    completed_events: u64,
    dropped_events: u64,
    events: VecDeque<TraceEvent>,
}

struct TraceState {
    buffer: RefCell<TraceBuffer>,
}

impl TraceState {
    fn new(max_events: usize) -> Self {
        Self {
            buffer: RefCell::new(TraceBuffer {
                max_events: max_events.max(1),
                execution_id: 0,
                sequence: 0,
                requested_events: 0,
                completed_events: 0,
                dropped_events: 0,
                events: VecDeque::new(),
            }),
        }
    }

    fn begin_execution(&self) {
        self.buffer.borrow_mut().execution_id += 1;
    }

    fn record(&self, tensor: *mut forge_sys::ggml_tensor, ask: bool) {
        if tensor.is_null() {
            return;
        }
        // SAFETY: the scheduler owns the tensor for the duration of the
        // callback; names and sizes are copied immediately.
        let node_name = unsafe { name_of(forge_sys::ggml_get_name(tensor)) };
        let op_name = unsafe { name_of(forge_sys::llama_tensor_op_name(tensor)) };
        let bytes = unsafe { forge_sys::ggml_nbytes(tensor) as u64 };
        let node_id = tensor as usize as u64;
        let mut dependencies = Vec::new();
        let mut dependency_ids = Vec::new();
        // SAFETY: read-only node metadata during the scheduler callback.
        let source_count = unsafe { forge_sys::llama_tensor_src_count(tensor) };
        for index in 0..source_count {
            if let Some(name) = unsafe { name_of(forge_sys::llama_tensor_src_name(tensor, index)) }
            {
                dependencies.push(name);
            }
            let id = unsafe { forge_sys::llama_tensor_src_id(tensor, index) };
            if id != 0 {
                dependency_ids.push(id);
            }
        }
        let mut buffer = self.buffer.borrow_mut();
        let phase = if ask {
            buffer.requested_events += 1;
            TracePhase::Requested
        } else {
            buffer.completed_events += 1;
            TracePhase::CompletedSynchronized
        };
        let event = TraceEvent {
            execution_id: buffer.execution_id,
            sequence: buffer.sequence,
            node_id,
            node_name,
            op_name,
            bytes,
            dependencies,
            dependency_ids,
            phase,
            completion: if ask {
                TraceCompletion::NotCompleted
            } else {
                TraceCompletion::BackendSynchronized
            },
        };
        buffer.sequence += 1;
        if buffer.events.len() >= buffer.max_events {
            buffer.events.pop_front();
            buffer.dropped_events += 1;
        }
        buffer.events.push_back(event);
    }

    fn events(&self) -> Vec<TraceEvent> {
        self.buffer.borrow().events.iter().cloned().collect()
    }

    fn take_events(&self) -> Vec<TraceEvent> {
        let mut buffer = self.buffer.borrow_mut();
        buffer.events.drain(..).collect()
    }

    fn clear(&self) {
        let mut buffer = self.buffer.borrow_mut();
        buffer.events.clear();
        buffer.dropped_events = 0;
    }

    fn summary(&self) -> TraceSummary {
        let buffer = self.buffer.borrow();
        TraceSummary {
            executions: buffer.execution_id,
            requested_events: buffer.requested_events,
            completed_events: buffer.completed_events,
            dropped_events: buffer.dropped_events,
            max_events: buffer.max_events,
            retained_events: buffer.events.len(),
        }
    }
}

unsafe extern "C" fn trace_callback(
    tensor: *mut forge_sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if !user_data.is_null() {
        // SAFETY: user_data points at the Box<TraceState> owned by the
        // context for the duration of the callback. No unwind crosses FFI.
        let state = unsafe { &*(user_data as *const TraceState) };
        state.record(tensor, ask);
    }
    // Returning true at both phases keeps execution going and asks the
    // scheduler to observe every node.
    true
}

unsafe fn name_of(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    // SAFETY: native name pointers are NUL-terminated and valid during the
    // callback; copy immediately.
    let value = unsafe { CStr::from_ptr(pointer) }.to_str().ok()?;
    (!value.is_empty()).then(|| value.to_owned())
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ContextOptions {
    /// Context (KV) length. `0` selects the model's training length;
    /// any other value is used verbatim, then padded up to a multiple
    /// of 256 and rounded to a multiple of `n_seq_max` by upstream.
    pub n_ctx: u32,
    /// Worker thread count for generation. Must be at least 1. The
    /// default favors determinism over throughput; RAMforge will set
    /// production counts.
    pub n_threads: u32,
    /// Logical maximum batch size for [`decode`](Context::decode).
    /// Must be at least 1 (upstream aborts while opening a context
    /// with `n_batch == 0`). Under causal attention upstream clamps
    /// this to the context length.
    pub n_batch: u32,
    /// Physical micro-batch size; larger batches are split internally.
    /// `0` selects `n_batch` (upstream rule). Under non-causal
    /// attention every batch must fit in one micro-batch.
    pub n_ubatch: u32,
    /// Maximum distinct sequences per batch (`0` behaves as 1;
    /// upstream refuses values above 256 with a creation failure).
    /// The context length is rounded to a multiple of this value.
    pub n_seq_max: u32,
    /// Worker thread count for batch processing. `0` (the default)
    /// follows `n_threads`, preserving the historical lockstep; any
    /// other value must be at least 1 and is used verbatim.
    pub n_threads_batch: u32,
    /// Graph-sizing hint: maximum outputs per micro-batch (`0`, the
    /// default, selects `n_batch`). Exceeding it only costs a
    /// re-reserve, never an error.
    pub n_outputs_max: u32,
    /// Graph-sizing hint: maximum outputs per sequence (`0` selects
    /// `n_outputs_max`).
    pub n_outputs_max_per_seq: u32,
    /// Advisory: allow the KQV ops (including KV cache buffers) to use
    /// GPU placement when a GPU is available. Upstream default `true`;
    /// a no-op on CPU-only systems (there is nothing to place) and
    /// never a refusal: this flag permits GPU use, it does not request
    /// GPU execution, so the P3 no-silent-fallback rule does not apply.
    pub offload_kqv: bool,
    /// Advisory: allow the scheduler to move host-weight ops onto a
    /// capable device. Same no-op-on-CPU semantics as `offload_kqv`.
    pub op_offload: bool,
    /// Use one unified KV buffer across input sequences instead of a
    /// per-sequence split. Required for coupled batches (one token
    /// shared across sequences); without it upstream fails such
    /// batches with a native error. Also lifts sequence-id validation
    /// to the native 256-sequence bound (enforced by upstream, since
    /// the constant lives in an internal header ForgeCore does not
    /// duplicate) and reports the full context length per sequence.
    pub kv_unified: bool,
    /// Extract embeddings alongside logits (see [`Embeddings`]).
    /// Embedding-*input* batches decode on any context; this flag only
    /// controls embedding *output* extraction.
    pub embeddings: bool,
    /// Pooled-embedding mode for embedding outputs.
    pub pooling: PoolingType,
    /// Attention masking.
    pub attention: AttentionType,
    /// Flash-attention selection.
    pub flash_attn: FlashAttnType,
    /// Data type for the K cache. Upstream default `F16`.
    /// Combinations upstream cannot build fail context creation
    /// (MLA/DeepSeek-V4 models require K and V to match; a quantized V
    /// cache requires flash attention; quantized block sizes must
    /// divide the head dimensions). Recurrent (Mamba/RWKV)
    /// architectures force `F32` regardless of these fields (verified
    /// in the native memory factory). There is no native getter for
    /// the effective types; the applied width shows up in
    /// [`Context::state_size`].
    pub type_k: DType,
    /// Data type for the V cache. See [`ContextOptions::type_k`].
    pub type_v: DType,
    /// Opt-in bounded graph eval tracing. `None` preserves the original
    /// execution path with no callback installed.
    pub trace: Option<ContextTraceOptions>,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            n_ctx: 0,
            n_threads: 1,
            // Upstream defaults, mirrored so default options keep
            // passing upstream exactly what plain opens always have.
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads_batch: 0,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 1,
            offload_kqv: true,
            op_offload: true,
            kv_unified: false,
            embeddings: false,
            pooling: PoolingType::Unspecified,
            attention: AttentionType::Unspecified,
            flash_attn: FlashAttnType::Auto,
            type_k: DType::F16,
            type_v: DType::F16,
            trace: None,
        }
    }
}

/// Owned logits for one decoded output position.
///
/// An **owned copy**: `values` holds exactly `n_vocab` `f32`s copied
/// out of the native buffer, so later decodes cannot invalidate it.
#[derive(Debug, Clone)]
pub struct Logits {
    values: Vec<f32>,
    n_vocab: usize,
}

impl Logits {
    /// Logit values, one per vocabulary id.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Vocabulary dimension (always equals `values().len()`).
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }
}

/// Owned embedding row for one decoded output.
///
/// An **owned copy** like [`Logits`]: per-token rows hold `n_embd_out`
/// floats; pooled per-sequence rows hold `n_embd_out` floats, or
/// `n_cls_out` floats under [`PoolingType::Rank`].
#[derive(Debug, Clone)]
pub struct Embeddings {
    values: Vec<f32>,
    width: usize,
}

impl Embeddings {
    /// Embedding values for the requested token or sequence.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Row width (always equals `values().len()`).
    pub fn width(&self) -> usize {
        self.width
    }
}

/// An owned libllama execution context bound to one [`Model`].
pub struct Context {
    raw: *mut forge_sys::llama_context,
    // Shared model handle: keeps the native model alive while any
    // context exists. Never read, only kept alive (Drop order frees
    // the context first, then releases this share).
    #[allow(dead_code)]
    model: Rc<ModelInner>,
    // Effective values, queried back from upstream at `open` (never
    // trusted from the request).
    n_ctx: u32,
    n_ctx_seq: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_vocab: u32,
    n_threads: u32,
    n_threads_batch: u32,
    causal_attn: bool,
    pooling: PoolingType,
    kv_unified: bool,
    // DeepSeek-V4 memory detection (read once at `open` from the
    // `general.architecture` metadata): the DSV4 memory class forces
    // per-sequence inner streams even when unified and asserts `seq <
    // n_seq_max` on every sequence op, so the unified relaxation of
    // sequence-id validation never applies to it.
    dsv4_memory: bool,
    n_embd_out: u32,
    n_cls_out: u32,
    // Native parallel-sequence limit (unified seq-id bound) and
    // whether position shifts are legal on this model (refused on
    // multi-position MROPE/IMROPE models, where upstream aborts).
    max_parallel_seqs: u32,
    shift_allowed: bool,
    trace: Option<Box<TraceState>>,
    model_instance_id: u64,
    model_raw: *const forge_sys::llama_model,
    dispatch: Option<Box<DispatchState>>,
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.dispatch.is_some() {
            // The context is exclusively borrowed and no decode is running.
            // Disable the native pointer before freeing the native context.
            unsafe {
                let _ = forge_sys::llama_context_set_node_dispatch_observer(
                    self.raw,
                    None,
                    std::ptr::null_mut(),
                );
            }
        }
        // SAFETY: raw came from a successful init, is freed exactly once;
        // the shared model outlives us via the Rc below (Drop runs before
        // fields are released).
        unsafe { forge_sys::llama_free(self.raw) };
    }
}

impl Context {
    /// Create a context for `model` with explicit [`ContextOptions`].
    ///
    /// Returns [`Error`] for invalid options (`n_threads == 0`,
    /// `n_batch == 0`, unrepresentable thread counts) or when upstream
    /// refuses the configuration (exhausted memory, `n_seq_max > 256`,
    /// tensor-split without flash attention, ...).
    pub fn open(model: &Model, options: &ContextOptions) -> Result<Self> {
        if options.n_threads == 0 {
            return Err(Error::invalid("context n_threads must be at least 1"));
        }
        if options.n_batch == 0 {
            // Not a native failure: upstream aborts inside the
            // constructor (assert on the output sizing), so this must
            // never reach the native call.
            return Err(Error::invalid("context n_batch must be at least 1"));
        }
        let model_instance_id = model.lifetime_state()?.instance_id;
        let n_threads = c_int::try_from(options.n_threads)
            .map_err(|_| Error::invalid("context n_threads too large"))?;
        let threads_batch = if options.n_threads_batch == 0 {
            options.n_threads
        } else {
            options.n_threads_batch
        };
        let n_threads_batch = c_int::try_from(threads_batch)
            .map_err(|_| Error::invalid("context n_threads_batch too large"))?;
        let trace_state = options
            .trace
            .map(|trace| Box::new(TraceState::new(trace.max_events)));
        let trace_ptr = trace_state
            .as_deref()
            .map_or(std::ptr::null_mut(), |state| {
                (state as *const TraceState).cast_mut().cast::<c_void>()
            });
        // SAFETY: default params are valid by construction; the model is
        // live; NULL return (init failure) is checked. The context keeps
        // a share of the model so it cannot outlive it.
        unsafe {
            let mut params = forge_sys::llama_context_default_params();
            params.n_ctx = options.n_ctx;
            params.n_batch = options.n_batch;
            params.n_ubatch = options.n_ubatch;
            params.n_seq_max = options.n_seq_max;
            params.n_outputs_max = options.n_outputs_max;
            params.n_outputs_max_per_seq = options.n_outputs_max_per_seq;
            params.n_threads = n_threads;
            params.n_threads_batch = n_threads_batch;
            params.pooling_type = options.pooling.as_native();
            params.attention_type = options.attention.as_native();
            params.flash_attn_type = options.flash_attn.as_native();
            params.embeddings = options.embeddings;
            params.offload_kqv = options.offload_kqv;
            params.op_offload = options.op_offload;
            params.kv_unified = options.kv_unified;
            params.type_k = options.type_k.ggml_type();
            params.type_v = options.type_v.ggml_type();
            params.cb_eval = trace_state.as_ref().map(|_| {
                trace_callback
                    as unsafe extern "C" fn(*mut forge_sys::ggml_tensor, bool, *mut c_void) -> bool
            });
            params.cb_eval_user_data = trace_ptr;
            let raw = forge_sys::llama_init_from_model(model.raw(), params);
            if raw.is_null() {
                return Err(Error::context("context creation failed"));
            }
            // Query the effective values back: upstream pads, clamps,
            // and resolves the request, and decode-time validation
            // must use the actuals.
            let n_ctx = forge_sys::llama_n_ctx(raw);
            let n_ctx_seq = forge_sys::llama_n_ctx_seq(raw);
            let n_batch = forge_sys::llama_n_batch(raw);
            let n_ubatch = forge_sys::llama_n_ubatch(raw);
            let n_seq_max = forge_sys::llama_n_seq_max(raw);
            let pooling = PoolingType::from_native(forge_sys::llama_pooling_type(raw));
            let max_parallel_seqs = u32::try_from(forge_sys::llama_max_parallel_sequences())
                .map_err(|_| Error::context("native parallel-sequence limit too large"))?;
            // Multi-position (MROPE/IMROPE) models carry 4 positions per
            // embedding and upstream aborts position shifts there, so
            // `Memory` refuses them up front. The rope family is a pure
            // function of the model weights; anything unrecognized is
            // treated as multi-position only when it matches exactly.
            let rope = forge_sys::llama_model_rope_type(model.raw());
            let shift_allowed =
                rope != forge_sys::rope_type::MROPE && rope != forge_sys::rope_type::IMROPE;
            // DeepSeek-V4 detection: `general.architecture ==
            // "deepseek4"` routes to the DSV4 memory class in the
            // native factory (the only such construction site at the
            // pin), and ForgeCore never sets `ctx_type`, so the
            // non-MTP branch always applies. A loaded model always
            // carries a recognized architecture string (the loader
            // throws otherwise), so only an exact match enables the
            // strict bound; anything else keeps the relaxed unified
            // behavior bit-for-bit.
            let dsv4_memory = {
                let mut buf = [0u8; 64];
                let len = forge_sys::llama_model_meta_val_str(
                    model.raw(),
                    c"general.architecture".as_ptr(),
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len(),
                );
                let n = usize::try_from(len).unwrap_or(0);
                n == b"deepseek4".len() && buf.get(..n) == Some(b"deepseek4".as_slice())
            };
            let n_vocab = model.vocab_size()?;
            let n_embd_out = model.n_embd_out()?;
            let n_cls_out = model.n_cls_out();
            // Upstream exposes no getter for the resolved causal flag,
            // so resolve it from what is known: explicit options are
            // exact; `Unspecified` follows the model default, which is
            // causal for decoder models and non-causal for encoders.
            // (A decoder GGUF explicitly marked non-causal would be
            // misclassified under `Unspecified` — the same exposure as
            // the upstream CLI, which validates nothing.)
            let causal_attn = match options.attention {
                AttentionType::Causal => true,
                AttentionType::NonCausal => false,
                AttentionType::Unspecified => !model.has_encoder(),
            };
            Ok(Self {
                raw,
                model: Rc::clone(model.inner()),
                n_ctx,
                n_ctx_seq,
                n_batch,
                n_ubatch,
                n_seq_max,
                n_vocab,
                n_threads: options.n_threads,
                n_threads_batch: threads_batch,
                causal_attn,
                pooling,
                kv_unified: options.kv_unified,
                dsv4_memory,
                n_embd_out,
                n_cls_out,
                max_parallel_seqs,
                shift_allowed,
                trace: trace_state,
                model_instance_id,
                model_raw: model.raw(),
                dispatch: None,
            })
        }
    }

    /// Effective context length (upstream pads the request up to a
    /// multiple of 256 and rounds to a multiple of `n_seq_max`, so
    /// this generally exceeds the requested value).
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Effective per-sequence context length (`n_ctx / n_seq_max`,
    /// padded by upstream).
    pub fn n_ctx_seq(&self) -> u32 {
        self.n_ctx_seq
    }

    /// Effective logical batch limit enforced by [`decode`](Context::decode).
    pub fn n_batch(&self) -> u32 {
        self.n_batch
    }

    /// Effective micro-batch size (also the per-decode token limit
    /// under non-causal attention).
    pub fn n_ubatch(&self) -> u32 {
        self.n_ubatch
    }

    /// Effective maximum distinct sequences per batch.
    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    /// Vocabulary dimension used for [`Logits`].
    pub fn n_vocab(&self) -> u32 {
        self.n_vocab
    }

    /// Worker thread count for generation.
    pub fn n_threads(&self) -> u32 {
        self.n_threads
    }

    /// Worker thread count for batch processing.
    pub fn n_threads_batch(&self) -> u32 {
        self.n_threads_batch
    }

    /// Effective causal-attention state (see [`AttentionType`]).
    pub fn causal_attn(&self) -> bool {
        self.causal_attn
    }

    /// Resolved pooling type (see [`PoolingType`]).
    pub fn pooling(&self) -> PoolingType {
        self.pooling
    }

    /// Change the worker thread counts mid-life. Both must be at least
    /// 1 (upstream consumes them verbatim at compute time, so 0 has no
    /// defined meaning). Takes effect on subsequent decodes.
    pub fn set_n_threads(&mut self, n_threads: u32, n_threads_batch: u32) -> Result<()> {
        if n_threads == 0 || n_threads_batch == 0 {
            return Err(Error::invalid("context thread counts must be at least 1"));
        }
        let gen = c_int::try_from(n_threads)
            .map_err(|_| Error::invalid("context n_threads too large"))?;
        let batch = c_int::try_from(n_threads_batch)
            .map_err(|_| Error::invalid("context n_threads_batch too large"))?;
        // SAFETY: raw is a live context; the setter is infallible and
        // only updates validated fields.
        unsafe { forge_sys::llama_set_n_threads(self.raw, gen, batch) };
        self.n_threads = n_threads;
        self.n_threads_batch = n_threads_batch;
        Ok(())
    }

    /// Switch causal attention mid-life (upstream also exposes this
    /// for mixed flows such as image tokens decoded non-causally).
    /// Takes effect on subsequent decodes; retained KV is not cleared.
    /// Prefer [`AttentionType`] at construction when the choice is
    /// known up front.
    pub fn set_causal_attn(&mut self, causal: bool) {
        // SAFETY: raw is a live context; the setter is infallible.
        unsafe { forge_sys::llama_set_causal_attn(self.raw, causal) };
        self.causal_attn = causal;
    }

    /// Wait for in-flight work to complete. A near-no-op on CPU
    /// execution; required before reading device-side state directly
    /// on GPU execution. Safe to call any time.
    pub fn synchronize(&mut self) {
        // SAFETY: raw is a live context; safe on an idle context.
        unsafe { forge_sys::llama_synchronize(self.raw) };
    }

    /// Native context identity and execution state.
    ///
    /// `synchronized` means no asynchronous graph work remains at this
    /// instant. It does not mean a model weight will not be reused by a
    /// later token or graph.
    pub fn lifetime_state(&self) -> Result<ContextLifetimeState> {
        let mut raw = forge_sys::llama_context_lifetime_state::default();
        // SAFETY: raw is a live context; out points to writable storage.
        let rc = unsafe { forge_sys::llama_context_lifetime_state(self.raw, &mut raw) };
        if rc != 0 {
            return Err(Error::context("native context lifetime query failed"));
        }
        if raw.context_id == 0 {
            return Err(Error::context("native context returned a zero id"));
        }
        Ok(ContextLifetimeState {
            context_id: raw.context_id,
            execution_generation: raw.execution_generation,
            inflight: raw.inflight != 0,
            synchronized: raw.synchronized != 0,
        })
    }

    /// Install or remove the opt-in CPU node-dispatch observer.
    ///
    /// Removal first synchronizes the context, so no callback can be in
    /// flight when the observation buffer is dropped. Returns an error when
    /// the backend does not expose the CPU observer.
    pub fn set_dispatch_observer(&mut self, options: Option<NodeDispatchOptions>) -> Result<()> {
        self.synchronize();
        // SAFETY: raw is live; the setter only changes the CPU backend field.
        unsafe {
            let _ = forge_sys::llama_context_set_node_dispatch_observer(
                self.raw,
                None,
                std::ptr::null_mut(),
            );
        }
        self.dispatch = None;
        let Some(options) = options else {
            return Ok(());
        };
        if options.max_events == 0 || options.max_events > MAX_NODE_DISPATCH_EVENTS {
            return Err(Error::invalid(format!(
                "node dispatch max_events must be in 1..={MAX_NODE_DISPATCH_EVENTS}"
            )));
        }
        if options.sample_pages > MAX_NODE_DISPATCH_SAMPLE_PAGES {
            return Err(Error::invalid(format!(
                "node dispatch sample_pages must be in 0..={MAX_NODE_DISPATCH_SAMPLE_PAGES}"
            )));
        }
        let context_id = self.lifetime_state()?.context_id;
        let state = Box::new(DispatchState::new(
            self.model_instance_id,
            context_id,
            self.model_raw,
            options.sample_pages,
            options.max_events,
        ));
        let pointer = (&*state as *const DispatchState)
            .cast_mut()
            .cast::<c_void>();
        // SAFETY: raw is live; state is boxed and retained in self.dispatch.
        let rc = unsafe {
            forge_sys::llama_context_set_node_dispatch_observer(
                self.raw,
                Some(dispatch_callback),
                pointer,
            )
        };
        if rc != 0 {
            // The observer was not installed; drop the state while no native
            // callback can reference it.
            return Err(Error::context(format!(
                "CPU node-dispatch observer unavailable (code {rc})"
            )));
        }
        self.dispatch = Some(state);
        Ok(())
    }

    /// Whether the CPU node-dispatch observer is currently installed.
    pub fn dispatch_observer_active(&self) -> bool {
        self.dispatch.is_some()
    }

    /// Bounded CPU node-dispatch event summary.
    pub fn dispatch_summary(&self) -> Option<NodeDispatchSummary> {
        self.dispatch.as_ref().map(|state| state.summary())
    }

    /// Copy retained CPU node-dispatch events.
    pub fn dispatch_events(&self) -> Vec<NodeDispatchEvent> {
        self.dispatch
            .as_ref()
            .map(|state| state.events())
            .unwrap_or_default()
    }

    /// Drain retained CPU node-dispatch events.
    pub fn take_dispatch_events(&self) -> Vec<NodeDispatchEvent> {
        self.dispatch
            .as_ref()
            .map(|state| state.take_events())
            .unwrap_or_default()
    }

    /// Clear retained CPU node-dispatch events.
    pub fn clear_dispatch_events(&self) {
        if let Some(state) = &self.dispatch {
            state.clear();
        }
    }

    /// Bounded graph trace summary, or `None` when tracing was not enabled.
    pub fn trace_summary(&self) -> Option<TraceSummary> {
        self.trace.as_ref().map(|trace| trace.summary())
    }

    /// Copy retained trace events. Empty when tracing was not enabled.
    pub fn trace_events(&self) -> Vec<TraceEvent> {
        self.trace
            .as_ref()
            .map(|trace| trace.events())
            .unwrap_or_default()
    }

    /// Drain retained trace events.
    pub fn take_trace_events(&self) -> Vec<TraceEvent> {
        self.trace
            .as_ref()
            .map(|trace| trace.take_events())
            .unwrap_or_default()
    }

    /// Clear retained events and reset the dropped-event counter.
    pub fn clear_trace(&self) {
        if let Some(trace) = &self.trace {
            trace.clear();
        }
    }

    /// Decode one [`Batch`], advancing this context's KV state.
    ///
    /// ForgeCore refuses up front what upstream cannot take safely: an
    /// empty batch, a batch larger than the effective `n_batch`, a
    /// batch larger than the effective `n_ubatch` under non-causal
    /// attention (both are upstream aborts, verified by probe), and
    /// sequence ids outside the effective `n_seq_max` (unless
    /// [`ContextOptions::kv_unified`] defers sequence validation to
    /// upstream, whose 256-sequence bound fails safely — except on
    /// DeepSeek-V4, whose unified memory still asserts `n_seq_max`,
    /// so the bound holds there too). Token ranges,
    /// position consecutiveness, and KV capacity stay upstream's
    /// domain: violations come back as error codes and map to
    /// [`Error::decode`] with the code's meaning — nothing is hidden
    /// or retried.
    pub fn decode(&mut self, batch: &Batch) -> Result<()> {
        if batch.n_tokens() == 0 {
            return Err(Error::invalid("decode: batch is empty"));
        }
        if batch.n_tokens() > self.n_batch as usize {
            return Err(Error::invalid(format!(
                "decode: batch of {} tokens exceeds context n_batch {}",
                batch.n_tokens(),
                self.n_batch
            )));
        }
        if !self.causal_attn && batch.n_tokens() > self.n_ubatch as usize {
            return Err(Error::invalid(format!(
                "decode: batch of {} tokens exceeds context n_ubatch {} \
                 (non-causal attention requires n_ubatch >= n_tokens)",
                batch.n_tokens(),
                self.n_ubatch
            )));
        }
        // Under unified KV, upstream validates sequence ids against
        // its internal 256-sequence bound (a safe native error, not an
        // abort), so there is nothing to pre-check — except on
        // DeepSeek-V4, whose unified memory asserts `n_seq_max`
        // instead (verified SIGABRT); otherwise the effective
        // n_seq_max applies.
        if !self.kv_unified || self.dsv4_memory {
            if let Some(bad) = batch.all_seq_ids().find(|seq| *seq >= self.n_seq_max) {
                return Err(Error::invalid(format!(
                    "decode: seq id {bad} out of range (context n_seq_max {})",
                    self.n_seq_max
                )));
            }
        }
        if let Some(trace) = &self.trace {
            trace.begin_execution();
        }
        if let Some(dispatch) = &self.dispatch {
            dispatch.begin_execution();
        }
        // SAFETY: raw is a live context; the batch view borrows only
        // arrays the `&Batch` owns, which outlive this call; the model
        // is alive via our Rc. Upstream takes the batch by const
        // reference (`llama_context::decode(const llama_batch &)`) and
        // retains no pointers after the call returns.
        let code = unsafe { forge_sys::llama_decode(self.raw, batch.as_sys()) };
        match code {
            0 => Ok(()),
            1 => Err(Error::decode(
                "no KV slot for batch (try a smaller batch or larger context)",
            )),
            2 => Err(Error::decode("execution aborted")),
            -1 => Err(Error::decode("backend rejected the batch as invalid")),
            other => Err(Error::decode(format!(
                "fatal backend failure (code {other})"
            ))),
        }
    }

    /// Copy the logits for one token of the most recently decoded batch.
    ///
    /// `token_index` is the position within that batch (not a token id):
    /// it must be below the batch length and that position must have had
    /// its logits flag set, otherwise upstream returns NULL. Takes `&mut
    /// self` to match upstream (which synchronizes internal state);
    /// returns an owned [`Logits`] copy of `n_vocab` floats, or
    /// [`Error::logits`] when the index has no logits.
    pub fn logits(&mut self, token_index: u32) -> Result<Logits> {
        let index = c_int::try_from(token_index)
            .map_err(|_| Error::invalid("logits token index too large"))?;
        // SAFETY: raw is a live context; NULL (no such output) is
        // checked; on success upstream guarantees a row of exactly
        // n_vocab floats, which is copied before returning.
        unsafe {
            let ptr = forge_sys::llama_get_logits_ith(self.raw, index);
            if ptr.is_null() {
                return Err(Error::logits(format!(
                    "no logits for batch index {token_index}"
                )));
            }
            let values = std::slice::from_raw_parts(ptr, self.n_vocab as usize).to_vec();
            Ok(Logits {
                values,
                n_vocab: self.n_vocab as usize,
            })
        }
    }

    /// Copy the embedding row for one token of the most recently
    /// decoded batch. Requires an embeddings context
    /// ([`ContextOptions::embeddings`]) and a flagged output position,
    /// like [`logits`](Context::logits); returns an owned
    /// [`Embeddings`] row of `n_embd_out` floats, or
    /// [`Error::embeddings`] when the index has no embedding row.
    pub fn embeddings(&mut self, token_index: u32) -> Result<Embeddings> {
        let index = c_int::try_from(token_index)
            .map_err(|_| Error::invalid("embeddings token index too large"))?;
        // SAFETY: raw is a live context; NULL (no such output) is
        // checked; on success upstream guarantees a row of exactly
        // n_embd_out floats, which is copied before returning.
        unsafe {
            let ptr = forge_sys::llama_get_embeddings_ith(self.raw, index);
            if ptr.is_null() {
                return Err(Error::embeddings(format!(
                    "no embedding row for batch index {token_index}"
                )));
            }
            let values = std::slice::from_raw_parts(ptr, self.n_embd_out as usize).to_vec();
            Ok(Embeddings {
                values,
                width: self.n_embd_out as usize,
            })
        }
    }

    /// Copy the pooled embedding row for one sequence of the most
    /// recently decoded batch. Requires an embeddings context with
    /// pooling enabled ([`PoolingType::None`] has no per-sequence rows)
    /// and a sequence that participated in the batch. Rows hold
    /// `n_embd_out` floats, or `n_cls_out` floats under
    /// [`PoolingType::Rank`].
    pub fn embeddings_seq(&mut self, seq: SeqId) -> Result<Embeddings> {
        let id =
            c_int::try_from(seq).map_err(|_| Error::invalid(format!("seq id {seq} too large")))?;
        let width = if self.pooling == PoolingType::Rank {
            self.n_cls_out
        } else {
            self.n_embd_out
        };
        // SAFETY: raw is a live context; NULL (no pooling, or sequence
        // absent) is checked; on success upstream guarantees a row of
        // exactly `width` floats for this pooling mode, which is copied
        // before returning.
        unsafe {
            let ptr = forge_sys::llama_get_embeddings_seq(self.raw, id);
            if ptr.is_null() {
                return Err(Error::embeddings(format!(
                    "no pooled embedding row for sequence {seq}"
                )));
            }
            let values = std::slice::from_raw_parts(ptr, width as usize).to_vec();
            Ok(Embeddings {
                values,
                width: width as usize,
            })
        }
    }

    /// Borrow this context's native memory (KV cache) object for
    /// sequence operations. Returns [`None`] when the model
    /// architecture has no memory object (BERT-family encoders):
    /// sequence surgery is meaningless there, while state snapshots
    /// still work (they carry just the architecture tag).
    pub fn memory(&mut self) -> Option<Memory<'_>> {
        // SAFETY: raw is a live context; the borrowed handle cannot
        // outlive this `&mut` borrow, and NULL (memory-less model) maps
        // to `None` rather than a dangling handle.
        let raw = unsafe { forge_sys::llama_get_memory(self.raw) };
        if raw.is_null() {
            None
        } else {
            Some(Memory::borrow(
                raw,
                self.n_seq_max,
                self.max_parallel_seqs,
                self.kv_unified,
                self.dsv4_memory,
                self.shift_allowed,
            ))
        }
    }

    /// Exact size in bytes of the current whole-context serialized
    /// state (what [`Context::export_state`] would emit).
    ///
    /// This is a *serialization* size — model-arch tag plus live KV
    /// cells — not allocated, resident, or device memory, which the
    /// pinned C API does not report (the breakdown helper is C++-only).
    /// RAMforge must not mistake it for residency.
    pub fn state_size(&mut self) -> Result<u64> {
        // SAFETY: raw is a live context; the native query catches its
        // own failures internally and reports 0, which is never a
        // valid state size (even an empty cache serializes headers).
        let size = unsafe { forge_sys::llama_state_get_size(self.raw) };
        if size == 0 {
            return Err(Error::state("native state size query failed"));
        }
        Ok(size as u64)
    }

    /// Export the whole-context state into caller-owned bytes. The
    /// size query and the copy happen under one `&mut` borrow, so no
    /// decode or sequence op can change the state between them.
    pub fn export_state(&mut self) -> Result<State> {
        let size = usize::try_from(self.state_size()?)
            .map_err(|_| Error::state("native state size exceeds addressable memory"))?;
        let mut bytes = vec![0u8; size];
        // SAFETY: raw is a live context; the buffer is caller-owned
        // with exactly `size` initialized bytes; native writes at most
        // `size` (overrun throws internally and is reported as 0, never
        // overflows) and returns the count written.
        let written =
            unsafe { forge_sys::llama_state_get_data(self.raw, bytes.as_mut_ptr(), size) };
        if written == 0 || written > size {
            return Err(Error::state("native state export failed"));
        }
        bytes.truncate(written);
        Ok(State::from_exported(bytes))
    }

    /// Replace the whole-context state from a snapshot previously
    /// produced by [`Context::export_state`] (possibly on another
    /// context over the same model). Corrupt, truncated, or
    /// architecturally mismatched bytes are rejected with
    /// [`Error::state`]; trailing garbage after a valid prefix is also
    /// rejected (exact consumption). On failure the cache is left
    /// empty (native clears before reporting the error).
    pub fn import_state(&mut self, state: &State) -> Result<()> {
        let bytes = state.as_bytes();
        // SAFETY: raw is a live context; the slice is borrowed for the
        // call; native reads at most `len` (overrun throws internally,
        // reported as 0) and returns the count consumed.
        let read =
            unsafe { forge_sys::llama_state_set_data(self.raw, bytes.as_ptr(), bytes.len()) };
        if read == 0 || read != bytes.len() {
            return Err(Error::state(
                "native state import rejected the bytes \
                 (corrupt, truncated, mismatched, or trailing garbage)",
            ));
        }
        Ok(())
    }

    /// Exact size in bytes of one sequence's serialized state (what
    /// [`Context::export_seq_state`] would emit for `seq`). Same
    /// serialization-not-residency semantics as [`Context::state_size`].
    pub fn seq_state_size(&mut self, seq: SeqId) -> Result<u64> {
        let id = self.check_state_seq(seq, "seq_state_size")?;
        // SAFETY: as in `state_size`; the id satisfies the general
        // bound, which collapses to `n_seq_max` on DeepSeek-V4 (its
        // export asserts that bound via the inner position query).
        let size = unsafe { forge_sys::llama_state_seq_get_size(self.raw, id) };
        if size == 0 {
            return Err(Error::state(format!(
                "native sequence state size query failed for seq {seq}"
            )));
        }
        Ok(size as u64)
    }

    /// Export one sequence's state into caller-owned bytes. The
    /// snapshot restores onto any `seq < n_seq_max` via
    /// [`Context::import_seq_state`], even on another context over the
    /// same model.
    pub fn export_seq_state(&mut self, seq: SeqId) -> Result<SeqState> {
        let id = self.check_state_seq(seq, "export_seq_state")?;
        let size = usize::try_from(self.seq_state_size(seq)?)
            .map_err(|_| Error::state("native sequence state size exceeds addressable memory"))?;
        let mut bytes = vec![0u8; size];
        // SAFETY: as in `export_state`.
        let written =
            unsafe { forge_sys::llama_state_seq_get_data(self.raw, bytes.as_mut_ptr(), size, id) };
        if written == 0 || written > size {
            return Err(Error::state(format!(
                "native sequence state export failed for seq {seq}"
            )));
        }
        bytes.truncate(written);
        Ok(SeqState::from_exported(bytes))
    }

    /// Restore a snapshot produced by [`Context::export_seq_state`]
    /// onto `seq`, replacing that sequence's previous cells. Requires
    /// `seq < n_seq_max` even on unified caches (the DSV4 and
    /// recurrent implementations assert that bound). Same strict
    /// byte-consumption rule as [`Context::import_state`].
    pub fn import_seq_state(&mut self, seq: SeqId, state: &SeqState) -> Result<()> {
        // Tight bound (not the general unified one): restore asserts
        // `n_seq_max` on DSV4/recurrent implementations.
        if seq >= self.n_seq_max {
            return Err(Error::invalid(format!(
                "import_seq_state: seq id {seq} out of range (limit {})",
                self.n_seq_max
            )));
        }
        let id = c_int::try_from(seq)
            .map_err(|_| Error::invalid(format!("import_seq_state: seq id {seq} too large")))?;
        let bytes = state.as_bytes();
        // SAFETY: as in `import_state`; the id satisfies the tightest
        // bound any implementation asserts.
        let read = unsafe {
            forge_sys::llama_state_seq_set_data(self.raw, bytes.as_ptr(), bytes.len(), id)
        };
        if read == 0 || read != bytes.len() {
            return Err(Error::state(format!(
                "native sequence state import rejected the bytes for seq {seq}"
            )));
        }
        Ok(())
    }

    /// Validate a sequence id for state *export*-side calls (general
    /// bound: `n_seq_max`, or the native parallel-sequence limit when
    /// unified — except on DeepSeek-V4, whose unified export asserts
    /// `n_seq_max`). Restore-side calls use the tighter `n_seq_max`
    /// inline; see [`Context::import_seq_state`].
    fn check_state_seq(&self, seq: SeqId, op: &str) -> Result<c_int> {
        let bound = if self.kv_unified && !self.dsv4_memory {
            self.max_parallel_seqs
        } else {
            self.n_seq_max
        };
        if seq >= bound {
            return Err(Error::invalid(format!(
                "{op}: seq id {seq} out of range (limit {bound})"
            )));
        }
        c_int::try_from(seq).map_err(|_| Error::invalid(format!("{op}: seq id {seq} too large")))
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("n_ctx", &self.n_ctx)
            .field("n_vocab", &self.n_vocab)
            .field("n_batch", &self.n_batch)
            .field("n_seq_max", &self.n_seq_max)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_options_default_is_model_ctx_single_thread() {
        let options = ContextOptions::default();
        assert_eq!(options.n_ctx, 0);
        assert_eq!(options.n_threads, 1);
    }

    #[test]
    fn context_options_default_matches_native() {
        let options = ContextOptions::default();
        assert_eq!(options.n_ctx, 0);
        assert_eq!(options.n_threads, 1);
        assert_eq!(options.n_batch, 2048);
        assert_eq!(options.n_ubatch, 512);
        assert_eq!(options.n_seq_max, 1);
        assert_eq!(options.n_threads_batch, 0);
        assert_eq!(options.n_outputs_max, 0);
        assert_eq!(options.n_outputs_max_per_seq, 1);
        assert!(options.offload_kqv);
        assert!(options.op_offload);
        assert!(!options.kv_unified);
        assert!(!options.embeddings);
        assert_eq!(options.pooling, PoolingType::Unspecified);
        assert_eq!(options.attention, AttentionType::Unspecified);
        assert_eq!(options.flash_attn, FlashAttnType::Auto);
        assert_eq!(options.type_k, DType::F16);
        assert_eq!(options.type_v, DType::F16);
    }

    #[test]
    fn pooling_type_maps_to_native() {
        assert_eq!(PoolingType::Unspecified.as_native(), -1);
        assert_eq!(PoolingType::None.as_native(), 0);
        assert_eq!(PoolingType::Mean.as_native(), 1);
        assert_eq!(PoolingType::Cls.as_native(), 2);
        assert_eq!(PoolingType::Last.as_native(), 3);
        assert_eq!(PoolingType::Rank.as_native(), 4);
        assert_eq!(PoolingType::from_native(1), PoolingType::Mean);
        assert_eq!(PoolingType::from_native(4), PoolingType::Rank);
        assert_eq!(PoolingType::from_native(0), PoolingType::None);
        assert_eq!(PoolingType::from_native(99), PoolingType::None);
    }

    #[test]
    fn attention_type_maps_to_native() {
        assert_eq!(AttentionType::Unspecified.as_native(), -1);
        assert_eq!(AttentionType::Causal.as_native(), 0);
        assert_eq!(AttentionType::NonCausal.as_native(), 1);
    }

    #[test]
    fn flash_attn_type_maps_to_native() {
        assert_eq!(FlashAttnType::Auto.as_native(), -1);
        assert_eq!(FlashAttnType::Disabled.as_native(), 0);
        assert_eq!(FlashAttnType::Enabled.as_native(), 1);
    }
}
