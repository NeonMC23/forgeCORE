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
use crate::model::{Model, ModelInner, PageSample, PageSampleStatus, TensorMappingInfo};
use crate::residency::ContextLifetimeState;
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
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
pub const MAX_NODE_DISPATCH_CACHE: usize = 65_536;
const NODE_NAME_CAPACITY: usize = 128;
const OP_NAME_CAPACITY: usize = 64;
const SOURCE_NAME_CAPACITY: usize = 128;
const MAX_DISPATCH_SOURCES: usize = 8;

/// Deterministic pre-dispatch sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeSamplingStrategy {
    /// Sample ranges in mapping/page order without adaptive reuse.
    Systematic,
    /// Prioritize ranges without a valid cached observation.
    #[default]
    Targeted,
    /// Reuse fresh complete range observations and sample stale/missing ranges.
    Adaptive,
}

/// Opt-in CPU node-dispatch observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeDispatchOptions {
    /// Maximum retained events. Must be in `1..=MAX_NODE_DISPATCH_EVENTS`.
    pub max_events: usize,
    /// Maximum pages sampled per merged dispatch range. Zero disables
    /// pre-dispatch residency sampling.
    pub sample_pages: u64,
    /// Maximum unique pages sampled per dispatch. Zero disables sampling.
    pub sample_budget: u64,
    pub strategy: NodeSamplingStrategy,
    /// Maximum cached range observations retained by the context.
    pub cache_capacity: usize,
    /// Number of event sequences for which a complete cached range is fresh.
    /// Zero disables cache reuse.
    pub cache_ttl_events: u64,
}

impl Default for NodeDispatchOptions {
    fn default() -> Self {
        Self {
            max_events: 4096,
            sample_pages: 0,
            sample_budget: 0,
            strategy: NodeSamplingStrategy::Targeted,
            cache_capacity: 1024,
            cache_ttl_events: 1024,
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
    /// Bounded per-source mapping ranges and reference counts. Residency is
    /// aggregated once in [`NodeDispatchEvent::dispatch_sample`].
    pub source_samples: Vec<NodeDispatchSource>,
    pub source_references: usize,
    pub sources_truncated: bool,
    /// Deduplicated page observation for the whole dispatch.
    pub dispatch_sample: Option<PageSample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDispatchSource {
    pub name: String,
    pub mapping_id: Option<u64>,
    pub mapping_generation: Option<u64>,
    pub page_start: Option<u64>,
    pub page_end: Option<u64>,
    /// Pages referenced by this source's validated mapping range.
    pub requested_pages: u64,
    /// Pages referenced only by this source within the dispatch.
    pub exclusive_pages: u64,
    /// Pages also referenced by at least one other source in this dispatch.
    pub shared_pages: u64,
    pub sampled_pages: u64,
    pub fresh_pages: u64,
    pub reused_pages: u64,
    pub resident_pages: u64,
    pub nonresident_pages: u64,
    pub unknown_pages: u64,
    pub failed_pages: u64,
    pub skipped_pages: u64,
    pub budget_excluded_pages: u64,
    pub unsampled_pages: u64,
    /// Reserved for future per-source samples. Currently always `None`;
    /// deduplicated residency is reported in
    /// [`NodeDispatchEvent::dispatch_sample`].
    pub sample: Option<PageSample>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NodeDispatchSummary {
    pub executions: u64,
    pub events: u64,
    pub dropped_events: u64,
    pub retained_events: usize,
    pub max_events: usize,
    pub cache_reuses: u64,
    pub cache_skips: u64,
    pub cache_evictions: u64,
    pub budget_total: u64,
    pub budget_remaining: u64,
}

#[derive(Clone, Copy)]
struct RawDispatchSource {
    name: [u8; SOURCE_NAME_CAPACITY],
    len: u16,
    truncated: bool,
    mapping_id: u64,
    mapping_generation: u64,
    page_start: u64,
    page_end: u64,
    page_size: u64,
    requested_pages: u64,
    exclusive_pages: u64,
    shared_pages: u64,
    sampled_pages: u64,
    fresh_pages: u64,
    reused_pages: u64,
    resident_pages: u64,
    nonresident_pages: u64,
    unknown_pages: u64,
    failed_pages: u64,
    skipped_pages: u64,
    budget_excluded_pages: u64,
    unsampled_pages: u64,
}

impl Default for RawDispatchSource {
    fn default() -> Self {
        Self {
            name: [0; SOURCE_NAME_CAPACITY],
            len: 0,
            truncated: false,
            mapping_id: 0,
            mapping_generation: 0,
            page_start: 0,
            page_end: 0,
            page_size: 0,
            requested_pages: 0,
            exclusive_pages: 0,
            shared_pages: 0,
            sampled_pages: 0,
            fresh_pages: 0,
            reused_pages: 0,
            resident_pages: 0,
            nonresident_pages: 0,
            unknown_pages: 0,
            failed_pages: 0,
            skipped_pages: 0,
            budget_excluded_pages: 0,
            unsampled_pages: 0,
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
    source_references: u16,
    sources_truncated: bool,
    dispatch_sample: PageSample,
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
            source_references: 0,
            sources_truncated: false,
            dispatch_sample: PageSample::default(),
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct DispatchRange {
    mapping_id: u64,
    mapping_generation: u64,
    page_start: u64,
    page_end: u64,
    page_size: u64,
}

fn merge_dispatch_ranges(
    mut ranges: [DispatchRange; MAX_DISPATCH_SOURCES],
    count: usize,
) -> ([DispatchRange; MAX_DISPATCH_SOURCES], usize) {
    ranges[..count].sort_unstable_by_key(|range| {
        (
            range.mapping_id,
            range.mapping_generation,
            range.page_start,
            range.page_end,
        )
    });
    let mut merged = [DispatchRange::default(); MAX_DISPATCH_SOURCES];
    let mut merged_count = 0usize;
    for range in ranges.iter().take(count) {
        if *range == DispatchRange::default() {
            continue;
        }
        if merged_count > 0 {
            let previous = &mut merged[merged_count - 1];
            if previous.mapping_id == range.mapping_id
                && previous.mapping_generation == range.mapping_generation
                && range.page_start <= previous.page_end
            {
                previous.page_end = previous.page_end.max(range.page_end);
                continue;
            }
        }
        merged[merged_count] = *range;
        merged_count += 1;
    }
    (merged, merged_count)
}

#[derive(Clone, Copy, Default)]
struct SourceStats {
    referenced_pages: u64,
    exclusive_pages: u64,
    shared_pages: u64,
    sampled_pages: u64,
    fresh_pages: u64,
    reused_pages: u64,
    resident_pages: u64,
    nonresident_pages: u64,
    unknown_pages: u64,
    failed_pages: u64,
    skipped_pages: u64,
    budget_excluded_pages: u64,
    unsampled_pages: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ObservedProvenance {
    Fresh,
    Reused,
    None,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UnobservedClass {
    BudgetExcluded,
    Unsampled,
}

struct DispatchBuffer {
    max_events: usize,
    next_event_sequence: u64,
    execution_sequence: u64,
    total_events: u64,
    dropped_events: u64,
    events: VecDeque<RawDispatchEvent>,
}

#[derive(Clone, Default)]
struct CachedRange {
    mapping_id: u64,
    mapping_generation: u64,
    page_start: u64,
    page_end: u64,
    last_event_sequence: u64,
    sample: PageSample,
    states: Vec<(u64, u8)>,
}

struct DispatchState {
    buffer: RefCell<DispatchBuffer>,
    model_instance_id: u64,
    context_id: u64,
    model: *const forge_sys::llama_model,
    mappings: BTreeMap<String, TensorMappingInfo>,
    sample_pages: u64,
    sample_budget: u64,
    execution_budget: RefCell<u64>,
    strategy: NodeSamplingStrategy,
    cache_capacity: usize,
    cache_ttl_events: u64,
    cache: RefCell<Vec<CachedRange>>,
    state_bytes: RefCell<Vec<u8>>,
    state_offsets: RefCell<Vec<u64>>,
    state_pairs: RefCell<Vec<(u64, u8)>>,
    cache_reuses: RefCell<u64>,
    cache_skips: RefCell<u64>,
    cache_evictions: RefCell<u64>,
}

impl DispatchState {
    fn new(
        model_instance_id: u64,
        context_id: u64,
        model: *const forge_sys::llama_model,
        mappings: BTreeMap<String, TensorMappingInfo>,
        options: NodeDispatchOptions,
    ) -> Self {
        Self {
            buffer: RefCell::new(DispatchBuffer {
                max_events: options.max_events,
                next_event_sequence: 0,
                execution_sequence: 0,
                total_events: 0,
                dropped_events: 0,
                events: VecDeque::with_capacity(options.max_events),
            }),
            model_instance_id,
            context_id,
            model,
            mappings,
            sample_pages: options.sample_pages,
            sample_budget: options.sample_budget,
            execution_budget: RefCell::new(options.sample_budget),
            strategy: options.strategy,
            cache_capacity: options.cache_capacity,
            cache_ttl_events: options.cache_ttl_events,
            cache: RefCell::new(Vec::with_capacity(options.cache_capacity)),
            state_bytes: RefCell::new(Vec::new()),
            state_offsets: RefCell::new(Vec::new()),
            state_pairs: RefCell::new(Vec::new()),
            cache_reuses: RefCell::new(0),
            cache_skips: RefCell::new(0),
            cache_evictions: RefCell::new(0),
        }
    }

    fn begin_execution(&self) {
        if let Ok(mut buffer) = self.buffer.try_borrow_mut() {
            buffer.execution_sequence = buffer.execution_sequence.saturating_add(1);
        }
        if let Ok(mut budget) = self.execution_budget.try_borrow_mut() {
            *budget = self.sample_budget;
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
        let mut ranges = [DispatchRange::default(); MAX_DISPATCH_SOURCES];
        let mut range_count = 0usize;
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
            event.source_references = source_total.min(u16::MAX as usize) as u16;
            event.sources_truncated = source_total > stored;
            for source_index in 0..stored {
                let index = i32::try_from(source_index).unwrap_or(0);
                let name_pointer = unsafe { forge_sys::llama_tensor_src_name(node, index) };
                let (len, truncated) =
                    unsafe { copy_c_name(name_pointer, &mut event.sources[source_index].name) };
                event.sources[source_index].len = len;
                event.sources[source_index].truncated = truncated;
                if !name_pointer.is_null() {
                    // SAFETY: native name is NUL-terminated for the callback.
                    if let Ok(name) = unsafe { CStr::from_ptr(name_pointer) }.to_str() {
                        if let Some(mapping) = self.mappings.get(name) {
                            event.sources[source_index].mapping_id =
                                mapping.mapping_id.unwrap_or(0);
                            event.sources[source_index].mapping_generation =
                                mapping.mapping_generation.unwrap_or(0);
                            event.sources[source_index].page_start = mapping.page_start;
                            event.sources[source_index].page_end = mapping.page_end;
                            event.sources[source_index].page_size = mapping.page_size;
                            if mapping.file_backed
                                && mapping.page_size > 0
                                && mapping.page_end > mapping.page_start
                            {
                                event.sources[source_index].requested_pages =
                                    (mapping.page_end - mapping.page_start) / mapping.page_size;
                                if let Some(mapping_id) = mapping.mapping_id {
                                    ranges[range_count] = DispatchRange {
                                        mapping_id,
                                        mapping_generation: mapping.mapping_generation.unwrap_or(0),
                                        page_start: mapping.page_start,
                                        page_end: mapping.page_end,
                                        page_size: mapping.page_size,
                                    };
                                    range_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Deterministic merge of overlapping page ranges within this dispatch.
        let (merged, merged_count) = merge_dispatch_ranges(ranges, range_count);
        let unique_pages = merged
            .iter()
            .take(merged_count)
            .map(|range| range.page_end.saturating_sub(range.page_start) / range.page_size.max(1))
            .fold(0u64, u64::saturating_add);
        let mut aggregate = PageSample {
            requested_pages: unique_pages,
            page_size: self
                .mappings
                .values()
                .find_map(|mapping| (mapping.page_size > 0).then_some(mapping.page_size))
                .unwrap_or(0),
            mapping_generation: self.model_instance_id,
            ..PageSample::default()
        };
        let sampling_enabled = self.sample_pages > 0 && self.sample_budget > 0;
        let stored_sources = event.source_count as usize;
        let mut source_stats = [SourceStats::default(); MAX_DISPATCH_SOURCES];
        for (index, source) in event.sources.iter().take(stored_sources).enumerate() {
            source_stats[index].referenced_pages = source.requested_pages;
            if source.requested_pages > 0 {
                let shared = source_shared_pages(&event.sources[..stored_sources], index);
                source_stats[index].shared_pages = shared;
                source_stats[index].exclusive_pages = source.requested_pages.saturating_sub(shared);
            }
        }
        if sampling_enabled && !self.model.is_null() {
            let Ok(mut execution_budget) = self.execution_budget.try_borrow_mut() else {
                return;
            };
            let mut budget = *execution_budget;
            for range in merged.iter().take(merged_count) {
                let pages =
                    range.page_end.saturating_sub(range.page_start) / range.page_size.max(1);
                if pages == 0 {
                    continue;
                }
                let cached = self.cached_range(range, buffer.next_event_sequence);
                if self.strategy == NodeSamplingStrategy::Targeted && cached.is_some() {
                    if let Ok(mut skips) = self.cache_skips.try_borrow_mut() {
                        *skips = skips.saturating_add(1);
                    }
                    add_skipped_attribution(
                        range,
                        &event.sources[..stored_sources],
                        &mut source_stats,
                        &mut aggregate,
                    );
                    continue;
                }
                if self.strategy == NodeSamplingStrategy::Adaptive {
                    if let Some(cached) = cached {
                        let mut sample = cached.sample;
                        sample.reused_pages = sample.sampled_pages;
                        sample.fresh_sampled_pages = 0;
                        sample.sampled_at_sequence = buffer.next_event_sequence;
                        if let Ok(mut reuses) = self.cache_reuses.try_borrow_mut() {
                            *reuses = reuses.saturating_add(1);
                        }
                        add_range_attribution(
                            range,
                            &event.sources[..stored_sources],
                            &mut source_stats,
                            &cached.states,
                            ObservedProvenance::Reused,
                            UnobservedClass::Unsampled,
                            &mut aggregate,
                        );
                        copy_aggregate_mapping(&mut aggregate, &sample);
                        continue;
                    }
                }
                if budget == 0 {
                    add_range_attribution(
                        range,
                        &event.sources[..stored_sources],
                        &mut source_stats,
                        &[],
                        ObservedProvenance::None,
                        UnobservedClass::BudgetExcluded,
                        &mut aggregate,
                    );
                    continue;
                }
                let max_pages = pages.min(self.sample_pages).min(budget);
                if max_pages == 0 {
                    add_range_attribution(
                        range,
                        &event.sources[..stored_sources],
                        &mut source_stats,
                        &[],
                        ObservedProvenance::None,
                        UnobservedClass::Unsampled,
                        &mut aggregate,
                    );
                    continue;
                }
                let Ok(mut state_bytes) = self.state_bytes.try_borrow_mut() else {
                    continue;
                };
                let Ok(mut state_offsets) = self.state_offsets.try_borrow_mut() else {
                    continue;
                };
                let Ok(mut state_pairs) = self.state_pairs.try_borrow_mut() else {
                    continue;
                };
                let capacity = usize::try_from(max_pages).unwrap_or(0);
                state_bytes.resize(capacity, 0);
                state_offsets.resize(capacity, 0);
                state_pairs.clear();
                let mut sample = forge_sys::llama_page_sample_result {
                    page_states: state_bytes.as_mut_ptr(),
                    page_offsets: state_offsets.as_mut_ptr(),
                    state_capacity: max_pages,
                    ..forge_sys::llama_page_sample_result::default()
                };
                let rc = unsafe {
                    forge_sys::llama_model_sample_mapping_pages(
                        self.model,
                        range.mapping_id,
                        range.mapping_generation,
                        range.page_start,
                        range.page_end.min(
                            range
                                .page_start
                                .saturating_add(max_pages.saturating_mul(range.page_size)),
                        ),
                        max_pages,
                        &mut sample,
                    )
                };
                let written = usize::try_from(sample.states_written)
                    .unwrap_or(0)
                    .min(capacity);
                state_pairs
                    .extend((0..written).map(|index| (state_offsets[index], state_bytes[index])));
                let sampled = page_sample_from_raw(rc, sample);
                debug_assert_eq!(
                    state_pairs.len() as u64,
                    sampled.sampled_pages,
                    "native page states must cover every sampled page"
                );
                let budget_after = budget.saturating_sub(sampled.sampled_pages);
                let unobserved_class = if budget_after == 0 && sampled.sampled_pages < pages {
                    UnobservedClass::BudgetExcluded
                } else {
                    UnobservedClass::Unsampled
                };
                add_range_attribution(
                    range,
                    &event.sources[..stored_sources],
                    &mut source_stats,
                    &state_pairs,
                    ObservedProvenance::Fresh,
                    unobserved_class,
                    &mut aggregate,
                );
                copy_aggregate_mapping(&mut aggregate, &sampled);
                aggregate.truncated |= sampled.truncated;
                if sampled.error != 0 {
                    aggregate.error = sampled.error;
                }
                budget = budget_after;
                *execution_budget = budget;
                if sampled.status == PageSampleStatus::Complete && state_pairs.len() as u64 == pages
                {
                    self.cache_range(
                        range,
                        buffer.next_event_sequence,
                        sampled,
                        state_pairs.clone(),
                    );
                }
            }
        }
        for (index, source) in event.sources.iter_mut().take(stored_sources).enumerate() {
            let stats = &mut source_stats[index];
            stats.unsampled_pages = stats
                .referenced_pages
                .saturating_sub(stats.sampled_pages)
                .saturating_sub(stats.skipped_pages)
                .saturating_sub(stats.budget_excluded_pages);
            source.exclusive_pages = stats.exclusive_pages;
            source.shared_pages = stats.shared_pages;
            source.sampled_pages = stats.sampled_pages;
            source.fresh_pages = stats.fresh_pages;
            source.reused_pages = stats.reused_pages;
            source.resident_pages = stats.resident_pages;
            source.nonresident_pages = stats.nonresident_pages;
            source.unknown_pages = stats.unknown_pages;
            source.failed_pages = stats.failed_pages;
            source.skipped_pages = stats.skipped_pages;
            source.budget_excluded_pages = stats.budget_excluded_pages;
            source.unsampled_pages = stats.unsampled_pages;
        }
        aggregate.unsampled_pages = aggregate
            .requested_pages
            .saturating_sub(aggregate.sampled_pages)
            .saturating_sub(aggregate.skipped_pages)
            .saturating_sub(aggregate.budget_excluded_pages);
        aggregate.status = if !sampling_enabled || aggregate.sampled_pages == 0 {
            PageSampleStatus::Unknown
        } else if aggregate.sampled_pages < aggregate.requested_pages
            || aggregate.unknown_pages > 0
            || aggregate.failed_pages > 0
            || aggregate.skipped_pages > 0
            || aggregate.budget_excluded_pages > 0
            || aggregate.truncated
        {
            PageSampleStatus::Partial
        } else {
            PageSampleStatus::Complete
        };
        aggregate.truncated |= aggregate.sampled_pages < aggregate.requested_pages
            || aggregate.skipped_pages > 0
            || aggregate.budget_excluded_pages > 0;
        if sampling_enabled {
            event.dispatch_sample = aggregate;
        }
        buffer.next_event_sequence = buffer.next_event_sequence.saturating_add(1);
        buffer.total_events = buffer.total_events.saturating_add(1);
        if buffer.events.len() >= buffer.max_events {
            buffer.events.pop_front();
            buffer.dropped_events = buffer.dropped_events.saturating_add(1);
        }
        buffer.events.push_back(event);
    }

    fn cached_range(&self, range: &DispatchRange, sequence: u64) -> Option<CachedRange> {
        if !matches!(
            self.strategy,
            NodeSamplingStrategy::Targeted | NodeSamplingStrategy::Adaptive
        ) || self.cache_ttl_events == 0
        {
            return None;
        }
        let cache = self.cache.try_borrow().ok()?;
        cache.iter().find_map(|entry| {
            (entry.mapping_id == range.mapping_id
                && entry.mapping_generation == range.mapping_generation
                && entry.page_start == range.page_start
                && entry.page_end == range.page_end
                && sequence.saturating_sub(entry.last_event_sequence) <= self.cache_ttl_events
                && entry.sample.status == PageSampleStatus::Complete
                && entry.states.len() as u64 == entry.sample.requested_pages)
                .then(|| entry.clone())
        })
    }

    fn cache_range(
        &self,
        range: &DispatchRange,
        sequence: u64,
        sample: PageSample,
        states: Vec<(u64, u8)>,
    ) {
        if self.cache_capacity == 0 || self.strategy == NodeSamplingStrategy::Systematic {
            return;
        }
        let Ok(mut cache) = self.cache.try_borrow_mut() else {
            return;
        };
        if let Some(entry) = cache.iter_mut().find(|entry| {
            entry.mapping_id == range.mapping_id
                && entry.mapping_generation == range.mapping_generation
                && entry.page_start == range.page_start
                && entry.page_end == range.page_end
        }) {
            entry.last_event_sequence = sequence;
            entry.sample = sample;
            entry.states = states;
            return;
        }
        if cache.len() >= self.cache_capacity {
            if let Some(oldest) = cache
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_event_sequence)
                .map(|(index, _)| index)
            {
                cache.remove(oldest);
                if let Ok(mut evictions) = self.cache_evictions.try_borrow_mut() {
                    *evictions = evictions.saturating_add(1);
                }
            }
        }
        cache.push(CachedRange {
            mapping_id: range.mapping_id,
            mapping_generation: range.mapping_generation,
            page_start: range.page_start,
            page_end: range.page_end,
            last_event_sequence: sequence,
            sample,
            states,
        });
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
            cache_reuses: self
                .cache_reuses
                .try_borrow()
                .map(|value| *value)
                .unwrap_or(0),
            cache_skips: self
                .cache_skips
                .try_borrow()
                .map(|value| *value)
                .unwrap_or(0),
            cache_evictions: self
                .cache_evictions
                .try_borrow()
                .map(|value| *value)
                .unwrap_or(0),
            budget_total: self.sample_budget,
            budget_remaining: self
                .execution_budget
                .try_borrow()
                .map(|value| *value)
                .unwrap_or(0),
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

fn source_shared_pages(sources: &[RawDispatchSource], index: usize) -> u64 {
    let source = &sources[index];
    if source.requested_pages == 0 || source.page_size == 0 {
        return 0;
    }
    let mut intervals = [(0u64, 0u64); MAX_DISPATCH_SOURCES];
    let mut count = 0usize;
    for (other_index, other) in sources.iter().enumerate() {
        if other_index == index
            || other.mapping_id != source.mapping_id
            || other.mapping_generation != source.mapping_generation
            || other.page_size != source.page_size
            || other.requested_pages == 0
        {
            continue;
        }
        let start = source.page_start.max(other.page_start);
        let end = source.page_end.min(other.page_end);
        if start < end {
            intervals[count] = (start, end);
            count += 1;
        }
    }
    intervals[..count].sort_unstable_by_key(|interval| interval.0);
    let mut merged = [(0u64, 0u64); MAX_DISPATCH_SOURCES];
    let mut merged_count = 0usize;
    for &(start, end) in intervals.iter().take(count) {
        if merged_count > 0 && start <= merged[merged_count - 1].1 {
            merged[merged_count - 1].1 = merged[merged_count - 1].1.max(end);
        } else {
            merged[merged_count] = (start, end);
            merged_count += 1;
        }
    }
    merged
        .iter()
        .take(merged_count)
        .map(|(start, end)| (end - start) / source.page_size)
        .fold(0u64, u64::saturating_add)
}

fn add_range_attribution(
    range: &DispatchRange,
    sources: &[RawDispatchSource],
    stats: &mut [SourceStats; MAX_DISPATCH_SOURCES],
    states: &[(u64, u8)],
    provenance: ObservedProvenance,
    unobserved_class: UnobservedClass,
    global: &mut PageSample,
) {
    let range_pages = range.page_end.saturating_sub(range.page_start) / range.page_size.max(1);
    let observed = u64::try_from(states.len())
        .unwrap_or(u64::MAX)
        .min(range_pages);
    global.sampled_pages = global.sampled_pages.saturating_add(observed);
    match provenance {
        ObservedProvenance::Fresh => {
            global.fresh_sampled_pages = global.fresh_sampled_pages.saturating_add(observed);
        }
        ObservedProvenance::Reused => {
            global.reused_pages = global.reused_pages.saturating_add(observed);
        }
        ObservedProvenance::None => {}
    }
    for &(_, state) in states {
        match state {
            1 => global.resident_pages = global.resident_pages.saturating_add(1),
            0 => global.nonresident_pages = global.nonresident_pages.saturating_add(1),
            2 => global.unknown_pages = global.unknown_pages.saturating_add(1),
            _ => global.failed_pages = global.failed_pages.saturating_add(1),
        }
    }
    let unobserved = range_pages.saturating_sub(observed);
    match unobserved_class {
        UnobservedClass::BudgetExcluded => {
            global.budget_excluded_pages = global.budget_excluded_pages.saturating_add(unobserved);
        }
        UnobservedClass::Unsampled => {
            global.unsampled_pages = global.unsampled_pages.saturating_add(unobserved);
        }
    }
    for (source_index, source) in sources.iter().enumerate() {
        if source.requested_pages == 0
            || source.mapping_id != range.mapping_id
            || source.mapping_generation != range.mapping_generation
            || source.page_size != range.page_size
        {
            continue;
        }
        let start = range.page_start.max(source.page_start);
        let end = range.page_end.min(source.page_end);
        if start >= end {
            continue;
        }
        let source_pages = (end - start) / source.page_size.max(1);
        let mut source_observed = 0u64;
        for &(offset, state) in states {
            if offset < start || offset >= end {
                continue;
            }
            source_observed = source_observed.saturating_add(1);
            match state {
                1 => {
                    stats[source_index].resident_pages =
                        stats[source_index].resident_pages.saturating_add(1)
                }
                0 => {
                    stats[source_index].nonresident_pages =
                        stats[source_index].nonresident_pages.saturating_add(1)
                }
                2 => {
                    stats[source_index].unknown_pages =
                        stats[source_index].unknown_pages.saturating_add(1)
                }
                _ => {
                    stats[source_index].failed_pages =
                        stats[source_index].failed_pages.saturating_add(1)
                }
            }
        }
        let source_observed = source_observed.min(source_pages);
        stats[source_index].sampled_pages = stats[source_index]
            .sampled_pages
            .saturating_add(source_observed);
        match provenance {
            ObservedProvenance::Fresh => {
                stats[source_index].fresh_pages = stats[source_index]
                    .fresh_pages
                    .saturating_add(source_observed);
            }
            ObservedProvenance::Reused => {
                stats[source_index].reused_pages = stats[source_index]
                    .reused_pages
                    .saturating_add(source_observed);
            }
            ObservedProvenance::None => {}
        }
        let source_unobserved = source_pages.saturating_sub(source_observed);
        match unobserved_class {
            UnobservedClass::BudgetExcluded => {
                stats[source_index].budget_excluded_pages = stats[source_index]
                    .budget_excluded_pages
                    .saturating_add(source_unobserved);
            }
            UnobservedClass::Unsampled => {
                stats[source_index].unsampled_pages = stats[source_index]
                    .unsampled_pages
                    .saturating_add(source_unobserved);
            }
        }
    }
}

fn add_skipped_attribution(
    range: &DispatchRange,
    sources: &[RawDispatchSource],
    stats: &mut [SourceStats; MAX_DISPATCH_SOURCES],
    global: &mut PageSample,
) {
    let pages = range.page_end.saturating_sub(range.page_start) / range.page_size.max(1);
    global.skipped_pages = global.skipped_pages.saturating_add(pages);
    for (source_index, source) in sources.iter().enumerate() {
        if source.requested_pages == 0
            || source.mapping_id != range.mapping_id
            || source.mapping_generation != range.mapping_generation
            || source.page_size != range.page_size
        {
            continue;
        }
        let start = range.page_start.max(source.page_start);
        let end = range.page_end.min(source.page_end);
        if start >= end {
            continue;
        }
        stats[source_index].skipped_pages = stats[source_index]
            .skipped_pages
            .saturating_add((end - start) / source.page_size.max(1));
    }
}

fn copy_aggregate_mapping(aggregate: &mut PageSample, sample: &PageSample) {
    if aggregate.mapping_size == 0 {
        aggregate.mapping_id = sample.mapping_id;
        aggregate.mapping_generation = sample.mapping_generation;
        aggregate.mapping_size = sample.mapping_size;
        aggregate.tensor_offset = sample.tensor_offset;
        aggregate.tensor_size = sample.tensor_size;
        aggregate.page_start = sample.page_start;
        aggregate.page_end = sample.page_end;
    }
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
        unsampled_pages: raw.requested_pages.saturating_sub(raw.sampled_pages),
        skipped_pages: 0,
        budget_excluded_pages: 0,
        fresh_sampled_pages: raw.sampled_pages,
        reused_pages: 0,
        resident_pages: raw.resident_pages,
        nonresident_pages: raw.nonresident_pages,
        unknown_pages: raw.unknown_pages,
        failed_pages: raw.failed_pages,
        page_size: raw.page_size,
        max_pages: raw.max_pages,
        mapping_id: raw.mapping_id,
        mapping_generation: raw.mapping_generation,
        mapping_size: raw.mapping_size,
        sampled_at_sequence: 0,
        tensor_offset: raw.tensor_offset,
        tensor_size: raw.tensor_size,
        page_start: raw.page_start,
        page_end: raw.page_end,
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
                mapping_id: (source.requested_pages > 0 || source.mapping_generation != 0)
                    .then_some(source.mapping_id),
                mapping_generation: (source.mapping_generation != 0)
                    .then_some(source.mapping_generation),
                page_start: (source.requested_pages > 0).then_some(source.page_start),
                page_end: (source.requested_pages > 0).then_some(source.page_end),
                requested_pages: source.requested_pages,
                exclusive_pages: source.exclusive_pages,
                shared_pages: source.shared_pages,
                sampled_pages: source.sampled_pages,
                fresh_pages: source.fresh_pages,
                reused_pages: source.reused_pages,
                resident_pages: source.resident_pages,
                nonresident_pages: source.nonresident_pages,
                unknown_pages: source.unknown_pages,
                failed_pages: source.failed_pages,
                skipped_pages: source.skipped_pages,
                budget_excluded_pages: source.budget_excluded_pages,
                unsampled_pages: source.unsampled_pages,
                sample: None,
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
        source_references: raw.source_references as usize,
        sources_truncated: raw.sources_truncated,
        dispatch_sample: (raw.dispatch_sample.requested_pages > 0
            || raw.dispatch_sample.error != 0)
            .then_some(raw.dispatch_sample),
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

/// Independent CUDA-resident layer storage installed in a context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedLayerInfo {
    pub layer: u32,
    pub tensor_count: u32,
    pub device_bytes: u64,
    pub graph_substitutions: u64,
}

/// One tensor in a staged layer and its device-visible byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedTensorInfo {
    pub name: String,
    pub device_ptr: u64,
    pub bytes: u64,
    pub layer: u32,
}

/// Native graph lifecycle phase. Observational: only successful graph
/// invalidation proves that no cached graph can consume previous staged sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphPhase {
    Idle,
    Building,
    Prepared,
    Executing,
    CompletionPending,
    Quiescing,
    Invalidating,
    Invalidated,
    Rebuilding,
    Failed,
    Destroying,
    Unknown(u32),
}

impl GraphPhase {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Idle,
            1 => Self::Building,
            2 => Self::Prepared,
            3 => Self::Executing,
            4 => Self::CompletionPending,
            5 => Self::Quiescing,
            6 => Self::Invalidating,
            7 => Self::Invalidated,
            8 => Self::Rebuilding,
            9 => Self::Failed,
            10 => Self::Destroying,
            other => Self::Unknown(other),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Building => "building",
            Self::Prepared => "prepared",
            Self::Executing => "executing",
            Self::CompletionPending => "completion_pending",
            Self::Quiescing => "quiescing",
            Self::Invalidating => "invalidating",
            Self::Invalidated => "invalidated",
            Self::Rebuilding => "rebuilding",
            Self::Failed => "failed",
            Self::Destroying => "destroying",
            Self::Unknown(_) => "unknown",
        }
    }
}

/// Coarse-grained native staged-residency lifecycle snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingLifecycleState {
    pub graph_phase: GraphPhase,
    pub graph_invalidated: bool,
    pub resident_layers: u32,
    pub pending_layers: u32,
    pub resident_bytes: u64,
    pub pending_bytes: u64,
    pub pool_bytes: u64,
    pub pool_limit_bytes: u64,
    pub pool_buffers: u64,
    pub commit_count: u64,
    pub allocation_count: u64,
    pub release_count: u64,
    pub reuse_count: u64,
    pub staging_failures: u64,
    pub graph_invalidation_count: u64,
    pub graph_rebuild_count: u64,
    pub scheduler_reset_count: u64,
    pub graph_build_ns: u64,
    pub last_graph_build_ns: u64,
    pub invalidation_failures: u64,
    pub quiescence_wait_ns: u64,
    pub last_quiescence_wait_ns: u64,
    pub eviction_supported: bool,
    pub allocation_reuse_supported: bool,
    pub transfer_compute_overlap_supported: bool,
}

fn staging_error(operation: &str, layer: Option<u32>, code: i32) -> Error {
    let kind = match code {
        1 => "not_staged",
        2 => "busy",
        3 => "no_cuda",
        4 => "allocation_failed",
        5 => "invalid_state",
        -1 => "invalid_argument",
        _ => "unknown",
    };
    match layer {
        Some(layer) => Error::staging(format!(
            "{operation} failed for layer {layer}: {kind} (native code {code})"
        )),
        None => Error::staging(format!("{operation} failed: {kind} (native code {code})")),
    }
}

/// One buffer-type entry from the native context memory breakdown.
///
/// `model_bytes`, `context_bytes`, and `compute_bytes` are disjoint within the
/// entry. Entries may share a device; do not sum across buffer types without
/// understanding that the native breakdown is grouped by buffer type, not a
/// unique physical allocation ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferMemory {
    pub buffer_type: String,
    pub device_name: String,
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub compute_bytes: u64,
}

impl BufferMemory {
    pub fn total_bytes(&self) -> u64 {
        self.model_bytes
            .saturating_add(self.context_bytes)
            .saturating_add(self.compute_bytes)
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
    model_mappings: BTreeMap<String, TensorMappingInfo>,
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
                model_mappings: model
                    .tensor_mappings()?
                    .into_iter()
                    .map(|mapping| (mapping.name.clone(), mapping))
                    .collect(),
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

    /// Allocate independent CUDA-resident storage for all `blk.N` weights.
    /// Must be called before the first graph build (native enforces no active
    /// graph); the caller then fills the exposed device pointers.
    pub fn stage_layer_cuda(&self, layer: u32) -> Result<()> {
        let layer =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        // SAFETY: self.raw is a live context; native validates layer/graph state.
        let rc = unsafe { forge_sys::llama_context_stage_layer_cuda(self.raw, layer) };
        if rc != 0 {
            return Err(Error::context(format!(
                "native CUDA staging failed for layer {layer}"
            )));
        }
        Ok(())
    }

    /// Exact aligned CUDA byte total for a layer, without allocating.
    pub fn staged_layer_bytes(&self, layer: u32) -> Result<u64> {
        let layer_i32 =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        let mut bytes = 0u64;
        // SAFETY: live context; native validates layer and writes out.
        let rc = unsafe {
            forge_sys::llama_context_layer_staging_bytes(self.raw, layer_i32, &mut bytes)
        };
        if rc != 0 {
            return Err(Error::context(format!(
                "native staging byte query failed for layer {layer}"
            )));
        }
        Ok(bytes)
    }

    pub fn staged_layer_info(&self, layer: u32) -> Result<StagedLayerInfo> {
        let layer_i32 =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        let mut raw = forge_sys::llama_staged_layer_info::default();
        // SAFETY: live context and writable out.
        let rc =
            unsafe { forge_sys::llama_context_staged_layer_info(self.raw, layer_i32, &mut raw) };
        if rc != 0 {
            return Err(Error::context(format!("layer {layer} is not staged")));
        }
        Ok(StagedLayerInfo {
            layer: u32::try_from(raw.layer).unwrap_or(layer),
            tensor_count: u32::try_from(raw.tensor_count).unwrap_or(0),
            device_bytes: raw.device_bytes,
            graph_substitutions: raw.graph_substitutions,
        })
    }

    pub fn staged_tensor_count(&self, layer: u32) -> Result<u32> {
        let layer =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        // SAFETY: live context; native validates.
        let count = unsafe { forge_sys::llama_context_staged_tensor_count(self.raw, layer) };
        u32::try_from(count).map_err(|_| Error::context(format!("layer {layer} is not staged")))
    }

    pub fn staged_tensor_info(&self, layer: u32, index: u32) -> Result<StagedTensorInfo> {
        let layer_i32 =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        let index_i32 =
            i32::try_from(index).map_err(|_| Error::invalid("tensor index exceeds i32::MAX"))?;
        let mut raw = forge_sys::llama_staged_tensor_info::default();
        // SAFETY: live context, valid index, writable out.
        let rc = unsafe {
            forge_sys::llama_context_staged_tensor_info(self.raw, layer_i32, index_i32, &mut raw)
        };
        if rc != 0 {
            return Err(Error::context(format!(
                "staged tensor {index} in layer {layer} is unavailable"
            )));
        }
        Ok(StagedTensorInfo {
            name: unsafe { CStr::from_ptr(raw.name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            device_ptr: raw.device_ptr,
            bytes: raw.bytes,
            layer,
        })
    }

    /// Quiesce outstanding work and invalidate all cached graph results and
    /// scheduler mappings. After success no cached graph can consume the
    /// current staged sources, so they may be evicted or added before the next
    /// graph build. Idempotent.
    pub fn invalidate_graph(&self) -> Result<()> {
        // SAFETY: self.raw is a live context; native performs quiescence and
        // only resets state it owns.
        let rc = unsafe { forge_sys::llama_context_invalidate_graph(self.raw) };
        if rc != 0 {
            return Err(staging_error("invalidate_graph", None, rc));
        }
        Ok(())
    }

    /// Release one staged layer after successful invalidation. The native layer
    /// refuses with `busy` when any graph, scheduler mapping or execution can
    /// still reference the allocation.
    pub fn evict_layer_cuda(&self, layer: u32) -> Result<()> {
        let layer_i32 =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        // SAFETY: live context; native validates state and layer.
        let rc = unsafe { forge_sys::llama_context_evict_layer_cuda(self.raw, layer_i32) };
        if rc != 0 {
            return Err(staging_error("evict_layer_cuda", Some(layer), rc));
        }
        Ok(())
    }

    /// Native lifecycle phases, staged bytes and mutation counters.
    pub fn staging_state(&self) -> Result<StagingLifecycleState> {
        let mut raw = forge_sys::llama_staging_state::default();
        // SAFETY: live context, writable out.
        let rc = unsafe { forge_sys::llama_context_staging_state(self.raw, &mut raw) };
        if rc != 0 {
            return Err(Error::context("native staging state query failed"));
        }
        Ok(StagingLifecycleState {
            graph_phase: GraphPhase::from_raw(raw.graph_phase),
            graph_invalidated: raw.graph_invalidated != 0,
            resident_layers: raw.resident_layers,
            pending_layers: raw.pending_layers,
            resident_bytes: raw.resident_bytes,
            pending_bytes: raw.pending_bytes,
            pool_bytes: raw.pool_bytes,
            pool_limit_bytes: raw.pool_limit_bytes,
            pool_buffers: raw.pool_buffers,
            commit_count: raw.commit_count,
            allocation_count: raw.allocation_count,
            release_count: raw.release_count,
            reuse_count: raw.reuse_count,
            staging_failures: raw.staging_failures,
            graph_invalidation_count: raw.graph_invalidation_count,
            graph_rebuild_count: raw.graph_rebuild_count,
            scheduler_reset_count: raw.scheduler_reset_count,
            graph_build_ns: raw.graph_build_ns,
            last_graph_build_ns: raw.last_graph_build_ns,
            invalidation_failures: raw.invalidation_failures,
            quiescence_wait_ns: raw.quiescence_wait_ns,
            last_quiescence_wait_ns: raw.last_quiescence_wait_ns,
            eviction_supported: raw.eviction_supported != 0,
            allocation_reuse_supported: raw.allocation_reuse_supported != 0,
            transfer_compute_overlap_supported: raw.transfer_compute_overlap_supported != 0,
        })
    }

    pub fn staged_layer_count(&self) -> Result<u32> {
        // SAFETY: live context.
        let count = unsafe { forge_sys::llama_context_staged_layer_count(self.raw) };
        u32::try_from(count).map_err(|_| Error::context("native staged layer count failed"))
    }

    pub fn staged_layer_at(&self, index: u32) -> Result<StagedLayerInfo> {
        let index_i32 =
            i32::try_from(index).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        let mut raw = forge_sys::llama_staged_layer_info::default();
        // SAFETY: live context, valid index per native check, writable out.
        let rc = unsafe { forge_sys::llama_context_staged_layer_at(self.raw, index_i32, &mut raw) };
        if rc != 0 {
            return Err(Error::context(format!(
                "staged layer at index {index} is unavailable"
            )));
        }
        Ok(StagedLayerInfo {
            layer: u32::try_from(raw.layer).unwrap_or(0),
            tensor_count: u32::try_from(raw.tensor_count).unwrap_or(0),
            device_bytes: raw.device_bytes,
            graph_substitutions: raw.graph_substitutions,
        })
    }

    /// Allocate a layer into the pending double-buffered slot. Pending tensors
    /// are not substituted into an active graph, so this is allowed while the
    /// active selection executes.
    pub fn stage_layer_pending_cuda(&self, layer: u32) -> Result<()> {
        let layer =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        // SAFETY: live context; native validates layer and slot state.
        let rc = unsafe { forge_sys::llama_context_stage_layer_pending_cuda(self.raw, layer) };
        if rc != 0 {
            return Err(staging_error(
                "stage_layer_pending_cuda",
                Some(layer as u32),
                rc,
            ));
        }
        Ok(())
    }

    pub fn pending_layer_info(&self, layer: u32) -> Result<StagedLayerInfo> {
        let layer_i32 =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        let mut raw = forge_sys::llama_staged_layer_info::default();
        // SAFETY: live context and writable out.
        let rc =
            unsafe { forge_sys::llama_context_pending_layer_info(self.raw, layer_i32, &mut raw) };
        if rc != 0 {
            return Err(Error::context(format!("layer {layer} is not pending")));
        }
        Ok(StagedLayerInfo {
            layer,
            tensor_count: u32::try_from(raw.tensor_count).unwrap_or(0),
            device_bytes: raw.device_bytes,
            graph_substitutions: 0,
        })
    }

    pub fn pending_tensor_count(&self, layer: u32) -> Result<u32> {
        let layer =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        // SAFETY: live context; native validates.
        let count = unsafe { forge_sys::llama_context_pending_tensor_count(self.raw, layer) };
        u32::try_from(count).map_err(|_| Error::context(format!("layer {layer} is not pending")))
    }

    pub fn pending_tensor_info(&self, layer: u32, index: u32) -> Result<StagedTensorInfo> {
        let layer_i32 =
            i32::try_from(layer).map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?;
        let index_i32 =
            i32::try_from(index).map_err(|_| Error::invalid("tensor index exceeds i32::MAX"))?;
        let mut raw = forge_sys::llama_staged_tensor_info::default();
        // SAFETY: live context, valid index, writable out.
        let rc = unsafe {
            forge_sys::llama_context_pending_tensor_info(self.raw, layer_i32, index_i32, &mut raw)
        };
        if rc != 0 {
            return Err(Error::context(format!(
                "pending tensor {index} in layer {layer} is unavailable"
            )));
        }
        Ok(StagedTensorInfo {
            name: unsafe { CStr::from_ptr(raw.name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            device_ptr: raw.device_ptr,
            bytes: raw.bytes,
            layer,
        })
    }

    /// Activate the pending selection. Requires a successfully invalidated graph.
    pub fn commit_pending_cuda(&self) -> Result<()> {
        // SAFETY: live context; native enforces invalidation and quiescence.
        let rc = unsafe { forge_sys::llama_context_commit_pending_cuda(self.raw) };
        if rc != 0 {
            return Err(staging_error("commit_pending_cuda", None, rc));
        }
        Ok(())
    }

    /// Atomically remove obsolete active layers, preserve all other active
    /// layers, and append the pending selection. Requires invalidation.
    pub fn commit_pending_delta_cuda(&self, remove_layers: &[u32]) -> Result<()> {
        let mut raw = Vec::with_capacity(remove_layers.len());
        for layer in remove_layers {
            raw.push(
                i32::try_from(*layer)
                    .map_err(|_| Error::invalid("layer index exceeds i32::MAX"))?,
            );
        }
        // SAFETY: live context; native validates the complete delta before mutation.
        let rc = unsafe {
            forge_sys::llama_context_commit_pending_delta_cuda(self.raw, raw.as_ptr(), raw.len())
        };
        if rc != 0 {
            return Err(staging_error("commit_pending_delta_cuda", None, rc));
        }
        Ok(())
    }

    pub fn discard_pending_cuda(&self) -> Result<()> {
        // SAFETY: live context; pending buffers are not graph-referenced.
        let rc = unsafe { forge_sys::llama_context_discard_pending_cuda(self.raw) };
        if rc != 0 {
            return Err(staging_error("discard_pending_cuda", None, rc));
        }
        Ok(())
    }

    pub fn set_staging_pool_limit(&self, bytes: u64) -> Result<()> {
        // SAFETY: live context; native trims retained buffers.
        let rc = unsafe { forge_sys::llama_context_set_staging_pool_limit(self.raw, bytes) };
        if rc != 0 {
            return Err(staging_error("set_staging_pool_limit", None, rc));
        }
        Ok(())
    }

    /// Native model/context/compute byte breakdown grouped by backend buffer
    /// type. The values are a point-in-time native snapshot after the context
    /// exists; they are not a process-wide GPU memory measurement.
    pub fn memory_breakdown(&self) -> Result<Vec<BufferMemory>> {
        // SAFETY: self.raw is a live context for this borrow.
        let count = unsafe { forge_sys::llama_context_memory_breakdown_count(self.raw) };
        if count < 0 {
            return Err(Error::context("native memory breakdown count failed"));
        }
        let mut entries = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut raw = forge_sys::llama_buffer_memory_info::default();
            // SAFETY: index is in range and raw is writable.
            let rc = unsafe {
                forge_sys::llama_context_memory_breakdown_entry(self.raw, index, &mut raw)
            };
            if rc != 0 {
                return Err(Error::context(format!(
                    "native memory breakdown entry {index} failed"
                )));
            }
            // SAFETY: the native bridge NUL-terminates both fixed arrays.
            let buffer_type = unsafe { CStr::from_ptr(raw.buffer_type.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: same as above.
            let device_name = unsafe { CStr::from_ptr(raw.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            entries.push(BufferMemory {
                buffer_type,
                device_name,
                model_bytes: u64::try_from(raw.model_bytes).unwrap_or(u64::MAX),
                context_bytes: u64::try_from(raw.context_bytes).unwrap_or(u64::MAX),
                compute_bytes: u64::try_from(raw.compute_bytes).unwrap_or(u64::MAX),
            });
        }
        Ok(entries)
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
        if options.sample_pages > MAX_NODE_DISPATCH_SAMPLE_PAGES
            || options.sample_budget > MAX_NODE_DISPATCH_SAMPLE_PAGES
        {
            return Err(Error::invalid(format!(
                "node dispatch sample limits must be in 0..={MAX_NODE_DISPATCH_SAMPLE_PAGES}"
            )));
        }
        if options.cache_capacity > MAX_NODE_DISPATCH_CACHE {
            return Err(Error::invalid(format!(
                "node dispatch cache_capacity must be in 0..={MAX_NODE_DISPATCH_CACHE}"
            )));
        }
        let context_id = self.lifetime_state()?.context_id;
        let state = Box::new(DispatchState::new(
            self.model_instance_id,
            context_id,
            self.model_raw,
            self.model_mappings.clone(),
            options,
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

#[cfg(test)]
mod dispatch_range_tests {
    use super::*;

    fn range(mapping: u64, generation: u64, start: u64, end: u64) -> DispatchRange {
        DispatchRange {
            mapping_id: mapping,
            mapping_generation: generation,
            page_start: start,
            page_end: end,
            page_size: 4096,
        }
    }

    fn merge(ranges: &[DispatchRange]) -> ([DispatchRange; MAX_DISPATCH_SOURCES], usize) {
        let mut input = [DispatchRange::default(); MAX_DISPATCH_SOURCES];
        input[..ranges.len()].copy_from_slice(ranges);
        merge_dispatch_ranges(input, ranges.len())
    }

    #[test]
    fn identical_and_overlapping_ranges_merge_to_unique_pages() {
        let (merged, count) = merge(&[range(1, 1, 0, 3), range(1, 1, 1, 4)]);
        assert_eq!(count, 1);
        assert_eq!(merged[0].page_start, 0);
        assert_eq!(merged[0].page_end, 4);
        assert_eq!(merged[0].page_end - merged[0].page_start, 4);
    }

    #[test]
    fn adjacent_ranges_merge_and_distinct_mappings_do_not() {
        let (merged, count) = merge(&[range(1, 1, 0, 2), range(1, 1, 2, 3)]);
        assert_eq!(count, 1);
        assert_eq!(merged[0].page_end, 3);
        let (_, count) = merge(&[range(1, 1, 0, 2), range(2, 1, 0, 2)]);
        assert_eq!(count, 2);
        let (_, count) = merge(&[range(1, 1, 0, 2), range(1, 2, 0, 2)]);
        assert_eq!(count, 2);
    }

    #[test]
    fn empty_and_default_ranges_are_ignored() {
        let (_, count) = merge(&[]);
        assert_eq!(count, 0);
        let (_, count) = merge(&[DispatchRange::default()]);
        assert_eq!(count, 0);
    }
}

#[cfg(test)]
mod dispatch_attribution_tests {
    use super::*;

    fn source(
        mapping: u64,
        generation: u64,
        start: u64,
        end: u64,
        page_size: u64,
    ) -> RawDispatchSource {
        RawDispatchSource {
            mapping_id: mapping,
            mapping_generation: generation,
            page_start: start,
            page_end: end,
            page_size,
            requested_pages: end.saturating_sub(start) / page_size,
            ..RawDispatchSource::default()
        }
    }

    fn range(start: u64, end: u64, page_size: u64) -> DispatchRange {
        DispatchRange {
            mapping_id: 1,
            mapping_generation: 1,
            page_start: start,
            page_end: end,
            page_size,
        }
    }

    #[test]
    #[ignore = "manual range-merge microbenchmark"]
    fn dispatch_range_merge_microbench() {
        let page = 4096u64;
        let mut input = [DispatchRange::default(); MAX_DISPATCH_SOURCES];
        for (index, slot) in input.iter_mut().enumerate() {
            let start = (index as u64 % 3) * 64 * page + (index as u64) * 8 * page;
            *slot = DispatchRange {
                mapping_id: index as u64 / 4,
                mapping_generation: 1,
                page_start: start,
                page_end: start + 32 * page,
                page_size: page,
            };
        }
        let iterations = std::env::var("FORGE_MICROBENCH_ITERATIONS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(100_000);
        let started = std::time::Instant::now();
        let mut merged_count = 0usize;
        for _ in 0..iterations {
            let (merged, count) = merge_dispatch_ranges(input, MAX_DISPATCH_SOURCES);
            merged_count = count;
            std::hint::black_box(merged[0]);
        }
        let elapsed = started.elapsed().as_nanos();
        eprintln!(
            "range_merge iterations={iterations} merged_count={merged_count} ns_per_iter={}",
            elapsed / u128::from(iterations)
        );
    }

    #[test]
    fn shared_page_count_merges_overlaps_without_double_counting() {
        let page = 4096;
        let sources = [
            source(1, 1, 0, 4 * page, page),
            source(1, 1, 0, 4 * page, page),
            source(1, 1, 2 * page, 6 * page, page),
        ];
        assert_eq!(source_shared_pages(&sources[..3], 0), 4);
        assert_eq!(source_shared_pages(&sources[..3], 1), 4);
        assert_eq!(source_shared_pages(&sources[..3], 2), 2);
    }

    #[test]
    fn shared_page_count_is_mapping_and_generation_aware() {
        let page = 4096;
        let sources = [
            source(1, 1, 0, 4 * page, page),
            source(2, 1, 0, 4 * page, page),
            source(1, 2, 0, 4 * page, page),
        ];
        assert_eq!(source_shared_pages(&sources, 0), 0);
        assert_eq!(source_shared_pages(&sources, 1), 0);
        assert_eq!(source_shared_pages(&sources, 2), 0);
    }

    #[test]
    fn attribution_counts_each_shared_page_once_globally() {
        let page = 4096;
        let sources = [
            source(1, 1, 0, 4 * page, page),
            source(1, 1, 0, 2 * page, page),
        ];
        let mut stats = [SourceStats::default(); MAX_DISPATCH_SOURCES];
        stats[0].referenced_pages = 4;
        stats[0].shared_pages = 2;
        stats[0].exclusive_pages = 2;
        stats[1].referenced_pages = 2;
        stats[1].shared_pages = 2;
        stats[1].exclusive_pages = 0;
        let mut global = PageSample {
            requested_pages: 4,
            ..PageSample::default()
        };
        let states = [(0, 1), (page, 1), (2 * page, 0), (3 * page, 0)];
        add_range_attribution(
            &range(0, 4 * page, page),
            &sources,
            &mut stats,
            &states,
            ObservedProvenance::Fresh,
            UnobservedClass::Unsampled,
            &mut global,
        );
        assert_eq!(global.sampled_pages, 4);
        assert_eq!(global.fresh_sampled_pages, 4);
        assert_eq!(global.resident_pages, 2);
        assert_eq!(global.nonresident_pages, 2);
        assert_eq!(global.unknown_pages, 0);
        assert_eq!(global.unsampled_pages, 0);
        assert_eq!(stats[0].sampled_pages, 4);
        assert_eq!(stats[0].resident_pages, 2);
        assert_eq!(stats[0].nonresident_pages, 2);
        assert_eq!(stats[1].sampled_pages, 2);
        assert_eq!(stats[1].resident_pages, 2);
        assert_eq!(stats[1].nonresident_pages, 0);
    }

    #[test]
    fn attribution_keeps_failed_pages_distinct_from_unknown() {
        let page = 4096;
        let sources = [source(1, 1, 0, 4 * page, page)];
        let mut stats = [SourceStats::default(); MAX_DISPATCH_SOURCES];
        stats[0].referenced_pages = 4;
        let mut global = PageSample {
            requested_pages: 4,
            ..PageSample::default()
        };
        let states = [(0, 1), (page, 2), (2 * page, 3)];
        add_range_attribution(
            &range(0, 4 * page, page),
            &sources,
            &mut stats,
            &states,
            ObservedProvenance::Fresh,
            UnobservedClass::Unsampled,
            &mut global,
        );
        assert_eq!(global.sampled_pages, 3);
        assert_eq!(global.resident_pages, 1);
        assert_eq!(global.unknown_pages, 1);
        assert_eq!(global.failed_pages, 1);
        assert_eq!(global.unsampled_pages, 1);
        assert_eq!(stats[0].failed_pages, 1);
        assert_eq!(stats[0].unknown_pages, 1);
    }

    #[test]
    fn attribution_keeps_budget_skip_and_unknown_distinct() {
        let page = 4096;
        let sources = [source(1, 1, 0, 4 * page, page)];
        let mut stats = [SourceStats::default(); MAX_DISPATCH_SOURCES];
        stats[0].referenced_pages = 4;
        let mut global = PageSample {
            requested_pages: 4,
            ..PageSample::default()
        };
        let states = [(0, 1), (page, 2)];
        add_range_attribution(
            &range(0, 4 * page, page),
            &sources,
            &mut stats,
            &states,
            ObservedProvenance::Reused,
            UnobservedClass::Unsampled,
            &mut global,
        );
        assert_eq!(global.sampled_pages, 2);
        assert_eq!(global.reused_pages, 2);
        assert_eq!(global.resident_pages, 1);
        assert_eq!(global.unknown_pages, 1);
        assert_eq!(global.unsampled_pages, 2);
        assert_eq!(global.budget_excluded_pages, 0);

        let mut budget_global = PageSample {
            requested_pages: 4,
            ..PageSample::default()
        };
        let mut budget_stats = [SourceStats::default(); MAX_DISPATCH_SOURCES];
        budget_stats[0].referenced_pages = 4;
        add_range_attribution(
            &range(0, 4 * page, page),
            &sources,
            &mut budget_stats,
            &[],
            ObservedProvenance::None,
            UnobservedClass::BudgetExcluded,
            &mut budget_global,
        );
        assert_eq!(budget_global.budget_excluded_pages, 4);
        assert_eq!(budget_global.unsampled_pages, 0);
        assert_eq!(budget_global.unknown_pages, 0);
        assert_eq!(budget_stats[0].budget_excluded_pages, 4);
    }

    #[test]
    fn skipped_attribution_preserves_shared_sources() {
        let page = 4096;
        let sources = [
            source(1, 1, 0, 4 * page, page),
            source(1, 1, 0, 2 * page, page),
        ];
        let mut stats = [SourceStats::default(); MAX_DISPATCH_SOURCES];
        stats[0].referenced_pages = 4;
        stats[1].referenced_pages = 2;
        let mut global = PageSample {
            requested_pages: 4,
            ..PageSample::default()
        };
        add_skipped_attribution(&range(0, 4 * page, page), &sources, &mut stats, &mut global);
        assert_eq!(global.skipped_pages, 4);
        assert_eq!(stats[0].skipped_pages, 4);
        assert_eq!(stats[1].skipped_pages, 2);
    }
}
