//! ForgeCore: thin Rust abstraction over llama.cpp/ggml.
//!
//! ForgeCore is the middle of `RAMforge → ForgeCore → llama.cpp/ggml →
//! hardware`. It owns no compute of its own: [`Backend`] opens ggml
//! backends, [`Tensor`] owns ggml tensors, [`runtime`] executes ggml
//! graphs, [`Model`] loads `.gguf` models through libllama,
//! [`Tokenizer`] encodes text through the model's vocabulary, and
//! [`Context`] runs the minimal CPU inference path (`Model` → `Context`
//! → [`Batch`] → `decode` → [`Logits`]), and [`SamplerChain`] samples
//! token ids from logits. The only `unsafe` in the crate
//! sits at the documented FFI boundary inside each module; no raw C
//! pointers appear in any public API.
//!
//! The pre-pivot scalar kernels survive frozen under [`mod@reference`] as
//! validation oracles — they cross-check ggml numerics in tests and are
//! never on an execution path.

pub mod backend;
pub mod batch;
pub mod buffer;
pub mod context;
pub mod device;
pub mod dtype;
pub mod error;
pub mod gguf;
pub mod graph;
pub mod memory;
pub mod model;
pub mod quant;
pub mod reference;
pub mod residency;
pub mod runtime;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

pub use backend::Backend;
pub use batch::{Batch, BatchBuilder, SeqId, TokenId};
pub use buffer::{Buffer, BufferType};
pub use context::{
    AttentionType, Context, ContextOptions, ContextTraceOptions, Embeddings, FlashAttnType, Logits,
    NodeDispatchBackend, NodeDispatchEvent, NodeDispatchOptions, NodeDispatchSource,
    NodeDispatchSummary, PoolingType, TraceCompletion, TraceEvent, TracePhase, TraceSummary,
    MAX_NODE_DISPATCH_EVENTS, MAX_NODE_DISPATCH_SAMPLE_PAGES,
};
pub use device::{
    enumerate_devices, max_devices, supports_gpu_offload, supports_mlock, supports_mmap,
    DeviceCaps, DeviceInfo, DeviceProps, DeviceType, OpSpec,
};
pub use dtype::DType;
pub use error::{Error, Result};
pub use gguf::{
    inspect as inspect_gguf, inspect_bytes as inspect_gguf_bytes,
    inspect_path as inspect_gguf_path, GgufInspection, GgufValueType, MetadataValue,
    TensorDescriptor,
};
pub use graph::{Graph, NodeInfo, Plan, DEFAULT_GRAPH_CAPACITY};
pub use memory::{Memory, SeqState, State};
pub use model::{
    GpuLayers, Model, ModelLoadMode, ModelOptions, PageReleaseResult, PageSample, PageSampleStatus,
    SplitMode, TensorMappingInfo,
};
pub use quant::{block_spec, row_bytes, tensor_bytes, BlockSpec, QuantFormat};
pub use residency::{
    ContextLifetimeState, ModelLifetimeState, ResidencyError, RESIDENCY_CONTRACT_VERSION,
};
pub use runtime::{
    add, concat, div, get_rows, matmul, mul, norm, rms_norm, rope, scale, silu, soft_max,
    soft_max_ext, sqr, sqrt, sub, RopeMode, RopeParams,
};
pub use sampler::{SampleConfig, SamplerChain};
pub use tensor::{Tensor, MAX_DIMS};
pub use tokenizer::{DecodeOptions, EncodeOptions, SpecialTokens, TokenAttr, Tokenizer, VocabType};
