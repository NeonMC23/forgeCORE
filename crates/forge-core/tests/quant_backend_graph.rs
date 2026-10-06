//! Quantized dtypes, backend/buffer/device facts, and graph structure.
//!
//! Three groups: (A) the quant dtype surface — block geometry pinned
//! against ggml, block-divisibility enforcement, the cast-pair
//! envelope (including the Q8_1/Q8_K no-dequantizer exclusions and
//! the K-quant quantize-depth audit boundary), and get_rows table
//! gating; (B) backend facts — CPU buffer-type constants, owned
//! buffer round-trip, alloc-size queries, device registry and CPU
//! props, and the CPU op-support matrix; (C) graph structure —
//! capacity rules, node snapshots, name lookup, backend matching,
//! overflow refusal, and plan re-execution updating outputs.
//!
//! CPU-only, single-threaded for determinism. No model fixture
//! needed. Numeric execution proofs live in `cpu_exec_proof`.

use forge_core::{
    add, get_rows, Backend, DType, DeviceInfo, DeviceType, Graph, OpSpec, Tensor,
    DEFAULT_GRAPH_CAPACITY,
};

fn open_test_cpu() -> Backend {
    let backend = Backend::open_cpu().expect("CPU backend must open");
    backend
        .set_cpu_threads(1)
        .expect("single-threaded CPU for determinism");
    backend
}

fn cpu_device() -> DeviceInfo {
    forge_core::enumerate_devices()
        .into_iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("registry holds a CPU device")
}

// -- A. quant dtypes ----------------------------------------------------------

#[test]
fn dtype_facts_match_ggml() {
    // (quantized, elements-per-block, bytes-per-block). K-quant
    // super-blocks span 256 elements, not 32.
    let table = [
        (DType::F32, false, 1, 4),
        (DType::F16, false, 1, 2),
        (DType::Q4_0, true, 32, 18),
        (DType::Q4_1, true, 32, 20),
        (DType::Q5_0, true, 32, 22),
        (DType::Q5_1, true, 32, 24),
        (DType::Q8_0, true, 32, 34),
        (DType::Q8_1, true, 32, 36),
        (DType::Q2_K, true, 256, 84),
        (DType::Q3_K, true, 256, 110),
        (DType::Q4_K, true, 256, 144),
        (DType::Q5_K, true, 256, 176),
        (DType::Q6_K, true, 256, 210),
        (DType::Q8_K, true, 256, 292),
        (DType::I8, false, 1, 1),
        (DType::I16, false, 1, 2),
        (DType::I32, false, 1, 4),
        (DType::F64, false, 1, 8),
        (DType::BF16, false, 1, 2),
    ];
    for (dtype, quant, block, size) in table {
        assert_eq!(dtype.is_quantized(), quant, "{dtype:?} quantized");
        assert_eq!(dtype.block_len(), block, "{dtype:?} block_len");
        assert_eq!(dtype.type_size(), size, "{dtype:?} type_size");
        assert_eq!(DType::from_ggml(dtype.ggml_type()), Ok(dtype));
    }
    assert_eq!(DType::Q4_K.name(), "Q4_K");
    assert_eq!(DType::BF16.name(), "BF16");
}

#[test]
fn quant_empty_requires_block_rows() {
    let backend = open_test_cpu();
    assert!(
        Tensor::empty(&backend, DType::Q4_0, &[31]).is_err(),
        "ragged Q4_0"
    );
    assert!(Tensor::empty(&backend, DType::Q4_0, &[32]).is_ok());
    assert!(
        Tensor::empty(&backend, DType::Q4_K, &[128]).is_err(),
        "ragged Q4_K"
    );
    assert!(Tensor::empty(&backend, DType::Q4_K, &[256]).is_ok());
    // Q8_1/Q8_K creation is fine (storage is inert); only
    // dequantizing out of them is refused.
    assert!(Tensor::empty(&backend, DType::Q8_1, &[32]).is_ok());
    assert!(Tensor::empty(&backend, DType::Q8_K, &[256]).is_ok());
}

#[test]
fn cast_pair_envelope_rejects_unaudited_pairs() {
    let backend = open_test_cpu();
    let f32t = Tensor::from_f32(&backend, &[256], &[0.5; 256]).expect("f32");

    // Rejected: K-quant targets (super-block depth unaudited), Q8_1
    // (one-way trap: no dequantizer), float/int crosses outside the
    // kernel table, quant-to-quant, and F64/I8/I16 endpoints.
    for target in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::Q8_K,
        DType::Q8_1,
        DType::F64,
        DType::I8,
        DType::I16,
    ] {
        assert!(
            f32t.cast(target).is_err(),
            "F32 -> {} refused",
            target.name()
        );
    }
    let f16t = f32t.cast(DType::F16).expect("F32 -> F16");
    assert!(f16t.cast(DType::I32).is_err(), "F16 -> I32 refused");
    assert!(f16t.cast(DType::Q8_0).is_err(), "F16 -> Q8_0 refused");
    let i32t = Tensor::from_i32(&backend, &[4], &[1, 2, 3, 4]).expect("i32");
    assert!(i32t.cast(DType::F16).is_err(), "I32 -> F16 refused");
    let q8 = Tensor::from_f32(&backend, &[32], &[0.5; 32])
        .expect("f32")
        .cast(DType::Q8_0)
        .expect("F32 -> Q8_0");
    assert!(q8.cast(DType::Q4_0).is_err(), "Q8_0 -> Q4_0 refused");

    // Same-type casts are byte copies (always sound).
    let same = f32t.cast(DType::F32).expect("F32 -> F32");
    assert_eq!(same.to_vec_f32().expect("download"), vec![0.5; 256]);
}

#[test]
fn cast_out_of_q8_family_without_dequantizer_is_refused() {
    let backend = open_test_cpu();
    // Q8_1/Q8_K have no `to_float` row in the ggml type table: the
    // kernel would call a NULL function pointer.
    let q8_1 = Tensor::from_bytes(&backend, DType::Q8_1, &[32], &[0u8; 36]).expect("q8_1 storage");
    assert!(q8_1.cast(DType::F32).is_err(), "Q8_1 -> F32 refused");
    let q8_k =
        Tensor::from_bytes(&backend, DType::Q8_K, &[256], &[0u8; 292]).expect("q8_k storage");
    assert!(q8_k.cast(DType::F32).is_err(), "Q8_K -> F32 refused");
}

#[test]
fn cast_out_of_k_quants_dequantizes() {
    let backend = open_test_cpu();
    // Zeroed K-quant super-blocks decode to exact zeros (zero
    // scales/mins/quants); the cast-out path is live for every
    // K-quant with a dequantizer.
    for (dtype, size) in [
        (DType::Q2_K, 84),
        (DType::Q3_K, 110),
        (DType::Q4_K, 144),
        (DType::Q5_K, 176),
        (DType::Q6_K, 210),
    ] {
        let zeros = vec![0u8; size];
        let q = Tensor::from_bytes(&backend, dtype, &[256], &zeros).expect("k-quant storage");
        let f = q.cast(DType::F32).expect("K-quant -> F32");
        assert_eq!(f.to_vec_f32().expect("download"), vec![0.0; 256]);
    }
}

#[test]
fn get_rows_gates_unsupported_tables() {
    let backend = open_test_cpu();
    let idx = Tensor::from_i32(&backend, &[1], &[0]).expect("indices");
    for (dtype, ne0, nbytes) in [
        (DType::Q8_1, 32, 36),
        (DType::Q8_K, 256, 292),
        (DType::I8, 4, 4),
        (DType::I16, 4, 8),
        (DType::F64, 4, 32),
    ] {
        let zeros = vec![0u8; 2 * nbytes];
        let table = Tensor::from_bytes(&backend, dtype, &[ne0, 2], &zeros).expect("table storage");
        assert!(
            get_rows(&table, &idx).is_err(),
            "get_rows refuses {dtype:?} tables"
        );
    }
    // A supported quant table gathers (zeroed Q4_0 rows decode to 0).
    let table = Tensor::from_bytes(&backend, DType::Q4_0, &[32, 2], &[0u8; 36]).expect("q4 table");
    let out = get_rows(&table, &idx).expect("get_rows on Q4_0");
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape(), &[32], "ne [32, 1] normalizes");
    assert_eq!(out.to_vec_f32().expect("download"), vec![0.0; 32]);
}

// -- B. backend / buffer / device ---------------------------------------------

#[test]
fn cpu_buffer_type_facts() {
    let backend = open_test_cpu();
    let buft = backend.default_buffer_type();
    assert_eq!(buft.name(), "CPU");
    assert_eq!(buft.alignment(), 32, "TENSOR_ALIGNMENT");
    assert_eq!(buft.max_size(), usize::MAX, "no max_size hook -> SIZE_MAX");
    assert!(buft.is_host(), "CPU memory is host-visible");
}

#[test]
fn alloc_buffer_round_trip() {
    let backend = open_test_cpu();
    let buft = backend.default_buffer_type();
    let dummy = buft.alloc_buffer(0).expect("zero-size dummy buffer");
    assert_eq!(dummy.size(), 0);
    let buf = buft.alloc_buffer(1024).expect("1 KiB buffer");
    assert_eq!(buf.size(), 1024);
    assert_eq!(buf.alignment(), 32);
    assert_eq!(buf.max_size(), usize::MAX);
    assert!(buf.is_host());
    assert!(!buf.name().is_empty());
    // `buf` frees on drop (double-free would abort the suite).
}

#[test]
fn tensor_alloc_size_matches_nbytes_on_cpu() {
    let backend = open_test_cpu();
    let buft = backend.default_buffer_type();
    let plain = Tensor::from_f32(&backend, &[4, 3], &[0.0; 12]).expect("plain");
    assert_eq!(buft.tensor_alloc_size(&plain), 48);
    // The base CPU type reports the span for views (metadata query;
    // views still reserve nothing at allocation time).
    let view = plain.view_2d([4, 2], 32, 0).expect("strided view");
    assert_eq!(buft.tensor_alloc_size(&view), 48);
}

#[test]
fn enumerate_devices_finds_cpu() {
    let devices = forge_core::enumerate_devices();
    assert!(!devices.is_empty(), "registry holds at least CPU");
    assert!(devices.iter().any(|d| d.device_type == DeviceType::Cpu));
    assert!(forge_core::max_devices() >= 1);
    // Total build queries (values are build-dependent, must not abort).
    let _ = forge_core::supports_mmap();
    let _ = forge_core::supports_mlock();
    assert!(
        !forge_core::supports_gpu_offload(),
        "CPU-only build offers no GPU offload"
    );
}

#[test]
fn device_props_cpu_sane() {
    let cpu = cpu_device();
    let props = cpu.props().expect("CPU props");
    assert_eq!(props.device_type, DeviceType::Cpu);
    assert!(!props.name.is_empty());
    assert!(props.memory_total > 0, "CPU reports host RAM");
    assert!(props.device_id.is_none(), "CPU has no device id");
    // Every cap field copies out (build-dependent values).
    let _ = (
        props.caps.async_compute,
        props.caps.host_buffer,
        props.caps.buffer_from_host_ptr,
        props.caps.events,
        props.caps.mmap,
    );
    assert!(props.description.is_empty() || !props.description.is_empty());
}

#[test]
fn supports_op_cpu_matrix() {
    let cpu = cpu_device();
    // Every fixed P6 op family is implemented on CPU.
    for op in [
        OpSpec::Add,
        OpSpec::Sub,
        OpSpec::Mul,
        OpSpec::Div,
        OpSpec::Matmul,
        OpSpec::Silu,
        OpSpec::Sqr,
        OpSpec::Sqrt,
        OpSpec::Scale,
        OpSpec::RmsNorm,
        OpSpec::Norm,
        OpSpec::Softmax,
        OpSpec::SoftmaxExt,
        OpSpec::Rope,
    ] {
        assert_eq!(cpu.supports_op(op), Ok(true), "{op:?} on CPU");
    }
    assert_eq!(
        cpu.supports_op(OpSpec::Cast {
            from: DType::F32,
            to: DType::Q8_0
        }),
        Ok(true)
    );
    assert_eq!(
        cpu.supports_op(OpSpec::GetRows { table: DType::Q4_K }),
        Ok(true)
    );
    assert_eq!(
        cpu.supports_op(OpSpec::Concat { dtype: DType::F32 }),
        Ok(true)
    );
    // Probes outside the P6 envelope fail instead of verdicting.
    assert!(cpu
        .supports_op(OpSpec::Cast {
            from: DType::F32,
            to: DType::Q8_1
        })
        .is_err());
    assert!(cpu
        .supports_op(OpSpec::Cast {
            from: DType::F16,
            to: DType::I32
        })
        .is_err());
    assert!(cpu
        .supports_op(OpSpec::GetRows { table: DType::Q8_K })
        .is_err());
    assert!(cpu
        .supports_op(OpSpec::GetRows { table: DType::I8 })
        .is_err());
}

#[test]
fn supports_op_and_props_reject_stale_index() {
    let bogus = DeviceInfo {
        index: usize::MAX,
        name: "bogus".to_string(),
        description: String::new(),
        device_type: DeviceType::Cpu,
        memory_free: 0,
        memory_total: 0,
    };
    assert!(bogus.props().is_err(), "stale props");
    assert!(bogus.supports_op(OpSpec::Add).is_err(), "stale probe");
}

// -- C. graph structure ---------------------------------------------------------

#[test]
fn graph_capacity_validates() {
    assert!(Graph::with_capacity(0).is_err(), "zero capacity");
    assert!(
        Graph::with_capacity(u32::MAX as usize + 1).is_err(),
        "past u32 range"
    );
    let graph = Graph::new().expect("default graph");
    assert_eq!(graph.capacity(), DEFAULT_GRAPH_CAPACITY);
    assert_eq!(graph.n_nodes(), 0);
    assert!(graph.node(0).is_err(), "no nodes yet");
    assert!(graph.find("anything").is_none());
    assert!(graph.find("a\0b").is_none(), "NUL name matches nothing");
}

#[test]
fn graph_add_output_builds_nodes() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[4], &[1.0, 2.0, 3.0, 4.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[4], &[10.0, 20.0, 30.0, 40.0]).expect("b");
    let c = add(&a, &b).expect("add");
    c.set_name("sum").expect("name");
    assert_eq!(c.op_name(), "ADD");

    let mut graph = Graph::new().expect("graph");
    graph.add_output(&c).expect("add output");
    assert_eq!(graph.n_nodes(), 1, "one ADD node; leaves are not nodes");
    let node = graph.node(0).expect("node 0");
    assert_eq!(node.op, "ADD");
    assert_eq!(node.name, "sum");
    assert_eq!(node.nelements, 4);
    assert_eq!(node.nbytes, 16);
    assert_eq!(node.n_dims, 1);
    assert!(node.is_contiguous);
    assert!(!node.is_view);
    assert!(graph.node(1).is_err(), "upper bound checked");

    let found = graph.find("sum").expect("find by name");
    assert_eq!(found, node);
    assert!(graph.find("missing").is_none());
}

#[test]
fn graph_names_anonymous_tensors_on_add() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[2], &[3.0, 4.0]).expect("b");
    let c = add(&a, &b).expect("add");
    let mut graph = Graph::new().expect("graph");
    graph.add_output(&c).expect("add output");
    // Whatever ggml names it, the name sticks to the tensor and the
    // graph resolves it.
    let name = c.name();
    assert!(!name.is_empty(), "anonymous output got a name");
    assert!(graph.find(&name).is_some(), "graph resolves {name}");
}

#[test]
fn graph_rejects_mixed_backend_and_empty_use() {
    let backend = open_test_cpu();
    let other = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[2], &[3.0, 4.0]).expect("b");
    let c = add(&a, &b).expect("add");
    let x = Tensor::from_f32(&other, &[2], &[5.0, 6.0]).expect("x");
    let y = Tensor::from_f32(&other, &[2], &[7.0, 8.0]).expect("y");
    let z = add(&x, &y).expect("add on other backend");

    // Empty graph: plan/compute refused before any backend exists.
    let mut graph = Graph::new().expect("graph");
    assert!(graph.plan(&backend).is_err(), "plan of empty graph");
    assert!(graph.compute(&backend).is_err(), "compute of empty graph");

    graph.add_output(&c).expect("first output");
    assert!(
        graph.add_output(&z).is_err(),
        "mixed-backend output refused"
    );
    assert!(graph.plan(&other).is_err(), "foreign plan backend refused");
    assert!(
        graph.compute(&other).is_err(),
        "foreign compute backend refused"
    );
    // Same-backend plan/compute succeed (values proven in cpu_exec_proof).
    let plan = graph.plan(&backend).expect("plan");
    plan.compute().expect("plan compute");
    graph.compute(&backend).expect("direct compute");
}

#[test]
fn graph_capacity_overflow_refused() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[2], &[3.0, 4.0]).expect("b");
    let c = add(&a, &b).expect("add"); // graph bound 1 + 1 + 1 = 3
    let mut tiny = Graph::with_capacity(1).expect("tiny graph");
    assert!(tiny.add_output(&c).is_err(), "bound 3 does not fit cap 1");
    let mut fits = Graph::with_capacity(4).expect("graph");
    fits.add_output(&c).expect("bound 3 fits cap 4");
}

#[test]
fn plan_reexecution_updates_outputs() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[2], &[10.0, 20.0]).expect("b");
    let c = add(&a, &b).expect("add");
    let mut graph = Graph::new().expect("graph");
    graph.add_output(&c).expect("add output");
    let plan = graph.plan(&backend).expect("plan");

    // Re-upload inputs, re-run the plan: the output tensor's storage
    // is recomputed in place.
    a.upload_f32(&[100.0, 200.0]).expect("re-upload a");
    b.upload_f32(&[7.0, 8.0]).expect("re-upload b");
    plan.compute().expect("recompute");
    assert_eq!(c.to_vec_f32().expect("download"), vec![107.0, 208.0]);
    // And once more, proving plans are reusable.
    a.upload_f32(&[-1.0, -2.0]).expect("re-upload a");
    plan.compute().expect("recompute");
    assert_eq!(c.to_vec_f32().expect("download"), vec![6.0, 6.0]);
}
