# CUPTI activity tracer

Optional, project-local validation tooling for profiler-backed attribution.
It is not linked into RAMforge and does not change runtime behavior.

Build:

```sh
g++ -O2 -shared -fPIC -o /tmp/libforge_cupti_trace.so tools/cupti_trace.cpp \
    -I/usr/include -L/usr/lib64 -lcupti -ldl
```

Run with any RAMforge benchmark:

```sh
LD_PRELOAD=/tmp/libforge_cupti_trace.so \
FORGE_CUPTI_OUT=/tmp/forge-cupti.csv \
ramforge benchmark MODEL --runs 1 ... --json
```

Set `FORGE_CUPTI_API=0` to collect only kernel and memcpy activity; runtime and
driver API tracing is heavy and perturbs throughput.

CSV columns:

- `kind`: kernel, memcpy, runtime, driver
- `start_ns`, `end_ns`: CUPTI timestamps
- `stream_or_thread`: stream id for GPU activity, thread id for API activity
- `detail`: kernel name, memcpy bytes/kind, or API callback id

Limitations observed on CUDA 13.2 / RTX 3050:

- Under CUDA graph replay, many kernel records have zero timestamps. Use
  `GGML_CUDA_DISABLE_GRAPHS=1` for kernel-timeline attribution; that changes
  performance and must not be used for throughput claims.
- Runtime/driver API tracing can slow inference by roughly 4x; use it only for
  relative host-submission attribution.
- Nsight Systems and Nsight Compute are not installed on the current machine.
