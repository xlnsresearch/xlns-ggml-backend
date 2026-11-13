# Architecture

How the LNS backends in this repository are organised, how they are wired
into ggml / llama.cpp, and how to add another one. Read this before changing
backend code; [CUDA.md](CUDA.md) covers the GPU backend in more depth.

## The four backends

| Device | Library | Runs on | Arithmetic | Source |
|---|---|---|---|---|
| `LNS16` | xlnscpp | CPU | xlns16, summation mode chosen at build time (`base` / `monte` / `lpvip` / `sum32`) | `lns-backend/src/` |
| `LNS32` | xlnscpp | CPU | xlns32 | `lns-backend/src32/` |
| `LNSCUDA16` | xlnscuda | CUDA GPU | xlns16, bit-identical to `LNS16` built in the same mode (`base` / `monte` / `lpvip` / `sum32`) | `lns-backend/cuda/` |
| `LNSCUDA32` | xlnscuda | CUDA GPU | xlns32, matches `LNS32` (see [CUDA.md](CUDA.md#numerics)) | `lns-backend/cuda/` |

All four are ggml backends of type `ACCEL` that compute on ggml's ordinary
host (CPU) buffers. Weights and activations stay in host memory, llama.cpp
needs no `-ngl`, and any node an LNS backend declines runs on the stock CPU
backend with no copies. The GPU backends copy operands to the device per op
and cache the weights there; see [CUDA.md](CUDA.md#memory-model).

## Layers

```
 llama.cpp / ggml scheduler
        │  supports_op / graph_compute
        ▼
 lns-backend/common/lns-backend-framework.h      host_backend<Traits>: the ggml vtables,
        │                                         op-table dispatch, selection, audit
        ▼
 per-backend Traits + op table                    ggml-lns.cpp, ggml-lns32.cpp, ggml-lns-cuda.cpp
        │
        ├── CPU kernels ──────────────► xlnscpp    lns-ops.cpp, lns32-ops.cpp
        │
        └── lns_cuda::context ── kernel_set ──► xlnscuda (+ xlnscpp for host tables)
            lns-cuda-context.cpp    lns-cuda-kernels.{h,cuh}, lns-cuda-xlns{16,32}.cu
            lns-cuda-ops.cpp
```

Each layer only knows the one below it through a narrow interface:

| Piece | Responsibility | Knows nothing about |
|---|---|---|
| `lns-backend-config.h` | `GGML_LNS_BACKEND` selection, `LNS_DISABLE_OPS`, `LNS_OP_AUDIT` | any particular backend |
| `lns-op-support.h` | what each ggml op means and when an LNS kernel can compute it exactly | how it is computed |
| `lns-backend-framework.h` | ggml registry / device / backend vtables, op-table lookup | LNS arithmetic |
| `ggml-lns*.cpp` | a backend's identity (name, GUID, description) and op table | ggml vtables |
| `lns-ops.cpp`, `lns32-ops.cpp` | CPU kernels | scheduling, selection |
| `lns-cuda-ops.cpp` | marshalling ggml nodes into GPU kernel calls | CUDA, LNS formats |
| `lns-cuda-context.cpp` | device, stream, scratch, weight cache, host↔device layout | which op runs |
| `lns-cuda-kernels.h` | the abstract GPU kernel interface (`kernel_set`) | ggml |
| `lns-cuda-kernels.cuh` | kernels written once as templates over an LNS format | which format |
| `lns-cuda-xlns{16,32}.cu` | one LNS format: includes the libraries, defines the arithmetic policy | ggml, ops |

## The backend framework

`lns::host_backend<Traits>` (in `lns-backend/common/lns-backend-framework.h`)
implements everything ggml requires of a backend. A backend only describes
itself:

```cpp
struct lns16_traits : lns::backend_traits_base {
    static constexpr const char * name = "LNS16";
    static const char * description();          // shown by llama.cpp; LNS16 includes "sum=<mode>"
    static ggml_guid_t  guid();
    static lns::op_table<context> kernels();    // the op table, below
    // optional: context / create_context, device_count, get_memory,
    //           log_init, audit_fields (defaults in backend_traits_base)
};
using lns16_backend = lns::host_backend<lns16_traits>;

ggml_backend_reg_t ggml_backend_lns_reg(void) { return lns16_backend::reg(); }
```

The **op table** pairs each supported op with its support predicate and its
kernel:

```cpp
const op_kernel<no_context> k_kernels[] = {
    { GGML_OP_MUL_MAT, any_subop,          lns::check_mul_mat,          stateless<lns_mul_mat> },
    { GGML_OP_UNARY,   GGML_UNARY_OP_SILU, lns::check_unary_activation, stateless<lns_silu>    },
    ...
};
```

`supports_op` and `graph_compute` both look nodes up in this one table, so a
backend can neither claim an op it does not implement nor implement one it
never claims. View-like nodes (`RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`,
`NONE`) are accepted and skipped by the framework for every backend.

Backends that keep per-instance state (the GPU backends' device memory and
weight cache) declare `using context = ...;` and `create_context()`; their
kernels then receive `context &`. Stateless CPU kernels are adapted with
`stateless<fn>`.

The predicates in `lns-op-support.h` are shared on purpose. A GPU backend
wraps the CPU predicate and only adds its own limits (e.g. grid sizes), so
"what does this op mean" is defined once and the backends cannot drift apart:

```cpp
op_support check_scale(const ggml_tensor * op) {
    return check_f32_result(op, lns::check_unary_like_f32(op));
}
```

## Backend selection and fallback

All LNS devices are ACCEL devices on host buffers, so the scheduler would
hand every op they all support to whichever registered first.
`GGML_LNS_BACKEND` decides which of them answer `supports_op` at all
(`lns-backend-config.h`):

| Value | Selected devices |
|---|---|
| unset, `16` | `LNS16` |
| `32` | `LNS32` |
| `cuda`, `cuda16` | `LNSCUDA16`, `LNS16` |
| `cuda32` | `LNSCUDA32`, `LNS32` |
| a device name, or a comma-separated list of them | exactly those devices |

A GPU alias also selects its CPU counterpart. The GPU backend registers first,
so it wins the ops it implements; ops it declines (e.g. `MUL_MAT_ID`,
`ARGSORT`) land on the CPU LNS backend instead of on the plain F32 CPU
backend, and the whole graph still runs in LNS. Selecting a device name alone
(`GGML_LNS_BACKEND=LNSCUDA16`) turns that fallback off. Unknown tokens are
reported on stderr and ignored.

Two diagnostics work identically for every backend:

- `LNS_OP_AUDIT=1` prints one JSON line per `supports_op` query, with the
  backend, the decision and the predicate's reason.
- `LNS_DISABLE_OPS=MUL_MAT,ffn_down` forces matching ops (by op name, or by a
  substring of the tensor name) back to the CPU backend.

## Build integration

Nothing in ggml or llama.cpp is forked. `setup.sh` checks out both at the
commits pinned in `patches/`, then:

1. **Hooks** (`scripts/integrate-lns.sh`): adds the `GGML_TYPE_LNS16` tensor
   type (`include/ggml.h`, `src/ggml.c`) and, for every backend, a CMake
   option, a `ggml_add_backend()` call and the registry include/registration.
   The per-backend hooks are generated blocks between marker comments
   (`>>> xlns-ggml-backend: ...`) that are rewritten on every run, so
   integration is idempotent and always follows the manifest. If an upstream
   change breaks an anchor, the patches in `patches/` are applied instead.
2. **Sources** (`copy_lns_backends`): copies each backend directory to
   `src/ggml-<name>/`, its public header to `include/`, and
   `lns-backend/common/` to `src/ggml-lns-common/`.
3. **Tests** (`copy_lns_tests`): adds the unit tests to the standalone ggml
   tree's `examples/simple`.

The single list of backends is the manifest, `scripts/lns-backends.sh`:

```bash
LNS_BACKENDS=(
    "LNSCUDA|lns-backend/cuda|ggml-lns-cuda.h|ggml_backend_lns_cuda16_reg ggml_backend_lns_cuda32_reg|ggml: use LNS on CUDA GPUs (xlnscuda)"
    "LNS|lns-backend/src|ggml-lns.h|ggml_backend_lns_reg|ggml: use LNS (Logarithmic Number System)"
    "LNS32|lns-backend/src32|ggml-lns32.h|ggml_backend_lns32_reg|ggml: use LNS32 (32-bit Logarithmic Number System)"
)
```

Row order is registration order, which is scheduling priority among the LNS
devices: GPU backends first.

Every backend's `CMakeLists.txt` includes `ggml-lns-common/lns-common.cmake`,
which adds the shared headers and `XLNSCPP_DIR` to the target and fails
clearly when xlnscpp is missing.

## Testing

| Test | What it proves |
|---|---|
| `lns-backend/tests/test-lns-backend.cpp` | each device against float references, per summation profile (`./bin/test-lns-backend LNS16 base`, `LNS32`, `LNSCUDA16 base`, `LNSCUDA32`) |
| `lns-backend/tests/test-lns-cuda-parity.cpp` | `LNSCUDA16` is bit-identical to `LNS16` of the same summation mode (Monte from equal LFSR states) and `LNSCUDA32` matches `LNS32`, op by op and through the scheduler fallback chain |
| `xlnscuda/tests/chkxlns16d_sum_modes.cu` | device Monte / LPVIP arithmetic is bit-identical to xlnscpp |
| `lns-backend/tests/test-lns16-sum32.cpp` | the sum32 accumulator's invariants, without ggml |
| `xlnscuda/tests/chkxlns{16,32}d_table.cu` | the xlnscuda device arithmetic is bit-identical to xlnscpp |
| `scripts/lns-vs-cpu-quality.sh` | end to end: an LNS backend against a pure-CPU llama.cpp (token divergence, perplexity) |

`./setup.sh --unit-test` runs the ggml-level tests; add `--cuda` for the GPU
ones.

## Adding a backend

The shape depends on what the new library offers.

### A new library computing on host buffers (CPU, or an accelerator with its own copies)

This is how all four existing backends work. Say the library is "foo":

1. **Sources.** Create `lns-backend/foo/` with
   - `ggml-lns-foo.cpp`: a `Traits` struct and an op table, as in
     `lns-backend/src32/ggml-lns32.cpp` (about 90 lines). Reuse the predicates
     in `lns-op-support.h`; wrap them where your kernels have extra limits.
   - the kernels, each `void kernel(ggml_tensor * dst)` (or
     `void kernel(context &, ggml_tensor *)` with a context).
   - `CMakeLists.txt`: `ggml_add_backend_library(ggml-lnsfoo ...)`, then
     `include(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-lns-common/lns-common.cmake)`
     and `lns_backend_common_setup(ggml-lnsfoo)`, plus your library's
     includes/links.
2. **Public header.** `lns-backend/include/ggml-lns-foo.h` declaring
   `ggml_backend_lnsfoo_reg()` (and `_init()`), as in `ggml-lns32.h`.
3. **Manifest.** Add a row to `scripts/lns-backends.sh`:
   `"LNSFOO|lns-backend/foo|ggml-lns-foo.h|ggml_backend_lnsfoo_reg|ggml: use LNS on foo"`.
   The directory in the ggml tree is `src/ggml-` plus the lowercased name, so
   `LNSFOO` lands in `src/ggml-lnsfoo/`.
4. **Selection.** Add an alias to `k_backend_aliases` in
   `lns-backend-config.h` if users should reach it with a short name, e.g.
   `{ "foo", "LNSFOO,LNS16" }` to fall back to LNS16.
5. **Build flag.** In `setup.sh`, pass `-DGGML_LNSFOO=ON` where you want it
   built (see how `--cuda` adds `GGML_LNSCUDA`).
6. **Tests.** Make `test-lns-backend` accept the device (it maps device names
   to tolerance profiles and fallbacks at the top of `main`), and, if the
   backend is meant to reproduce another, add a parity test in the style of
   `test-lns-cuda-parity.cpp`.

No other script, framework or backend file changes.

### A new LNS format on the GPU

Write a translation unit like `lns-cuda-xlns16.cu`: include the library, define
an arithmetic policy struct (`from_f32`, `add`, `mul`, ... as listed at the top
of `lns-cuda-kernels.cuh`), and expose a `kernel_set` factory. Then add a
`Format` struct in `ggml-lns-cuda.cpp` and register it like `xlns32_format`.
The kernels, context and op table are reused unchanged.

### A new op

Add a predicate to `lns-op-support.h` (if it is not there), the kernel, and one
op-table row in each backend that implements it. For the GPU backends, add the
kernel to `kernel_set` and `lns-cuda-kernels.cuh`, and the marshalling to
`lns-cuda-ops.cpp`; then add a parity case.

### A device-resident GPU backend

The GPU backends deliberately keep tensors in host memory (see
[CUDA.md](CUDA.md#memory-model)). A backend whose tensors live in device memory
needs its own buffer type and ggml's copy paths, so it would not use
`host_backend`. It could still reuse `lns-op-support.h`, `lns-backend-config.h`
and the whole `kernel_set` layer.

## Design principles, concretely

- **Single responsibility.** Each file in the layer table above has one job;
  for instance the GPU op handlers never touch device memory directly, and the
  context never decides which op runs.
- **Open/closed.** Backends, ops and LNS formats are added by adding a row,
  a file or a policy struct (manifest, op table, `kernel_set` factory); the
  framework, scripts and existing backends stay as they are.
- **Liskov substitution.** Every backend is a complete ggml backend; the
  scheduler and llama.cpp treat them interchangeably, and the parity tests
  check that a GPU backend is a drop-in substitute for its CPU counterpart.
- **Interface segregation.** The GPU host code depends only on `kernel_set`
  and device pointers, never on xlnscuda; kernels depend only on the
  arithmetic policy, never on ggml.
- **Dependency inversion.** The framework depends on the `Traits` contract
  and the op table, not on concrete backends; the GPU context depends on the
  abstract `kernel_set`, and the concrete formats plug into it.
