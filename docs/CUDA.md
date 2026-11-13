# LNSCUDA: the LNS backends on CUDA GPUs

`LNSCUDA16` and `LNSCUDA32` run the LNS backends' op set on an NVIDIA GPU
through the [xlnscuda](https://github.com/xlnsresearch/xlnscuda) submodule.
They exist to make LNS inference fast enough to use; they are not a different
numerical experiment:

- **LNSCUDA16 is bit-identical to LNS16** built with the same
  `GGML_LNS16_SUM`: `base`, `monte`, `lpvip` or `sum32`, the same four modes.
  Every logit, and so every generated token, matches the CPU backend.
- **LNSCUDA32 matches LNS32** bit for bit in everything integer-only (mat-mul,
  add/mul/div, rms_norm, rope); its float-based activations can differ in the
  last ulp (see [Numerics](#numerics)).

For how this fits next to the CPU backends, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- An NVIDIA GPU (any architecture CUDA supports; developed on an RTX 2050,
  sm_86, 4 GB) and its driver.
- The CUDA toolkit (`nvcc`, cudart). Developed with CUDA 12.0. ggml's own CUDA
  backend is not used or built.
- A host compiler your `nvcc` accepts. CUDA 12.0 rejects GCC 13; `setup.sh`
  probes the installed `g++` versions and uses the first one that works
  (`g++-12` on Ubuntu 24.04). Override with `CUDAHOSTCXX=/path/to/g++`.
- The xlnscuda and xlnscpp submodules at their pinned commits (xlnscuda
  `aa5ccf0` or later for `xlns16d_monte.cu`, `xlns32d_lpvip.cu` and
  `tests/chkxlns16d_sum_modes.cu`; xlnscpp `41843b0` or later for the
  `xlns16.cpp` / `xlns32lpvip.cpp` fixes).

## Build, test, run

```bash
git submodule update --init                       # xlnscpp + xlnscuda

# all four modes at once (base, monte, lpvip, sum32), tests only
./setup.sh --cuda --unit-test --skip-inference

# one GPU mode at a time: build + tests (no model needed)
./setup.sh --cuda --sum-mode=base  --unit-test --skip-inference
./setup.sh --cuda --sum-mode=monte --unit-test --skip-inference
./setup.sh --cuda --sum-mode=lpvip --unit-test --skip-inference
./setup.sh --cuda --sum-mode=sum32 --unit-test --skip-inference

# one GPU mode at a time: build + tests + SmolLM2 inference (CPU and GPU)
./setup.sh --cuda --sum-mode=base  --unit-test
./setup.sh --cuda --sum-mode=monte --unit-test
./setup.sh --cuda --sum-mode=lpvip --unit-test
./setup.sh --cuda --sum-mode=sum32 --unit-test
# Monte through LPVIP (GGML_LNS16_MONTE_BYLPVIP=ON) is not a setup.sh option;
# build it by hand after setup has prepared build/ggml:
cmake -S build/ggml -B build/ggml/build-lns-monte-bylpvip -DGGML_LNS=ON -DGGML_LNS32=ON -DGGML_LNSCUDA=ON \
  -DGGML_LNS16_SUM=monte -DGGML_LNS16_MONTE_BYLPVIP=ON -DGGML_BLAS=OFF -DGGML_CPU_REPACK=OFF \
  -DXLNSCPP_DIR=$PWD/xlnscpp -DXLNSCUDA_DIR=$PWD/xlnscuda -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_BUILD_TYPE=Release
cmake --build build/ggml/build-lns-monte-bylpvip --target test-lns-cuda-parity test-lns-backend -j6
(cd build/ggml/build-lns-monte-bylpvip && ./bin/test-lns-cuda-parity && ./bin/test-lns-backend LNSCUDA16 monte)

# run after setup (no rebuild)
./run.sh --backend=cuda16 --sum-mode=base  -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=monte -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=lpvip -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=sum32 -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=all   -- --temp 0 --seed 1   # all four modes in turn
./run.sh --backend=16     --sum-mode=monte -- --temp 0 --seed 1   # same tree on the CPU: same text
./run.sh --backend=cuda32                                         # LNS32 arithmetic on the GPU
```

`--cuda` (or `LNS_CUDA=1`) adds `-DGGML_LNSCUDA=ON` to every LNS build. With
`--unit-test` it also runs `xlnscuda/tests/chkxlns16d_sum_modes.cu` (device
Monte/LPVIP vs xlnscpp) and, in each mode's build (LNSCUDA32 once):

- `test-lns-cuda-parity`: LNSCUDA16 against LNS16 and LNSCUDA32 against LNS32,
  op by op (with quantized, F16 and F32 weights, broadcasting, permuted KV-cache
  views, cold and warm weight cache) and through a scheduled block that mixes
  GPU, CPU-LNS and CPU nodes;
- `test-lns-backend LNSCUDA16 base` and `test-lns-backend LNSCUDA32`: the
  regular float-reference suite on the GPU devices.

On machines with little RAM, limit build parallelism with `JOBS=4 ./setup.sh ...`.

The manual equivalent of the ggml part:

```bash
cmake -S build/ggml -B build/ggml/build-lns-base \
  -DGGML_LNS=ON -DGGML_LNS32=ON -DGGML_LNSCUDA=ON -DGGML_LNS16_SUM=base \
  -DGGML_BLAS=OFF -DGGML_METAL=OFF -DGGML_CPU_REPACK=OFF -DCMAKE_BUILD_TYPE=Release \
  -DXLNSCPP_DIR=$PWD/xlnscpp -DXLNSCUDA_DIR=$PWD/xlnscuda \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build/ggml/build-lns-base --target test-lns-cuda-parity test-lns-backend -j4
(cd build/ggml/build-lns-base && ./bin/test-lns-cuda-parity)
```

`CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the GPUs in the build
machine); set it explicitly to build for other GPUs.

## Selecting the backend

| `GGML_LNS_BACKEND` | What runs where |
|---|---|
| `cuda`, `cuda16` | LNSCUDA16 for the ops it implements, LNS16 (CPU) for the other LNS ops |
| `cuda32` | LNSCUDA32, with LNS32 as the fallback |
| `LNSCUDA16` | LNSCUDA16 only; everything else on the plain CPU backend |

Other settings:

| Variable | Default | Meaning |
|---|---|---|
| `GGML_LNS_CUDA_DEVICE` | `0` | which GPU to use |
| `GGML_LNS_CUDA_CACHE_MB` | free VRAM − 768 MiB | VRAM budget for weights converted to LNS |
| `LNS_OP_AUDIT=1` | off | log every `supports_op` decision (backend, op, reason) |
| `LNS_DISABLE_OPS=...` | none | force named ops back to the CPU backend |

The GPU stays untouched until the first op runs, so a binary built with
`--cuda` costs no VRAM when another backend is selected.

## What runs on the GPU

| Op | Notes |
|---|---|
| `MUL_MAT` | F32, F16, BF16 and every quantized weight type (dequantized on the host once, then cached as LNS); GQA broadcasting; strided (permuted) KV-cache views |
| `ADD`, `MUL`, `DIV` | with ggml broadcasting |
| `SCALE` | including `ggml_scale_bias` |
| `RMS_NORM`, `SOFT_MAX` | softmax with scale and F32 mask; no ALiBi, no attention sinks (also declined by the CPU backends) |
| `ROPE` | normal and NEOX, with optional frequency factors (Llama 3); YaRN / linear scaling / multi-section modes are declined, by the CPU backends too |
| `UNARY` | `SILU`, `GELU`, `RELU`, `TANH` |
| `GLU` | `SWIGLU`, `GEGLU`, split and fused (and swapped) |

Declined, and run by the CPU LNS backend under `cuda16` / `cuda32`:
`MUL_MAT_ID` (MoE), `GET_ROWS`, `ARGSORT`, `CLAMP`, `DIAG_MASK_INF`, and copies
of LNS16-typed tensors. `LNS_OP_AUDIT=1` shows each decision with its reason.

## Numerics

The CPU backends read each operand as F32, convert it to LNS, compute, and
write F32 back, op by op. The GPU kernels do exactly the same operations, in
the same order, with the same tables:

- **Conversions and lookups.** LNS16 is built with `xlns16_table`: float↔xlns16
  conversion, sigmoid, exp and log are table lookups, and xlns16 addition uses
  a 1280-entry sb/db table. xlnscuda previously ignored `xlns16_table`; the new
  `xlns16d_table` mode copies those tables to the device (about 900 KB). xlns32
  addition uses an interpolated sb table and a cotransformed db; xlnscuda only
  had the ideal (libm) adder, and now ports the table version (about 75 KB).
- **Reductions are sequential.** LNS addition is not associative, so the sum
  inside a dot product, softmax normalizer or RMSNorm depends on its order.
  The GPU parallelises only across independent outputs; each dot product runs
  over `k` in order, exactly like `xlns16_vec_dot`. (Each mul_mat thread
  interleaves four independent output columns to hide the latency of the table
  lookup inside every addition.)
- **Host-side constants.** RoPE's cos/sin values (including the division by
  frequency factors) are computed on the host with the same float recurrence
  as the CPU kernel, then converted on the device.

The result: LNSCUDA16 is bit-identical to LNS16 `base` in every parity case,
in the full `test-lns-backend` suite (identical output, line for line), and
end to end: the complete final-token logit vectors of SmolLM2-135M and
Qwen2.5-0.5B (dumped with `scripts/rc-logits.cpp`) are byte-for-byte equal,
and greedy generation is token-identical.
LNSCUDA32 is bit-identical except in silu, gelu, tanh and the exp inside
softmax: xlns32 evaluates those with float `expf` / `tanhf`, where CUDA's and
glibc's results can differ in the last ulp. The parity test allows LNSCUDA32 a scaled
error (relative for |values| ≥ 1, absolute below) of 1e-5; observed: at most
2.2e-7 for a single op and 1.9e-6 through a small multi-op block, and every
difference is reported.
Through a whole model these last-ulp differences accumulate: on SmolLM2 and
Qwen2.5-0.5B the final logits differ from LNS32's by at most 0.0016 (absolute,
on logits of order 10), with the same top-5 tokens.

### Summation modes

`GGML_LNS16_SUM` selects the reduction (mat-mul dot products, softmax sum,
RMS-norm sum of squares) of LNS16 and LNSCUDA16 alike; other additions are
unchanged. A mode is a separate build (`build-lns-<mode>`); `run.sh` picks it
and checks the device description (`sum=monte, monte=pure16`) before running.

- `lpvip`: xlns16 products widened into a 32-bit LPVIP accumulator, truncated
  once at the end, as in `xlns32lpvip.cpp`. Its singularity branch uses double
  `log`/`pow`; the library test compares all 2^23 inputs of it (0 mismatches
  with CUDA 12.0 vs glibc). On consumer GPUs that FP64 path makes lpvip about
  5x slower than base (SmolLM2: 3.0 vs 14.5 tok/s on an RTX 2050).
- `monte`: `xlns16monte.cpp` (or `xlns16monte_bylpvip.cpp` with
  `GGML_LNS16_MONTE_BYLPVIP=ON`). The LFSR (seed 7, one draw per Monte add)
  is owned by LNS16 (`ggml-lns.h`); each GPU op reserves the draws the CPU
  would consume at that point of the graph and every output jumps ahead to
  its own start, so GPU, CPU-fallback ops (`MUL_MAT_ID`) and an all-CPU run
  use the same sequence. The state keeps advancing across ops and tokens;
  each process starts at 7. SmolLM2: 11.9 tok/s.
- `sum32`: xlns16 terms widened exactly into an xlns32 accumulator, added
  with xlnscuda's table-based `xlns32d_add` (the same xlns32 configuration as
  LNS32: table `sb`, cotransformation `db`), narrowed once with
  round-to-nearest, as in `lns-backend/src/lns16-sum32.h`. The most accurate
  mode, and it needs no extra xlnscuda code: CUDA32 already uses that adder.

## Memory model

Tensors stay in ggml host buffers, as with the CPU LNS backends. For each op
the backend uploads the inputs, runs the kernel, downloads the result and
waits for it, so the next node can run on any backend. This keeps the backend
simple and composable: no device buffer type, no copies inserted by the
scheduler, CPU fallback at no cost.

What makes it fast is the **weight cache**: the first time a weight
(a tensor in a buffer marked as model weights) is used, it is dequantized on
the host, uploaded in chunks, converted to LNS on the GPU and kept in VRAM in
the layout the mat-mul kernel reads. From the second token on, only
activations cross the PCIe bus.

The cache needs 2 bytes per parameter for LNSCUDA16 and 4 for LNSCUDA32,
whatever the GGUF quantization. When the budget is exhausted the backend says
so once, and converts the remaining weights on every use (correct, but slow).
At exit it reports what it cached:

```
ggml-lns-cuda: LNSCUDA16 on GPU 0 (NVIDIA GeForce RTX 2050, sm_86, 3679 MiB free), weight cache budget 2911 MiB
...
ggml-lns-cuda: LNSCUDA16 cached 211 weight tensors (256.5 MiB)
```

## Performance

Measured on an RTX 2050 (4 GB, laptop) with a 12-thread CPU, CUDA 12.0,
greedy decoding, `-p "The capital of France is" -n 32` (5-token prompt), after
the first run had filled the weight cache. "pp" is prompt processing, "tg" is
token generation.

| Model (Q4_K_M) | LNS16 (CPU) tg | LNSCUDA16 pp / tg | LNSCUDA32 pp / tg | VRAM cache 16 / 32 |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | 0.44 tok/s | 32.7 / **12.2** tok/s | 11.6 / 5.0 tok/s | 257 / 513 MiB |
| Qwen2.5-0.5B-Instruct | 0.12 tok/s | 22.6 / **9.7** tok/s | 7.5 / 3.6 tok/s | 942 / 1884 MiB |
| Llama-3.2-1B-Instruct | — | 17.0 / **7.1** tok/s | does not fit | 2357 MiB / — |

That is roughly 28× (SmolLM2) to 80× (Qwen2.5-0.5B) faster than the CPU
backend for the same, bit-identical, numbers.

Where the time goes (Qwen2.5-0.5B, LNSCUDA16): about three quarters is
mat-mul, the rest the other GPU ops and their transfers. Mat-mul is latency
bound: every output is one sequential chain of LNS additions, each with a
dependent table lookup, so a single token uses only as many threads as the
matrix has rows. Loading the operands of eight steps ahead of the chain
(software pipelining, numerically a no-op) made generation 1.7× faster; block
size and deeper prefetch made no further difference.

## Source layout

| File | Role |
|---|---|
| `ggml-lns-cuda.cpp` | the two devices: names, GUIDs, which `kernel_set` each uses |
| `lns-cuda-ops.{h,cpp}` | op table: support predicates and handlers marshalling nodes into kernel calls |
| `lns-cuda-context.{h,cpp}` | device and stream (lazy), scratch buffers, weight cache, host↔device layout conversion |
| `lns-cuda-kernels.h` | `kernel_set`, the interface between host code and GPU code |
| `lns-cuda-kernels.cuh` | the kernels, written once as templates over an arithmetic policy |
| `lns-cuda-xlns16.cu`, `lns-cuda-xlns32.cu` | one LNS format each: include xlnscpp + xlnscuda, define the policy |

The `.cpp` files never include CUDA device code or xlnscuda; the `.cu` files
never include ggml. Adding an LNS format is one more `.cu` file and a
`Format` struct (see [ARCHITECTURE.md](ARCHITECTURE.md#a-new-lns-format-on-the-gpu)).

## Limitations and future work

- **Per-op transfers.** Every GPU op round-trips its activations over PCIe,
  with a synchronisation per op. A device-resident buffer type (activations
  and KV cache in VRAM) would remove that, at the cost of implementing the
  remaining ops (`GET_ROWS`, `SET_ROWS`, copies) on the GPU too.
- **Sequential reductions** keep parity but limit single-token mat-mul to one
  thread per output row; prompt processing parallelises across tokens.
- Monte and LPVIP keep their algorithmic limits on the GPU (LPVIP is poor on
  mixed-sign cancellation).
- RoPE: no YaRN or linear scaling (declined, so those nodes run on the CPU
  backend in F32).
- One GPU per process (`GGML_LNS_CUDA_DEVICE`); `GGML_BACKEND_DL=ON` builds are
  not supported (the library registers two devices).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nvcc is missing or rejects the host compiler` at configure time | pass a supported host compiler: `CUDAHOSTCXX=/usr/bin/g++-12 ./setup.sh --cuda` |
| `predates device table support` | update the xlnscuda submodule |
| `test-lns-cuda-parity: SKIP` | no CUDA device visible (`nvidia-smi`), or built without `--cuda` |
| `weight cache full` | the model's LNS weights do not fit in VRAM: use a smaller model, LNSCUDA16 instead of 32, or raise `GGML_LNS_CUDA_CACHE_MB` if other memory is free |
| nothing runs on the GPU | check `GGML_LNS_BACKEND` (`cuda16`), and `LNS_OP_AUDIT=1` for the reasons ops are declined |
