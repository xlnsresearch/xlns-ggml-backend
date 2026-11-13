# xlns-ggml-backend

GSoC 2026 code for **"Support for Logarithmic Number Systems in Large Language Models"**
([xlnsresearch](https://github.com/xlnsresearch) / ggml + llama.cpp).

In a logarithmic number system (LNS), a value is stored as its sign plus the
base-2 logarithm of its magnitude, so multiplication and division become integer
addition and subtraction while addition needs an approximation — see
[xlnscpp](https://github.com/xlnsresearch/xlnscpp) for the full background.

This repo implements ggml backends on top of the xlnscpp (CPU) and xlnscuda
(GPU) libraries and hooks them into llama.cpp, so a stock `llama-completion`
runs an LLM's compute in LNS arithmetic:

- **LNS16** — 16-bit LNS (xlns16) arithmetic on the CPU, with a compile-time
  choice of summation algorithm (`base` / `monte` / `lpvip` / `sum32`) for the
  reduction-heavy ops (softmax, rms_norm, and the vec_dot inside mat-mul).
- **LNS32** — 32-bit LNS (xlns32) arithmetic on the CPU, structural parity port
  of the same op set.
- **LNSCUDA16 / LNSCUDA32** — the same arithmetic on an NVIDIA GPU through
  xlnscuda. LNSCUDA16 supports all four summations (`base`, `monte`,
  `lpvip`, `sum32`) and is bit-identical to LNS16 built in the same mode, so a GPU run
  reproduces the CPU results token for token, much faster. See
  [docs/CUDA.md](docs/CUDA.md).

| Backend | Summation modes |
|---|---|
| LNS16 (CPU) | `base`, `monte`, `lpvip`, `sum32` |
| LNSCUDA16 (GPU) | `base`, `monte`, `lpvip`, `sum32` |
| LNS32 / LNSCUDA32 | xlns32 (no modes) |

How the backends share one framework, and how to add another library, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick start

```bash
git clone https://github.com/xlnsresearch/xlns-ggml-backend
cd xlns-ggml-backend
git submodule update --init   # optional — setup.sh also initialises the submodules

# Model files are not shipped; download SmolLM2 (105 MB):
pip install -U huggingface_hub
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir .

./setup.sh                  # clone+build ggml/llama.cpp at pinned commits, run inference per sum mode
./run.sh --sum-mode=sum32   # re-run a variant without rebuilding

# With an NVIDIA GPU and the CUDA toolkit:
./setup.sh --cuda --sum-mode=monte --unit-test
./run.sh --backend=cuda16 --sum-mode=monte
```

Expected output (SmolLM2, LNS16, `sum32`):

```
The capital of France is Paris. Paris is the political, cultural, and economic center of France. ...
```

See [SETUP.md](SETUP.md) for prerequisites, every flag, the other models, and
what success looks like.

---

## LNS16 backend

A proper `ggml_backend` (taking inspiration from the BLAS and CPU backends) that performs all
supported ops in **xlns16** arithmetic.  It registers as the `"LNS16"` device
with `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

Compilation uses `xlns16_table` (O(1) float↔xlns16 via a 65536-entry lookup
table) and `xlns16_alt` (reduced-branching addition path).  `xlns16_ideal` is
not defined — this matches a realistic hardware target rather than ideal
arithmetic.

### Summation variants (`GGML_LNS16_SUM`)

Long xlns16 summations "stick": once the partial sum dwarfs each new addend,
`xlns16_add` cannot represent the increment and the result is systematically
biased (see `xlnscpp/summation16bit.MD`).  The reduction inside softmax
(normalizer), rms_norm (sum of squares), and mat-mul vec_dot can therefore be
switched between four implementations at build time:

```bash
cmake .. -DGGML_LNS=ON -DGGML_LNS16_SUM=sum32   # base (default xlns16) | monte | lpvip | sum32
```

| Mode | Implementation | Character |
|---|---|---|
| `base` | sequential `xlns16_add` (`xlns16.cpp`) | pure 16-bit; ~4% bias observed on an n=576 sum of squares |
| `monte` | Monte-Carlo LNS dither (`xlns16monte.cpp`) | pure 16-bit; unbiased in expectation, individual reductions can err more than base |
| `lpvip` | 32-bit LPVIP accumulator (`xlns32lpvip.cpp`) | ~5× better than base on same-sign long sums (0.77% on the same n=576 case), cheaper than a full xlns32 accumulator — but `xlns32_add_lpvip` is inaccurate on mixed-sign cancellation, so it is *worse* than base inside mat-mul (57% vs 4.4% on a K=1536 dot) |
| `sum32` | pure xlns32 accumulator, rounded back to xlns16 (`lns-backend/src/lns16-sum32.h`) | the accuracy ceiling: two-term sums come out correctly rounded and the ~370-term stagnation disappears (K=1536 dot 0.31%, n=576 sum of squares 0.45%). Also the slowest — see the throughput note below |

`sum32` is the option `xlnscpp/summation16bit.MD` describes but does not ship
("one could sum such `xlns16` values using an `xlns32` accumulator, but that is
very expensive").  Each term is widened to xlns32 — exact, the formats are
bit-aligned — accumulated with `xlns32_add`, and the accumulator is rounded
back to xlns16 with round-to-nearest at the end.  `xlns32_add` is the *only*
xlns32 arithmetic involved: operands, results and storage stay xlns16
throughout, and each reduction narrows exactly once.  It uses the same `xlns32_alt` configuration as the
LNS32 backend (table + interpolation `sb`, cotransformation `db`; no
`xlns32_ideal`).  Measured cost: roughly 1.66x slower than `base`
end-to-end on SmolLM2-135M.

Numbers from `test-lns-backend` (the `mul mat long` case, K=1536 mixed-sign —
the only case in the suite that can see an accumulator at all; every mode
scores an identical 0.59% on the K=2 `mul mat`):

| | base | monte | lpvip | sum32 | LNS32 |
|---|---|---|---|---|---|
| `mul mat long` NRMSE | 4.39% | 4.72% | 57.21% | **0.31%** | 0.012% |
| `rms_norm 576` | 4.07% | 5.68% | 0.77% | **0.45%** | 0.0012% |

`monte` has a second compile-time knob, `-DGGML_LNS16_MONTE_BYLPVIP=ON`, which
swaps the pure-16-bit LFSR implementation for the `xlns32_add_lpvip`-based one
(`xlns16monte_bylpvip.cpp`).  The two define identical symbols, so this choice
is inherently compile-time.

The summation variant is used in exactly four places: softmax's normalizer,
rms_norm's sum of squares, and the vec_dot inside `MUL_MAT` and `MUL_MAT_ID`.
The remaining implemented ops are elementwise, pairwise (ROPE, the GLU ops),
data movement (GET_ROWS, CPY/CONT/DUP, views), or a row-wise sort (ARGSORT).
If layernorm (`GGML_OP_NORM`) or `SUM`/`SUM_ROWS`/`MEAN` are added later,
xlnscpp already ships `_monte`/`_lpvip32` versions of them, and
`lns16-sum32.h` ships `xlns16_layernorm_sum32` / `xlns16_sum_sum32`.

The `sum32` family lives in `lns-backend/src/lns16-sum32.h` rather than in
xlnscpp for now; it is written to move to `xlnscpp/xlns16sum32.cpp` verbatim
once upstreamed.  Its invariants (exact widen/narrow round-trip, no
stagnation, long mixed-sign dots) are checked by a standalone program that
needs no ggml:

```bash
g++ -O2 -std=c++17 -I xlnscpp -I lns-backend/src lns-backend/tests/test-lns16-sum32.cpp -o /tmp/test-lns16-sum32 && /tmp/test-lns16-sum32
```

## LNS32 backend

A parallel backend (`"LNS32"` device, option `GGML_LNS32`) porting the same op
set to **xlns32** (1 sign / 8 int / 23 frac log bits — F32-class accuracy,
unit tests pass at 0.1% tolerance).  Built with `xlns32_alt`; `xlns32_ideal`
is not defined.

Two structural caveats:

- **No conversion table exists for xlns32** (`fp2xlns32` is a libm `log()` per
  element), and weights are converted per-op, so inference is much slower than
  LNS16.  A per-tensor weight cache / custom buffer type is documented future
  work.
- There is no `GGML_TYPE_LNS32`; all tensors flow as F32.

## LNSCUDA16 / LNSCUDA32 backends (GPU)

The same op set on an NVIDIA GPU, through the xlnscuda submodule. The GPU
kernels repeat the CPU kernels' LNS operations in the same order with the same
lookup tables (support for which was added to xlnscuda for this), so LNSCUDA16
output is bit-identical to LNS16 built with the same `GGML_LNS16_SUM` (`base`,
`monte`, `lpvip` or `sum32`), and
LNSCUDA32 matches LNS32 except for last-ulp libm differences in its
activations. Weights are converted to LNS once and cached in VRAM. Build with
`./setup.sh --cuda`; details, numbers and limits are in
[docs/CUDA.md](docs/CUDA.md).

### Backend selection at runtime

All LNS backends register as ACCEL devices sharing the CPU buffer type, so the
scheduler would give every op to whichever registered first. The
`GGML_LNS_BACKEND` env var decides which backends claim ops; the others
decline with audit reason `disabled_by_env` (observable with
`LNS_OP_AUDIT=1`):

| `GGML_LNS_BACKEND` | Devices that claim ops |
|---|---|
| unset, `16` | LNS16 |
| `32` | LNS32 |
| `cuda`, `cuda16` | LNSCUDA16, with LNS16 for the ops the GPU backend does not implement |
| `cuda32` | LNSCUDA32, with LNS32 as fallback |
| device names, comma-separated | exactly those (e.g. `LNSCUDA16` alone disables the fallback) |

Unknown values are reported and ignored (falling back to LNS16).

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [xlnscpp](https://github.com/xlnsresearch/xlnscpp) | C++ LNS library — included as a **git submodule** |
| [xlnscuda](https://github.com/xlnsresearch/xlnscuda) | CUDA LNS backend — included as a **git submodule** |
| [ggml](https://github.com/ggml-org/ggml) | Tensor runtime for the backend and unit test |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | End-to-end LLM inference validation |

`setup.sh` uses each dependency at the commit pinned in
`patches/*-base-commit.txt`; a plain run never fetches upstream.  Only
`./setup.sh --update` fetches latest (`origin/main` for xlnscpp,
`origin/master` for ggml/llama.cpp), and it rolls xlnscpp back to its pin if
the build fails against that HEAD.  See
[Upstream integration](SETUP.md#upstream-integration) in SETUP.md.

For manual builds without `setup.sh`, populate the submodule first:

```bash
git submodule update --init
```

This ensures `#include "xlns16.cpp"` resolves correctly in the backend.

---

## Repository layout

| Path | Contents |
|---|---|
| `lns-backend/include/` | Public API headers (`ggml-lns.h`, `ggml-lns32.h`, `ggml-lns-cuda.h`) |
| `lns-backend/common/` | Shared by all backends: the backend framework, op-support predicates, runtime configuration, CMake helper ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) |
| `lns-backend/src/` | LNS16 backend (`ggml-lns.cpp`, `lns-ops.h`, `lns-ops.cpp`, `lns16-sum32.h`, `CMakeLists.txt`) |
| `lns-backend/src32/` | LNS32 backend (`ggml-lns32.cpp`, `lns32-ops.h`, `lns32-ops.cpp`, `CMakeLists.txt`) |
| `lns-backend/cuda/` | LNSCUDA16/32 backends: device registration, op handlers, device context and weight cache, kernels ([docs/CUDA.md](docs/CUDA.md)) |
| `lns-backend/tests/test-lns-backend.cpp` | Validation tests (runs against any LNS device) |
| `lns-backend/tests/test-lns-cuda-parity.cpp` | GPU-vs-CPU parity: LNSCUDA16 vs LNS16, LNSCUDA32 vs LNS32 |
| `lns-backend/tests/test-lns16-sum32.cpp` | Standalone sum32 invariant checks (needs no ggml) |
| `xlnscpp/` | xlnscpp submodule (16- and 32-bit LNS library); used at the pinned commit, advanced only by `--update` |
| `xlnscuda/` | xlnscuda submodule (CUDA LNS library); used as checked out |
| `docs/` | Architecture, GPU backend, and local-LLM guides |
| `patches/` | Pinned base commits for xlnscpp/ggml/llama.cpp + fallback patches (`ggml-core.patch`, `llama-core.patch`) |
| `scripts/lns-backends.sh` | The manifest: every backend and how it hooks into ggml |
| `scripts/integrate-lns.sh` | Integration of the LNS hooks into ggml/llama.cpp, driven by the manifest |
| `scripts/validate-latest.sh` | Compatibility gate against latest upstream |
| `scripts/lns-vs-cpu-quality.sh` | LNS-vs-CPU token-match + perplexity gate |
| `scripts/rc-logits.cpp` | Dump one forward pass's final-token logit vector |
| `scripts/lns-op-coverage.py` | Summarise an `LNS_OP_AUDIT=1` log: which backend runs each op, and which ops no LNS backend claims |
| `scripts/logit-compare.py` | Grade two logit dumps (rms / corr / rank) |
| `results/sum32-evidence/` | Committed evidence artefacts for the summation-mode measurements (see [SETUP.md](SETUP.md#regenerating-the-evidence-artefacts); bulk logit dumps are not shipped) |
| `setup.sh` | Automated build and run script |
| `run.sh` | Re-run inference after a successful setup (no rebuild) |
| `build/` | Created by `setup.sh` (ggml/llama.cpp clones + build trees); gitignored |

Model `.gguf` files are deliberately gitignored — download them per
[SETUP.md](SETUP.md#2-models-what-to-download).

---

## Core ggml changes (applied automatically by `setup.sh`)

The LNS hooks are applied by `scripts/integrate-lns.sh` for every backend in
the manifest `scripts/lns-backends.sh`, falling back to
`patches/ggml-core.patch` / `patches/llama-core.patch` if an anchor no longer
matches. The per-backend hooks are generated blocks between
`>>> xlns-ggml-backend:` marker comments, rewritten on every run. Paths below
are relative to the fetched ggml tree, which `setup.sh` keeps at `build/ggml`
(and inside llama.cpp's vendored `ggml/`):

| File | Change |
|---|---|
| `include/ggml.h` | Insert `GGML_TYPE_LNS16` at the current `GGML_TYPE_COUNT` value, then bump `GGML_TYPE_COUNT` by 1 |
| `src/ggml.c` | Register LNS16 type traits (`blck_size=1`, `type_size=2`, not quantized) |
| `CMakeLists.txt` | Add `GGML_LNSCUDA`, `GGML_LNS` and `GGML_LNS32` CMake options (default OFF) |
| `src/CMakeLists.txt` | Add `ggml_add_backend(...)` for each |
| `src/ggml-backend-reg.cpp` | Add each backend's include and registration (GPU backends first) |

### Build and validate

The supported path needs no model:

```bash
./setup.sh --unit-test --skip-inference
```

It clones ggml at the pinned commit into `build/ggml`, applies the hooks,
copies the backend sources and the test into the tree, and builds + runs
`test-lns-backend` once per summation mode plus once for LNS32. With
`--cuda` it also builds the GPU backends and runs `test-lns-cuda-parity` and
`test-lns-backend` on LNSCUDA16/LNSCUDA32.

The manual equivalent, after `setup.sh` has prepared `build/ggml` at least
once.  Every flag matters: BLAS and Metal default ON on macOS, and a
BLAS-enabled "LNS" build silently runs its mat-muls on BLAS — exactly the
contamination `scripts/lns-vs-cpu-quality.sh` refuses:

```bash
cmake -S build/ggml -B build/ggml/build-lns-base \
  -DGGML_LNS=ON -DGGML_LNS32=ON -DGGML_LNS16_SUM=base \
  -DGGML_LNS16_MONTE_BYLPVIP=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF \
  -DGGML_CPU_REPACK=OFF -DCMAKE_BUILD_TYPE=Release -DXLNSCPP_DIR=$PWD/xlnscpp
cmake --build build/ggml/build-lns-base --target test-lns-backend -j4

# Validate (per backend; second arg = tolerance profile, which MUST match the
# GGML_LNS16_SUM the binary was built with — a mismatch is refused, not graded)
cd build/ggml/build-lns-base
GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 base
GGML_LNS_BACKEND=32 ./bin/test-lns-backend LNS32
```

### End-to-end inference (SmolLM2-135M-Instruct, Q4_K_M)

```bash
# Builds all four LNS16 sum-mode variants + the pure-CPU reference (build-ref),
# then runs SmolLM2 per mode (n=20) plus one short LNS32 run (n=4)
./setup.sh
./setup.sh --sum-mode=lpvip      # build/run only ONE variant
./setup.sh --sum-mode=sum32

# Or with Llama 3.2 1B as well
MODEL_LLAMA32=./Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh

# After setup: re-run without rebuilding (see SETUP.md for flags)
./run.sh
./run.sh --backend=32
```

`scripts/validate-latest.sh` runs the full gate against latest upstream
ggml/llama.cpp: unit tests for all four sum modes plus LNS32, SmolLM2
inference in every configuration, then Llama 3.2 1B (LNS16). Both GGUFs are
required — see [SETUP.md](SETUP.md).

---

## License

MIT — see [LICENSE](LICENSE).
