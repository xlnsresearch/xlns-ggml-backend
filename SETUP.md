# Reproducing the Results

## Prerequisites

- `git`, `cmake`, `g++`, `make`, `perl`
- `python3` with `numpy` — used by `scripts/logit-compare.py` and the
  perplexity math in `scripts/lns-vs-cpu-quality.sh`
- the `hf` CLI for model downloads (`pip install -U huggingface_hub`)
- Disk: ~2.5 GB for the source clones and the default four-variant build,
  plus the models you download (0.1–10 GB each)
- Platform: the CPU backends were developed and measured on macOS / Apple
  Silicon (M3, 16 GB); the GPU backends on Ubuntu 24.04 with an RTX 2050.
  All builds deliberately pass `-DGGML_BLAS=OFF -DGGML_METAL=OFF` so the LNS
  backends actually execute the math (see the CPU-reference warning below).
- On 64-bit ARM Linux the compilers must have the NEON intrinsics llama.cpp
  uses (`vld1q_s8_x4` and friends); `setup.sh` checks this first. Ubuntu
  18.04's default GCC 7.5 (e.g. NVIDIA Jetson with JetPack 4) does not: install
  a newer one (`sudo add-apt-repository ppa:ubuntu-toolchain-r/test && sudo apt
  install gcc-11 g++-11`) and run `CC=gcc-11 CXX=g++-11 ./setup.sh`. Build
  directories configured with a different compiler are wiped and rebuilt
  automatically.
- Optional, for the GPU backends (`--cuda`): an NVIDIA GPU and driver, the CUDA
  toolkit (`nvcc`), and a host compiler `nvcc` accepts (`setup.sh` finds one,
  e.g. `g++-12` for CUDA 12.0). See [docs/CUDA.md](docs/CUDA.md).

## Steps

### 1. Clone and initialise

```bash
git clone https://github.com/xlnsresearch/xlns-ggml-backend
cd xlns-ggml-backend
git submodule update --init   # optional — setup.sh also initialises xlnscpp (and xlnscuda with --cuda)
```

`setup.sh` uses `xlnscpp/` (and, with `--cuda`, `xlnscuda/`) at the pinned commits in `patches/xlnscpp-base-commit.txt` and `patches/xlnscuda-base-commit.txt` (which match the recorded submodule state). Only `./setup.sh --update` advances them to latest `origin/main`.

### 2. Models: what to download

No `.gguf` ships with the repo (they are gitignored on purpose). Download what
you need into the repo root:

| Model | Size | Needed for |
|---|---|---|
| SmolLM2-135M-Instruct-Q4_K_M | 105 MB | default model for `setup.sh` / `run.sh` — start here |
| Llama-3.2-1B-Instruct-Q4_K_M | 808 MB | the `MODEL_LLAMA32` runs and `scripts/validate-latest.sh` |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M | 1.1 GB | reproducing the sum32 rescue ([Model compatibility](#model-compatibility)) |
| gemma-2-2b-it-Q4_K_M | 1.7 GB | additional model for the robustness grading |
| DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M | 10.4 GB | additional model for the robustness grading |

```bash
pip install -U huggingface_hub   # provides the `hf` CLI
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir .
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir .
hf download bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --local-dir .
hf download bartowski/gemma-2-2b-it-GGUF gemma-2-2b-it-Q4_K_M.gguf --local-dir .
hf download bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf --local-dir .
```

On huggingface_hub < 1.0 the older `huggingface-cli download` spelling still
works; it was removed in v1.0.

### 3. Run

```bash
# Build ALL four xlns16 summation variants + the pure-CPU reference, then run
# end-to-end LNS inference on SmolLM2 per variant (default model path is the
# repo-root .gguf). Runs LNS16 (n=20 tokens) per mode plus one short LNS32 run
# (n=4; disable with LNS32_INFERENCE=0)
./setup.sh

# Unit test + SmolLM2 inference (Llama 3.2 only if MODEL_LLAMA32 is set)
./setup.sh --unit-test

# Unit test only (no model needed) — runs the LNS16 suite per sum mode and the LNS32 suite
./setup.sh --unit-test --skip-inference

# Build only ONE summation variant instead of all four: base | monte | lpvip | sum32
./setup.sh --sum-mode=lpvip

# Update to latest upstream (the only mode that fetches new upstream work); on
# success the pins in patches/ are rewritten to the new commits ("last known good")
./setup.sh --update

# Force-reset the source trees to the pinned commits (discards in-tree edits)
./setup.sh --clean

# Also build the GPU backends (LNSCUDA16/LNSCUDA32), test them against the CPU
# backends, and run GPU inference; combine with any of the above. Limit the
# build's parallelism on machines with little RAM with JOBS=N.
./setup.sh --cuda --unit-test --skip-inference          # all four modes on CPU and GPU
./setup.sh --cuda --sum-mode=monte --unit-test --skip-inference

# Explicit model paths (MODEL_LLAMA32 enables the Llama 3.2 1B inference run)
MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
./setup.sh
```

Build layout: one directory per summation mode —
`build/llama.cpp/build-lns-{base,monte,lpvip,sum32}` (each contains
`llama-completion` and `llama-perplexity`), plus `build/llama.cpp/build-ref`
(pure CPU, no LNS/BLAS — the baseline for `scripts/lns-vs-cpu-quality.sh`).
The LNS trees are also configured with `-DGGML_BLAS=OFF -DGGML_METAL=OFF`,
since the quality gate refuses BLAS-enabled LNS binaries.
`build/llama.cpp/build-lns` is a symlink to the first mode built (`base` unless
`--sum-mode` restricted the set), so older paths keep working. With
`--unit-test`, the standalone ggml tree gets the same per-mode directories and
`build-lns` symlink under `build/ggml/` (containing `test-lns-backend`; there
is no `build-ref` there).

`MODEL=/path/to/model.gguf` remains supported as an alias for `MODEL_SMOLLM`.

At runtime, `GGML_LNS_BACKEND=16|32|cuda16|cuda32` (default `16`) selects which
backends claim ops — LNS16 and LNS32 are always built, the GPU backends with
`--cuda`. `LNS_OP_AUDIT=1` prints one JSON line per `supports_op` query
(including which backend answered and, for LNS16, the compiled sum mode);
`scripts/lns-op-coverage.py` summarises such a log.

## What success looks like

`./setup.sh` ends with `==> Done.` after running inference for every built
mode. The SmolLM2 LNS16 continuations look like this (`sum32` shown):

```
The capital of France is Paris. Paris is the political, cultural, and economic center of France. Paris is also the political
```

LNS16 output *diverging* from the pure-CPU reference after a few tokens while
staying coherent is expected — xlns16 rounding flips near-tie argmaxes even on
healthy models. Unit tests (`--unit-test`) print one `... result: PASS` line
per case (compare against the committed
`results/sum32-evidence/unit-tests-*.txt`) and abort on any failure.

Approximate wall-clock on an M3 MacBook (the backends are single-threaded):

| Run | Speed |
|---|---|
| SmolLM2-135M, LNS16 `base` | ~1.4 s/token (`sum32` ~1.66x slower) |
| SmolLM2-135M, LNS32 | ~2.9 s/token |
| Llama-3.2-1B, LNS16 | ~15 s/token |
| DeepSeek-1.5B, LNS16 | ~17–25 s/token |
| `llama-perplexity`, 1.5B model, one 256-token chunk | ~20 min |

With `--cuda`, on an RTX 2050 laptop GPU (LNSCUDA16 gives the same tokens as
LNS16 `base`): SmolLM2-135M ~12 tokens/s, Qwen2.5-0.5B ~10 tokens/s,
Llama-3.2-1B ~7 tokens/s; LNSCUDA32 about half that. See
[docs/CUDA.md](docs/CUDA.md#performance).

A first `./setup.sh` run is build-dominated (two clones + five llama.cpp build
trees) — expect tens of minutes. Later runs skip whatever is already built.

## Re-run inference

After a successful `./setup.sh`, re-run the model without rebuilding:

```bash
./run.sh                          # LNS16, base sum mode, default SmolLM2 + prompt
./run.sh --sum-mode=sum32         # pick a summation-variant binary
./run.sh --sum-mode=all -n 8      # same prompt across all four variants, in sequence
./run.sh --backend=32             # LNS32; default -n 4
./run.sh --backend=cuda16 --sum-mode=lpvip   # GPU (needs ./setup.sh --cuda); any of the 4 modes
./run.sh -m ./other_model_file.gguf -n 40
./run.sh -p "Hello" -- --temp 0.2 # extra args after -- go to llama-completion
```

`run.sh` selects `build/llama.cpp/build-lns-<mode>/bin/llama-completion`
and checks via `--list-devices` that it really runs that mode (and has the
requested device); the legacy `build-lns` binary is used only if it passes. If no binary exists, it tells you to run `./setup.sh` first.
Each summation mode is a separate compile-time binary produced by `setup.sh`;
`run.sh` only chooses among them.

### `run.sh` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--sum-mode=MODE` | `base` (or `$LNS16_SUM`) | `base`\|`monte`\|`lpvip`\|`sum32`\|`all` — which variant binary; `all` runs every variant in turn |
| `--backend=B` | `16` (or `$GGML_LNS_BACKEND`) | Which LNS backend claims ops: `16`, `32`, `cuda16` (alias `cuda`), `cuda32` |
| `-m`, `--model PATH` | `$MODEL` / `$MODEL_SMOLLM` / repo-root SmolLM2 GGUF | Model file |
| `-p`, `--prompt TEXT` | `The capital of France is` | Prompt |
| `-n`, `--tokens N` | `4` for CPU LNS32, else `20` | Tokens to predict |
| `-h`, `--help` | — | Show usage |
| `-- …` | — | Forward remaining args to `llama-completion` |

Space-separated forms (`--sum-mode MODE`, `--backend N`) are accepted too.
Always passed to `llama-completion`: `-fa off -no-cnv`.

Useful env vars: `GGML_LNS_BACKEND`, `MODEL` / `MODEL_SMOLLM`, `LNS_OP_AUDIT=1`,
and for the GPU backends `GGML_LNS_CUDA_DEVICE` and `GGML_LNS_CUDA_CACHE_MB`.

### Important `llama-completion` flags

Verified from `build/llama.cpp/build-lns/bin/llama-completion --help`. Pass any
of these after `--` on `./run.sh`, or call the binary directly. Full list:
`llama-completion --help`.

| Flag | Meaning |
|------|---------|
| `-m`, `--model FNAME` | Model path to load |
| `-p`, `--prompt PROMPT` | Prompt to start generation with |
| `-n`, `--predict`, `--n-predict N` | Tokens to predict (`-1` = infinity) |
| `-c`, `--ctx-size N` | Prompt context size (`0` = from model) |
| `-t`, `--threads N` | CPU threads for generation |
| `-fa`, `--flash-attn [on\|off\|auto]` | Flash Attention (`run.sh` sets `off`) |
| `-cnv` / `-no-cnv` | Conversation mode on/off (`run.sh` forces `-no-cnv`) |
| `-sys`, `--system-prompt PROMPT` | System prompt (chat template) |
| `-f`, `--file FNAME` | Read prompt from a file |
| `-s`, `--seed SEED` | RNG seed (`-1` = random) |
| `--temp N` | Temperature (default `0.80`) |
| `--top-k N` / `--top-p N` / `--min-p N` | Sampling controls |

## Model compatibility

All runs greedy (`--temp 0 --seed 1`, `-fa off`); CPU reference = the
`build-ref` pure-CPU build:

| Model | LNS16 `base` | LNS16 `sum32` | LNS32 |
|---|---|---|---|
| SmolLM2-135M | coherent | coherent | coherent |
| Llama-3.2-1B | coherent | **token-identical to CPU** | coherent |
| gemma-2-2b-it | coherent | coherent | coherent |
| DeepSeek-R1-Distill-Qwen-1.5B | degenerate | coherent | coherent |
| DeepSeek-Coder-V2-Lite-Instruct | coherent | coherent | coherent |

Graded across 3 prompts x 2 qwen2-family models, `sum32` is coherent in 6/6
prompt-model combinations, `base` in 0/6 —
[results/sum32-evidence/robust-sum32.txt](results/sum32-evidence/robust-sum32.txt).

Why the qwen2 family breaks on `base` — two independent consequences of
xlns16 having only 7 fraction bits:

1. **Sequential summation stagnates.** `xlns16_add(S, t)` returns `S`
   unchanged once `t/S` drops below half an ULP, so a same-sign running sum
   saturates after ~370 terms (summing 1.0 forever converges to 369.97 — see
   [results/sum32-evidence/standalone-sum32-invariants.txt](results/sum32-evidence/standalone-sum32-invariants.txt)).
   Every reduction in a 1.5B qwen2 model is far past that point (rms_norm
   n=1536, mat-mul K=1536 and K=8960). This is what `-DGGML_LNS16_SUM=sum32`
   fixes: each term is widened to xlns32, accumulated with `xlns32_add`, and
   rounded back to xlns16 once per reduction.
2. **Some attention scores are not representable.** qwen2 carries per-head
   attention biases that push |Q·K| to ~2.3e5. xlns16 has constant *relative*
   precision, so at that magnitude one ULP is ~1230 absolute, while the
   key-to-key differences softmax must resolve are only ~700 — distinct keys
   collapse onto the same 16-bit code. No accumulator, however wide, recovers
   an ordering the storage format cannot express. The same storage-grid effect
   shows up even under `sum32`: on "The largest planet in our solar system is"
   the CPU reference separates `" Jupiter"` from `" Earth"` by 0.82 logits,
   while LNS16 lands both on the same xlns16 code (gap 0.000) and the
   tie-break emits `" Earth"` (see
   [results/sum32-evidence/robust-sum32.txt](results/sum32-evidence/robust-sum32.txt)).

Practical guidance:

- Prefer `--sum-mode=sum32` on LNS16. For precision-sensitive models run
  `GGML_LNS_BACKEND=32` (xlns32, ~23 fraction bits, near-f32 — fully coherent
  and correct on every model above).
- Always run reasoning-distill models with their chat template
  (`-cnv --jinja`); raw-completion prompts are off-distribution.
- To verify a suspected LNS regression quantitatively, use the quality gate
  (uses the build-ref pure-CPU binaries as reference):

```bash
# greedy divergence report + perplexity delta vs the CPU reference
./scripts/lns-vs-cpu-quality.sh -m ./DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
# quick SmolLM2-only check (default model), divergence only:
./scripts/lns-vs-cpu-quality.sh --skip-ppl
```

Note: perplexity on the LNS backends is single-threaded — one 256-token chunk
(`-c 256 --chunks 1`, the default) on a 1.5B model takes on the order of 20
minutes on LNS16.

> **Warning: what counts as a "CPU reference".** Inside the LNS-enabled
> `llama-completion` binary, neither `--no-op-offload` nor `--device none`
> gives you CPU-only execution. `--no-op-offload` only disables the
> weight-buffer offload path in `ggml-backend.cpp`, and `--device none` only
> affects layer offload; the scheduler still assigns ops to LNS16/LNS32/BLAS
> (verify with `GGML_SCHED_DEBUG=2 ... -v` and look at the `SPLIT #n: BACKEND`
> lines). The only trustworthy CPU reference is a separate build with
> `-DGGML_LNS=OFF -DGGML_LNS32=OFF -DGGML_BLAS=OFF` (this repo keeps one at
> `build/llama.cpp/build-ref`; `--list-devices` on that binary prints
> "(none)"). An earlier version of this section compared LNS16 against a
> contaminated "CPU" run that was itself mostly LNS16 — the two matched
> bit-for-bit precisely because both were the same backend, and the wrong
> conclusion ("not an LNS bug") followed.

Debug/bisection knobs (LNS16 and LNS32; `LNS_MM_ROUNDTRIP` is LNS16-only):

- `LNS_DISABLE_OPS="ADD,ROPE,..."` — force listed ops (by ggml op name, or by
  substring of the tensor name, e.g. `ffn_down`, `result_output`; substrings
  are case-sensitive) to fall back to CPU. No rebuild needed.
- `LNS_MM_ROUNDTRIP=1|2|3` — in `lns_mul_mat`, round operands to xlns16 but
  accumulate the dot product in IEEE float (1=both operands, 2=weights only,
  3=activations only). Isolates operand quantization from LNS accumulation.
- `LNS_OP_AUDIT=1` — log every `supports_op` query as JSON on stderr.
- `GGML_SCHED_DEBUG=2` + `-v` — ggml prints which backend every graph node is
  assigned to; the only reliable way to check what actually ran where.

## Regenerating the evidence artefacts

Everything in `results/sum32-evidence/` can be re-measured:

- `unit-tests-lns16-{base,monte,lpvip,sum32}.txt`, `unit-tests-lns32.txt` —
  the per-mode `test-lns-backend` output from
  `./setup.sh --unit-test --skip-inference`.
- `standalone-sum32-invariants.txt` — the no-ggml standalone check:

  ```bash
  g++ -O2 -std=c++17 -I xlnscpp -I lns-backend/src lns-backend/tests/test-lns16-sum32.cpp -o /tmp/test-lns16-sum32 && /tmp/test-lns16-sum32
  ```

- `robust-sum32.txt` — for each of the three prompts and two models in the
  file's header, a greedy run per configuration, e.g.:

  ```bash
  ./run.sh --sum-mode=sum32 -n 12 -m ./gemma-2-2b-it-Q4_K_M.gguf \
      -p "The capital of France is" -- --temp 0 --seed 1
  ./run.sh --sum-mode=sum32 -n 12 -m ./DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf \
      -p "The capital of France is" -- --temp 0 --seed 1
  ```

  (repeat with `--sum-mode=base`, and against
  `build/llama.cpp/build-ref/bin/llama-completion` for the reference rows).
- `deepseek-logit-ladder.txt` — build `scripts/rc-logits.cpp` once per
  llama.cpp tree and grade with `scripts/logit-compare.py`:

  ```bash
  T=build/llama.cpp/build-lns-sum32
  g++ -O2 -std=c++17 -I build/llama.cpp/include -I build/llama.cpp/ggml/include \
      scripts/rc-logits.cpp -o /tmp/rc-logits-sum32 \
      -L $T/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$T/bin
  GGML_LNS_BACKEND=16 /tmp/rc-logits-sum32 ./DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
      "The capital of France is" /tmp/sum32.bin 20
  python3 scripts/logit-compare.py /tmp/ref.bin /tmp/sum32.bin --token 12095
  ```

  (`/tmp/ref.bin` comes from the same `rc-logits` built against `build-ref`.
  Read rms/corr/rank together — a rank 0 reached with a top-1/top-2 gap of
  0.000 is a coin flip, not a fix.)

Note: the artefact headers name the CPU reference `build-ref-v2` — that was
the reference build's directory name at measurement time; it is the same
configuration `setup.sh` now produces at `build/llama.cpp/build-ref`.

## Upstream integration

By default `setup.sh` is pin-first: each dependency is used at the commit
recorded in `patches/*-base-commit.txt` (cloned on first run; a tree already
at its pin is left untouched, so manual in-tree experiments survive re-runs).
`./setup.sh --update` is the only mode that fetches new upstream work — setup
still clones missing trees on first run, initialises the submodule, and
fetches once if a pinned commit is absent from the local object store:

| Dependency | Pin file | `--update` target | Integration |
|---|---|---|---|
| [xlnscpp](https://github.com/xlnsresearch/xlnscpp) | `patches/xlnscpp-base-commit.txt` | `origin/main` | submodule init + checkout |
| [ggml](https://github.com/ggml-org/ggml) | `patches/ggml-base-commit.txt` | `origin/master` | regex edits in [`scripts/integrate-lns.sh`](scripts/integrate-lns.sh) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `patches/llama-base-commit.txt` | `origin/master` | regex edits in [`scripts/integrate-lns.sh`](scripts/integrate-lns.sh) |
| [xlnscuda](https://github.com/xlnsresearch/xlnscuda) (`--cuda` only) | `patches/xlnscuda-base-commit.txt` | `origin/main` | submodule init + checkout; must provide `xlns16d_table_init` |

The edits are driven by the backend manifest `scripts/lns-backends.sh` and
written as generated blocks between `>>> xlns-ggml-backend:` markers, so
re-running integration is idempotent; trees hooked by an older version of the
script are restored from git and re-integrated automatically. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#build-integration).

If the regex integration fails (in any mode), setup resets that tree to its
pinned commit and applies `patches/ggml-core.patch` /
`patches/llama-core.patch`. Under `--update`, if the *build* fails, only
xlnscpp is rolled back to its pin and the build retried; ggml/llama.cpp remain
at the fetched HEAD (a build failure in default pin-first mode just errors
out). When an `--update` run succeeds end-to-end, the pin files and fallback
patches are rewritten to the new commits ("last known good") — commit the
`patches/` diff to make the new pins permanent.

To re-validate all three against latest upstream before refreshing the fallback pins
(requires both GGUFs — defaults to the repo-root SmolLM2 and Llama 3.2 1B paths):
```bash
MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
./scripts/validate-latest.sh
```

The gate runs unit tests for all four LNS16 sum modes plus LNS32, SmolLM2
inference in every configuration, then a Llama 3.2 1B LNS16 run.

## What is in this repo

See the [Repository layout](README.md#repository-layout) table in the README.
