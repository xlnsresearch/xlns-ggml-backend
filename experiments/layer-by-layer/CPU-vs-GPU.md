# CPU vs GPU: LNS16 vs LNSCUDA16 (base, monte, lpvip, sum32) and LNS32 vs LNSCUDA32

Measured 2026-11-10 on an RTX 2050 laptop GPU (sm_86, 4 GB), CUDA 12.0 with g++-12,
12-thread x86-64 CPU, Ubuntu 24.04, with the post-merge library versions listed in
[../README.md](../README.md). Prompt "The capital of France is". The CPU backends are
single-threaded.

**Summary.** In every xlns16 summation mode the GPU computes exactly what the CPU
computes: every op-level parity case, every unit-test error figure, every intermediate
tensor of a forward pass (SmolLM2-135M and Qwen2.5-0.5B) and the greedy text are
identical. LNS32 and LNSCUDA32 differ only in the last ulp of float libm functions
(`expf`, `tanhf`), which shows first in the softmax and stays at ~1e-6 relative.

## 1. End to end (`cpu-vs-gpu.sh`, SmolLM2-135M-Instruct Q4_K_M)

| Mode | parity (GPU vs CPU) | unit-test figures equal | greedy text equal | prompt tok/s CPU → GPU | gen tok/s CPU → GPU | perplexity CPU | perplexity GPU |
|---|---|---|---|---|---|---|---|
| base | 37/37 | 68 / 68 | yes | 1.16 → 36.58 | 0.87 → 14.21 | 1.1874 (+11.65%) | 1.1874 (+11.65%) |
| monte | 75/75 | 68 / 68 | yes | 0.60 → 23.98 | 0.49 → 10.32 | 1.1758 (+10.56%) | 1.1758 (+10.56%) |
| lpvip | 37/37 | 68 / 68 | yes | 0.76 → 4.43 | 0.59 → 2.91 | 2.1556 (+102.69%) | 2.1556 (+102.69%) |
| sum32 | 37/37 | 68 / 68 | yes | 0.56 → 13.69 | 0.43 → 7.31 | 1.1088 (+4.26%) | 1.1088 (+4.26%) |
| xlns32 | 37/37 | 66 / 66 | yes | 0.49 → 12.40 | 0.28 → 5.47 | 1.0580 (−0.52%) | 1.0580 (−0.52%) |

Perplexity is one 256-token chunk against the F32 reference (1.0635). Parity for xlns16
is bit-identical; for xlns32 it is within 1e-5 scaled error. The monte parity run has
extra cases: identical LFSR state and draw counts after each graph, two runs without a
reset, a second backend instance, and a CPU-fallback op inside a GPU graph.

lpvip is slower on the GPU because its opposite-sign branch evaluates `log`/`pow` in
double precision (1/64 rate on consumer GPUs).

## 2. Layer by layer (`run.sh`)

Every named tensor of one forward pass, CPU against GPU, bit for bit:

| Pair | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| base | 696 / 696 identical | 630 / 630 identical |
| monte | 696 / 696 | 630 / 630 |
| lpvip | 696 / 696 | 630 / 630 |
| sum32 | 696 / 696 | 630 / 630 |
| xlns32 | 12 / 696, first `kq_soft_max-0`, max abs diff 0.0383 | 15 / 630, first `kq_soft_max-0`, max abs diff 1.39 |

Final-logit error against F32 (same on CPU and GPU; † = top-1 token differs from F32):

| Model | base | monte | lpvip | sum32 | xlns32 |
|---|---|---|---|---|---|
| SmolLM2-135M | 37.4% | 25.6% | 131% † | 17.5% | 6.88% |
| Qwen2.5-0.5B | 42.8% † | 28.3% † | 69.1% † | 16.0% † | 3.91% |

The per-layer and per-tensor-type breakdowns are in [results/](results/).

## How to reproduce

```bash
./setup.sh --cuda --unit-test                  # builds every mode, CPU and GPU, + tests

# section 1
./experiments/layer-by-layer/cpu-vs-gpu.sh -m SmolLM2-135M-Instruct-Q4_K_M.gguf

# section 2
./experiments/layer-by-layer/run.sh -m SmolLM2-135M-Instruct-Q4_K_M.gguf \
    -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

# single pieces, MODE = base | monte | lpvip | sum32
cd build/ggml/build-lns-MODE
./bin/test-lns-cuda-parity --only=cuda16             # --only=cuda32 in build-lns-base
GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 MODE
./bin/test-lns-backend LNSCUDA16 MODE
cd -
./run.sh --backend=16     --sum-mode=MODE -n 16 -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=MODE -n 16 -- --temp 0 --seed 1
./scripts/lns-vs-cpu-quality.sh -b cuda16 --sum-mode=MODE -m SmolLM2-135M-Instruct-Q4_K_M.gguf
```
