# CPU vs GPU report — LNS16 vs LNSCUDA16 (base, monte, lpvip) and LNS32 vs LNSCUDA32

Measured 2026-11-10 on an RTX 2050 laptop GPU (sm_86, 4 GB, driver 595.99.02), CUDA 12.0
with g++-12, 12-thread x86-64 CPU, Ubuntu 24.04. xlnscpp `ac891da` + `the xlnscpp changes proposed upstream (patches/xlnscpp-PR.md)`,
xlnscuda `864d0c9` + `the xlnscuda additions proposed upstream (patches/xlnscuda-PR.md)`. Model: SmolLM2-135M-Instruct Q4_K_M
unless noted. Prompt: "The capital of France is". The CPU backends are single-threaded.

**Summary.** In every summation mode the GPU computes exactly what the CPU computes: every
op-level parity check, every unit-test error figure, all 49,152 final-token logits and the
greedy text are identical. LNS32/LNSCUDA32 differ only in the last ulp of libm activations
(as before this work). The GPU is 17-21x faster in base and monte, 5x in lpvip.

## 1. Op-level parity (`test-lns-cuda-parity`)

| Build | Result |
|---|---|
| base | 37/37 bit-identical |
| monte (pure16) | 75/75 — incl. same LFSR snapshots, draw counts, two runs without reset, 2nd backend instance, MUL_MAT_ID CPU fallback inside a scheduled GPU graph |
| monte (bylpvip) | 75/75 |
| lpvip | 37/37 |
| xlns32 (LNSCUDA32 vs LNS32) | 37/37 within 1e-5 scaled error (last-ulp libm differences in silu/gelu/tanh/softmax exp) |
| sum32 | LNSCUDA16 pair refused: there is no GPU sum32 |

Cases: mul_mat with F32/F16/Q4_0/Q8_0/Q4_K weights, K=1/1536/2048/4096, tail columns,
N=1, GQA and 4D broadcasting, permuted F16 KV view; add/mul/div/scale/silu/gelu/relu/tanh/
swiglu/geglu; rms_norm (576, 2048, singleton rows); softmax (plain, masked, long rows,
singleton rows); rope (normal, neox, frequency factors); and a mixed attention/FFN block.

## 2. Unit-test error figures (`test-lns-backend`, CPU device vs GPU device)

Each cell is "CPU / GPU" against the float reference; 38/38 cases pass on both devices in
every xlns16 mode (37/37 for xlns32). Monte starts from seed 7 on both.

| test-lns-backend metric | base CPU / GPU | monte CPU / GPU | lpvip CPU / GPU | monte-bylpvip CPU / GPU | lns32 CPU / GPU |
|---|---|---|---|---|---|
| mul mat | 0.5895% / 0.5895% | 0.5895% / 0.5895% | 1.6114% / 1.6114% | 1.1263% / 1.1263% | 0.0000% / 0.0000% |
| mul mat long | 4.3942% / 4.3942% | 4.7150% / 4.7150% | 57.2137% / 57.2137% | 56.6310% / 56.6310% | 0.0122% / 0.0122% |
| mul_mat_id non-broadcast | 1.3414% / 1.3414% | 1.3414% / 1.3414% | 2.3691% / 2.3691% | 1.7269% / 1.7269% | 0.0001% / 0.0001% |
| mul_mat_id broadcast ne11=1 | 1.0163% / 1.0163% | 1.0163% / 1.0163% | 1.6549% / 1.6549% | 2.1669% / 2.1669% | 0.0000% / 0.0000% |
| mul_mat_id partial broadcast ne11=2 | 0.7606% / 0.7606% | 0.7606% / 0.7606% | 1.7662% / 1.7662% | 1.8797% / 1.8797% | 0.0000% / 0.0000% |
| mul_mat_id dst=LNS16 | 1.0163% / 1.0163% | 1.0163% / 1.0163% | 1.6549% / 1.6549% | 2.3982% / 2.3982% | - / - |
| div | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.0000% / 0.0000% |
| div scalar broadcast | 0.4728% / 0.4728% | 0.4728% / 0.4728% | 0.4728% / 0.4728% | 0.4728% / 0.4728% | 0.0000% / 0.0000% |
| div 2d broadcast | 0.4728% / 0.4728% | 0.4728% / 0.4728% | 0.4728% / 0.4728% | 0.4728% / 0.4728% | 0.0000% / 0.0000% |
| swiglu split | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.0000% / 0.0000% |
| swiglu fused | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.0000% / 0.0000% |
| swiglu fused swapped | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.3682% / 0.3682% | 0.0000% / 0.0000% |
| geglu split | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.0000% / 0.0000% |
| geglu fused | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.0000% / 0.0000% |
| geglu fused swapped | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.5821% / 0.5821% | 0.0000% / 0.0000% |
| tanh | 0.2921% / 0.2921% | 0.2921% / 0.2921% | 0.2921% / 0.2921% | 0.2921% / 0.2921% | 0.0000% / 0.0000% |
| clamp | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% |
| clamp +inf max | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.0000% / 0.0000% |
| clamp -inf min | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.1119% / 0.1119% | 0.0000% / 0.0000% |
| clamp min>max | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% |
| clamp 2d | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% | 0.0000% / 0.0000% |
| softmax short | 0.002120 / 0.002120 | 0.002120 / 0.002120 | 0.002018 / 0.002018 | 0.002120 / 0.002120 | 0.000000 / 0.000000 |
| softmax long | 0.000805 / 0.000805 | 0.000819 / 0.000819 | 0.000361 / 0.000361 | 0.000720 / 0.000720 | 0.000000 / 0.000000 |
| softmax scaled | 0.000206 / 0.000206 | 0.000458 / 0.000458 | 0.000074 / 0.000074 | 0.000137 / 0.000137 | 0.000000 / 0.000000 |
| softmax masked | 0.002455 / 0.002455 | 0.004015 / 0.004015 | 0.004318 / 0.004318 | 0.005650 / 0.005650 | 0.000000 / 0.000000 |
| softmax masked scaled | 0.000287 / 0.000287 | 0.000287 / 0.000287 | 0.000129 / 0.000129 | 0.000345 / 0.000345 | 0.000000 / 0.000000 |
| rms_norm short | 0.4374% / 0.4374% | 0.4374% / 0.4374% | 1.2722% / 1.2722% | 1.5099% / 1.5099% | 0.0000% / 0.0000% |
| rms_norm 576 | 4.0681% / 4.0681% | 5.6847% / 5.6847% | 0.7656% / 0.7656% | 3.4992% / 3.4992% | 0.0012% / 0.0012% |
| add bias broadcast (qwen2) | 1.6432% / 1.6432% | 1.6432% / 1.6432% | 1.6432% / 1.6432% | 1.6432% / 1.6432% | 0.0000% / 0.0000% |
| rope neox full-dim (qwen2) | 0.7887% / 0.7887% | 0.7887% / 0.7887% | 0.7887% / 0.7887% | 0.7887% / 0.7887% | 0.0001% / 0.0001% |

## 3. Final-token logits (`scripts/rc-logits.cpp`, 49,152 logits)

| Mode | CPU vs GPU logits: bit-identical / max abs diff | CPU vs F32 ref: rms / corr / rank of ref top token | GPU vs F32 ref: rms / corr / rank |
|---|---|---|---|
| base | 49152/49152 / 0 | 1.597 / 0.9560 / 0 | 1.597 / 0.9560 / 0 |
| monte | 49152/49152 / 0 | 1.095 / 0.9613 / 0 | 1.095 / 0.9613 / 0 |
| lpvip | 49152/49152 / 0 | 5.612 / 0.7757 / 25 | 5.612 / 0.7757 / 25 |
| xlns32 (LNS32 / LNSCUDA32) | 19/49152 / 0.00164 | 0.297 / 0.9986 / 0 | 0.297 / 0.9986 / 0 |

Rank = position of the F32 reference's top token (" Paris") in the LNS logits (0 = chosen).

## 4. Generation and speed (`llama-completion -n 16 --temp 0 --seed 1`)

| Mode | CPU greedy text == GPU | CPU prompt / gen tok/s | GPU prompt / gen tok/s | gen speedup |
|---|---|---|---|---|
| base | yes | 1.16 / 0.87 | 36.79 / 14.41 | 17x |
| monte | yes | 0.60 / 0.49 | 24.08 / 10.20 | 21x |
| lpvip | yes | 0.76 / 0.59 | 4.45 / 2.92 | 5x |
| xlns32 | yes | 0.50 / 0.28 | 12.41 / 5.47 | 20x |


Greedy texts (identical on CPU and GPU):
- base: "Paris. Paris is the political, cultural, and economic center of France. Paris"
- monte: "Paris. Paris is the largest city in France and is located in the northern part"
- lpvip: "the capital of the French, and the French, and the French, and the"
- xlns32: "Paris. Paris is a city known for its historical landmarks, art museums, fashion"

Llama-3.2-1B-Instruct Q4_K_M, 6 tokens: identical CPU/GPU text in monte ("Paris. Paris is
famous for") and lpvip ("the the the ...").

## 5. Perplexity (`scripts/lns-vs-cpu-quality.sh`, one 256-token chunk; F32 reference 1.0635)

| Mode | CPU LNS16 | GPU LNSCUDA16 | Gate (15%) |
|---|---|---|---|
| base | 1.1874 (+11.65%) | 1.1874 (+11.65%) | pass |
| monte | 1.1758 (+10.56%) | 1.1758 (+10.56%) | pass |
| lpvip | 2.1556 (+102.69%) | 2.1556 (+102.69%) | fail — LPVIP's mixed-sign cancellation defect, on both devices |

## 6. Arithmetic library (`xlnscuda/tests/chkxlns16d_sum_modes.cu`, both Monte variants)

0 mismatches in: LFSR step and jump-ahead (67,535), Monte add result and state
(70,451,200), LPVIP singularity branch (all 8,388,607 inputs), LPVIP add (72,548,352),
Monte and LPVIP dot products (K=1, 1536, 70,001).

## 7. Why lpvip is slower on the GPU

16,384 dot products of K=576: base 2.9 ms, monte 4.9 ms, lpvip 16.2 ms on mixed-sign
data, lpvip 6.6 ms on same-sign data. The opposite-sign "singularity" branch evaluates
`log`/`pow` in double precision (1/64 rate on consumer GPUs): ~2.5x; the 32-bit
accumulator: ~1.9x over base.

## How to reproduce

```bash
# builds + all model-free tests (CPU-only, then every GPU mode)
./setup.sh --unit-test --skip-inference
./setup.sh --cuda --unit-test --skip-inference
./setup.sh --cuda --sum-mode=monte --unit-test --skip-inference      # or base / lpvip
./setup.sh --cuda                                                    # + llama.cpp builds and inference

# 1 + 2: parity and unit tests in one mode's ggml tree (MODE = base|monte|lpvip)
cd build/ggml/build-lns-MODE
./bin/test-lns-cuda-parity                      # --only=cuda16|cuda32, --expect-sum=MODE
GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 MODE
./bin/test-lns-backend LNSCUDA16 MODE
GGML_LNS_BACKEND=32 ./bin/test-lns-backend LNS32 && ./bin/test-lns-backend LNSCUDA32
cd -

# 3: logits (build rc-logits once per tree)
for t in build-lns-base build-lns-monte build-lns-lpvip build-ref; do
  g++ -O2 -std=c++17 -I build/llama.cpp/include -I build/llama.cpp/ggml/include scripts/rc-logits.cpp \
      -o /tmp/rc-$t -L build/llama.cpp/$t/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/build/llama.cpp/$t/bin
done
M=SmolLM2-135M-Instruct-Q4_K_M.gguf
/tmp/rc-build-ref $M "The capital of France is" /tmp/ref.bin
GGML_LNS_BACKEND=16     /tmp/rc-build-lns-monte $M "The capital of France is" /tmp/cpu.bin
GGML_LNS_BACKEND=cuda16 /tmp/rc-build-lns-monte $M "The capital of France is" /tmp/gpu.bin
cmp /tmp/cpu.bin /tmp/gpu.bin && echo identical
python3 scripts/logit-compare.py /tmp/ref.bin /tmp/gpu.bin      # needs numpy

# 4: generation, CPU vs GPU, same binary
./run.sh --backend=16     --sum-mode=monte -n 16 -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=monte -n 16 -- --temp 0 --seed 1
./run.sh --backend=cuda16 --sum-mode=all   -n 16 -- --temp 0 --seed 1   # base, monte, lpvip
./run.sh --backend=cuda16 --sum-mode=sum32                              # refused: no GPU sum32

# 5: perplexity vs the F32 reference
./scripts/lns-vs-cpu-quality.sh -b 16     --sum-mode=monte -m $M
./scripts/lns-vs-cpu-quality.sh -b cuda16 --sum-mode=monte -m $M

# 6: library test
nvcc -std=c++14 -ccbin g++-12 -I xlnscpp xlnscuda/tests/chkxlns16d_sum_modes.cu -o /tmp/chk && /tmp/chk
nvcc -std=c++14 -ccbin g++-12 -I xlnscpp -Dxlns16d_monte_bylpvip xlnscuda/tests/chkxlns16d_sum_modes.cu -o /tmp/chk2 && /tmp/chk2

# against the latest upstream ggml / llama.cpp (needs network and both GGUFs)
CUDAHOSTCXX=/usr/bin/g++-12 ./scripts/validate-latest.sh --cuda
```
