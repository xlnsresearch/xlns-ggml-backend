# Layer by layer: LNS backends and summation modes against F32, and CPU against GPU

Two questions, for every LNS backend and every xlns16 summation mode:

1. **Where does LNS error enter an LLM, and how does it grow through the
   layers?** Every intermediate tensor of one forward pass is compared with
   the same tensor from the plain F32 CPU build.
2. **Does the GPU compute exactly what the CPU computes?** Every tensor of the
   GPU run is compared bit for bit with the CPU run of the same arithmetic.

## Files

| File | Role |
|---|---|
| `layer-dump.cpp` | One forward pass of a prompt; saves the output of every named F32 tensor (llama.cpp's `cb_eval` hook): per layer the norms, Q/K/V, `kq`, `kq_soft_max`, `kqv`, `attn_out`, the FFN tensors and the residual stream `l_out`; then `result_norm` and the logits `result_output`. |
| `layer-compare.py` | Compares dumps: relative RMS error of every tensor against the reference, and bit-for-bit CPU vs GPU. Writes a markdown report. Plain Python. |
| `run.sh` | Builds `layer-dump` against each build tree, runs every configuration (see [../README.md](../README.md)) and writes `results/<model>.md`. |
| `cpu-vs-gpu.sh` | End-to-end CPU vs GPU per mode: parity test, unit-test figures, greedy text and speed, perplexity. |
| [CPU-vs-GPU.md](CPU-vs-GPU.md) | The CPU vs GPU report: results and every command behind them. |
| `results/` | Per-model layer-by-layer reports. |

## How to run

```bash
./setup.sh --cuda --unit-test                         # all 4 modes, CPU and GPU
./experiments/layer-by-layer/run.sh -m SmolLM2-135M-Instruct-Q4_K_M.gguf \
    -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
./experiments/layer-by-layer/cpu-vs-gpu.sh -m SmolLM2-135M-Instruct-Q4_K_M.gguf
```

`run.sh` takes `-p PROMPT` (default "The capital of France is") and
`-o OUT_DIR`; `SKIP_CPU32=1` skips the slow xlns32 CPU run. A configuration
whose build or device is missing is skipped and listed at the end of the report.

## Findings

- **CPU = GPU.** Every xlns16 mode is bit-identical on the two devices in every
  tensor of both models. xlns32 differs only from `kq_soft_max-0` on, by float
  libm last-ulp differences. Details in [CPU-vs-GPU.md](CPU-vs-GPU.md).
- **Accuracy order** (final logits vs F32): xlns32 (4–7%) < sum32 (16–18%) <
  monte (26–28%) < base (37–43%) < lpvip (69–131%). On Qwen2.5-0.5B every xlns16
  mode changes the top-1 token; xlns32 keeps it.
- **Where the error comes from**: the largest per-tensor errors are in the
  attention output `kqv` and the FFN chain (`ffn_swiglu`, `ffn_out`); the softmax
  itself adds little. On SmolLM2 the residual stream error grows ~1% per layer in
  the 16-bit modes (layers 11–27) and stays flat at ~1.2% in xlns32.

## What the report contains

For each model (`results/<model>.md`):

1. **Final logits** per configuration: relative error, cosine similarity, and
   whether the top-1 token matches the F32 reference.
2. **Residual stream (`l_out`) error per layer**: how the error accumulates
   with depth, one column per configuration.
3. **Error per tensor type** (mean / max over layers): which operation adds
   the error (e.g. the softmax or the FFN dot products).
4. **CPU vs GPU**: for every mode, how many of the tensors are bit-identical
   on the two devices, the first one that differs and the largest difference.

Relative error is ||x − ref|| / ||ref|| over the whole tensor. Each process
starts the Monte random sequence at seed 7, so the `monte` CPU and GPU runs
draw the same random numbers.
