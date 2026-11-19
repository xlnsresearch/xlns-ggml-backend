# Dot-product error scaling

How does the error of one LNS dot product grow with its length K, for each
summation mode, and is it a bias or random? These are the mechanics behind
the per-layer numbers of the layer-by-layer experiment.

## Files

| File | Role |
|---|---|
| `dot-bench.cpp` | Drives the backend's own `mul_mat` (and `scale`, to read back converted values) through the ggml API on one LNS device. `synth` uses random inputs for K = 16 .. 16384. `replay` uses real weights from the GGUF and real activations from an F32 `layer-dump`. `convert` measures the F32 → LNS conversion. Writes one JSON line per point. |
| `run.py` | The queue: convert, synth, Monte averaging and replay, one process at a time, resumable. |
| `report.py` | `results/` → `results/REPORT.md`: error tables, power-law fits, Monte averaging, replay against prediction. |
| `long-prompt.txt` | About 440 tokens of text for the replay dumps; `kqv` at position t is a dot product of length t+1. |

## Method

- **What `acc` measures.** The kernel gets the original F32 inputs. The
  reference is the exact long-double sum of the products of the values the
  kernel actually uses after conversion, read back as `to_f32(from_f32(x))`
  from the same device with a `scale(x, 1)`. LNS multiplication is exact, so
  `acc` is exactly the error the accumulation adds. `total` is against the
  unconverted inputs and also includes the conversion.
- **Conversion is not idempotent.** The xlns16 table conversion is not a
  projection: converting a decoded value again changes about half the codes.
  So the decoded values cannot be fed back in as inputs, and the kernel always
  gets the originals. For xlns32 the read-back through F32 is rounded
  (about 6e-8), which sets a floor on `acc` at small K.
- **`convert`.** 4M values with log-uniform magnitudes (2^-20 .. 2^20) and
  random sign are converted once, twice and three times. This measures the
  bias and spread of the F32 → LNS conversion that every LNS op applies to
  its inputs.
- **Error of one dot product.** e = (computed − exact) / Σ|aᵢbᵢ|. The report
  gives `bias` = mean(e), `sd`, `rms` and the relative L2 over all outputs.
- **Configurations:** LNSCUDA16 with base, monte, lpvip and sum32 (one build
  tree each), and LNSCUDA32. They are bit-identical to the CPU backends,
  except that xlns32 can differ in the last ulp of libm.
- **Distributions:**
  - `signed`: N(0,1) × N(0,1), like the projection matmuls.
  - `positive`: |N(0,1)| × |N(0,1)|, like sums of squares and softmax
    denominators.
  - `attn`: N(0,1) × softmax(2·N(0,1)), an attention-weighted sum of V where
    K is the context length.
- **Monte averaging.** The same inputs are summed 32 times, each time with a
  different `ggml_backend_lns16_monte_set_state` seed. The averages of 1, 2,
  4 … 32 repeats test whether the dither is unbiased (error falls as 1/√R).
- **Replay.** For the first, middle and last layer of each model it replays
  `attn_q`, `ffn_up`, `ffn_down`, `attn_output` (weights dequantized from the
  GGUF) and `kqv` (softmax probabilities × V rounded through F16 like the KV
  cache, per head, bucketed by effective K = t+1). Each is compared with the
  synthetic curve of the matching distribution at the same K. The two
  normalizers are replayed too:
  - `rms_norm` on `ffn_inp-L`. Its final multiply is exact in LNS, so
    output / converted input gives back the 1/rms the kernel used. That is
    compared with the exact `1/sqrt(mean(x^2) + eps)`.
  - `softmax` on `kq-L` with the causal mask and scale 1/8. The divide is exact,
    so 1/sum(outputs) - 1 is the error of the accumulated denominator, bucketed
    by K = position + 1.

## How to run

`run.sh` does everything:
- builds the trees (`setup.sh --cuda`, which also runs its short inference checks);
- downloads the two Q4_K_M models if they are missing;
- runs synth, Monte averaging and replay;
- writes `results/REPORT.md`.

```bash
experiments/dot-product-scaling/run.sh                     # build, run, report
SKIP_SETUP=1 experiments/dot-product-scaling/run.sh --parts synth,monte --kmax 4096
```

**Stopping and resuming.** Ctrl-C or `kill` is safe at any point:
- the configuration in progress is discarded, while every finished one stays
  in `results/`, written as one complete line;
- temporary files are removed;
- the next `run.sh` resumes where it stopped;
- a second `run.sh` started while one is running is refused.

After a complete run the work directory `build/exp-dot` (binaries, reference logits,
dumps) is removed. Set `KEEP_WORK=1` to keep it. `JOBS` (default 4) limits
build parallelism.
