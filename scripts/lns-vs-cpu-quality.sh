#!/usr/bin/env bash
# lns-vs-cpu-quality.sh — quantitative quality gate: LNS backend vs a TRUE CPU
# reference.
#
# Motivation: "does the output look coherent?" is anecdotal. This script turns
# it into repeatable numbers per model.
#
# IMPORTANT (learned the hard way): the CPU side MUST come from a separate
# llama.cpp build WITHOUT the LNS/BLAS backends (build-ref). Inside the
# LNS-enabled binary, neither --no-op-offload nor --device none prevents the
# scheduler from assigning ops to LNS/BLAS — op_offload only disables the
# weight-buffer offload path, and --device none only affects layer offload.
# Verified with GGML_SCHED_DEBUG=2: with --no-op-offload, SmolLM2 still ran
# its MUL_MATs on LNS16+BLAS. Comparing "LNS16" against that mixture proved
# nothing; the two matched bit-for-bit precisely because both were mostly
# LNS16.
#
# Metrics per model:
#   1. Greedy first-divergence — llama-completion with --temp 0 --seed 1,
#      CPU-ref vs LNS; report the first token where the continuations differ.
#      NOTE: exact match is NOT expected for LNS16 (xlns16 applies ~0.3%
#      rounding to every matmul operand, which flips argmaxes even on healthy
#      models). Informational only.
#   2. Perplexity delta — llama-perplexity on a small fixed eval text, CPU-ref
#      vs LNS; pass/fail on the relative delta.
#
# Usage:
#   ./scripts/lns-vs-cpu-quality.sh                      # default: SmolLM2, quick
#   ./scripts/lns-vs-cpu-quality.sh -m model.gguf [-m model2.gguf ...]
#   ./scripts/lns-vs-cpu-quality.sh -b 32                # LNS32 instead of LNS16
#   ./scripts/lns-vs-cpu-quality.sh -b cuda16            # LNS16 arithmetic on the GPU
#                                                        # (needs ./setup.sh --cuda)
#   ./scripts/lns-vs-cpu-quality.sh -b cuda16 --sum-mode=monte  # one mode's build
#   ./scripts/lns-vs-cpu-quality.sh --skip-ppl           # divergence only
#   ./scripts/lns-vs-cpu-quality.sh -n 40 --ppl-chunks 2 # more tokens / chunks
#
# Exit code: 0 if (unless --skip-ppl) every relative PPL delta is within
# --ppl-tol percent.
#
# NOTE: perplexity on the LNS backends is single-threaded and slow — one
# 256-token chunk (-c 256 --chunks 1, the default) on a 1.5B model takes
# ~20 min on LNS16. Keep the eval text short.
#
# Reference binaries: auto-detected from build/llama.cpp/build-ref/bin (a
# build configured with -DGGML_LNS=OFF -DGGML_LNS32=OFF -DGGML_BLAS=OFF).
# Override with REF_BIN_DIR=/path/to/bin if you keep it elsewhere.
#
# The LNS side defaults to build/llama.cpp/build-lns/bin. Since the summation
# mode is compile-time, grading a specific mode means pointing at that mode's
# tree: --sum-mode=MODE uses build/llama.cpp/build-lns-MODE/bin (or set
# LLAMA_BIN_DIR), and checks that the binary reports sum=MODE.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_BIN_DIR_SET="${LLAMA_BIN_DIR:-}"
REF_BIN_DIR="${REF_BIN_DIR:-$REPO_ROOT/build/llama.cpp/build-ref/bin}"
REF_COMPLETION="$REF_BIN_DIR/llama-completion"
REF_PERPLEXITY="$REF_BIN_DIR/llama-perplexity"

BACKEND="${GGML_LNS_BACKEND:-16}"
SUM_MODE=""
N_TOKENS=20
PPL_CHUNKS=1
PPL_TOL_PCT=15
SKIP_PPL=0
MODELS=()

usage() { sed -n '2,51p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)       usage ;;
        -b|--backend)    BACKEND="$2"; shift 2 ;;
        --sum-mode=*)    SUM_MODE="${1#*=}"; shift ;;
        --sum-mode)      SUM_MODE="$2"; shift 2 ;;
        -m|--model)      MODELS+=("$2"); shift 2 ;;
        -n|--tokens)     N_TOKENS="$2"; shift 2 ;;
        --ppl-chunks)    PPL_CHUNKS="$2"; shift 2 ;;
        --ppl-tol)       PPL_TOL_PCT="$2"; shift 2 ;;
        --skip-ppl)      SKIP_PPL=1; shift ;;
        *) echo "ERROR: unknown option '$1'" >&2; usage 1 ;;
    esac
done

case "$BACKEND" in
    16|32|cuda16|cuda32) ;;
    cuda) BACKEND=cuda16 ;;
    *) echo "ERROR: backend must be 16, 32, cuda16 or cuda32" >&2; exit 1 ;;
esac
case "$SUM_MODE" in
    ""|base|monte|lpvip|sum32) ;;
    *) echo "ERROR: --sum-mode must be base, monte, lpvip or sum32" >&2; exit 1 ;;
esac
if [ -n "$SUM_MODE" ]; then
    LLAMA_BIN_DIR="${LLAMA_BIN_DIR_SET:-$REPO_ROOT/build/llama.cpp/build-lns-$SUM_MODE/bin}"
else
    LLAMA_BIN_DIR="${LLAMA_BIN_DIR_SET:-$REPO_ROOT/build/llama.cpp/build-lns/bin}"
fi
COMPLETION="$LLAMA_BIN_DIR/llama-completion"
PERPLEXITY="$LLAMA_BIN_DIR/llama-perplexity"

if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=("$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf")
fi

for bin in "$COMPLETION" "$PERPLEXITY"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: $(basename "$bin") not found at $bin" >&2
        echo "Build with: cmake --build $(dirname "$LLAMA_BIN_DIR") --target $(basename "$bin")" >&2
        exit 1
    fi
done

for bin in "$REF_COMPLETION" "$REF_PERPLEXITY"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: CPU reference $(basename "$bin") not found at $bin" >&2
        echo "Build a pure-CPU reference (no LNS, no BLAS) with:" >&2
        echo "  cmake -B $REPO_ROOT/build/llama.cpp/build-ref -S $REPO_ROOT/build/llama.cpp \\" >&2
        echo "    -DGGML_LNS=OFF -DGGML_LNS32=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF \\" >&2
        echo "    -DGGML_CPU_REPACK=OFF -DCMAKE_BUILD_TYPE=Release" >&2
        echo "  cmake --build $REPO_ROOT/build/llama.cpp/build-ref --target llama-completion llama-perplexity -j" >&2
        echo "(or point REF_BIN_DIR at it)" >&2
        exit 1
    fi
done

# The LNS binary must have the backend's device and, for the xlns16 backends,
# the requested summation mode.
case "$BACKEND" in 16) DEV=LNS16 ;; 32) DEV=LNS32 ;; cuda16) DEV=LNSCUDA16 ;; cuda32) DEV=LNSCUDA32 ;; esac
DEV_LINE="$("$COMPLETION" --list-devices 2>/dev/null | grep -m1 "^ *$DEV:" || true)"
if [ -z "$DEV_LINE" ]; then
    echo "ERROR: $COMPLETION has no $DEV device (built without --cuda, or no usable GPU)" >&2
    exit 1
fi
if [ -n "$SUM_MODE" ] && [ "$DEV" != LNS32 ] && [ "$DEV" != LNSCUDA32 ] && ! echo "$DEV_LINE" | grep -q "sum=$SUM_MODE[,)]"; then
    echo "ERROR: $COMPLETION does not run sum=$SUM_MODE: $DEV_LINE" >&2
    exit 1
fi
echo "==> LNS binary: $COMPLETION"
echo "   $DEV_LINE"

# Sanity: the reference must not register any LNS device, otherwise we are
# comparing the backend against itself again.
if "$REF_COMPLETION" --list-devices 2>/dev/null | grep -q "LNS"; then
    echo "ERROR: $REF_COMPLETION registers an LNS device — not a pure CPU reference" >&2
    exit 1
fi

# Symmetric check on the LNS side. ggml's BLAS backend claims any MUL_MAT whose
# ne0/ne1/ne10 are all >= 32, and it is registered ahead of LNS, so at the ubatch
# sizes llama-perplexity uses (-c 256) BLAS executes almost every mat-mul in f32
# and the result gets attributed to LNS. Generation (ubatch 1) is unaffected,
# which is why this stayed invisible: measured on SmolLM2, the LNS16 lpvip PPL
# delta reads +3.25% (PASS) on a BLAS build against +102.96% (FAIL) without it.
for _bin in "$COMPLETION" "$PERPLEXITY"; do
    if "$_bin" --list-devices 2>/dev/null | grep -q "BLAS"; then
        echo "ERROR: $_bin registers a BLAS device — its mat-muls are not LNS." >&2
        echo "Rebuild the LNS tree with -DGGML_BLAS=OFF (setup.sh does this since 2026-08-19)." >&2
        exit 1
    fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Small fixed eval text (written for this purpose). llama-perplexity requires
# at least 2*ctx tokens, so the paragraph is repeated to exceed 512 tokens
# for the -c 256 evaluation below.
EVAL_TXT="$WORK/eval.txt"
para="The city of Paris has been the capital of France for many centuries. Its boulevards, bridges, and museums attract millions of visitors every year. The Seine runs through the heart of the city, and the Eiffel Tower stands on its left bank. Bakers sell fresh baguettes every morning, and cafes fill with people reading newspapers and drinking coffee."
: > "$EVAL_TXT"
for _ in $(seq 1 10); do
    echo "$para" >> "$EVAL_TXT"
done

PROMPT="The capital of France is"
overall_rc=0

for model in "${MODELS[@]}"; do
    name="$(basename "$model")"
    echo "==> $name (GGML_LNS_BACKEND=$BACKEND vs CPU reference)"

    # --- 1. greedy first-divergence (informational) --------------------------
    "$REF_COMPLETION" -m "$model" -p "$PROMPT" -n "$N_TOKENS" -fa off -no-cnv \
        --temp 0 --seed 1 2>/dev/null > "$WORK/cpu.txt"
    GGML_LNS_BACKEND="$BACKEND" "$COMPLETION" -m "$model" -p "$PROMPT" \
        -n "$N_TOKENS" -fa off -no-cnv --temp 0 --seed 1 2>/dev/null > "$WORK/lns.txt"

    if cmp -s "$WORK/cpu.txt" "$WORK/lns.txt"; then
        echo "    greedy divergence ($N_TOKENS tokens): none (identical)"
    else
        # index of first differing token, counted on the concatenated text
        diverge_at=$(python3 - "$WORK/cpu.txt" "$WORK/lns.txt" <<'EOF'
import sys
a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print(i)
EOF
)
        echo "    greedy divergence ($N_TOKENS tokens): first diff at char $diverge_at"
        echo "      CPU-ref:  $(head -c 120 "$WORK/cpu.txt")"
        echo "      LNS$BACKEND:  $(head -c 120 "$WORK/lns.txt")"
    fi

    # --- 2. perplexity delta (pass/fail) --------------------------------------
    ppl_ok=1
    if [ "$SKIP_PPL" -eq 0 ]; then
        # NOTE: llama.cpp prints the "Final estimate" line to stderr.
        ppl_cpu=$( "$REF_PERPLEXITY" -m "$model" -f "$EVAL_TXT" --chunks "$PPL_CHUNKS" -c 256 \
            -fa off 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | tail -1 | awk '{print $5}')
        ppl_lns=$(GGML_LNS_BACKEND="$BACKEND" "$PERPLEXITY" -m "$model" -f "$EVAL_TXT" \
            --chunks "$PPL_CHUNKS" -c 256 -fa off 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | tail -1 | awk '{print $5}')
        if [ -z "$ppl_cpu" ] || [ -z "$ppl_lns" ]; then
            echo "    perplexity: ERROR (failed to parse output)"; ppl_ok=0
        else
            rel=$(awk -v a="$ppl_cpu" -v b="$ppl_lns" 'BEGIN{printf "%.2f", (b-a)/a*100}')
            echo "    perplexity: CPU-ref=$ppl_cpu  LNS$BACKEND=$ppl_lns  (delta ${rel}%)"
            awk -v d="$rel" -v t="$PPL_TOL_PCT" 'BEGIN{d=d<0?-d:d; exit (d<=t)?0:1}' || ppl_ok=0
        fi
    fi

    if [ "$ppl_ok" -eq 0 ]; then
        overall_rc=1
    fi
done

if [ "$overall_rc" -ne 0 ]; then
    echo "==> FAIL: see per-model results above"
elif [ "$SKIP_PPL" -eq 1 ]; then
    # --skip-ppl removes the only pass/fail measurement, so there is nothing to
    # pass. The greedy-divergence report above is informational only.
    echo "==> SKIPPED: divergence reported, no perplexity measured (--skip-ppl)"
else
    echo "==> PASS: LNS$BACKEND matches CPU reference within tolerance"
fi
exit "$overall_rc"
