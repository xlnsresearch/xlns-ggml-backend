#!/usr/bin/env bash
# cpu-vs-gpu.sh — CPU vs GPU for every summation mode, end to end.
#
# Usage: ./experiments/layer-by-layer/cpu-vs-gpu.sh [-m model.gguf] [--skip-ppl]
#
# Per mode (base, monte, lpvip, sum32: CPU16 vs CUDA16) and for xlns32
# (CPU32 vs CUDA32), prints a markdown table of:
#   - test-lns-cuda-parity result (build/ggml/build-lns-<mode>)
#   - test-lns-backend: how many printed error figures are equal on both devices
#   - greedy generation (16 tokens): same text?, tokens/s on each device
#   - perplexity on CPU and GPU (scripts/lns-vs-cpu-quality.sh; F32 reference)
# Needs ./setup.sh --cuda --unit-test (ggml test trees and llama.cpp trees).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODEL="$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf"
SKIP_PPL=0
while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODEL="$2"; shift 2 ;;
        --skip-ppl) SKIP_PPL=1; shift ;;
        *) echo "usage: $0 [-m model.gguf] [--skip-ppl]" >&2; exit 1 ;;
    esac
done
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PROMPT="The capital of France is"

metrics() { grep -E "error|NRMSE|result" "$1"; }
tps() { grep -oE "[0-9.]+ tokens per second" "$1" | sed -n "${2}p" | cut -d' ' -f1; }
ppl() {
    "$REPO_ROOT/scripts/lns-vs-cpu-quality.sh" -b "$1" --sum-mode="$2" -m "$MODEL" 2>&1 |
        grep -oE "LNS ?$1=[0-9.]+ +\(delta [-0-9.]+%\)" | sed 's/^LNS *[^=]*=//'
}

echo "| Mode | parity (GPU vs CPU) | unit-test figures equal | greedy text equal | prompt tok/s CPU → GPU | gen tok/s CPU → GPU | perplexity CPU | perplexity GPU |"
echo "|---|---|---|---|---|---|---|---|"
for mode in base monte lpvip sum32 xlns32; do
    if [ "$mode" = xlns32 ]; then
        cpu=32; gpu=cuda32; tree=base; dev_cpu=LNS32; dev_gpu=LNSCUDA32; only=cuda32; prof=""
    else
        cpu=16; gpu=cuda16; tree=$mode; dev_cpu=LNS16; dev_gpu=LNSCUDA16; only=cuda16; prof=$mode
    fi
    gdir="$REPO_ROOT/build/ggml/build-lns-$tree"
    bin="$REPO_ROOT/build/llama.cpp/build-lns-$tree/bin/llama-completion"
    parity="$( (cd "$gdir" && ./bin/test-lns-cuda-parity --only=$only) 2>&1 | grep -E '^(PASS|FAIL|ERROR)' | head -1)"
    (cd "$gdir" && GGML_LNS_BACKEND=$cpu ./bin/test-lns-backend $dev_cpu $prof) > "$WORK/c.txt" 2>&1
    (cd "$gdir" && ./bin/test-lns-backend $dev_gpu $prof) > "$WORK/g.txt" 2>&1
    total=$(metrics "$WORK/g.txt" | wc -l)
    same=$(diff <(metrics "$WORK/c.txt") <(metrics "$WORK/g.txt") > /dev/null && echo "$total" || \
           paste <(metrics "$WORK/c.txt") <(metrics "$WORK/g.txt") | awk -F'\t' '$1==$2' | wc -l)
    for b in $cpu $gpu; do
        GGML_LNS_BACKEND=$b "$bin" -m "$MODEL" -p "$PROMPT" -n 16 -fa off -no-cnv --temp 0 --seed 1 \
            > "$WORK/t$b.txt" 2> "$WORK/e$b.txt"
    done
    cmp -s "$WORK/t$cpu.txt" "$WORK/t$gpu.txt" && text=yes || text=NO
    p_cpu="-"; p_gpu="-"
    if [ "$SKIP_PPL" -eq 0 ]; then
        q=$mode; [ "$mode" = xlns32 ] && q=base
        p_cpu="$(ppl $cpu $q)"; p_gpu="$(ppl $gpu $q)"
    fi
    echo "| $mode | ${parity:-?} | $same / $total | $text | $(tps "$WORK/e$cpu.txt" 1) → $(tps "$WORK/e$gpu.txt" 1) | $(tps "$WORK/e$cpu.txt" 2) → $(tps "$WORK/e$gpu.txt" 2) | ${p_cpu:-?} | ${p_gpu:-?} |"
done
