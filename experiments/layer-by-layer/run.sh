#!/usr/bin/env bash
# layer-experiment.sh — layer-by-layer comparison of every LNS backend and
# summation mode against the F32 reference, and of CPU against GPU.
#
# Usage:
#   ./experiments/layer-by-layer/run.sh -m model.gguf [-m model2.gguf ...] [-p PROMPT] [-o OUT_DIR]
#
# For each model, one forward pass over the prompt is dumped tensor by tensor
# (layer-dump.cpp) for these configurations:
#   f32                        build-ref (pure CPU, no LNS)
#   cpu16-MODE, cuda16-MODE    build-lns-MODE, MODE = base monte lpvip sum32
#   cpu32, cuda32              build-lns-base
# and compared with layer-compare.py. The report (OUT_DIR/<model>.md,
# default experiments/layer-by-layer/results) has: final logits per config,
# the residual stream error per layer, the error per tensor type, and CPU vs
# GPU bit-identity for every mode.
#
# Needs ./setup.sh --cuda (all modes) for the trees. A configuration whose
# device is missing (no GPU, CPU-only build) is skipped and listed.
# Environment: LLAMA_ROOT (default build/llama.cpp), TREE_PREFIX (default
# build-lns-), SKIP_CPU32=1 to skip the slow CPU32 run on big models.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-$REPO_ROOT/build/llama.cpp}"
TREE_PREFIX="${TREE_PREFIX:-build-lns-}"
PROMPT="The capital of France is"
OUT_DIR="$HERE/results"
MODELS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -m) MODELS+=("$2"); shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -o) OUT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown option '$1'" >&2; exit 1 ;;
    esac
done
[ "${#MODELS[@]}" -gt 0 ] || { echo "ERROR: give at least one model with -m" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT_DIR"

# layer-dump binary per tree
build_dump() {
    local tree="$1" out="$WORK/ld-$(basename "$1")"
    if [ ! -x "$out" ]; then
        g++ -O2 -std=c++17 -I "$LLAMA_ROOT/include" -I "$LLAMA_ROOT/ggml/include" \
            "$HERE/layer-dump.cpp" -o "$out" \
            -L "$tree/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$tree/bin" || return 1
    fi
    echo "$out"
}

# has_device TREE DEVICE MODE — the tree's binary reports DEVICE (with sum=MODE)
has_device() {
    "$1/bin/llama-completion" --list-devices 2>/dev/null | grep -q "^ *$2:.*${3:+sum=$3[,)]}"
}

for model in "${MODELS[@]}"; do
    name="$(basename "$model" .gguf)"
    echo "==> $name"
    runs=()
    pairs=()
    skipped=()
    run_cfg() { # label tree backend device mode
        local label="$1" tree="$LLAMA_ROOT/$2" backend="$3" dev="$4" mode="$5" bin
        if [ ! -x "$tree/bin/llama-completion" ] || { [ -n "$dev" ] && ! has_device "$tree" "$dev" "$mode"; }; then
            skipped+=("$label")
            return 0
        fi
        bin="$(build_dump "$tree")" || { skipped+=("$label"); return 0; }
        mkdir -p "$WORK/$name/$label"
        echo "    $label"
        if GGML_LNS_BACKEND="$backend" "$bin" "$model" "$PROMPT" "$WORK/$name/$label" > "$WORK/$name/$label.log" 2>&1; then
            runs+=("--run" "$label=$WORK/$name/$label")
        else
            skipped+=("$label (failed, see log)")
        fi
    }
    run_cfg f32 build-ref "" "" ""
    [ "${#runs[@]}" -gt 0 ] || { echo "ERROR: the F32 reference (build-ref) did not run" >&2; exit 1; }
    ref="${runs[1]}"
    runs=()
    for mode in base monte lpvip sum32; do
        run_cfg "cpu16-$mode"  "$TREE_PREFIX$mode" 16     LNS16     "$mode"
        run_cfg "cuda16-$mode" "$TREE_PREFIX$mode" cuda16 LNSCUDA16 "$mode"
        pairs+=("--pair" "cpu16-$mode:cuda16-$mode")
    done
    if [ "${SKIP_CPU32:-0}" != 1 ]; then
        run_cfg cpu32 "${TREE_PREFIX}base" 32 LNS32 ""
    fi
    run_cfg cuda32 "${TREE_PREFIX}base" cuda32 LNSCUDA32 ""
    pairs+=("--pair" "cpu32:cuda32")

    # keep only pairs whose both sides ran
    ok_pairs=()
    for ((i = 1; i < ${#pairs[@]}; i += 2)); do
        a="${pairs[$i]%%:*}"; b="${pairs[$i]##*:}"
        if [[ " ${runs[*]} " == *" $a="* && " ${runs[*]} " == *" $b="* ]]; then
            ok_pairs+=("--pair" "${pairs[$i]}")
        fi
    done

    report="$OUT_DIR/$name.md"
    python3 "$HERE/layer-compare.py" --title "Layer-by-layer: $name, prompt \"$PROMPT\"" \
        --ref "${ref#--run }" "${runs[@]}" "${ok_pairs[@]+"${ok_pairs[@]}"}" > "$report"
    if [ "${#skipped[@]}" -gt 0 ]; then
        printf '\nSkipped (missing build/device or failed run): %s\n' "${skipped[*]}" >> "$report"
    fi
    echo "    report: $report"
done
