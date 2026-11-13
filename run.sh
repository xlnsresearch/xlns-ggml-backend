#!/usr/bin/env bash
# run.sh — re-run LNS inference after a successful ./setup.sh build.
#
# Usage:
#   ./run.sh                          # LNS16, base sum mode, default model/prompt
#   ./run.sh --sum-mode=sum32         # pick a summation-variant binary
#   ./run.sh --sum-mode=all           # run the same prompt across all four variants
#   ./run.sh --backend=32             # LNS32; default -n 4
#   ./run.sh --backend=cuda16 --sum-mode=monte   # LNS16 on the GPU (./setup.sh --cuda);
#                                     # cuda16 has the same 4 modes as the CPU
#   ./run.sh -m ./other.gguf -n 40
#   ./run.sh -p "Hello" -- --temp 0.2 # extra args after -- go to llama-completion
#
# Summation mode is compile-time: ./setup.sh builds one binary per mode under
# build/llama.cpp/build-lns-{base,monte,lpvip,sum32}; this script selects among
# them and checks (llama-completion --list-devices) that the binary really
# runs that mode. GGML_LNS_BACKEND (--backend) selects the devices inside it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BUILD_ROOT="${LLAMA_BUILD_ROOT:-$REPO_ROOT/build/llama.cpp}"
DEFAULT_MODEL="$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf"
ALL_MODES=(base monte lpvip sum32)

BACKEND="${GGML_LNS_BACKEND:-16}"
MODEL="${MODEL:-${MODEL_SMOLLM:-$DEFAULT_MODEL}}"
SUM_MODE="${LNS16_SUM:-base}"
PROMPT="The capital of France is"
N_TOKENS=""
N_TOKENS_SET=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./run.sh [options] [-- llama-completion-args...]

Re-run LNS inference using the binaries built by ./setup.sh.
Does not rebuild; each summation mode is a separate compile-time binary.

Options:
  --sum-mode=MODE     base|monte|lpvip|sum32|all (default: base, or $LNS16_SUM)
                      'all' runs the same prompt through every variant in turn
  --backend=B         LNS backend (default: 16, or $GGML_LNS_BACKEND):
                        16      LNS16 on the CPU
                        32      LNS32 on the CPU
                        cuda16  LNS16 arithmetic on a CUDA GPU (alias: cuda);
                                sum modes base|monte|lpvip|sum32, as on the CPU
                        cuda32  LNS32 arithmetic on a CUDA GPU
                      The GPU backends need ./setup.sh --cuda; ops they do not
                      implement run on the matching CPU LNS backend.
  -m, --model PATH    GGUF model path (default: $MODEL / $MODEL_SMOLLM /
                      SmolLM2-135M-Instruct-Q4_K_M.gguf in repo root)
  -p, --prompt TEXT   Prompt (default: "The capital of France is")
  -n, --tokens N      Tokens to predict (default: 4 for CPU LNS32, else 20)
  -h, --help          Show this help

Environment:
  LNS16_SUM           Same as --sum-mode
  GGML_LNS_BACKEND    Same as --backend
  GGML_LNS_CUDA_DEVICE  GPU index for the cuda backends (default 0)
  GGML_LNS_CUDA_CACHE_MB
                      VRAM budget for weights converted to LNS (default:
                      free VRAM minus 768 MiB)
  MODEL / MODEL_SMOLLM
                      Default model path
  LNS_OP_AUDIT=1      Print one JSON line per supports_op query

Examples:
  ./run.sh
  ./run.sh --sum-mode=sum32
  ./run.sh --sum-mode=all -n 8
  ./run.sh --backend=32
  ./run.sh --backend=cuda16 -m ./qwen2.5-0.5b-instruct-q8_0.gguf -n 64
  ./run.sh -m ./Llama-3.2-1B-Instruct-Q4_K_M.gguf -n 40
  ./run.sh -p "Hello" -- --temp 0.2 --seed 1

Pass-through: anything after -- is forwarded to llama-completion.
See SETUP.md for important llama-completion flags, or run:
  ./build/llama.cpp/build-lns/bin/llama-completion --help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --sum-mode=*)
            SUM_MODE="${1#*=}"
            shift
            ;;
        --sum-mode)
            if [ $# -lt 2 ]; then
                echo "ERROR: --sum-mode requires base|monte|lpvip|sum32|all" >&2
                exit 1
            fi
            SUM_MODE="$2"
            shift 2
            ;;
        --backend=*)
            BACKEND="${1#*=}"
            shift
            ;;
        --backend)
            if [ $# -lt 2 ]; then
                echo "ERROR: --backend requires 16, 32, cuda16 or cuda32" >&2
                exit 1
            fi
            BACKEND="$2"
            shift 2
            ;;
        -m|--model)
            if [ $# -lt 2 ]; then
                echo "ERROR: $1 requires a path" >&2
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -p|--prompt)
            if [ $# -lt 2 ]; then
                echo "ERROR: $1 requires a prompt string" >&2
                exit 1
            fi
            PROMPT="$2"
            shift 2
            ;;
        -n|--tokens)
            if [ $# -lt 2 ]; then
                echo "ERROR: $1 requires a number" >&2
                exit 1
            fi
            N_TOKENS="$2"
            N_TOKENS_SET=1
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "ERROR: unknown option '$1' (use -- to forward args to llama-completion)" >&2
            echo "Run ./run.sh --help for usage." >&2
            exit 1
            ;;
        *)
            echo "ERROR: unexpected argument '$1'" >&2
            echo "Run ./run.sh --help for usage." >&2
            exit 1
            ;;
    esac
done

case "$SUM_MODE" in
    base|monte|lpvip|sum32|all) ;;
    *)
        echo "ERROR: --sum-mode must be base, monte, lpvip, sum32, or all (got '$SUM_MODE')" >&2
        exit 1
        ;;
esac

case "$BACKEND" in
    16|32|cuda16|cuda32) ;;
    cuda) BACKEND=cuda16 ;;
    *)
        echo "ERROR: --backend must be 16, 32, cuda16 or cuda32 (got '$BACKEND')" >&2
        exit 1
        ;;
esac

if [ "$BACKEND" = 32 ] || [ "$BACKEND" = cuda32 ]; then
    if [ "$SUM_MODE" != base ]; then
        echo "NOTE: --sum-mode does not apply to --backend=$BACKEND (xlns32 has one summation); running once" >&2
    fi
    [ "$SUM_MODE" = all ] && SUM_MODE=base
fi

if [ "$N_TOKENS_SET" -eq 0 ]; then
    if [ "$BACKEND" = "32" ]; then
        N_TOKENS=4
    else
        N_TOKENS=20
    fi
fi

# The binary for one sum mode, checked against what it reports: the device
# for the backend must exist and the xlns16 devices must say sum=<mode>. The
# legacy build-lns path is used only if it passes the same check.
bin_for_mode() {
    local mode="$1" bin dev
    case "$BACKEND" in
        16) dev=LNS16 ;; 32) dev=LNS32 ;; cuda16) dev=LNSCUDA16 ;; cuda32) dev=LNSCUDA32 ;;
    esac
    for bin in "$LLAMA_BUILD_ROOT/build-lns-$mode/bin/llama-completion" "$LLAMA_BUILD_ROOT/build-lns/bin/llama-completion"; do
        [ -x "$bin" ] || continue
        local devices
        devices="$("$bin" --list-devices 2>/dev/null)" || continue
        if ! printf '%s\n' "$devices" | grep -q "^ *$dev:"; then
            echo "NOTE: $bin has no $dev device (built without --cuda, or no usable GPU)" >&2
            continue
        fi
        if [ "$dev" = LNS16 ] || [ "$dev" = LNSCUDA16 ]; then
            if ! printf '%s\n' "$devices" | grep "^ *\(LNS16\|$dev\):" | grep -qv "sum=$mode[,)]" &&
               printf '%s\n' "$devices" | grep -q "^ *LNS16:.*sum=$mode[,)]"; then
                echo "$bin"
                return 0
            fi
            echo "NOTE: $bin does not run sum=$mode" >&2
            continue
        fi
        echo "$bin"
        return 0
    done
    echo "ERROR: no llama-completion under $LLAMA_BUILD_ROOT runs --backend=$BACKEND with sum mode '$mode'." >&2
    if [ "$BACKEND" = cuda16 ] || [ "$BACKEND" = cuda32 ]; then
        echo "Build it with: ./setup.sh --cuda --sum-mode=$mode" >&2
    else
        echo "Build it with: ./setup.sh --sum-mode=$mode" >&2
    fi
    return 1
}

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"

run_one() {
    local mode="$1"
    local llama_bin
    llama_bin="$(bin_for_mode "$mode")" || exit 1

    echo "==> Running LNS inference (GGML_LNS_BACKEND=$BACKEND, sum mode: $mode, n=$N_TOKENS)"
    echo "    model: $MODEL"
    echo "    bin:   $llama_bin"

    GGML_LNS_BACKEND="$BACKEND" "$llama_bin" \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n "$N_TOKENS" \
        -fa off \
        -no-cnv \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

if [ "$SUM_MODE" = "all" ]; then
    for mode in "${ALL_MODES[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  sum mode: $mode"
        echo "════════════════════════════════════════════════════════════════"
        run_one "$mode"
    done
else
    run_one "$SUM_MODE"
fi
