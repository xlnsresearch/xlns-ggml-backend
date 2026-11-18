REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JOBS="${JOBS:-4}"
MODELS_DIR="$REPO_ROOT/build/models"
MODEL_SMOLLM="${MODEL_SMOLLM:-$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf}"
MODEL_QWEN="${MODEL_QWEN:-$REPO_ROOT/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
URL_SMOLLM="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
URL_QWEN="https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

LIB_TMP=()
LIB_PID=""

lib_cleanup() {
    local f
    for f in "${LIB_TMP[@]+"${LIB_TMP[@]}"}"; do
        rm -rf -- "$f"
    done
    LIB_TMP=()
}

lib_on_signal() {
    trap '' INT TERM HUP
    if [ -n "$LIB_PID" ] && kill -0 "$LIB_PID" 2>/dev/null; then
        kill -TERM -- "-$LIB_PID" 2>/dev/null || kill -TERM "$LIB_PID" 2>/dev/null || true
        wait "$LIB_PID" 2>/dev/null || true
    fi
    lib_cleanup
    echo "stopped: finished results are kept, temporary files removed; run again to resume" >&2
    exit 130
}

trap lib_on_signal INT TERM HUP
trap lib_cleanup EXIT

lib_run() {
    local status=0
    set -m
    "$@" < /dev/null &
    LIB_PID=$!
    set +m
    wait "$LIB_PID" || status=$?
    LIB_PID=""
    return "$status"
}

lock_experiment() {
    mkdir -p "$REPO_ROOT/build"
    exec 9>"$REPO_ROOT/build/.$1.lock"
    if ! flock -n 9; then
        echo "ERROR: $1 is already running (lock $REPO_ROOT/build/.$1.lock)" >&2
        exit 1
    fi
}

setup_build() {
    if [ "${SKIP_SETUP:-0}" = 1 ]; then
        echo "==> SKIP_SETUP=1: using the existing build trees"
        return 0
    fi
    download "$URL_SMOLLM" "$MODEL_SMOLLM"
    echo "==> building every LNS tree and checking it with short runs: setup.sh --cuda (JOBS=$JOBS)"
    lib_run env MODEL_SMOLLM="$MODEL_SMOLLM" LNS32_INFERENCE="${LNS32_INFERENCE:-0}" \
        bash -c 'cd "$1" && ./setup.sh --cuda' _ "$REPO_ROOT"
}

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "==> downloading $(basename "$dest")"
    LIB_TMP+=("$dest.part")
    lib_run curl -L --fail --retry 3 -o "$dest.part" "$url"
    mv -f "$dest.part" "$dest"
}

ensure_models() {
    download "$URL_SMOLLM" "$MODEL_SMOLLM"
    download "$URL_QWEN" "$MODEL_QWEN"
}

write_report() {
    local out="${!#}"
    local tmp="$out.tmp"
    LIB_TMP+=("$tmp")
    python3 "${@:1:$#-1}" > "$tmp"
    mv -f "$tmp" "$out"
    echo "    report: $out"
}

finish_work() {
    local work="$1"
    if [ "${KEEP_WORK:-0}" = 1 ]; then
        echo "==> KEEP_WORK=1: keeping $work"
    else
        rm -rf -- "$work"
        echo "==> removed the temporary work directory $work"
    fi
}
