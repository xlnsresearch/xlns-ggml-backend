#!/usr/bin/env bash
# setup.sh — reproduce the LNS backend results from scratch.
#
# Usage:
#   ./setup.sh                    # build ALL sum-mode variants at the pinned commits
#   ./setup.sh --unit-test        # also build and run test-lns-backend (standalone ggml)
#   ./setup.sh --unit-test --skip-inference
#   ./setup.sh --sum-mode=monte   # build only ONE xlns16 summation variant:
#                                 # base|monte|lpvip|sum32 (also settable via LNS16_SUM env)
#   ./setup.sh --update           # fetch latest ggml/llama.cpp/xlnscpp, integrate, and on
#                                 # success rewrite the pins in patches/ (last known good)
#   ./setup.sh --clean            # force-reset the source trees to the pins first
#                                 # (discards any manual in-tree edits)
#   ./setup.sh --cuda             # also build the LNSCUDA16/LNSCUDA32 GPU backends
#                                 # (xlnscuda; needs nvcc), test them against the
#                                 # CPU backends and run GPU inference (LNS_CUDA=1).
#                                 # LNSCUDA16 has the same 4 sum modes as LNS16.
#                                 # LNS_REQUIRE_GPU=1 fails instead of skipping
#                                 # GPU tests when no GPU is usable.
#
# Upstream trees: by default the three source trees (build/ggml, build/llama.cpp,
# xlnscpp) are used at the pinned commits recorded in patches/*-base-commit.txt —
# no network access, and a tree already at its pin is left untouched (manual
# in-tree experiments survive re-runs). Trees are cloned on first run. Only
# --update fetches upstream; only --update/--clean (or a tree at the wrong
# commit) reset anything.
#
# Builds: one build directory per xlns16 summation mode,
#   build/llama.cpp/build-lns-{base,monte,lpvip,sum32}
# plus a pure-CPU reference build (build/llama.cpp/build-ref, no LNS/BLAS) for
# scripts/lns-vs-cpu-quality.sh. build/llama.cpp/build-lns is a symlink to the
# first mode built (base, unless --sum-mode restricted the set) so older docs
# and scripts keep working. Select a variant at run time with
# ./run.sh --sum-mode=<mode>.
#
# Backends: both LNS16 (xlns16) and LNS32 (xlns32) are built into every binary,
# plus LNSCUDA16/LNSCUDA32 with --cuda. At runtime the GGML_LNS_BACKEND env var
# (16|32|cuda16|cuda32, default 16) selects which backends claim ops. The LNS32
# SmolLM2 inference run can be skipped with LNS32_INFERENCE=0 (it is slow:
# float<->xlns32 conversion has no lookup table).
#
# Models (at least one required for inference):
#   MODEL_SMOLLM  — SmolLM2-135M-Instruct-Q4_K_M.gguf (in repo root by default)
#   MODEL_LLAMA32   — Llama-3.2-1B-Instruct-Q4_K_M.gguf
#   MODEL           — alias for MODEL_SMOLLM (backward compatible)

set -e
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
XLNSCPP_DIR="$REPO_ROOT/xlnscpp"
XLNSCUDA_DIR="$REPO_ROOT/xlnscuda"
BUILD_DIR="$REPO_ROOT/build"
LLAMA_BASE_COMMIT="$(cat "$REPO_ROOT/patches/llama-base-commit.txt")"
GGML_BASE_COMMIT="$(cat "$REPO_ROOT/patches/ggml-base-commit.txt")"
XLNSCPP_BASE_COMMIT="$(cat "$REPO_ROOT/patches/xlnscpp-base-commit.txt")"
XLNSCUDA_BASE_COMMIT="$(cat "$REPO_ROOT/patches/xlnscuda-base-commit.txt")"

MODEL_SMOLLM="${MODEL_SMOLLM:-${MODEL:-}}"
# The repo-root SmolLM2 GGUF is only a default when it actually exists (models
# are not shipped); a fresh clone without it should get the download hint
# below, not a bare "not found" for a path the user never set.
if [ -z "$MODEL_SMOLLM" ] && [ -f "$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf" ]; then
    MODEL_SMOLLM="$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf"
fi
MODEL_LLAMA32="${MODEL_LLAMA32:-}"

RUN_UNIT_TEST=0
RUN_INFERENCE=1
DO_UPDATE=0
DO_CLEAN=0
SUM_MODE_ARG=""
LNS16_SUM="${LNS16_SUM:-}"
LNS32_INFERENCE="${LNS32_INFERENCE:-1}"
LNS_CUDA="${LNS_CUDA:-0}"
LNS_REQUIRE_GPU="${LNS_REQUIRE_GPU:-0}"

for arg in "$@"; do
    case "$arg" in
        --unit-test)         RUN_UNIT_TEST=1 ;;
        --skip-inference)    RUN_INFERENCE=0 ;;
        --update)            DO_UPDATE=1 ;;
        --clean)             DO_CLEAN=1 ;;
        --cuda)              LNS_CUDA=1 ;;
        --sum-mode=*)        SUM_MODE_ARG="${arg#*=}" ;;
        --sum-mode)          echo "ERROR: use --sum-mode=<base|monte|lpvip|sum32> (with '=')" >&2; exit 1 ;;
        *)                   echo "WARNING: ignoring unknown argument '$arg'" >&2 ;;
    esac
done

if [ -n "$SUM_MODE_ARG" ]; then
    LNS16_SUM="$SUM_MODE_ARG"
fi

# Empty LNS16_SUM (the default) builds every variant; a value restricts the
# build to that single mode.
if [ -n "$LNS16_SUM" ]; then
    case "$LNS16_SUM" in
        base|monte|lpvip|sum32) MODES=("$LNS16_SUM") ;;
        *) echo "ERROR: --sum-mode must be base, monte, lpvip, or sum32 (got '$LNS16_SUM')" >&2; exit 1 ;;
    esac
else
    MODES=(base monte lpvip sum32)
fi

resolve_model_path() {
    local var_name="$1"
    local path="$2"
    if [ -n "$path" ]; then
        if [ ! -f "$path" ]; then
            echo "ERROR: $var_name model not found: $path" >&2
            exit 1
        fi
        echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
}

# Model paths are only needed (and only validated) when inference will run;
# --skip-inference must work on a clean checkout with no .gguf present.
if [ "$RUN_INFERENCE" -eq 1 ]; then
    MODEL_SMOLLM="$(resolve_model_path MODEL_SMOLLM "$MODEL_SMOLLM")"
    MODEL_LLAMA32="$(resolve_model_path MODEL_LLAMA32 "$MODEL_LLAMA32")"
fi

if [ "$RUN_INFERENCE" -eq 1 ] && [ -z "$MODEL_SMOLLM" ] && [ -z "$MODEL_LLAMA32" ]; then
    echo "ERROR: At least one model path required for inference." >&2
    echo "  MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    echo "  MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    echo "" >&2
    echo "Model files are not shipped with the repo (see SETUP.md, 'Models: what to download'):" >&2
    echo "  hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir $REPO_ROOT" >&2
    exit 1
fi

# Parallel build jobs: $JOBS, or one per CPU. Lower it on machines with little
# RAM (each compiler process can take ~1 GB on the larger llama.cpp files).
if [ -z "${JOBS:-}" ]; then
    if command -v nproc &>/dev/null; then
        JOBS=$(nproc)
    else
        JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
    fi
fi

# shellcheck source=scripts/integrate-lns.sh
source "$REPO_ROOT/scripts/integrate-lns.sh"

check_deps() {
    for cmd in git cmake g++ make perl; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: '$cmd' not found. Please install it and retry." >&2
            exit 1
        fi
    done
    if [ "$LNS_CUDA" -eq 1 ] && ! command -v nvcc &>/dev/null; then
        echo "ERROR: --cuda needs the CUDA toolkit ('nvcc' not found on PATH)." >&2
        exit 1
    fi
    check_neon_intrinsics
}

# llama.cpp's aarch64 NEON code calls vld1q_*_x2/_x4 intrinsics that older
# GCCs lack (7.5, Ubuntu 18.04's default, has none; 8.1 and 9.1 lack the _x4
# forms, later point releases have them), so probe the compilers CMake will
# pick rather than fail deep in the build with "implicit declaration" errors.
check_neon_intrinsics() {
    case "$(uname -m)" in
        aarch64|arm64) ;;
        *) return 0 ;;
    esac
    local probe_dir var lang cc
    probe_dir="$(mktemp -d)"
    cat > "$probe_dir/probe.c" <<'EOF'
#include <arm_neon.h>
int probe(const int8_t * b, const int16_t * s) {
    int8x16x4_t x = vld1q_s8_x4(b);
    int16x8x2_t y = vld1q_s16_x2(s);
    return vgetq_lane_s8(x.val[0], 0) + vgetq_lane_s16(y.val[0], 0);
}
EOF
    for var in CC CXX; do
        if [ "$var" = CC ]; then
            cc="${CC:-cc}"; lang=c
        else
            cc="${CXX:-c++}"; lang=c++
        fi
        # Unquoted: CC/CXX may carry a launcher or flags (e.g. "ccache gcc").
        # shellcheck disable=SC2086
        if ! $cc -x "$lang" -Werror=implicit-function-declaration \
                -c "$probe_dir/probe.c" -o "$probe_dir/probe.o" &>/dev/null; then
            rm -rf "$probe_dir"
            echo "ERROR: the $var compiler '$cc' cannot compile the ARM NEON intrinsics llama.cpp" >&2
            echo "       needs (vld1q_s8_x4, vld1q_s16_x2); GCC 7 (Ubuntu 18.04's default) lacks them." >&2
            echo "       Install a newer GCC and select it with CC/CXX, e.g. on Ubuntu 18.04:" >&2
            echo "         sudo add-apt-repository ppa:ubuntu-toolchain-r/test" >&2
            echo "         sudo apt update && sudo apt install gcc-11 g++-11" >&2
            echo "         CC=gcc-11 CXX=g++-11 ./setup.sh" >&2
            exit 1
        fi
    done
    rm -rf "$probe_dir"
}

# nvcc refuses host compilers newer than it supports (CUDA 12.0 stops at
# GCC 12). Unless CUDAHOSTCXX is already set, use the first g++ it accepts.
pick_cuda_host_compiler() {
    if [ -z "${CUDAHOSTCXX:-}" ]; then
        local probe_dir cxx
        probe_dir="$(mktemp -d)"
        echo '__global__ void k() {} int main() { k<<<1, 1>>>(); return 0; }' > "$probe_dir/probe.cu"
        # Probe with absolute paths, as CMake passes them: given a bare
        # "g++", some nvcc packages (e.g. Ubuntu's) substitute their own
        # wrapper for a supported version, so a bare-name probe would pass
        # for a compiler CMake then cannot use.
        for cxx in g++ g++-14 g++-13 g++-12 g++-11 g++-10; do
            cxx="$(command -v "$cxx")" || continue
            if nvcc -ccbin "$cxx" -c "$probe_dir/probe.cu" -o "$probe_dir/probe.o" &>/dev/null; then
                CUDAHOSTCXX="$cxx"
                break
            fi
        done
        rm -rf "$probe_dir"
        if [ -z "${CUDAHOSTCXX:-}" ]; then
            echo "ERROR: nvcc accepts none of the installed g++ versions; install one it supports" >&2
            echo "       (e.g. g++-12 for CUDA 12.0) or set CUDAHOSTCXX=/path/to/g++" >&2
            exit 1
        fi
    fi
    export CUDAHOSTCXX
    echo "    CUDA host compiler: $CUDAHOSTCXX"
}

# Same pin-first policy as xlnscpp (patches/xlnscuda-base-commit.txt; --update
# moves to origin/main). The GPU backend needs xlnscuda's device lookup tables,
# so check for them up front.
prepare_xlnscuda() {
    if [ ! -e "$XLNSCUDA_DIR/.git" ]; then
        echo "    Initialising xlnscuda submodule..."
        git -C "$REPO_ROOT" submodule update --init xlnscuda
    fi
    ensure_repo_at "$XLNSCUDA_DIR" main "$XLNSCUDA_BASE_COMMIT" || exit 1
    if ! grep -q 'xlns16d_table_init' "$XLNSCUDA_DIR/xlns16d.cu"; then
        echo "ERROR: $XLNSCUDA_DIR predates device table support (xlns16d_table_init);" >&2
        echo "       update the xlnscuda submodule." >&2
        exit 1
    fi
}

# NOTE: functions below are invoked from contexts where `set -e` is suppressed
# (the `if ! run_build_and_test` retry wrapper disables errexit inside the
# whole call tree — a bash rule), so failures must be propagated explicitly
# with `|| return 1`.

# Bring a source tree to the wanted commit. Default: the pinned commit, with no
# network access and no reset when the tree is already there (preserves manual
# in-tree experiments; the regex integration is idempotent). --clean forces a
# pristine reset to the pin; --update fetches and moves to origin/<branch>.
ensure_repo_at() {
    local repo_dir="$1"
    local branch="$2"
    local pin="$3"
    cd "$repo_dir" || return 1
    if [ "$DO_UPDATE" -eq 1 ]; then
        echo "    Fetching latest ($branch)"
        git fetch origin || return 1
        git reset --hard "origin/$branch" || return 1
    else
        local head
        head="$(git rev-parse HEAD)" || return 1
        if [ "$head" != "$pin" ] || [ "$DO_CLEAN" -eq 1 ]; then
            if ! git cat-file -e "$pin^{commit}" 2>/dev/null; then
                echo "    Pinned commit not present locally; fetching"
                git fetch origin || return 1
            fi
            echo "    Resetting to pinned commit $pin"
            git reset --hard "$pin" || return 1
        else
            echo "    Already at pinned commit (tree left untouched)"
        fi
    fi
    echo "    at $(git rev-parse --short HEAD)"
}

prepare_xlnscpp() {
    # NB: in a submodule .git is a gitfile, not a directory — test with -e,
    # otherwise `submodule update --init` runs every time and would clobber a
    # manually moved xlnscpp checkout.
    if [ ! -e "$XLNSCPP_DIR/.git" ]; then
        echo "    Initialising xlnscpp submodule..."
        git -C "$REPO_ROOT" submodule update --init
    fi
    ensure_repo_at "$XLNSCPP_DIR" main "$XLNSCPP_BASE_COMMIT" || exit 1
}

fallback_xlnscpp() {
    echo "    xlnscpp latest failed; using pinned commit"
    cd "$XLNSCPP_DIR"
    git reset --hard "$XLNSCPP_BASE_COMMIT"
    echo "    at $(git rev-parse --short HEAD)"
    if [ "$LNS_CUDA" -eq 1 ]; then
        cd "$XLNSCUDA_DIR"
        git reset --hard "$XLNSCUDA_BASE_COMMIT"
        echo "    xlnscuda at $(git rev-parse --short HEAD)"
    fi
}

prepare_ggml() {
    local ggml_dir="$1"
    if [ ! -d "$ggml_dir/.git" ]; then
        echo "    Cloning ggml..."
        git clone https://github.com/ggml-org/ggml.git "$ggml_dir" || return 1
    fi
    ensure_repo_at "$ggml_dir" master "$GGML_BASE_COMMIT" || return 1
    integrate_lns_or_fallback "$ggml_dir" \
        "$REPO_ROOT/patches/ggml-core.patch" "$GGML_BASE_COMMIT" || return 1
    copy_lns_backends "$ggml_dir" || return 1
}

prepare_llama() {
    local llama_dir="$1"
    if [ ! -d "$llama_dir/.git" ]; then
        echo "    Cloning llama.cpp (this may take a few minutes)..."
        git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir" || return 1
    fi
    ensure_repo_at "$llama_dir" master "$LLAMA_BASE_COMMIT" || return 1
    integrate_lns_or_fallback "$llama_dir" \
        "$REPO_ROOT/patches/llama-core.patch" "$LLAMA_BASE_COMMIT" || return 1
    copy_lns_backends "$llama_dir/ggml" || return 1
}

# Absolute, symlink-resolved path of a compiler command (its first word, so
# "ccache gcc" resolves ccache), for comparing against CMake's cached path.
resolve_compiler() {
    local path
    path="$(command -v "${1%% *}")" || return 1
    readlink -f "$path" 2>/dev/null || echo "$path"
}

# Wipe a build dir that CMake would keep configuring wrongly:
#  - CC/CXX name a different compiler than the cached one: CMake reads them
#    only on a first configure and otherwise silently keeps the old compiler.
#  - an earlier configure stopped mid compiler detection (e.g. CXX was not
#    found), leaving a compiler record with no compile features. CMake reuses
#    it without re-testing, and every later configure then fails with
#    "target_compile_features no known features for C compiler".
reset_stale_build_dir() {
    local bdir="$1"
    local cache="$bdir/CMakeCache.txt"
    local reason="" lang var cached wanted
    for lang in C CXX; do
        if grep -qs "set(CMAKE_${lang}_COMPILE_FEATURES \"\")" \
                "$bdir"/CMakeFiles/*/CMake${lang}Compiler.cmake; then
            reason="an earlier configure stopped before finishing $lang compiler detection"
        fi
        var=CC
        [ "$lang" = CXX ] && var=CXX
        if [ -n "${!var:-}" ] && [ -f "$cache" ]; then
            cached="$(sed -n "s/^CMAKE_${lang}_COMPILER:[A-Z]*=//p" "$cache")"
            wanted="$(resolve_compiler "${!var}")" || continue
            if [ -n "$cached" ] && [ "$(resolve_compiler "$cached")" != "$wanted" ]; then
                reason="$var=${!var} differs from the cached $cached"
            fi
        fi
    done
    if [ -n "$reason" ]; then
        echo "    Removing stale $(basename "$bdir") ($reason)"
        rm -rf "$bdir" || return 1
    fi
}

# Configure and build one LNS variant build dir (src_dir/build-lns-<mode>).
# Remaining args are the cmake targets.
build_lns_variant() {
    local src_dir="$1"
    local mode="$2"
    shift 2
    local bdir="$src_dir/build-lns-$mode"
    local cuda_args=(-DGGML_LNSCUDA=OFF)
    if [ "$LNS_CUDA" -eq 1 ]; then
        cuda_args=(-DGGML_LNSCUDA=ON -DXLNSCUDA_DIR="$XLNSCUDA_DIR" -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX")
    fi
    reset_stale_build_dir "$bdir" || return 1
    mkdir -p "$bdir" || return 1
    cd "$bdir" || return 1
    cmake .. \
        -DGGML_LNS=ON \
        -DGGML_LNS32=ON \
        "${cuda_args[@]}" \
        -DGGML_LNS16_SUM="$mode" \
        -DGGML_LNS16_MONTE_BYLPVIP=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_CPU_REPACK=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DXLNSCPP_DIR="$XLNSCPP_DIR" || return 1
    cmake --build . --target "$@" -j"$JOBS" || return 1
}

# Pure-CPU reference build (no LNS, no BLAS) — the comparison baseline that
# scripts/lns-vs-cpu-quality.sh expects at build/llama.cpp/build-ref/bin.
build_cpu_ref() {
    local src_dir="$1"
    local bdir="$src_dir/build-ref"
    reset_stale_build_dir "$bdir" || return 1
    mkdir -p "$bdir" || return 1
    cd "$bdir" || return 1
    cmake .. \
        -DGGML_LNS=OFF \
        -DGGML_LNS32=OFF \
        -DGGML_LNSCUDA=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_CPU_REPACK=OFF \
        -DCMAKE_BUILD_TYPE=Release || return 1
    cmake --build . --target llama-completion llama-perplexity -j"$JOBS" || return 1
}

# Keep the legacy build-lns path working: symlink it to the first mode built.
# A pre-existing real build-lns directory (from the old single-build layout)
# carries a cmake cache with an unknown sum mode, so replace it once the new
# variant dirs exist.
link_build_compat() {
    local src_dir="$1"
    local target="build-lns-${MODES[0]}"
    if [ -d "$src_dir/build-lns" ] && [ ! -L "$src_dir/build-lns" ]; then
        echo "    Replacing legacy build-lns directory with symlink -> $target"
        rm -rf "$src_dir/build-lns" || return 1
    fi
    ln -sfn "$target" "$src_dir/build-lns" || return 1
}

run_inference() {
    local label="$1"
    local model_path="$2"
    local llama_bin="$3"
    local backend="${4:-16}"   # GGML_LNS_BACKEND: 16, 32, cuda16 or cuda32
    local n_tokens="${5:-20}"
    local mode="${6:-${MODES[0]}}"

    echo ""
    echo "==> Running LNS inference: $label (GGML_LNS_BACKEND=$backend, sum mode: $mode)"
    GGML_LNS_BACKEND="$backend" "$llama_bin" \
        -m "$model_path" \
        -p "The capital of France is" \
        -n "$n_tokens" \
        -fa 0 \
        -no-cnv || return 1
}

# Runs a GPU test; exit 77 means no usable GPU (a skip, or a failure with
# LNS_REQUIRE_GPU=1). Returns 0 ran, 77 skipped, 1 failed.
gpu_test() {
    local status=0
    "$@" || status=$?
    if [ "$status" -eq 77 ]; then
        echo "SKIPPED: no usable CUDA device for: $*" >&2
        [ "$LNS_REQUIRE_GPU" = 1 ] && return 1
        return 77
    fi
    [ "$status" -eq 0 ] || return 1
}

# xlnscuda Monte/LPVIP vs xlnscpp, then per supported mode LNSCUDA16 parity
# and unit tests; LNSCUDA32 once (it has no summation modes).
run_cuda_unit_tests() {
    local mode status bin="$GGML_DIR/chkxlns16d_sum_modes"
    echo ""
    echo "==> Running xlnscuda Monte/LPVIP tests (chkxlns16d_sum_modes, both Monte variants)"
    for variant in "" -Dxlns16d_monte_bylpvip; do
        nvcc -std=c++14 -O2 -ccbin "$CUDAHOSTCXX" -w -I "$XLNSCPP_DIR" $variant \
            "$XLNSCUDA_DIR/tests/chkxlns16d_sum_modes.cu" -o "$bin" || return 1
        status=0
        gpu_test "$bin" || status=$?
        [ "$status" -eq 1 ] && return 1
    done
    local cuda32_done=0
    for mode in "${MODES[@]}"; do
        echo ""
        echo "==> Running test-lns-cuda-parity + test-lns-backend (LNSCUDA16, sum mode: $mode)"
        status=0
        gpu_test bash -c 'cd "$1" && ./bin/test-lns-cuda-parity --only=cuda16 --expect-sum="$2"' _ \
            "$GGML_DIR/build-lns-$mode" "$mode" || status=$?
        [ "$status" -eq 1 ] && return 1
        if [ "$status" -eq 0 ]; then
            ( cd "$GGML_DIR/build-lns-$mode" && ./bin/test-lns-backend LNSCUDA16 "$mode" ) || return 1
        fi
        if [ "$cuda32_done" -eq 0 ]; then
            cuda32_done=1
            echo ""
            echo "==> Running test-lns-cuda-parity + test-lns-backend (LNSCUDA32)"
            status=0
            gpu_test bash -c 'cd "$1" && ./bin/test-lns-cuda-parity --only=cuda32' _ "$GGML_DIR/build-lns-$mode" || status=$?
            [ "$status" -eq 1 ] && return 1
            if [ "$status" -eq 0 ]; then
                ( cd "$GGML_DIR/build-lns-$mode" && ./bin/test-lns-backend LNSCUDA32 ) || return 1
            fi
        fi
    done
}

run_build_and_test() {
    local mode

    # ── Unit test (standalone ggml) ───────────────────────────────────────────

    if [ "$RUN_UNIT_TEST" -eq 1 ]; then
        echo ""
        echo "==> Setting up standalone ggml for unit test"

        GGML_DIR="$BUILD_DIR/ggml"
        prepare_ggml "$GGML_DIR" || return 1
        GGML_PREPARED=1
        copy_lns_tests "$GGML_DIR" || return 1

        local test_targets=(test-lns-backend)
        if [ "$LNS_CUDA" -eq 1 ]; then
            test_targets+=(test-lns-cuda-parity)
        fi
        for mode in "${MODES[@]}"; do
            echo "    Building ${test_targets[*]} (sum mode: $mode)"
            build_lns_variant "$GGML_DIR" "$mode" "${test_targets[@]}" || return 1
        done
        link_build_compat "$GGML_DIR" || return 1

        for mode in "${MODES[@]}"; do
            echo ""
            echo "==> Running test-lns-backend (LNS16, tolerance profile: $mode)"
            ( cd "$GGML_DIR/build-lns-$mode" && \
                GGML_LNS_BACKEND=16 ./bin/test-lns-backend LNS16 "$mode" ) || return 1
        done

        echo ""
        echo "==> Running test-lns-backend (LNS32)"
        ( cd "$GGML_DIR/build-lns-${MODES[0]}" && \
            GGML_LNS_BACKEND=32 ./bin/test-lns-backend LNS32 ) || return 1

        if [ "$LNS_CUDA" -eq 1 ]; then
            run_cuda_unit_tests || return 1
        fi
    fi

    # ── End-to-end inference (llama.cpp) ─────────────────────────────────────

    if [ "$RUN_INFERENCE" -eq 1 ]; then
        echo ""
        echo "==> Setting up llama.cpp for end-to-end LNS inference"

        LLAMA_DIR="$BUILD_DIR/llama.cpp"
        prepare_llama "$LLAMA_DIR" || return 1
        LLAMA_PREPARED=1

        for mode in "${MODES[@]}"; do
            echo "    Building llama-completion + llama-perplexity (sum mode: $mode)"
            build_lns_variant "$LLAMA_DIR" "$mode" llama-completion llama-perplexity || return 1
        done
        link_build_compat "$LLAMA_DIR" || return 1

        echo "    Building CPU reference (build-ref: no LNS, no BLAS)"
        build_cpu_ref "$LLAMA_DIR" || return 1

        for mode in "${MODES[@]}"; do
            local llama_bin="$LLAMA_DIR/build-lns-$mode/bin/llama-completion"
            if [ -n "$MODEL_SMOLLM" ]; then
                run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" "$llama_bin" 16 20 "$mode" || return 1
            fi
            if [ -n "$MODEL_LLAMA32" ]; then
                run_inference "Llama-3.2-1B-Instruct" "$MODEL_LLAMA32" "$llama_bin" 16 20 "$mode" || return 1
            fi
        done

        if [ -n "$MODEL_SMOLLM" ] && [ "$LNS32_INFERENCE" -eq 1 ]; then
            # Short run: the LNS32 backend converts every weight element
            # through libm log() per matmul (no lookup table exists).
            # Sum mode does not apply to LNS32; any variant binary will do.
            run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" \
                "$LLAMA_DIR/build-lns-${MODES[0]}/bin/llama-completion" 32 4 "${MODES[0]}" || return 1
        fi

        if [ "$LNS_CUDA" -eq 1 ]; then
            # LNSCUDA16 from each mode's own build; LNSCUDA32 once
            for mode in "${MODES[@]}"; do
                local cuda_bin="$LLAMA_DIR/build-lns-$mode/bin/llama-completion"
                if ! "$cuda_bin" --list-devices 2>/dev/null | grep -q "LNSCUDA16:.*sum=$mode[,)]"; then
                    echo "SKIPPED: cuda16 inference ($mode): no usable LNSCUDA16 device" >&2
                    [ "$LNS_REQUIRE_GPU" = 1 ] && return 1
                    continue
                fi
                if [ -n "$MODEL_SMOLLM" ]; then
                    run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" "$cuda_bin" cuda16 20 "$mode" || return 1
                fi
                if [ -n "$MODEL_LLAMA32" ]; then
                    run_inference "Llama-3.2-1B-Instruct" "$MODEL_LLAMA32" "$cuda_bin" cuda16 20 "$mode" || return 1
                fi
            done
            if [ -n "$MODEL_SMOLLM" ] && "$LLAMA_DIR/build-lns-${MODES[0]}/bin/llama-completion" --list-devices 2>/dev/null | grep -q LNSCUDA32; then
                run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" \
                    "$LLAMA_DIR/build-lns-${MODES[0]}/bin/llama-completion" cuda32 20 "${MODES[0]}" || return 1
            fi
        fi
    fi
}

# After a successful --update run, record the new tree states as the pinned
# "last known good": rewrite the base-commit files and regenerate the fallback
# patches from the integrated trees (tracked-file diffs = exactly the hooks;
# the ggml pathspec excludes the separately-applied examples/simple test hook).
update_pins() {
    echo ""
    echo "==> Recording updated trees as pinned commits (last known good)"
    git -C "$XLNSCPP_DIR" rev-parse HEAD > "$REPO_ROOT/patches/xlnscpp-base-commit.txt"
    echo "    xlnscpp  $(git -C "$XLNSCPP_DIR" rev-parse --short HEAD)"
    if [ "$LNS_CUDA" -eq 1 ]; then
        git -C "$XLNSCUDA_DIR" rev-parse HEAD > "$REPO_ROOT/patches/xlnscuda-base-commit.txt"
        echo "    xlnscuda $(git -C "$XLNSCUDA_DIR" rev-parse --short HEAD)"
    fi
    if [ "$LLAMA_PREPARED" -eq 1 ]; then
        git -C "$BUILD_DIR/llama.cpp" rev-parse HEAD > "$REPO_ROOT/patches/llama-base-commit.txt"
        git -C "$BUILD_DIR/llama.cpp" diff -- ggml/ > "$REPO_ROOT/patches/llama-core.patch"
        echo "    llama.cpp $(git -C "$BUILD_DIR/llama.cpp" rev-parse --short HEAD) (+ llama-core.patch)"
    fi
    if [ "$GGML_PREPARED" -eq 1 ]; then
        git -C "$BUILD_DIR/ggml" rev-parse HEAD > "$REPO_ROOT/patches/ggml-base-commit.txt"
        git -C "$BUILD_DIR/ggml" diff -- \
            CMakeLists.txt include/ggml.h src/CMakeLists.txt \
            src/ggml-backend-reg.cpp src/ggml.c \
            > "$REPO_ROOT/patches/ggml-core.patch"
        echo "    ggml     $(git -C "$BUILD_DIR/ggml" rev-parse --short HEAD) (+ ggml-core.patch)"
    fi
    echo "    (commit the patches/ changes to make the new pins permanent)"
}

# ── Prerequisites ─────────────────────────────────────────────────────────────

check_deps

echo "==> Preparing xlnscpp"
prepare_xlnscpp

if [ "$LNS_CUDA" -eq 1 ]; then
    echo "==> Preparing xlnscuda"
    prepare_xlnscuda
    pick_cuda_host_compiler
fi

mkdir -p "$BUILD_DIR"

LLAMA_PREPARED=0
GGML_PREPARED=0

if ! run_build_and_test; then
    if [ "$DO_UPDATE" -eq 1 ]; then
        # Latest xlnscpp / xlnscuda may be the culprit; retry against their pinned commits.
        fallback_xlnscpp
        run_build_and_test
    else
        echo "ERROR: build or test failed at the pinned commits." >&2
        exit 1
    fi
fi

if [ "$DO_UPDATE" -eq 1 ]; then
    update_pins
fi

echo ""
echo "==> Done."
