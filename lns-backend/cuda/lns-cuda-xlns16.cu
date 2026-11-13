// xlns16 kernels for the LNSCUDA16 device.
//
// Configured exactly like the LNS16 CPU backend (xlns16_table + xlns16_alt,
// no xlns16_ideal), with xlnscuda's xlns16d_table mirroring the host tables
// on the device. Every conversion, addition and lookup therefore returns the
// same bits on both sides; xlnscuda/tests/chkxlns16d_table.cu checks that.
// Reductions follow GGML_LNS16_SUM like LNS16: base, monte (xlns16d_monte.cu),
// lpvip (xlns32d_lpvip.cu) or sum32 (xlns32d.cu's table-based xlns32 adder, as
// lns-backend/src/lns16-sum32.h uses xlns32.cpp's).

// System headers first: the library includes below sit in a namespace, where
// their own nested includes of these headers must be no-ops.
#include <iostream>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "lns-cuda-kernels.h"

#if defined(GGML_LNS16_SUM_LPVIP) || (defined(GGML_LNS16_SUM_MONTE) && defined(GGML_LNS16_MONTE_BYLPVIP))
#define LNS_CUDA_LPVIP
#endif

// The xlnscpp/xlnscuda sources carry non-inline definitions and their own
// globals, so each may appear in one translation unit per link unit; the
// anonymous namespace keeps this copy private to this file (the LNS16 CPU
// backend has another one).
namespace {

// Both libraries declare xlns16_float / xlns16d_float overloads of these libm
// names in this namespace, which would hide the global ones from the
// libraries' own calls. Import the global overload sets.
using ::abs; using ::atan; using ::cos; using ::exp; using ::fabs; using ::log;
using ::pow; using ::sin; using ::sqrt; using ::tanh;

#define xlns16_table
#define xlns16_alt
#define xlns16d_table
#include "xlns16.cpp"
#ifdef LNS_CUDA_LPVIP
#include "xlns32lpvip.cpp"   // xlns32 with xlns32_ideal + xlns32_aicasb, private to this file
#endif
#ifdef GGML_LNS16_SUM_SUM32
#include "xlns32.cpp"        // xlns32_alt (from CMake): LNS32's table sb + cotransformation db
#endif
#include "xlns16d.cu"
#ifdef LNS_CUDA_LPVIP
#include "xlns32d_lpvip.cu"
#endif
#ifdef GGML_LNS16_SUM_SUM32
#include "xlns32d.cu"
#endif
#ifdef GGML_LNS16_SUM_MONTE
#ifdef GGML_LNS16_MONTE_BYLPVIP
#define xlns16d_monte_bylpvip
#endif
#include "xlns16d_monte.cu"
#endif

struct xlns16_arith {
    using value = xlns16;
    static constexpr const char * name = "xlns16";
    static constexpr value zero    = xlns16_zero;
    static constexpr value neg_inf = xlns16_neg_inf;

#if defined(GGML_LNS16_SUM_MONTE)
    static constexpr const char * sum_mode = "monte";
  #ifdef GGML_LNS16_MONTE_BYLPVIP
    static constexpr const char * monte_impl = "bylpvip";
  #else
    static constexpr const char * monte_impl = "pure16";
  #endif
    // xlns16 accumulator, one LFSR draw per addition
    struct reduction {
        using acc   = xlns16;
        using state = xlns16;
        static constexpr bool stochastic = true;
        __device__ static state start(unsigned seed, uint64_t index, uint64_t draws) {
            return xlns16_lfsr_jump((xlns16) seed, xlns16_lfsr_offset(index, draws));
        }
        __device__ static acc    from(xlns16 v)                    { return v; }
        __device__ static acc    add(acc a, xlns16 t, state & s)   { return xlns16d_add_monte(a, t, &s); }
        __device__ static xlns16 finish(acc a)                     { return a; }
    };
#elif defined(GGML_LNS16_SUM_LPVIP)
    static constexpr const char * sum_mode   = "lpvip";
    static constexpr const char * monte_impl = nullptr;
    // 32-bit LPVIP accumulator; xlns16 storage is unchanged
    struct reduction {
        using acc = xlns32;
        struct state {};
        static constexpr bool stochastic = false;
        __device__ static state  start(unsigned, uint64_t, uint64_t) { return {}; }
        __device__ static acc    from(xlns16 v)                     { return xlns16_lpvip32_widen(v); }
        __device__ static acc    add(acc a, xlns16 t, state &)      { return xlns16d_lpvip32_add(a, t); }
        __device__ static xlns16 finish(acc a)                      { return xlns16_lpvip32_narrow(a); }
    };
#elif defined(GGML_LNS16_SUM_SUM32)
    static constexpr const char * sum_mode   = "sum32";
    static constexpr const char * monte_impl = nullptr;
    // xlns32 accumulator (lns16-sum32.h): exact widening, xlns32 add, one
    // round-to-nearest narrowing; xlns16 storage is unchanged
    struct reduction {
        using acc = xlns32;
        struct state {};
        static constexpr bool stochastic = false;
        __device__ static state  start(unsigned, uint64_t, uint64_t) { return {}; }
        __device__ static acc    from(xlns16 v)                     { return ((xlns32) v) << 16; }
        __device__ static acc    add(acc a, xlns16 t, state &)      { return xlns32d_add(a, ((xlns32) t) << 16); }
        __device__ static xlns16 finish(acc s) {
            if ((s & xlns32_logmask) >= (xlns32) 0x7FFF8000u) {
                return (xlns16) ((s >> 16) | xlns16_logmask); // saturate
            }
            return (xlns16) ((s + 0x8000u) >> 16);
        }
    };
#else
    static constexpr const char * sum_mode   = "base";
    static constexpr const char * monte_impl = nullptr;
    struct reduction {
        using acc = xlns16;
        struct state {};
        static constexpr bool stochastic = false;
        __device__ static state  start(unsigned, uint64_t, uint64_t) { return {}; }
        __device__ static acc    from(xlns16 v)                     { return v; }
        __device__ static acc    add(acc a, xlns16 t, state &)      { return xlns16d_add(a, t); }
        __device__ static xlns16 finish(acc a)                      { return a; }
    };
#endif

    static cudaError_t upload_tables() {
        cudaError_t err = xlns16d_table_init();
#ifdef GGML_LNS16_SUM_SUM32
        if (err == cudaSuccess) {
            err = xlns32d_table_init();
        }
#endif
        return err;
    }

    __device__ static value from_f32(float x)        { return fp2xlns16d(x); }
    __device__ static float to_f32(value v)          { return xlns16d2fp(v); }
    __device__ static value add(value a, value b)    { return xlns16d_add(a, b); }
    __device__ static value sub(value a, value b)    { return xlns16d_sub(a, b); }
    __device__ static value mul(value a, value b)    { return xlns16d_mul(a, b); }
    __device__ static value div(value a, value b)    { return xlns16d_div(a, b); }
    __device__ static bool  gt(value a, value b)     { return xlns16d_gt(a, b); }
    __device__ static bool  is_zero(value v)         { return xlns16d_is_zero(v); }
    __device__ static value exp(value v)             { return xlns16d_exp(v); }
    __device__ static value sqrt(value v)            { return xlns16_sqrt(v); }
    __device__ static value recip(value v)           { return xlns16_recip(v); }
    __device__ static value silu(value v)            { return xlns16d_silu(v); }
    __device__ static value gelu(value v)            { return xlns16d_gelu(v); }
    __device__ static value relu(value v)            { return xlns16d_relu(v); }
    __device__ static value tanh(value v)            { return xlns16d_tanh(v); }

    // ggml marks masked-out positions with a huge negative float rather than
    // xlns16's own -inf encoding (lns-ops.cpp: f32_mask_to_lns16).
    __device__ static value mask_from_f32(float m)   { return m <= -3.0e38f ? neg_inf : from_f32(m); }
};

} // namespace

#include "lns-cuda-kernels.cuh"

const lns_cuda::kernel_set & lns_cuda::xlns16_kernels() {
    static const detail::kernel_set_impl<xlns16_arith> kernels{};
    return kernels;
}
