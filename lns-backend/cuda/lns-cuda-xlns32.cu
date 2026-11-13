// xlns32 kernels for the LNSCUDA32 device.
//
// Configured exactly like the LNS32 CPU backend (xlns32_alt, no
// xlns32_ideal): addition uses xlnscuda's port of the interpolated sb table
// and cotransformed db, bit-identical to the host. Conversions and the
// activations go through libm (log / pow / exp / tanh) on both sides, where
// CUDA's and glibc's results can differ in the last ulp; see
// xlnscuda/tests/chkxlns32d_table.cu.

// System headers first: the library includes below sit in a namespace, where
// their own nested includes of these headers must be no-ops.
#include <iostream>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "lns-cuda-kernels.h"

// See lns-cuda-xlns16.cu for why the libraries live in an anonymous namespace.
namespace {

using ::abs; using ::atan; using ::cos; using ::exp; using ::fabs; using ::log;
using ::pow; using ::sin; using ::sqrt; using ::tanh;

#define xlns32_alt
#include "xlns32.cpp"
#include "xlns32d.cu"

struct xlns32_arith {
    using value = xlns32;
    static constexpr const char * name = "xlns32";
    static constexpr const char * sum_mode   = nullptr; // GGML_LNS16_SUM is xlns16-only
    static constexpr const char * monte_impl = nullptr;
    struct reduction {
        using acc = xlns32;
        struct state {};
        static constexpr bool stochastic = false;
        __device__ static state  start(unsigned, uint64_t, uint64_t) { return {}; }
        __device__ static acc    from(xlns32 v)                     { return v; }
        __device__ static acc    add(acc a, xlns32 t, state &)      { return xlns32d_add(a, t); }
        __device__ static xlns32 finish(acc a)                      { return a; }
    };
    static constexpr value zero    = xlns32_zero;
    static constexpr value neg_inf = xlns32_neg_inf;

    static cudaError_t upload_tables() { return xlns32d_table_init(); }

    __device__ static value from_f32(float x)        { return fp2xlns32d(x); }
    __device__ static float to_f32(value v)          { return xlns32d2fp(v); }
    __device__ static value add(value a, value b)    { return xlns32d_add(a, b); }
    __device__ static value sub(value a, value b)    { return xlns32d_sub(a, b); }
    __device__ static value mul(value a, value b)    { return xlns32d_mul(a, b); }
    __device__ static value div(value a, value b)    { return xlns32d_div(a, b); }
    __device__ static bool  gt(value a, value b)     { return xlns32d_gt(a, b); }
    __device__ static bool  is_zero(value v)         { return xlns32d_is_zero(v); }
    __device__ static value exp(value v)             { return xlns32d_exp(v); }
    __device__ static value sqrt(value v)            { return xlns32_sqrt(v); }
    __device__ static value recip(value v)           { return xlns32_recip(v); }
    __device__ static value silu(value v)            { return xlns32d_silu(v); }
    __device__ static value gelu(value v)            { return xlns32d_gelu(v); }
    __device__ static value relu(value v)            { return xlns32d_relu(v); }
    __device__ static value tanh(value v)            { return xlns32d_tanh(v); }

    // lns32-ops.cpp: f32_mask_to_lns32
    __device__ static value mask_from_f32(float m)   { return m <= -3.0e38f ? neg_inf : from_f32(m); }
};

} // namespace

#include "lns-cuda-kernels.cuh"

const lns_cuda::kernel_set & lns_cuda::xlns32_kernels() {
    static const detail::kernel_set_impl<xlns32_arith> kernels{};
    return kernels;
}
