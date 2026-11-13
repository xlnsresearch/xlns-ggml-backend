#include "lns-ops.h"
#include "ggml-lns.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================
// xlnscpp inclusion
// ============================================================
// System headers are hoisted to global scope here; the nested includes inside
// the anonymous namespace below are then no-ops (include guards).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#if (defined(GGML_LNS16_SUM_MONTE) + defined(GGML_LNS16_SUM_LPVIP) + defined(GGML_LNS16_SUM_SUM32)) > 1
#error "GGML_LNS16_SUM_{MONTE,LPVIP,SUM32} are mutually exclusive (set GGML_LNS16_SUM to one of base|monte|lpvip|sum32)"
#endif

// The xlnscpp .cpp files carry non-inline definitions and mutable global
// tables, so they may appear in exactly one translation unit per link unit.
// The anonymous namespace gives this TU a private (internal-linkage) copy so
// it cannot collide at link/ODR level with ggml-lns32's differently-configured
// copy of xlns32.cpp.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
namespace {

// xlnscpp declares xlns16_float/xlns32_float overloads of these libm names
// inside this namespace, which would otherwise hide the global libm functions
// from the library's own internal calls (unqualified lookup stops at the
// nearest scope that has the name). Import the global overload sets so both
// participate in overload resolution.
using ::sin; using ::cos; using ::exp; using ::log; using ::atan;
using ::sqrt; using ::pow; using ::fabs; using ::abs;

// xlns16 defines — must be set before including xlns16.cpp
// (also set via CMake target_compile_definitions; guarded here for clarity).
// xlns16_table: O(1) float<->xlns16 via 65536-entry lookup table.
// xlns16_alt: reduced-branching sb/db addition path.
// xlns16_ideal is NOT defined (matches a realistic hardware target).
#ifndef xlns16_table
#define xlns16_table
#endif
#ifndef xlns16_alt
#define xlns16_alt
#endif

#include "xlns16.cpp"          // must be FIRST: defines xlns16_zero, which
                               // gates the xlns16_*_lpvip32 and _sum32 families
                               // below

#if defined(GGML_LNS16_SUM_SUM32)
// Pure xlns32 accumulator. xlns32.cpp has NO include guard and must be pulled
// in HERE, before xlns32lpvip.cpp: that file bootstraps xlns32.cpp with
// xlns32_ideal (the libm log/pow adder), and whichever include lands first
// fixes the configuration for the whole TU. Same config as ggml-lns32's
// lns32-ops.cpp: xlns32_alt, xlns32_ideal deliberately NOT defined, i.e. the
// table + interpolation sb / cotransformation db path.
// NOTE (thread safety): that non-ideal xlns32_sb is a macro writing the
// globals xlns32_z / xlns32_zh. Safe while graph_compute is single-threaded;
// must become thread_local if the backend is ever parallelized.
  #ifndef xlns32_zero
    #ifndef xlns32_alt
    #define xlns32_alt
    #endif
    #include "xlns32.cpp"
  #endif
  #include "lns16-sum32.h"     // xlns16_*_sum32 family (local until upstreamed
                               // to xlnscpp as xlns16sum32.cpp)
#endif

#if defined(GGML_LNS16_SUM_LPVIP) || \
    (defined(GGML_LNS16_SUM_MONTE) && defined(GGML_LNS16_MONTE_BYLPVIP))
#include "xlns32lpvip.cpp"     // bootstraps xlns32.cpp with xlns32_ideal +
                               // xlns32_aicasb (aicasb is required for
                               // accurate lpvip summation)
#endif

#if defined(GGML_LNS16_SUM_MONTE)
// The two monte implementations define identical symbols (xlns16_add_monte,
// xlns16_randombits, ...) and are mutually exclusive — include exactly one.
// NOTE: xlns16_randombits (LFSR PRNG state) is a mutable global, hit on every
// monte add. Safe while graph_compute is single-threaded; must become
// thread_local (with distinct nonzero per-thread seeds — 0 is an absorbing
// state) if the backend is ever parallelized. LNSCUDA16 draws from it too.
  #ifdef GGML_LNS16_MONTE_BYLPVIP
    #include "xlns16monte_bylpvip.cpp"   // MUST come after xlns32lpvip.cpp
  #else
    #include "xlns16monte.cpp"           // default: pure 16-bit MCLNS (LFSR)
  #endif
#endif

} // anonymous namespace
#pragma GCC diagnostic pop

// ============================================================
// Summation-variant selection (compile-time)
// ============================================================
// GGML_LNS16_SUM_{MONTE,LPVIP,SUM32} are set by CMake from GGML_LNS16_SUM.
// The four families have identical signatures and aliasing guarantees
// (in-place c == a is safe), so the aliases are drop-in at the call sites.

#if defined(GGML_LNS16_SUM_MONTE)
  #define LNS16_VEC_DOT        xlns16_vec_dot_monte
  #define LNS16_SOFTMAX        xlns16_softmax_monte
  #define LNS16_SOFTMAX_MASKED xlns16_softmax_masked_monte
  #define LNS16_RMS_NORM       xlns16_rms_norm_monte
  #define LNS16_SUM_MODE_NAME  "monte"
#elif defined(GGML_LNS16_SUM_LPVIP)
  #define LNS16_VEC_DOT        xlns16_vec_dot_lpvip32
  #define LNS16_SOFTMAX        xlns16_softmax_lpvip32
  #define LNS16_SOFTMAX_MASKED xlns16_softmax_masked_lpvip32
  #define LNS16_RMS_NORM       xlns16_rms_norm_lpvip32
  #define LNS16_SUM_MODE_NAME  "lpvip"
#elif defined(GGML_LNS16_SUM_SUM32)
  #define LNS16_VEC_DOT        xlns16_vec_dot_sum32
  #define LNS16_SOFTMAX        xlns16_softmax_sum32
  #define LNS16_SOFTMAX_MASKED xlns16_softmax_masked_sum32
  #define LNS16_RMS_NORM       xlns16_rms_norm_sum32
  #define LNS16_SUM_MODE_NAME  "sum32"
#else
  #define LNS16_VEC_DOT        xlns16_vec_dot
  #define LNS16_SOFTMAX        xlns16_softmax
  #define LNS16_SOFTMAX_MASKED xlns16_softmax_masked
  #define LNS16_RMS_NORM       xlns16_rms_norm
  #define LNS16_SUM_MODE_NAME  "base"
#endif

const char * lns16_sum_mode_name(void) {
    return LNS16_SUM_MODE_NAME;
}

const char * lns16_monte_impl_name(void) {
#if defined(GGML_LNS16_SUM_MONTE) && defined(GGML_LNS16_MONTE_BYLPVIP)
    return "bylpvip";
#elif defined(GGML_LNS16_SUM_MONTE)
    return "pure16";
#else
    return nullptr;
#endif
}

// ============================================================
// Monte state (ggml-lns.h): xlns16_randombits is the one Monte sequence of
// the process; LNSCUDA16 reserves blocks of it in graph order.
// ============================================================

bool ggml_backend_lns16_monte_get_state(uint16_t * state) {
#if defined(GGML_LNS16_SUM_MONTE)
    *state = xlns16_randombits;
    return true;
#else
    (void) state;
    return false;
#endif
}

bool ggml_backend_lns16_monte_set_state(uint16_t state) {
#if defined(GGML_LNS16_SUM_MONTE)
    if (state == 0) {
        return false; // absorbing LFSR state
    }
    xlns16_randombits = state;
    return true;
#else
    (void) state;
    return false;
#endif
}

bool ggml_backend_lns16_monte_reserve(uint64_t draws, uint16_t * start) {
#if defined(GGML_LNS16_SUM_MONTE)
    *start = xlns16_randombits;
    // the LFSR step of xlns16monte*.cpp; its period is 65535
    for (uint64_t i = draws % 65535; i > 0; i--) {
        const xlns16 bit = xlns16_randombits & 1;
        xlns16_randombits >>= 1;
        if (bit) {
            xlns16_randombits ^= 0xB400u;
        }
    }
    return true;
#else
    (void) draws;
    (void) start;
    return false;
#endif
}

// ============================================================
// C-linkage conversion helpers (registered as type-trait hooks)
// ============================================================

extern "C" {

void lns16_to_f32_row(const void * GGML_RESTRICT src, float * GGML_RESTRICT dst, int64_t n) {
    const xlns16 * s = (const xlns16 *)src;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = xlns162fp(s[i]);
    }
}

void f32_to_lns16_row(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t n) {
    xlns16_batch_from_float(src, (xlns16 *)dst, (size_t)n);
}

} // extern "C"

// ============================================================
// Helper: convert a row from any ggml type to xlns16
// Handles F32, LNS16, and any quantized type with to_float.
// ============================================================

static void row_to_xlns16(const void * src, enum ggml_type type, xlns16 * dst, float * f32_scratch, int64_t n) {
    if (type == GGML_TYPE_LNS16) {
        // Already xlns16 — direct copy, no conversion needed
        memcpy(dst, src, (size_t)n * sizeof(xlns16));
    } else if (type == GGML_TYPE_F32) {
        xlns16_batch_from_float((const float *)src, dst, (size_t)n);
    } else {
        const auto * traits = ggml_get_type_traits(type);
        GGML_ASSERT(traits->to_float != NULL);
        traits->to_float(src, f32_scratch, n);
        xlns16_batch_from_float(f32_scratch, dst, (size_t)n);
    }
}

// ============================================================
// Element-wise typed read/write helpers
// These use array indexing within a row — strides across rows
// are handled by the caller via the nb[] byte offsets.
// ============================================================

static inline xlns16 read_elem_xlns16(const void * row, int64_t i, enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_LNS16 || type == GGML_TYPE_F32);
    if (type == GGML_TYPE_LNS16) {
        return ((const xlns16 *)row)[i];
    }
    return fp2xlns16(((const float *)row)[i]);
}

static inline void write_elem_xlns16(void * row, int64_t i, enum ggml_type type, xlns16 val) {
    GGML_ASSERT(type == GGML_TYPE_LNS16 || type == GGML_TYPE_F32);
    if (type == GGML_TYPE_LNS16) {
        ((xlns16 *)row)[i] = val;
    } else {
        ((float *)row)[i] = xlns162fp(val);
    }
}

// ============================================================
// MUL_MAT: dst = src0^T * src1
// src0: weights (any type with to_float, or F32/LNS16)
// src1: activations (F32 or LNS16)
// dst:  F32 or LNS16 depending on dst->type
// ============================================================

// LNS_MM_ROUNDTRIP: bisection aid — quantize operands to xlns16, convert
// straight back to float, accumulate in IEEE float.
//   1 = round both operands
//   2 = round weights only (activations kept f32)
//   3 = round activations only (weights kept f32)
// If coherent in mode 1, the failure is in xlns16 accumulation; if still
// incoherent, the 16-bit operand quantization alone is fatal. Modes 2/3
// isolate which operand's rounding does the damage.
static int lns_mm_roundtrip_mode() {
    static const int v = []() {
        const char * env = getenv("LNS_MM_ROUNDTRIP");
        return env ? atoi(env) : 0;
    }();
    return v;
}

static float lns_mm_dot_roundtrip(const xlns16 * a, const xlns16 * b,
                                  const float * a_f32, const float * b_f32,
                                  int64_t K, int mode) {
    float sum = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        const float av = (mode == 3) ? a_f32[i] : xlns162fp(a[i]);
        const float bv = (mode == 2) ? b_f32[i] : xlns162fp(b[i]);
        sum += av * bv;
    }
    return sum;
}

void lns_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type0 = src0->type;
    const int64_t K = ne00;
    GGML_ASSERT(ne10 == K);

    const int mm_rt_mode = lns_mm_roundtrip_mode();

    std::vector<float>  f32_scratch(K);
    std::vector<xlns16> a_row_lns(K);
    // Pre-converted activation rows for one (i12,i13) batch slice: ne11 rows of K.
    // Converting src1 once per slice (rather than once per weight row) so this avoids
    // re-converting each weight row ne11 times.
    std::vector<xlns16> b_lns((size_t)ne11 * (size_t)K);
    // F32 row copies for roundtrip modes 2/3 (see lns_mm_roundtrip_mode).
    std::vector<float>  a_row_f32(mm_rt_mode == 3 ? (size_t)K : 0);
    std::vector<float>  b_f32(mm_rt_mode == 2 ? (size_t)ne11 * (size_t)K : 0);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            // Convert all src1 (activation) rows for this slice just once.
            for (int64_t i1 = 0; i1 < ne11; i1++) {
                const void * b_row_raw = (const char *)src1->data
                    + i1*nb11 + i12*nb12 + i13*nb13;
                row_to_xlns16(b_row_raw, src1->type, b_lns.data() + (size_t)i1 * (size_t)K,
                              f32_scratch.data(), K);
                if (mm_rt_mode == 2) {
                    float * bf = b_f32.data() + (size_t)i1 * (size_t)K;
                    const xlns16 * bl = b_lns.data() + (size_t)i1 * (size_t)K;
                    if (src1->type == GGML_TYPE_F32) {
                        memcpy(bf, b_row_raw, (size_t)K * sizeof(float));
                    } else if (src1->type == GGML_TYPE_LNS16) {
                        for (int64_t i = 0; i < K; i++) {
                            bf[i] = xlns162fp(bl[i]);
                        }
                    } else {
                        // scratch holds the dequantized f32 row produced by row_to_xlns16
                        memcpy(bf, f32_scratch.data(), (size_t)K * sizeof(float));
                    }
                }
            }

            for (int64_t i0 = 0; i0 < ne01; i0++) {
                // src0 (weights): any supported type — convert each weight row only once per slice.
                const void * a_row_raw = (const char *)src0->data
                    + i0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
                row_to_xlns16(a_row_raw, type0, a_row_lns.data(), f32_scratch.data(), K);
                if (mm_rt_mode == 3) {
                    if (type0 == GGML_TYPE_F32) {
                        memcpy(a_row_f32.data(), a_row_raw, (size_t)K * sizeof(float));
                    } else if (type0 == GGML_TYPE_LNS16) {
                        // weights already xlns16 — best available f32 is the roundtrip
                        for (int64_t i = 0; i < K; i++) {
                            a_row_f32[(size_t)i] = xlns162fp(a_row_lns[(size_t)i]);
                        }
                    } else {
                        memcpy(a_row_f32.data(), f32_scratch.data(), (size_t)K * sizeof(float));
                    }
                }

                for (int64_t i1 = 0; i1 < ne11; i1++) {
                    xlns16 dot;
                    if (mm_rt_mode != 0) {
                        dot = fp2xlns16(lns_mm_dot_roundtrip(a_row_lns.data(),
                                                             b_lns.data() + (size_t)i1 * (size_t)K,
                                                             a_row_f32.data(),
                                                             b_f32.empty() ? nullptr : b_f32.data() + (size_t)i1 * (size_t)K,
                                                             K, mm_rt_mode));
                    } else {
                        dot = LNS16_VEC_DOT(a_row_lns.data(),
                                            b_lns.data() + (size_t)i1 * (size_t)K, K);
                    }

                    // Write result: keep as xlns16 if dst is LNS16, else convert to F32
                    void * dst_ptr = (char *)dst->data + i0*nb0 + i1*nb1 + i12*nb2 + i13*nb3;
                    if (dst->type == GGML_TYPE_LNS16) {
                        *(xlns16 *)dst_ptr = dot;
                    } else {
                        *(float *)dst_ptr = xlns162fp(dot);
                    }
                }
            }
        }
    }
}

// ============================================================
// MUL_MAT_ID: batched/grouped MUL_MAT for MoE expert FFNs
// src0: expert weights [n_embd, n_ff, n_expert]
// src1: activations    [n_embd, n_expert_used_bcast, n_tokens] (F32)
// src2: ids            [n_expert_used, n_tokens] (I32) — selected expert per slot
// dst:  [n_ff, n_expert_used, n_tokens]
// ============================================================

void lns_mul_mat_id(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const enum ggml_type type0 = src0->type;
    const int64_t K = src0->ne[0]; // n_embd
    const int64_t N = src0->ne[1]; // n_ff
    const int64_t n_expert = src0->ne[2];
    GGML_ASSERT(src1->ne[0] == K);

    // Match ggml_mul_mat_id / CPU kernel invariants
    GGML_ASSERT(src0->nb[0] == ggml_type_size(type0));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ids->ne[1] == src1->ne[2]); // one id list per activation token

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t ne11          = src1->ne[1]; // activation broadcast rows per token

    GGML_ASSERT(ne11 > 0);
    GGML_ASSERT(n_expert_used % ne11 == 0); // can broadcast activations over slots
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT(dst->ne[1] == n_expert_used);
    GGML_ASSERT(dst->ne[2] == n_tokens);

    std::vector<float>  f32_scratch(K);
    std::vector<xlns16> w_row_lns(K);
    // Activation rows for the current token, converted once and reused across
    // every expert slot that broadcasts to the same row (ne11 is 1 in practice).
    std::vector<xlns16> act_lns((size_t)ne11 * (size_t)K);

    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            const void * act_row_raw = (const char *)src1->data + i11*src1->nb[1] + t*src1->nb[2];
            row_to_xlns16(act_row_raw, src1->type, act_lns.data() + (size_t)i11 * (size_t)K,
                          f32_scratch.data(), K);
        }

        for (int64_t s = 0; s < n_expert_used; s++) {
            const int32_t expert = *(const int32_t *)((const char *)ids->data
                + t*ids->nb[1] + s*ids->nb[0]);
            GGML_ASSERT(expert >= 0 && expert < n_expert);

            // Slot-based broadcast (matches CPU one_chunk): act row = s % ne11, not expert % ne11.
            const int64_t i11 = s % ne11;
            const xlns16 * act_row_lns = act_lns.data() + (size_t)i11 * (size_t)K;

            const char * expert_base = (const char *)src0->data + expert*src0->nb[2];
            char * dst_col = (char *)dst->data + s*dst->nb[1] + t*dst->nb[2];

            for (int64_t i0 = 0; i0 < N; i0++) {
                const void * w_row_raw = expert_base + i0*src0->nb[1];
                row_to_xlns16(w_row_raw, type0, w_row_lns.data(), f32_scratch.data(), K);

                xlns16 dot = LNS16_VEC_DOT(w_row_lns.data(), act_row_lns, K);

                void * dst_ptr = dst_col + i0*dst->nb[0];
                if (dst->type == GGML_TYPE_LNS16) {
                    *(xlns16 *)dst_ptr = dot;
                } else {
                    *(float *)dst_ptr = xlns162fp(dot);
                }
            }
        }
    }
}

// ============================================================
// ADD: dst = src0 + src1 (with broadcasting)
// ============================================================

void lns_add(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // Match ggml-cpu/binary-ops.cpp apply_binary_op(): full repeat-compatibility
    // across all dims (not just dim 0), plus dst must match src0's shape.
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns16 a = read_elem_xlns16(s0, i0, src0->type);
                    xlns16 b = read_elem_xlns16(s1, i0 % ne10, src1->type);
                    write_elem_xlns16(d, i0, dst->type, xlns16_add(a, b));
                }
            }
        }
    }
}

// ============================================================
// MUL: dst = src0 * src1 (element-wise with broadcasting)
// ============================================================

void lns_mul(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // Match ggml-cpu/binary-ops.cpp apply_binary_op(): full repeat-compatibility
    // across all dims (not just dim 0), plus dst must match src0's shape.
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns16 a = read_elem_xlns16(s0, i0, src0->type);
                    xlns16 b = read_elem_xlns16(s1, i0 % ne10, src1->type);
                    write_elem_xlns16(d, i0, dst->type, xlns16_mul(a, b));
                }
            }
        }
    }
}

// ============================================================
// DIV: dst = src0 / src1 (element-wise with broadcasting)
// ============================================================

void lns_div(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns16 a = read_elem_xlns16(s0, i0, src0->type);
                    xlns16 b = read_elem_xlns16(s1, i0 % ne10, src1->type);
                    write_elem_xlns16(d, i0, dst->type, xlns16_div(a, b));
                }
            }
        }
    }
}

// ============================================================
// SCALE: dst = src0 * s + b
// ============================================================

void lns_scale(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float s, b;
    memcpy(&s, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (float *) dst->op_params + 1, sizeof(float));

    const xlns16 s_lns = fp2xlns16(s);
    const xlns16 b_lns = fp2xlns16(b);

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns16 val;
        if (src0->type == GGML_TYPE_LNS16) {
            val = ((const xlns16 *)src0->data)[i];
        } else {
            val = fp2xlns16(((const float *)src0->data)[i]);
        }
        val = xlns16_mul(val, s_lns);
        if (b != 0.0f) {
            val = xlns16_add(val, b_lns);
        }
        if (dst->type == GGML_TYPE_LNS16) {
            ((xlns16 *)dst->data)[i] = val;
        } else {
            ((float *)dst->data)[i] = xlns162fp(val);
        }
    }
}

// ============================================================
// CLAMP: dst = max(min(x, max), min)
// ============================================================

void lns_clamp(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float min_f, max_f;
    memcpy(&min_f, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_f, (float *) dst->op_params + 1, sizeof(float));

    const xlns16 min_lns = fp2xlns16(min_f);
    const xlns16 max_lns = fp2xlns16(max_f);

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns16 v = (src0->type == GGML_TYPE_LNS16)
            ? ((const xlns16 *)src0->data)[i]
            : fp2xlns16(((const float *)src0->data)[i]);
        xlns16 r = xlns16_max(xlns16_min(v, max_lns), min_lns);
        if (dst->type == GGML_TYPE_LNS16) {
            ((xlns16 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns162fp(r);
        }
    }
}

// Convert one F32 mask entry to xlns16 (mask tensor is always F32 in ggml).
// ggml represents "masked out" as a very large negative float (its own
// -infinity convention), not xlns16's own -inf encoding, so this sentinel
// mapping belongs here (ggml-tensor extraction) rather than in xlns16.cpp.
static inline xlns16 f32_mask_to_lns16(float m) {
    if (m <= -3.0e38f) return xlns16_neg_inf;
    return fp2xlns16(m);
}

// ============================================================
// SOFT_MAX: dst = softmax(src0 * scale + mask)
// ============================================================

void lns_soft_max(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1]; // mask (optional)

    // TODO: sink support if required? (models such as openai MoE, MiMo), Llama, SmolLM, don't require them
    // TODO: implement ALiBi -- similar to sinks, not required by current set of models

    float scale, max_bias;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    (void)max_bias; // ignored for now (see TODO above)

    const int64_t ne00 = src0->ne[0]; // sequence length
    const int64_t ne01 = src0->ne[1]; // num queries
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const xlns16 scale_lns = fp2xlns16(scale);

    std::vector<xlns16> row(ne00);
    std::vector<xlns16> mask_lns(src1 ? (size_t)ne00 : 0);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0_row = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * d_row = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    row[i0] = read_elem_xlns16(s0_row, i0, src0->type);
                }

                if (src1) {
                    const int64_t ne12 = src1->ne[2];
                    const float * mask = (const float *)((const char *)src1->data
                        + i1*src1->nb[1] + (i2 % ne12)*src1->nb[2] + i3*src1->nb[3]);
                    for (int64_t i0 = 0; i0 < ne00; i0++) {
                        mask_lns[(size_t)i0] = f32_mask_to_lns16(mask[i0]);
                    }
                    LNS16_SOFTMAX_MASKED(row.data(), mask_lns.data(), row.data(), (size_t)ne00, scale_lns);
                } else {
                    LNS16_SOFTMAX(row.data(), row.data(), (size_t)ne00, scale_lns);
                }

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    write_elem_xlns16(d_row, i0, dst->type, row[i0]);
                }
            }
        }
    }
}

// ============================================================
// RMS_NORM: dst = x / sqrt(mean(x^2) + eps)
// ============================================================

void lns_rms_norm(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const xlns16 eps_lns = fp2xlns16(eps);

    std::vector<xlns16> row((size_t)ne00);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * d = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    row[(size_t)i0] = read_elem_xlns16(s, i0, src0->type);
                }

                LNS16_RMS_NORM(row.data(), row.data(), (size_t)ne00, eps_lns);

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    write_elem_xlns16(d, i0, dst->type, row[(size_t)i0]);
                }
            }
        }
    }
}

// ============================================================
// DIAG_MASK_INF: causal attention mask
// ============================================================

void lns_diag_mask_inf(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    int n_past;
    memcpy(&n_past, dst->op_params, sizeof(int32_t));

    const int64_t nc = src0->ne[0]; // columns
    const int64_t nr = src0->ne[1]; // rows
    const int64_t n  = ggml_nrows(src0);
    const int64_t nz = n / nr;

    const bool inplace = (src0->data == dst->data);
    if (!inplace) {
        // safety: xlns16 is 2 bytes vs F32's 4 bytes — dst buffer was F32-sized, so always fits
        GGML_ASSERT(ggml_nbytes(dst) <= ggml_nbytes(src0));
        memcpy(dst->data, src0->data, ggml_nbytes(dst));
    }

    if (dst->type == GGML_TYPE_LNS16) {
        const xlns16 neg_inf = xlns16_neg_inf;
        xlns16 * data = (xlns16 *)dst->data;
        for (int64_t k = 0; k < nz; k++) {
            for (int64_t j = 0; j < nr; j++) {
                for (int64_t i = n_past + j + 1; i < nc; i++) {
                    data[k*nr*nc + j*nc + i] = neg_inf;
                }
            }
        }
    } else {
        float * data = (float *)dst->data;
        for (int64_t k = 0; k < nz; k++) {
            for (int64_t j = 0; j < nr; j++) {
                for (int64_t i = n_past + j + 1; i < nc; i++) {
                    data[k*nr*nc + j*nc + i] = -INFINITY;
                }
            }
        }
    }
}

// ============================================================
// SILU: dst = x * sigmoid(x)
// ============================================================

void lns_silu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns16 v = (src0->type == GGML_TYPE_LNS16)
            ? ((const xlns16 *)src0->data)[i]
            : fp2xlns16(((const float *)src0->data)[i]);
        xlns16 r = xlns16_silu(v);
        if (dst->type == GGML_TYPE_LNS16) {
            ((xlns16 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns162fp(r);
        }
    }
}

// ============================================================
// SWIGLU: dst = silu(gate) * up
// ============================================================

void lns_swiglu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_LNS16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32 || dst->type  == GGML_TYPE_LNS16);

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_LNS16);
    } else {
        GGML_ASSERT(src0->ne[0] % 2 == 0);
    }

    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);
    if (src1) {
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(ggml_nrows(src1) == nr);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const size_t src0_elem_size = ggml_type_size(src0->type);
    const size_t src1_elem_size = src1 ? ggml_type_size(src1->type) : src0_elem_size;

    const char * src0_data = (const char *)src0->data;
    const char * src1_data = src1 ? (const char *)src1->data : src0_data;
    char * dst_data = (char *)dst->data;

    for (int64_t i1 = 0; i1 < nr; i1++) {
        const char * gate_row = src0_data + i1*src0->nb[1];
        const char * up_row   = src1_data + i1*(src1 ? src1->nb[1] : src0->nb[1]);
        char * dst_row = dst_data + i1*dst->nb[1];

        if (!src1) {
            gate_row += (swapped ? nc : 0) * src0_elem_size;
            up_row   += (swapped ? 0 : nc) * src1_elem_size;
        }

        for (int64_t k = 0; k < nc; k++) {
            const xlns16 gate = read_elem_xlns16(gate_row, k, src0->type);
            const xlns16 up   = read_elem_xlns16(up_row,   k, src1 ? src1->type : src0->type);
            write_elem_xlns16(dst_row, k, dst->type, xlns16_mul(xlns16_silu(gate), up));
        }
    }
}

// ============================================================
// GEGLU: dst = gelu(gate) * up
// Same structure as SWIGLU (lns_swiglu above), just gelu instead of silu.
// ============================================================

void lns_geglu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_LNS16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32 || dst->type  == GGML_TYPE_LNS16);

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_LNS16);
    } else {
        GGML_ASSERT(src0->ne[0] % 2 == 0);
    }

    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);
    if (src1) {
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(ggml_nrows(src1) == nr);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const size_t src0_elem_size = ggml_type_size(src0->type);
    const size_t src1_elem_size = src1 ? ggml_type_size(src1->type) : src0_elem_size;

    const char * src0_data = (const char *)src0->data;
    const char * src1_data = src1 ? (const char *)src1->data : src0_data;
    char * dst_data = (char *)dst->data;

    for (int64_t i1 = 0; i1 < nr; i1++) {
        const char * gate_row = src0_data + i1*src0->nb[1];
        const char * up_row   = src1_data + i1*(src1 ? src1->nb[1] : src0->nb[1]);
        char * dst_row = dst_data + i1*dst->nb[1];

        if (!src1) {
            gate_row += (swapped ? nc : 0) * src0_elem_size;
            up_row   += (swapped ? 0 : nc) * src1_elem_size;
        }

        for (int64_t k = 0; k < nc; k++) {
            const xlns16 gate = read_elem_xlns16(gate_row, k, src0->type);
            const xlns16 up   = read_elem_xlns16(up_row,   k, src1 ? src1->type : src0->type);
            write_elem_xlns16(dst_row, k, dst->type, xlns16_mul(xlns16_gelu(gate), up));
        }
    }
}

// ============================================================
// GELU: dst = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// ============================================================

void lns_gelu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns16 v = (src0->type == GGML_TYPE_LNS16)
            ? ((const xlns16 *)src0->data)[i]
            : fp2xlns16(((const float *)src0->data)[i]);
        xlns16 r = xlns16_gelu(v);
        if (dst->type == GGML_TYPE_LNS16) {
            ((xlns16 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns162fp(r);
        }
    }
}

// ============================================================
// RELU: dst = max(0, x)
// ============================================================

void lns_relu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns16 v = (src0->type == GGML_TYPE_LNS16)
            ? ((const xlns16 *)src0->data)[i]
            : fp2xlns16(((const float *)src0->data)[i]);
        xlns16 r = xlns16_relu(v);
        if (dst->type == GGML_TYPE_LNS16) {
            ((xlns16 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns162fp(r);
        }
    }
}

// ============================================================
// TANH: dst = tanh(x)
// ============================================================

void lns_tanh(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns16 v = (src0->type == GGML_TYPE_LNS16)
            ? ((const xlns16 *)src0->data)[i]
            : fp2xlns16(((const float *)src0->data)[i]);
        xlns16 r = xlns16_tanh(v);
        if (dst->type == GGML_TYPE_LNS16) {
            ((xlns16 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns162fp(r);
        }
    }
}

// ============================================================
// GET_ROWS: gather rows from src0 using indices in src1
// ============================================================

void lns_get_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // embedding table
    const struct ggml_tensor * src1 = dst->src[1]; // indices (I32)

    const int64_t nc = src0->ne[0]; // embedding dim
    const enum ggml_type type = src0->type;
    const auto * traits = ggml_get_type_traits(type);

    const int64_t nr = ggml_nelements(src1); // number of rows to gather

    std::vector<float> f32_scratch(nc);

    for (int64_t i = 0; i < nr; i++) {
        // Compute multi-dimensional index into src1
        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12*ne11*ne10) / ne10;
        const int64_t i10 = i - i12*ne11*ne10 - i11*ne10;

        // Get row index
        const int32_t row_idx = ((const int32_t *)((const char *)src1->data
            + i10*src1->nb[0] + i11*src1->nb[1] + i12*src1->nb[2]))[0];
        GGML_ASSERT(row_idx >= 0 && row_idx < src0->ne[1]);

        // Source row (embedding table — never LNS16, always a weight type)
        const void * src_row = (const char *)src0->data
            + row_idx*src0->nb[1] + i11*src0->nb[2] + i12*src0->nb[3];

        if (dst->type == GGML_TYPE_LNS16) {
            // Output directly as xlns16: dequantize to F32, then convert
            xlns16 * dst_row = (xlns16 *)((char *)dst->data
                + i10*dst->nb[1] + i11*dst->nb[2] + i12*dst->nb[3]);
            if (type == GGML_TYPE_F32) {
                xlns16_batch_from_float((const float *)src_row, dst_row, (size_t)nc);
            } else {
                GGML_ASSERT(traits->to_float != NULL);
                traits->to_float(src_row, f32_scratch.data(), nc);
                xlns16_batch_from_float(f32_scratch.data(), dst_row, (size_t)nc);
            }
        } else {
            // Output as F32 (original behaviour, for non-LNS mode)
            float * dst_row = (float *)((char *)dst->data
                + i10*dst->nb[1] + i11*dst->nb[2] + i12*dst->nb[3]);
            if (type == GGML_TYPE_F32) {
                memcpy(dst_row, src_row, (size_t)nc * sizeof(float));
            } else {
                GGML_ASSERT(traits->to_float != NULL);
                traits->to_float(src_row, dst_row, nc);
            }
        }
    }
}

// ============================================================
// CPY / CONT / DUP: copy src0 to dst (possibly with type conversion)
// ============================================================

void lns_cpy(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    // Same-type contiguous tensors: raw memcpy
    if (src0->type == dst->type && ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        memcpy(dst->data, src0->data, ggml_nbytes(dst));
        return;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // F32 -> F32 (with stride differences)
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * s = (const float *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    float * d = (float *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    memcpy(d, s, (size_t)ne00 * sizeof(float));
                }
        return;
    }

    // F32 -> F16 (common for KV cache)
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * s = (const float *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    ggml_fp16_t * d = (ggml_fp16_t *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    for (int64_t i0 = 0; i0 < ne00; i0++) {
                        d[i0] = ggml_fp32_to_fp16(s[i0]);
                    }
                }
        return;
    }

    // LNS16 -> F32 (e.g., output logits conversion)
    if (src0->type == GGML_TYPE_LNS16 && dst->type == GGML_TYPE_F32) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const xlns16 * s = (const xlns16 *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    float * d = (float *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    lns16_to_f32_row(s, d, ne00);
                }
        return;
    }

    // F32 -> LNS16
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_LNS16) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * s = (const float *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    xlns16 * d = (xlns16 *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    f32_to_lns16_row(s, d, ne00);
                }
        return;
    }

    // LNS16 -> LNS16 (with stride differences — same-type contiguous handled above)
    if (src0->type == GGML_TYPE_LNS16 && dst->type == GGML_TYPE_LNS16) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const xlns16 * s = (const xlns16 *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    xlns16 * d = (xlns16 *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    memcpy(d, s, (size_t)ne00 * sizeof(xlns16));
                }
        return;
    }

    GGML_ABORT("lns_cpy: unsupported type conversion %s -> %s\n",
               ggml_type_name(src0->type), ggml_type_name(dst->type));
}

// ============================================================
// ARGSORT: dst = permutation of indices sorting src0's rows
// Reference: ggml-cpu/ops.cpp ggml_compute_forward_argsort_f32()
// Pure comparison, no arithmetic — xlns16_lt/xlns16_gt (via xlns16_canon)
// ============================================================

void lns_argsort(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const enum ggml_sort_order order = (enum ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    std::vector<xlns16> row((size_t)nc);

    for (int64_t i = 0; i < nr; i++) {
        const void * src_row = (const char *)src0->data + i*src0->nb[1];
        int32_t * dst_row = (int32_t *)((char *)dst->data + i*dst->nb[1]);

        for (int64_t j = 0; j < nc; j++) {
            row[(size_t)j] = read_elem_xlns16(src_row, j, src0->type);
            dst_row[j] = (int32_t)j;
        }

        switch (order) {
            case GGML_SORT_ORDER_ASC:
                std::sort(dst_row, dst_row + nc, [&row](int32_t a, int32_t b) {
                    return xlns16_lt(row[(size_t)a], row[(size_t)b]);
                });
                break;
            case GGML_SORT_ORDER_DESC:
                std::sort(dst_row, dst_row + nc, [&row](int32_t a, int32_t b) {
                    return xlns16_gt(row[(size_t)a], row[(size_t)b]);
                });
                break;
            default:
                GGML_ABORT("lns_argsort: invalid sort order");
        }
    }
}

// ============================================================
// ROPE: Rotary Position Embeddings
// ============================================================

void lns_rope(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // Q or K tensor
    const struct ggml_tensor * src1 = dst->src[1]; // position indices (I32)

    const int64_t ne0 = src0->ne[0]; // head dim
    const int64_t ne1 = src0->ne[1]; // num heads
    const int64_t ne2 = src0->ne[2]; // seq len
    const int64_t ne3 = src0->ne[3]; // batch

    const int32_t n_dims     = ((int32_t *) dst->op_params)[1];
    const int32_t mode       = ((int32_t *) dst->op_params)[2];
    const int32_t n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const bool is_neox = (mode & 2) != 0; // NEOX-style: pairs at distance n_dims/2

    // YaRN, linear scaling and attention scaling are declined by
    // lns::check_rope; only the optional frequency factors (src[2]) apply.
    (void)n_ctx_orig; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)freq_scale;
    const float * freq_factors = dst->src[2] ? (const float *) dst->src[2]->data : nullptr;

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int32_t pos = ((const int32_t *)src1->data)[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const void * src_row = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * dst_row = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                float t = (float)pos;
                for (int64_t j = 0; j < n_dims; j += 2) {
                    const float theta = freq_factors ? t / freq_factors[j / 2] : t;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);

                    // Convert cos/sin to xlns16 for the rotation
                    xlns16 c     = fp2xlns16(cos_t);
                    xlns16 s_val = fp2xlns16(sin_t);

                    int64_t i0_0, i0_1;
                    if (is_neox) {
                        // NEOX: pairs at (j/2, j/2 + n_dims/2)
                        i0_0 = j / 2;
                        i0_1 = j / 2 + n_dims / 2;
                    } else {
                        // Normal: adjacent pairs (j, j+1)
                        i0_0 = j;
                        i0_1 = j + 1;
                    }

                    xlns16 x0 = read_elem_xlns16(src_row, i0_0, src0->type);
                    xlns16 x1 = read_elem_xlns16(src_row, i0_1, src0->type);

                    // Rotation: [cos -sin; sin cos] * [x0; x1]
                    xlns16 r0 = xlns16_sub(xlns16_mul(x0, c), xlns16_mul(x1, s_val));
                    xlns16 r1 = xlns16_add(xlns16_mul(x0, s_val), xlns16_mul(x1, c));

                    write_elem_xlns16(dst_row, i0_0, dst->type, r0);
                    write_elem_xlns16(dst_row, i0_1, dst->type, r1);

                    t *= theta_scale;
                }

                // Copy non-rotated dimensions (passthrough in xlns16 or F32)
                for (int64_t j = n_dims; j < ne0; j++) {
                    xlns16 v = read_elem_xlns16(src_row, j, src0->type);
                    write_elem_xlns16(dst_row, j, dst->type, v);
                }
            }
        }
    }
}
