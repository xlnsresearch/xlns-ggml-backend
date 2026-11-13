#include "lns-cuda-ops.h"

#include "ggml-impl.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace lns_cuda {
namespace {

using lns::op_support;

shape4 shape_of(const ggml_tensor * t) {
    return { { t->ne[0], t->ne[1], t->ne[2], t->ne[3] } };
}

float op_param_f32(const ggml_tensor * t, int i) {
    float v;
    std::memcpy(&v, (const int32_t *) t->op_params + i, sizeof(float));
    return v;
}

// Every handler writes an F32 result of its first input's shape.
op_support check_f32_result(const ggml_tensor * op, op_support base) {
    if (!base.supported) {
        return base;
    }
    if (op->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, op->src[0])) {
        return { false, "lns_cuda_requires_f32_dst_of_src0_shape" };
    }
    return base;
}

// ---- MUL_MAT ---------------------------------------------------------------

op_support check_mul_mat(const ggml_tensor * op) {
    const op_support base = lns::check_mul_mat(op);
    if (!base.supported) {
        return base;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->type != GGML_TYPE_F32) {
        return { false, "lns_cuda_mul_mat_requires_f32_dst" };
    }
    // grid limits of k_mul_mat (65535 in y and z)
    if ((src1->ne[1] + 3) / 4 > 65535 || src1->ne[2] * src1->ne[3] > 65535) {
        return { false, "lns_cuda_mul_mat_grid_too_large" };
    }
    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return { false, "lns_cuda_mul_mat_requires_whole_broadcast" };
    }
    return { true, "lns_cuda_mul_mat" };
}

void compute_mul_mat(context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const kernel_set & k = ctx.kernels();

    const void *  w     = ctx.lns_matrix(src0);
    const float * x_f32 = ctx.upload(src1, role::src1);
    void *        x     = ctx.device_lns(role::lns1, ggml_nelements(src1));
    LNS_CUDA_CHECK(k.to_lns(x_f32, x, ggml_nelements(src1), ctx.stream()));

    float * out = ctx.device_f32(role::dst, ggml_nelements(dst));
    mul_mat_args args = {
        w, x, out,
        src0->ne[0], src0->ne[1], src1->ne[1],
        src0->ne[2], src0->ne[3], src1->ne[2], src1->ne[3],
        0,
    };
    args.monte_seed = ctx.reserve_monte(k.mul_mat_draws(args));
    LNS_CUDA_CHECK(k.mul_mat(args, ctx.stream()));
    ctx.download(out, dst);
}

// ---- ADD / MUL / DIV -------------------------------------------------------

op_support check_binary(const ggml_tensor * op) {
    const op_support base = check_f32_result(op, lns::check_binary_f32(op));
    if (base.supported && !ggml_can_repeat(op->src[1], op->src[0])) {
        return { false, "lns_cuda_binary_requires_repeatable_src1" };
    }
    return base;
}

template <binary_op OP>
void compute_binary(context & ctx, ggml_tensor * dst) {
    const float * a   = ctx.upload(dst->src[0], role::src0);
    const float * b   = ctx.upload(dst->src[1], role::src1);
    float *       out = ctx.device_f32(role::dst, ggml_nelements(dst));
    LNS_CUDA_CHECK(ctx.kernels().binary(OP, a, shape_of(dst->src[0]), b, shape_of(dst->src[1]), out, ctx.stream()));
    ctx.download(out, dst);
}

// ---- SCALE -----------------------------------------------------------------

op_support check_scale(const ggml_tensor * op) {
    return check_f32_result(op, lns::check_contiguous_f32(op));
}

void compute_scale(context & ctx, ggml_tensor * dst) {
    const int64_t n   = ggml_nelements(dst);
    const float * src = ctx.upload(dst->src[0], role::src0);
    float *       out = ctx.device_f32(role::dst, n);
    LNS_CUDA_CHECK(ctx.kernels().scale(src, out, n, op_param_f32(dst, 0), op_param_f32(dst, 1), ctx.stream()));
    ctx.download(out, dst);
}

// ---- SILU / GELU / RELU / TANH ---------------------------------------------

op_support check_unary(const ggml_tensor * op) {
    return check_f32_result(op, lns::check_unary_activation(op));
}

template <unary_op OP>
void compute_unary(context & ctx, ggml_tensor * dst) {
    const int64_t n   = ggml_nelements(dst);
    const float * src = ctx.upload(dst->src[0], role::src0);
    float *       out = ctx.device_f32(role::dst, n);
    LNS_CUDA_CHECK(ctx.kernels().unary(OP, src, out, n, ctx.stream()));
    ctx.download(out, dst);
}

// ---- SWIGLU / GEGLU (lns_swiglu / lns_geglu) --------------------------------

template <glu_op OP>
void compute_glu(context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t ncols = dst->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * a = ctx.upload(src0, role::src0);
    const float * gate;
    const float * up;
    int64_t gate_stride = src0->ne[0];
    int64_t up_stride   = src0->ne[0];
    if (src1 != nullptr) {
        gate      = a;
        up        = ctx.upload(src1, role::src1);
        up_stride = src1->ne[0];
    } else {
        // fused: both halves live in each src0 row, gate first unless swapped
        const bool swapped = ggml_get_op_params_i32(dst, 1) != 0;
        gate = a + (swapped ? ncols : 0);
        up   = a + (swapped ? 0 : ncols);
    }
    float * out = ctx.device_f32(role::dst, ggml_nelements(dst));
    LNS_CUDA_CHECK(ctx.kernels().glu(OP, gate, up, gate_stride, up_stride, out, ncols, nrows, ctx.stream()));
    ctx.download(out, dst);
}

// ---- RMS_NORM --------------------------------------------------------------

op_support check_rms_norm(const ggml_tensor * op) {
    return check_f32_result(op, lns::check_unary_like_f32(op));
}

void compute_rms_norm(context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const kernel_set & k = ctx.kernels();
    const float * src = ctx.upload(src0, role::src0);
    float *       out = ctx.device_f32(role::dst, ggml_nelements(dst));
    const uint16_t seed = ctx.reserve_monte(k.rms_norm_draws(src0->ne[0], ggml_nrows(src0)));
    LNS_CUDA_CHECK(k.rms_norm(src, out, src0->ne[0], ggml_nrows(src0), op_param_f32(dst, 0), seed, ctx.stream()));
    ctx.download(out, dst);
}

// ---- SOFT_MAX --------------------------------------------------------------

op_support check_soft_max(const ggml_tensor * op) {
    const op_support base = check_f32_result(op, lns::check_soft_max(op));
    const ggml_tensor * mask = op->src[1];
    if (base.supported && mask != nullptr &&
        (mask->ne[0] < op->ne[0] || mask->ne[1] < op->ne[1])) {
        return { false, "lns_cuda_soft_max_mask_smaller_than_src0" };
    }
    return base;
}

void compute_soft_max(context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * mask = dst->src[1];
    soft_max_args args = {};
    args.src_shape = shape_of(src0);
    args.src       = ctx.upload(src0, role::src0);
    if (mask != nullptr) {
        args.mask       = ctx.upload(mask, role::src1);
        args.mask_shape = shape_of(mask);
    } else {
        args.mask_shape = { { 1, 1, 1, 1 } };
    }
    args.scale = op_param_f32(dst, 0);
    args.dst   = ctx.device_f32(role::dst, ggml_nelements(dst));
    args.work  = ctx.device_lns(role::work, ggml_nelements(src0));
    args.monte_seed = ctx.reserve_monte(ctx.kernels().soft_max_draws(src0->ne[0], ggml_nrows(src0)));
    LNS_CUDA_CHECK(ctx.kernels().soft_max(args, ctx.stream()));
    ctx.download(args.dst, dst);
}

// ---- ROPE (lns_rope) --------------------------------------------------------

op_support check_rope(const ggml_tensor * op) {
    const op_support base = check_f32_result(op, lns::check_rope(op));
    if (!base.supported) {
        return base;
    }
    const int n_dims = ggml_get_op_params_i32(op, 1);
    if (op->src[1] == nullptr || op->src[1]->type != GGML_TYPE_I32 || !ggml_is_contiguous(op->src[1]) ||
        op->src[1]->ne[0] < op->src[0]->ne[2]) {
        return { false, "lns_cuda_rope_requires_i32_positions" };
    }
    if (n_dims <= 0 || n_dims % 2 != 0 || n_dims > op->src[0]->ne[0]) {
        return { false, "lns_cuda_rope_requires_even_n_dims" };
    }
    return { true, "lns_cuda_rope" };
}

void compute_rope(context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int32_t * pos     = static_cast<const int32_t *>(dst->src[1]->data);
    const float *   freq_factors = dst->src[2] ? static_cast<const float *>(dst->src[2]->data) : nullptr;
    const int       n_dims  = ggml_get_op_params_i32(dst, 1);
    const int       mode    = ggml_get_op_params_i32(dst, 2);
    const float     theta_scale = powf(op_param_f32(dst, 5), -2.0f / n_dims);

    // (cos, sin) per position and rotated pair, computed in the same float
    // sequence as lns_rope so the device converts identical values.
    const int64_t half = n_dims / 2;
    std::vector<float> cos_sin((size_t) (src0->ne[2] * half * 2));
    for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
        float t = (float) pos[i2];
        for (int64_t p = 0; p < half; p++) {
            const float theta = freq_factors ? t / freq_factors[p] : t;
            cos_sin[(size_t) ((i2 * half + p) * 2 + 0)] = cosf(theta);
            cos_sin[(size_t) ((i2 * half + p) * 2 + 1)] = sinf(theta);
            t *= theta_scale;
        }
    }

    rope_args args = {};
    args.shape   = shape_of(src0);
    args.src     = ctx.upload(src0, role::src0);
    args.cos_sin = ctx.upload(cos_sin.data(), (int64_t) cos_sin.size(), role::src1);
    args.n_dims  = n_dims;
    args.neox    = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    args.dst     = ctx.device_f32(role::dst, ggml_nelements(dst));
    LNS_CUDA_CHECK(ctx.kernels().rope(args, ctx.stream()));
    ctx.download(args.dst, dst);
}

using lns::any_subop;
using kernel = lns::op_kernel<context>;

const kernel k_kernels[] = {
    { GGML_OP_MUL_MAT,  any_subop,          check_mul_mat,  compute_mul_mat                    },
    { GGML_OP_ADD,      any_subop,          check_binary,   compute_binary<binary_op::add>     },
    { GGML_OP_MUL,      any_subop,          check_binary,   compute_binary<binary_op::mul>     },
    { GGML_OP_DIV,      any_subop,          check_binary,   compute_binary<binary_op::div>     },
    { GGML_OP_SCALE,    any_subop,          check_scale,    compute_scale                      },
    { GGML_OP_RMS_NORM, any_subop,          check_rms_norm, compute_rms_norm                   },
    { GGML_OP_SOFT_MAX, any_subop,          check_soft_max, compute_soft_max                   },
    { GGML_OP_ROPE,     any_subop,          check_rope,     compute_rope                       },
    { GGML_OP_UNARY,    GGML_UNARY_OP_SILU, check_unary,    compute_unary<unary_op::silu>      },
    { GGML_OP_UNARY,    GGML_UNARY_OP_GELU, check_unary,    compute_unary<unary_op::gelu>      },
    { GGML_OP_UNARY,    GGML_UNARY_OP_RELU, check_unary,    compute_unary<unary_op::relu>      },
    { GGML_OP_UNARY,    GGML_UNARY_OP_TANH, check_unary,    compute_unary<unary_op::tanh>      },
    { GGML_OP_GLU,      GGML_GLU_OP_SWIGLU, lns::check_glu, compute_glu<glu_op::swiglu>        },
    { GGML_OP_GLU,      GGML_GLU_OP_GEGLU,  lns::check_glu, compute_glu<glu_op::geglu>         },
};

} // namespace

lns::op_table<context> op_table() {
    return lns::make_op_table(k_kernels);
}

} // namespace lns_cuda
