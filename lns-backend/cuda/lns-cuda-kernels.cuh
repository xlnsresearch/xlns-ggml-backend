#pragma once

// Templated CUDA kernels of the LNSCUDA backend, written once for every LNS
// format. Include after xlnscpp + xlnscuda and after defining an arithmetic
// policy A with:
//
//   using value = <LNS storage type>;
//   static constexpr value zero, neg_inf;
//   __device__ static value from_f32(float), to_f32(value) -> float
//   __device__ static value add, sub, mul, div (value, value)
//   __device__ static bool  gt(value, value), is_zero(value)
//   __device__ static value exp, sqrt, recip, silu, gelu, relu, tanh (value)
//   __device__ static value mask_from_f32(float)
//   static constexpr const char * sum_mode, * monte_impl   (may be nullptr)
//   struct reduction {          how mul_mat / softmax / rms_norm accumulate
//       using acc, state;       accumulator (wider for lpvip), Monte LFSR
//       static constexpr bool stochastic;
//       __device__ static state start(unsigned seed, uint64_t index, uint64_t draws);
//       __device__ static acc   from(value);
//       __device__ static acc   add(acc, value, state &);
//       __device__ static value finish(acc);
//   };
//
// Every kernel mirrors one CPU kernel in lns-backend/src/lns-ops.cpp (and its
// xlns32 twin), operation for operation, so results are bit-identical; the
// comment on each kernel names the code it mirrors. Parallelism only ever
// spans independent outputs: anything that accumulates is sequential. Other
// additions (ADD, mask bias, RoPE, epsilon) always use A::add.

#include "lns-cuda-kernels.h"

#include <cuda_runtime.h>

namespace lns_cuda {
namespace detail {

constexpr int k_block_size     = 256;
constexpr int k_mul_mat_block  = 64;
// Output columns per mul_mat thread. Each column is its own sequential
// accumulation; interleaving several hides the latency of the dependent
// table lookup inside every LNS addition.
constexpr int k_mul_mat_cols   = 4;
// k steps whose operands are loaded ahead of the accumulation (see k_mul_mat).
constexpr int k_mul_mat_ahead  = 8;

inline unsigned blocks_for(int64_t n, int block = k_block_size) {
    int64_t blocks = (n + block - 1) / block;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535; // the kernels below are grid-stride loops
    return (unsigned) blocks;
}

#define LNS_CUDA_GRID_STRIDE(i, n) \
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (int64_t) blockDim.x * gridDim.x)

// ---- conversions -----------------------------------------------------------

template <class A>
__global__ void k_to_lns(const float * src, typename A::value * dst, int64_t n) {
    LNS_CUDA_GRID_STRIDE(i, n) {
        dst[i] = A::from_f32(src[i]);
    }
}

// Reads are coalesced along cols; the scattered writes are acceptable because
// weights are converted once and then cached.
template <class A>
__global__ void k_to_lns_transposed(const float * src, typename A::value * dst, int64_t rows, int64_t cols,
                                    int64_t dst_ld) {
    LNS_CUDA_GRID_STRIDE(i, rows * cols) {
        const int64_t r = i / cols;
        const int64_t c = i - r * cols;
        dst[c * dst_ld + r] = A::from_f32(src[i]);
    }
}

// ---- MUL_MAT (lns_mul_mat: one vec_dot per output, LNS16_VEC_DOT order) ---

template <class A>
__global__ void k_mul_mat(const typename A::value * w, const typename A::value * x, float * dst,
                          int64_t K, int64_t M, int64_t N, int64_t ne02, int64_t ne12, int64_t r2, int64_t r3,
                          unsigned monte_seed) {
    using value = typename A::value;
    using R     = typename A::reduction;
    constexpr int NB = k_mul_mat_cols;

    const int64_t m = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) {
        return;
    }
    const int64_t n0  = (int64_t) blockIdx.y * NB;
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i13 = blockIdx.z / ne12;
    const int     nb  = (int) (N - n0 < NB ? N - n0 : NB);

    const value * ws = w + ((i13 / r3) * ne02 + i12 / r2) * K * M + m;
    const value * xs = x + (int64_t) blockIdx.z * N * K + n0 * K;
    float *       ds = dst + (int64_t) blockIdx.z * N * M + n0 * M + m;

    if (K == 0) {
        for (int j = 0; j < nb; j++) {
            ds[j * M] = A::to_f32(A::zero);
        }
        return;
    }
    // index of this output in the CPU's loop order (i13, i12, m, n)
    const uint64_t out0 = ((uint64_t) blockIdx.z * (uint64_t) M + (uint64_t) m) * (uint64_t) N + (uint64_t) n0;
    typename R::acc   acc[NB];
    typename R::state st[NB];
    const value w0 = ws[0];
#pragma unroll
    for (int j = 0; j < NB; j++) {
        if (j < nb) {
            acc[j] = R::from(A::mul(w0, xs[j * K]));
            st[j]  = R::start(monte_seed, out0 + j, (uint64_t) K - 1);
        }
    }
    // Operands for the next k_mul_mat_ahead steps are loaded before any of
    // them is accumulated, so the global-memory latency overlaps instead of
    // stalling every step of the dependent addition chain. The accumulation
    // order is unchanged.
    constexpr int U = k_mul_mat_ahead;
    int64_t k = 1;
    for (; k + U <= K; k += U) {
        value wv[U];
        value xv[NB][U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            wv[u] = ws[(k + u) * M];
        }
#pragma unroll
        for (int j = 0; j < NB; j++) {
#pragma unroll
            for (int u = 0; u < U; u++) {
                xv[j][u] = j < nb ? xs[j * K + k + u] : A::zero;
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
#pragma unroll
            for (int j = 0; j < NB; j++) {
                if (j < nb) acc[j] = R::add(acc[j], A::mul(wv[u], xv[j][u]), st[j]);
            }
        }
    }
    for (; k < K; k++) {
        const value wk = ws[k * M];
#pragma unroll
        for (int j = 0; j < NB; j++) {
            if (j < nb) acc[j] = R::add(acc[j], A::mul(wk, xs[j * K + k]), st[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < NB; j++) {
        if (j < nb) ds[j * M] = A::to_f32(R::finish(acc[j]));
    }
}

// ---- elementwise (lns_add / lns_mul / lns_div / lns_scale / activations) ---

template <class A, binary_op OP>
__global__ void k_binary(const float * a, shape4 sa, const float * b, shape4 sb, float * dst) {
    LNS_CUDA_GRID_STRIDE(i, sa.ne[0] * sa.ne[1] * sa.ne[2] * sa.ne[3]) {
        int64_t t = i;
        const int64_t i0 = t % sa.ne[0]; t /= sa.ne[0];
        const int64_t i1 = t % sa.ne[1]; t /= sa.ne[1];
        const int64_t i2 = t % sa.ne[2];
        const int64_t i3 = t / sa.ne[2];
        const int64_t j = (((i3 % sb.ne[3]) * sb.ne[2] + i2 % sb.ne[2]) * sb.ne[1] + i1 % sb.ne[1]) * sb.ne[0]
                        + i0 % sb.ne[0];
        const typename A::value x = A::from_f32(a[i]);
        const typename A::value y = A::from_f32(b[j]);
        dst[i] = A::to_f32(OP == binary_op::add ? A::add(x, y) :
                           OP == binary_op::mul ? A::mul(x, y) : A::div(x, y));
    }
}

template <class A>
__global__ void k_scale(const float * src, float * dst, int64_t n, float s, float b) {
    const typename A::value s_lns = A::from_f32(s);
    const typename A::value b_lns = A::from_f32(b);
    LNS_CUDA_GRID_STRIDE(i, n) {
        typename A::value v = A::mul(A::from_f32(src[i]), s_lns);
        if (b != 0.0f) {
            v = A::add(v, b_lns);
        }
        dst[i] = A::to_f32(v);
    }
}

template <class A, unary_op OP>
__device__ __forceinline__ typename A::value apply_unary(typename A::value v) {
    switch (OP) {
        case unary_op::silu: return A::silu(v);
        case unary_op::gelu: return A::gelu(v);
        case unary_op::relu: return A::relu(v);
        default:             return A::tanh(v);
    }
}

template <class A, unary_op OP>
__global__ void k_unary(const float * src, float * dst, int64_t n) {
    LNS_CUDA_GRID_STRIDE(i, n) {
        dst[i] = A::to_f32(apply_unary<A, OP>(A::from_f32(src[i])));
    }
}

// lns_swiglu / lns_geglu: act(gate) * up
template <class A, glu_op OP>
__global__ void k_glu(const float * gate, const float * up, int64_t gate_stride, int64_t up_stride,
                      float * dst, int64_t ncols, int64_t nrows) {
    LNS_CUDA_GRID_STRIDE(i, ncols * nrows) {
        const int64_t r = i / ncols;
        const int64_t c = i - r * ncols;
        const typename A::value g = A::from_f32(gate[r * gate_stride + c]);
        const typename A::value u = A::from_f32(up[r * up_stride + c]);
        const typename A::value act = OP == glu_op::swiglu ? A::silu(g) : A::gelu(g);
        dst[i] = A::to_f32(A::mul(act, u));
    }
}

// ---- RMS_NORM (xlns16_rms_norm), one block per row -------------------------

template <class A>
__global__ void k_rms_norm(const float * src, float * dst, int64_t ncols, float eps, unsigned monte_seed) {
    using value = typename A::value;
    using R     = typename A::reduction;
    __shared__ value s_inv_rms;

    const float * x = src + (int64_t) blockIdx.x * ncols;
    float *       y = dst + (int64_t) blockIdx.x * ncols;
    if (ncols == 0) {
        return;
    }
    if (threadIdx.x == 0) {
        typename R::state st = R::start(monte_seed, blockIdx.x, (uint64_t) ncols);
        typename R::acc sum_sq = R::from(A::zero);
        for (int64_t i = 0; i < ncols; i++) {
            const value v = A::from_f32(x[i]);
            sum_sq = R::add(sum_sq, A::mul(v, v), st);
        }
        const value mean = A::mul(R::finish(sum_sq), A::from_f32(1.0f / (float) ncols));
        s_inv_rms = A::recip(A::sqrt(A::add(mean, A::from_f32(eps))));
    }
    __syncthreads();
    const value inv_rms = s_inv_rms;
    for (int64_t i = threadIdx.x; i < ncols; i += blockDim.x) {
        y[i] = A::to_f32(A::mul(A::from_f32(x[i]), inv_rms));
    }
}

// ---- SOFT_MAX (xlns16_softmax / xlns16_softmax_masked), one block per row --

template <class A>
__global__ void k_soft_max(const float * src, shape4 s, const float * mask, shape4 ms, float scale,
                           float * dst, typename A::value * work, unsigned monte_seed) {
    using value = typename A::value;
    using R     = typename A::reduction;
    __shared__ value s_max[k_block_size];
    __shared__ value s_total;

    const int64_t n   = s.ne[0];
    const int64_t row = blockIdx.x;
    if (n == 0) {
        return;
    }
    const int64_t i1 = row % s.ne[1];
    const int64_t i2 = (row / s.ne[1]) % s.ne[2];
    const int64_t i3 = row / (s.ne[1] * s.ne[2]);

    const float * x = src + row * n;
    value *       c = work + row * n;
    const float * m = mask == nullptr ? nullptr :
        mask + (((i3 % ms.ne[3]) * ms.ne[2] + i2 % ms.ne[2]) * ms.ne[1] + i1) * ms.ne[0];
    const value scale_lns = A::from_f32(scale);

    // scale, mask, and the row maximum. The maximum is exact in any order:
    // canonical LNS comparison is a total order on distinct bit patterns.
    value local_max = A::neg_inf;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        value v = A::from_f32(x[i]);
        if (v != A::neg_inf) {
            v = A::mul(v, scale_lns);
            if (m != nullptr) {
                const value mk = A::mask_from_f32(m[i]);
                if (mk == A::neg_inf) {
                    v = A::neg_inf;
                } else if (!A::is_zero(mk)) {
                    v = A::add(v, mk);
                }
            }
        }
        c[i] = v;
        if (A::gt(v, local_max)) local_max = v;
    }
    s_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride && A::gt(s_max[threadIdx.x + stride], s_max[threadIdx.x])) {
            s_max[threadIdx.x] = s_max[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const value maxval = s_max[0];

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        c[i] = A::exp(A::sub(c[i], maxval));
    }
    __syncthreads();

    // xlns16_sum: starts from the first element, adds the rest in order
    if (threadIdx.x == 0) {
        typename R::state st = R::start(monte_seed, (uint64_t) row, (uint64_t) n - 1);
        typename R::acc total = R::from(c[0]);
        for (int64_t i = 1; i < n; i++) {
            total = R::add(total, c[i], st);
        }
        s_total = R::finish(total);
    }
    __syncthreads();
    const value total = s_total;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[row * n + i] = A::to_f32(A::div(c[i], total));
    }
}

// ---- ROPE (lns_rope) --------------------------------------------------------

template <class A>
__global__ void k_rope(const float * src, shape4 s, const float * cos_sin, int n_dims, bool neox, float * dst) {
    const int64_t half   = n_dims / 2;
    const int64_t pass   = s.ne[0] - n_dims;      // dims past n_dims are copied through LNS
    const int64_t per    = half + pass;
    const int64_t nrows  = s.ne[1] * s.ne[2] * s.ne[3];
    LNS_CUDA_GRID_STRIDE(i, nrows * per) {
        const int64_t row = i / per;
        const int64_t j   = i - row * per;
        const int64_t i2  = (row / s.ne[1]) % s.ne[2];
        const float * x   = src + row * s.ne[0];
        float *       y   = dst + row * s.ne[0];
        if (j < half) {
            const int64_t i0 = neox ? j : 2 * j;
            const int64_t i1 = neox ? j + half : 2 * j + 1;
            const typename A::value c  = A::from_f32(cos_sin[(i2 * half + j) * 2 + 0]);
            const typename A::value sn = A::from_f32(cos_sin[(i2 * half + j) * 2 + 1]);
            const typename A::value x0 = A::from_f32(x[i0]);
            const typename A::value x1 = A::from_f32(x[i1]);
            y[i0] = A::to_f32(A::sub(A::mul(x0, c), A::mul(x1, sn)));
            y[i1] = A::to_f32(A::add(A::mul(x0, sn), A::mul(x1, c)));
        } else {
            const int64_t i0 = n_dims + (j - half);
            y[i0] = A::to_f32(A::from_f32(x[i0]));
        }
    }
}

#undef LNS_CUDA_GRID_STRIDE

// ---- kernel_set ------------------------------------------------------------

template <class A>
class kernel_set_impl final : public kernel_set {
public:
    using value = typename A::value;

    const char * format_name() const override { return A::name; }
    size_t       value_size()  const override { return sizeof(value); }
    const char * sum_mode()    const override { return A::sum_mode; }
    const char * monte_impl()  const override { return A::monte_impl; }

    uint64_t mul_mat_draws(const mul_mat_args & a) const override {
        if (!A::reduction::stochastic || a.K == 0) {
            return 0;
        }
        return (uint64_t) a.ne12 * (uint64_t) a.ne13 * (uint64_t) a.M * (uint64_t) a.N * (uint64_t) (a.K - 1);
    }
    uint64_t soft_max_draws(int64_t ncols, int64_t nrows) const override {
        return A::reduction::stochastic && ncols > 0 ? (uint64_t) nrows * (uint64_t) (ncols - 1) : 0;
    }
    uint64_t rms_norm_draws(int64_t ncols, int64_t nrows) const override {
        return A::reduction::stochastic ? (uint64_t) nrows * (uint64_t) ncols : 0;
    }

    cudaError_t upload_tables() const override { return A::upload_tables(); }

    cudaError_t to_lns(const float * src, void * dst, int64_t n, cudaStream_t stream) const override {
        k_to_lns<A><<<blocks_for(n), k_block_size, 0, stream>>>(src, static_cast<value *>(dst), n);
        return cudaGetLastError();
    }

    cudaError_t to_lns_transposed(const float * src, void * dst, int64_t rows, int64_t cols, int64_t dst_ld,
                                  cudaStream_t stream) const override {
        k_to_lns_transposed<A><<<blocks_for(rows * cols), k_block_size, 0, stream>>>(
            src, static_cast<value *>(dst), rows, cols, dst_ld);
        return cudaGetLastError();
    }

    cudaError_t mul_mat(const mul_mat_args & a, cudaStream_t stream) const override {
        if (a.M == 0 || a.N == 0 || a.ne12 * a.ne13 == 0) {
            return cudaSuccess;
        }
        const dim3 grid((unsigned) ((a.M + k_mul_mat_block - 1) / k_mul_mat_block),
                        (unsigned) ((a.N + k_mul_mat_cols - 1) / k_mul_mat_cols),
                        (unsigned) (a.ne12 * a.ne13));
        k_mul_mat<A><<<grid, k_mul_mat_block, 0, stream>>>(
            static_cast<const value *>(a.w), static_cast<const value *>(a.x), a.dst,
            a.K, a.M, a.N, a.ne02, a.ne12, a.ne12 / a.ne02, a.ne13 / a.ne03, a.monte_seed);
        return cudaGetLastError();
    }

    cudaError_t binary(binary_op op, const float * a, shape4 sa, const float * b, shape4 sb, float * dst,
                       cudaStream_t stream) const override {
        const unsigned blocks = blocks_for(sa.nelements());
        switch (op) {
            case binary_op::add: k_binary<A, binary_op::add><<<blocks, k_block_size, 0, stream>>>(a, sa, b, sb, dst); break;
            case binary_op::mul: k_binary<A, binary_op::mul><<<blocks, k_block_size, 0, stream>>>(a, sa, b, sb, dst); break;
            case binary_op::div: k_binary<A, binary_op::div><<<blocks, k_block_size, 0, stream>>>(a, sa, b, sb, dst); break;
        }
        return cudaGetLastError();
    }

    cudaError_t scale(const float * src, float * dst, int64_t n, float s, float b,
                      cudaStream_t stream) const override {
        k_scale<A><<<blocks_for(n), k_block_size, 0, stream>>>(src, dst, n, s, b);
        return cudaGetLastError();
    }

    cudaError_t unary(unary_op op, const float * src, float * dst, int64_t n, cudaStream_t stream) const override {
        const unsigned blocks = blocks_for(n);
        switch (op) {
            case unary_op::silu: k_unary<A, unary_op::silu><<<blocks, k_block_size, 0, stream>>>(src, dst, n); break;
            case unary_op::gelu: k_unary<A, unary_op::gelu><<<blocks, k_block_size, 0, stream>>>(src, dst, n); break;
            case unary_op::relu: k_unary<A, unary_op::relu><<<blocks, k_block_size, 0, stream>>>(src, dst, n); break;
            case unary_op::tanh: k_unary<A, unary_op::tanh><<<blocks, k_block_size, 0, stream>>>(src, dst, n); break;
        }
        return cudaGetLastError();
    }

    cudaError_t glu(glu_op op, const float * gate, const float * up, int64_t gate_stride, int64_t up_stride,
                    float * dst, int64_t ncols, int64_t nrows, cudaStream_t stream) const override {
        const unsigned blocks = blocks_for(ncols * nrows);
        switch (op) {
            case glu_op::swiglu:
                k_glu<A, glu_op::swiglu><<<blocks, k_block_size, 0, stream>>>(gate, up, gate_stride, up_stride, dst, ncols, nrows);
                break;
            case glu_op::geglu:
                k_glu<A, glu_op::geglu><<<blocks, k_block_size, 0, stream>>>(gate, up, gate_stride, up_stride, dst, ncols, nrows);
                break;
        }
        return cudaGetLastError();
    }

    cudaError_t rms_norm(const float * src, float * dst, int64_t ncols, int64_t nrows, float eps,
                         uint16_t monte_seed, cudaStream_t stream) const override {
        if (nrows == 0) {
            return cudaSuccess;
        }
        k_rms_norm<A><<<(unsigned) nrows, k_block_size, 0, stream>>>(src, dst, ncols, eps, monte_seed);
        return cudaGetLastError();
    }

    cudaError_t soft_max(const soft_max_args & a, cudaStream_t stream) const override {
        const int64_t nrows = a.src_shape.ne[1] * a.src_shape.ne[2] * a.src_shape.ne[3];
        if (nrows == 0) {
            return cudaSuccess;
        }
        k_soft_max<A><<<(unsigned) nrows, k_block_size, 0, stream>>>(
            a.src, a.src_shape, a.mask, a.mask_shape, a.scale, a.dst, static_cast<value *>(a.work), a.monte_seed);
        return cudaGetLastError();
    }

    cudaError_t rope(const rope_args & a, cudaStream_t stream) const override {
        const int64_t per = a.n_dims / 2 + (a.shape.ne[0] - a.n_dims);
        k_rope<A><<<blocks_for(a.shape.ne[1] * a.shape.ne[2] * a.shape.ne[3] * per), k_block_size, 0, stream>>>(
            a.src, a.shape, a.cos_sin, a.n_dims, a.neox, a.dst);
        return cudaGetLastError();
    }
};

} // namespace detail
} // namespace lns_cuda
