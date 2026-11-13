#pragma once

// The GPU side of the LNSCUDA backend, behind one interface.
//
// kernel_set is implemented once, as templates over an LNS format
// (lns-cuda-kernels.cuh), and instantiated per format in its own translation
// unit (lns-cuda-xlns16.cu, lns-cuda-xlns32.cu), because xlnscpp/xlnscuda
// can only be included once per translation unit. The host side
// (lns-cuda-context, lns-cuda-ops) sees only this header and device pointers:
// it is compiled by the host compiler and shared by both formats.
//
// Conventions:
//   * Kernels read F32 and write F32 at op boundaries and do all arithmetic in
//     LNS, converting with the same tables and in the same order as the CPU
//     backends. That is what makes the results bit-identical to them.
//   * Reductions (mul_mat's dot products, softmax's normalizer, rms_norm's
//     sum of squares) are sequential in element order, like the CPU's
//     xlns16_vec_dot / xlns16_sum: LNS addition is not associative, so a
//     parallel tree reduction would give different (if not worse) numbers.
//     How they accumulate follows the xlns16 summation mode (GGML_LNS16_SUM:
//     base, monte, lpvip or sum32), as in the CPU backend.
//   * Monte reductions take random draws. Each call gets the LFSR state where
//     its first reduction starts (monte_seed); reduction i starts i * draws
//     later, i.e. in the CPU's sequential order.
//   * Tensors are contiguous arrays in ggml element order (ne[0] fastest).
//   * Every call launches asynchronously on the given stream.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace lns_cuda {

enum class binary_op { add, mul, div };
enum class unary_op  { silu, gelu, relu, tanh };
enum class glu_op    { swiglu, geglu };

struct shape4 {
    int64_t ne[4];

    int64_t nelements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

// dst = W^T x for every activation slice, sequential over k.
//   w:   LNS, K x M per weight slice, k-major (element (k, m) at k*M + m),
//        ne02 * ne03 slices
//   x:   LNS, N x K per activation slice, ne12 * ne13 slices
//   dst: F32, N x M per activation slice
// Activation slice (i12, i13) uses weight slice (i12 / (ne12/ne02), i13 / (ne13/ne03)).
struct mul_mat_args {
    const void * w;
    const void * x;
    float *      dst;
    int64_t      K, M, N;
    int64_t      ne02, ne03, ne12, ne13;
    uint16_t     monte_seed;
};

// dst = softmax(src * scale + mask) over rows of src.ne[0]. mask (optional)
// broadcasts over dims 2 and 3. work holds src.nelements() LNS values.
struct soft_max_args {
    const float * src;
    shape4        src_shape;
    const float * mask;
    shape4        mask_shape;
    float         scale;
    float *       dst;
    void *        work;
    uint16_t      monte_seed;
};

// Rotary embedding of src [ne0, ne1 (heads), ne2 (positions), ne3]. cos_sin
// holds, per position i2, n_dims/2 (cos, sin) float pairs computed on the
// host exactly as the CPU kernel computes them.
struct rope_args {
    const float * src;
    shape4        shape;
    const float * cos_sin;
    int           n_dims;
    bool          neox;
    float *       dst;
};

class kernel_set {
public:
    virtual ~kernel_set() = default;

    virtual const char * format_name() const = 0; // "xlns16" / "xlns32"
    virtual size_t       value_size()  const = 0; // bytes per LNS value

    // "base" | "monte" | "lpvip" for xlns16, nullptr for xlns32
    virtual const char * sum_mode()   const = 0;
    // "pure16" | "bylpvip" for monte, else nullptr
    virtual const char * monte_impl() const = 0;

    // Random draws these calls consume (0 unless monte).
    virtual uint64_t mul_mat_draws(const mul_mat_args & args) const = 0;
    virtual uint64_t soft_max_draws(int64_t ncols, int64_t nrows) const = 0;
    virtual uint64_t rms_norm_draws(int64_t ncols, int64_t nrows) const = 0;

    // Uploads the arithmetic lookup tables to the current device. Must run
    // once per device before any other call.
    virtual cudaError_t upload_tables() const = 0;

    // dst[i] = lns(src[i])
    virtual cudaError_t to_lns(const float * src, void * dst, int64_t n, cudaStream_t stream) const = 0;

    // Converts a row-major rows x cols F32 block into the k-major weight
    // layout: dst[c * dst_ld + r] = lns(src[r * cols + c]).
    virtual cudaError_t to_lns_transposed(const float * src, void * dst, int64_t rows, int64_t cols,
                                          int64_t dst_ld, cudaStream_t stream) const = 0;

    virtual cudaError_t mul_mat(const mul_mat_args & args, cudaStream_t stream) const = 0;

    // dst = a op b, with b repeated to the shape of a (ggml_can_repeat(b, a)).
    virtual cudaError_t binary(binary_op op, const float * a, shape4 a_shape, const float * b, shape4 b_shape,
                               float * dst, cudaStream_t stream) const = 0;

    // dst = src * s, plus b when b != 0 (ggml_scale_bias).
    virtual cudaError_t scale(const float * src, float * dst, int64_t n, float s, float b,
                              cudaStream_t stream) const = 0;

    virtual cudaError_t unary(unary_op op, const float * src, float * dst, int64_t n,
                              cudaStream_t stream) const = 0;

    // dst[r][c] = act(gate[r * gate_stride + c]) * up[r * up_stride + c]
    virtual cudaError_t glu(glu_op op, const float * gate, const float * up, int64_t gate_stride,
                            int64_t up_stride, float * dst, int64_t ncols, int64_t nrows,
                            cudaStream_t stream) const = 0;

    // dst = src / sqrt(mean(src^2) + eps) per row of ncols.
    virtual cudaError_t rms_norm(const float * src, float * dst, int64_t ncols, int64_t nrows, float eps,
                                 uint16_t monte_seed, cudaStream_t stream) const = 0;

    virtual cudaError_t soft_max(const soft_max_args & args, cudaStream_t stream) const = 0;

    virtual cudaError_t rope(const rope_args & args, cudaStream_t stream) const = 0;
};

const kernel_set & xlns16_kernels();
const kernel_set & xlns32_kernels();

} // namespace lns_cuda
