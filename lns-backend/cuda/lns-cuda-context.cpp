#include "lns-cuda-context.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#ifdef GGML_LNSCUDA_HAS_LNS16
#include "ggml-lns.h" // Monte state
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace lns_cuda {

// Device memory kept free for scratch when sizing the weight cache: the
// largest activations, softmax work, and a weight converted per use when the
// cache is full.
static constexpr size_t k_scratch_reserve = 768ull << 20;

// Host -> device chunk size when converting a matrix, in floats (64 MiB).
static constexpr int64_t k_convert_chunk = 16ll << 20;

static constexpr size_t MiB = 1u << 20;

void fail(cudaError_t err, const char * expr, const char * file, int line) {
    GGML_ABORT("ggml-lns-cuda: %s failed at %s:%d: %s", expr, file, line, cudaGetErrorString(err));
}

int selected_device() {
    static const int device = []() {
        const char * env = std::getenv("GGML_LNS_CUDA_DEVICE");
        const int wanted = env ? std::atoi(env) : 0;
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) {
            cudaGetLastError(); // no driver or no GPU: not an error, just no device
            return -1;
        }
        if (wanted < 0 || wanted >= count) {
            if (count > 0) {
                std::fprintf(stderr, "ggml-lns-cuda: GGML_LNS_CUDA_DEVICE=%d, but there are %d CUDA device(s); "
                                     "LNSCUDA disabled\n", wanted, count);
            }
            return -1;
        }
        return wanted;
    }();
    return device;
}

// ============================================================
// Memory
// ============================================================

device_buffer::device_buffer(device_buffer && other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr   = nullptr;
    other.bytes = 0;
}

device_buffer & device_buffer::operator=(device_buffer && other) noexcept {
    std::swap(ptr, other.ptr);
    std::swap(bytes, other.bytes);
    return *this;
}

device_buffer::~device_buffer() {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

bool device_buffer::try_reserve(size_t n) {
    if (n <= bytes) {
        return true;
    }
    if (ptr != nullptr) {
        LNS_CUDA_CHECK(cudaFree(ptr));
        ptr   = nullptr;
        bytes = 0;
    }
    const cudaError_t err = cudaMalloc(&ptr, n);
    if (err == cudaErrorMemoryAllocation) {
        cudaGetLastError(); // clear the sticky error so later calls do not report it
        ptr = nullptr;
        return false;
    }
    LNS_CUDA_CHECK(err);
    bytes = n;
    return true;
}

void * device_buffer::reserve(size_t n) {
    if (!try_reserve(n)) {
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        GGML_ABORT("ggml-lns-cuda: out of GPU memory allocating %zu MiB of scratch (%zu MiB free). "
                   "Leave more room for activations with a smaller GGML_LNS_CUDA_CACHE_MB, "
                   "or reduce the batch / context size (-ub, -c).",
                   n / MiB, free / MiB);
    }
    return ptr;
}

host_buffer::~host_buffer() {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

void * host_buffer::reserve(size_t n) {
    if (n > bytes) {
        if (ptr != nullptr) {
            LNS_CUDA_CHECK(cudaFreeHost(ptr));
            ptr = nullptr;
        }
        LNS_CUDA_CHECK(cudaMallocHost(&ptr, n));
        bytes = n;
    }
    return ptr;
}

weight_cache::key::key(const ggml_tensor * t) : data(t->data), type(t->type) {
    std::copy(t->ne, t->ne + GGML_MAX_DIMS, ne.begin());
    std::copy(t->nb, t->nb + GGML_MAX_DIMS, nb.begin());
}

bool weight_cache::key::operator<(const key & o) const {
    return std::tie(data, type, ne, nb) < std::tie(o.data, o.type, o.ne, o.nb);
}

const void * weight_cache::find(const ggml_tensor * t) const {
    const auto it = cache.find(key(t));
    return it == cache.end() ? nullptr : it->second.data();
}

void * weight_cache::insert(const ggml_tensor * t, size_t n) {
    if (used_bytes + n > budget) {
        return nullptr;
    }
    device_buffer buf;
    if (!buf.try_reserve(n)) {
        return nullptr;
    }
    void * ptr = buf.data();
    cache.emplace(key(t), std::move(buf));
    used_bytes += n;
    return ptr;
}

// ============================================================
// Host-side layout conversion
// ============================================================

// Rows [first, first + count) of the (i2, i3) slice of t, as contiguous F32.
// Quantized rows are always contiguous along ne[0]; F32 / F16 / BF16 rows may
// be strided (permuted views of the KV cache).
static void gather_rows_f32(const ggml_tensor * t, int64_t i2, int64_t i3, int64_t first, int64_t count,
                            float * out) {
    const int64_t ne0 = t->ne[0];
    const size_t  ts  = ggml_type_size(t->type);
    const auto *  traits = ggml_get_type_traits(t->type);
    for (int64_t r = 0; r < count; r++) {
        const char * row = (const char *) t->data + (first + r) * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        float * o = out + r * ne0;
        if (t->nb[0] == ts || ggml_is_quantized(t->type)) {
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(o, row, ne0 * sizeof(float));
            } else {
                traits->to_float(row, o, ne0);
            }
        } else if (t->type == GGML_TYPE_F32) {
            for (int64_t i = 0; i < ne0; i++) {
                std::memcpy(o + i, row + i * t->nb[0], sizeof(float));
            }
        } else {
            for (int64_t i = 0; i < ne0; i++) {
                traits->to_float(row + i * t->nb[0], o + i, 1);
            }
        }
    }
}

static void gather_f32(const ggml_tensor * t, float * out) {
    const int64_t rows = t->ne[1];
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            gather_rows_f32(t, i2, i3, 0, rows, out + ((i3 * t->ne[2] + i2) * rows) * t->ne[0]);
        }
    }
}

static void scatter_f32(const float * in, ggml_tensor * t) {
    const int64_t ne0 = t->ne[0];
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                const float * src = in + ((i3 * t->ne[2] + i2) * t->ne[1] + i1) * ne0;
                char * row = (char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                for (int64_t i = 0; i < ne0; i++) {
                    std::memcpy(row + i * t->nb[0], src + i, sizeof(float));
                }
            }
        }
    }
}

static bool is_weight(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// ============================================================
// context
// ============================================================

context::context(const kernel_set & kernels, const char * backend_name) : arith(kernels), name(backend_name) {}

context::~context() {
    if (device < 0) {
        return;
    }
    if (weights.entries() > 0) {
        std::fprintf(stderr, "ggml-lns-cuda: %s cached %zu weight tensors (%.1f MiB)\n",
                     name.c_str(), weights.entries(), (double) weights.used() / MiB);
    }
    cudaSetDevice(device);
    if (cuda_stream != nullptr) {
        cudaStreamDestroy(cuda_stream);
    }
}

void context::init_device() {
    device = selected_device();
    GGML_ASSERT(device >= 0 && "LNSCUDA backend used without a CUDA device");
    LNS_CUDA_CHECK(cudaSetDevice(device));
    LNS_CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
    LNS_CUDA_CHECK(arith.upload_tables());

    size_t free = 0, total = 0;
    LNS_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    size_t budget = free > k_scratch_reserve ? free - k_scratch_reserve : 0;
    if (const char * env = std::getenv("GGML_LNS_CUDA_CACHE_MB")) {
        budget = (size_t) std::max(0ll, std::atoll(env)) * MiB;
    }
    weights.set_budget(budget);

    cudaDeviceProp prop;
    LNS_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::fprintf(stderr, "ggml-lns-cuda: %s on GPU %d (%s, sm_%d%d, %zu MiB free), weight cache budget %zu MiB\n",
                 name.c_str(), device, prop.name, prop.major, prop.minor, free / MiB, budget / MiB);
}

uint16_t context::reserve_monte(uint64_t draws) {
    if (arith.monte_impl() == nullptr) {
        return 0;
    }
#ifdef GGML_LNSCUDA_HAS_LNS16
    uint16_t start = 0;
    if (ggml_backend_lns16_monte_reserve(draws, &start)) {
        return start;
    }
#endif
    (void) draws;
    GGML_ABORT("ggml-lns-cuda: %s needs an LNS16 backend built with GGML_LNS16_SUM=monte", name.c_str());
}

cudaStream_t context::stream() {
    if (cuda_stream == nullptr) {
        init_device();
    } else {
        // the current device is per host thread, and ggml may call from any
        LNS_CUDA_CHECK(cudaSetDevice(device));
    }
    return cuda_stream;
}

float * context::device_f32(role r, int64_t n) {
    return static_cast<float *>(device_scratch[(size_t) r].reserve(std::max<int64_t>(n, 1) * sizeof(float)));
}

void * context::device_lns(role r, int64_t n) {
    return device_scratch[(size_t) r].reserve(std::max<int64_t>(n, 1) * arith.value_size());
}

const float * context::upload(const ggml_tensor * t, role r) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t)) {
        return upload(static_cast<const float *>(t->data), n, r);
    }
    float * staged = static_cast<float *>(host_scratch[(size_t) r].reserve(std::max<int64_t>(n, 1) * sizeof(float)));
    gather_f32(t, staged);
    return upload(staged, n, r);
}

const float * context::upload(const float * host, int64_t n, role r) {
    cudaStream_t s = stream();
    float * dev = device_f32(r, n);
    LNS_CUDA_CHECK(cudaMemcpyAsync(dev, host, n * sizeof(float), cudaMemcpyHostToDevice, s));
    return dev;
}

void context::download(const float * src, ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    cudaStream_t s = stream();
    const int64_t n = ggml_nelements(t);
    if (ggml_is_contiguous(t)) {
        LNS_CUDA_CHECK(cudaMemcpyAsync(t->data, src, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        LNS_CUDA_CHECK(cudaStreamSynchronize(s));
        return;
    }
    float * staged = static_cast<float *>(host_scratch[(size_t) role::dst].reserve(std::max<int64_t>(n, 1) * sizeof(float)));
    LNS_CUDA_CHECK(cudaMemcpyAsync(staged, src, n * sizeof(float), cudaMemcpyDeviceToHost, s));
    LNS_CUDA_CHECK(cudaStreamSynchronize(s));
    scatter_f32(staged, t);
}

const void * context::lns_matrix(const ggml_tensor * t) {
    stream(); // the cache budget is only known once the device is up
    const int64_t n = ggml_nelements(t);
    if (is_weight(t)) {
        if (const void * cached = weights.find(t)) {
            return cached;
        }
        if (void * slot = weights.insert(t, n * arith.value_size())) {
            convert_matrix(t, slot);
            return slot;
        }
        if (!cache_full_reported) {
            cache_full_reported = true;
            std::fprintf(stderr, "ggml-lns-cuda: %s weight cache full (%zu MiB cached; budget or GPU memory "
                                 "exhausted); remaining weights are converted on every use, which is slow "
                                 "(use a smaller model, or adjust GGML_LNS_CUDA_CACHE_MB)\n",
                         name.c_str(), weights.used() / MiB);
        }
    }
    void * dst = device_lns(role::lns0, n);
    convert_matrix(t, dst);
    return dst;
}

// Converts slice by slice and in row chunks, so that even a large weight
// (e.g. a 128k-vocabulary output matrix) only needs a bounded F32 staging
// buffer on either side.
void context::convert_matrix(const ggml_tensor * t, void * dst) {
    cudaStream_t s = stream();
    const int64_t K = t->ne[0];
    const int64_t M = t->ne[1];
    const int64_t chunk_rows = std::max<int64_t>(1, std::min<int64_t>(M, k_convert_chunk / std::max<int64_t>(K, 1)));
    float * staged = static_cast<float *>(host_scratch[(size_t) role::src0].reserve(chunk_rows * K * sizeof(float)));
    float * dev    = device_f32(role::src0, chunk_rows * K);
    char *  out    = static_cast<char *>(dst);
    const size_t vs = arith.value_size();
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            char * slice = out + (size_t) ((i3 * t->ne[2] + i2) * K * M) * vs;
            for (int64_t m0 = 0; m0 < M; m0 += chunk_rows) {
                const int64_t rows = std::min(chunk_rows, M - m0);
                gather_rows_f32(t, i2, i3, m0, rows, staged);
                LNS_CUDA_CHECK(cudaMemcpyAsync(dev, staged, rows * K * sizeof(float), cudaMemcpyHostToDevice, s));
                LNS_CUDA_CHECK(arith.to_lns_transposed(dev, slice + (size_t) m0 * vs, rows, K, M, s));
                // the staging buffers are reused by the next chunk
                LNS_CUDA_CHECK(cudaStreamSynchronize(s));
            }
        }
    }
}

} // namespace lns_cuda
