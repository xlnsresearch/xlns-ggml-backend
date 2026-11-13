#pragma once

// Host-side state of one LNSCUDA backend instance.
//
// The backend computes on ggml host tensors (like the CPU LNS backends), so
// every op uploads its inputs, runs a kernel_set kernel and downloads the
// result. This file owns everything around that: the CUDA device and stream,
// scratch memory, the cache of weights already converted to LNS, and the
// conversions between ggml tensor layouts and the contiguous F32 arrays the
// kernels take. Ops (lns-cuda-ops.cpp) only describe what to move and run.

#include "ggml.h"

#include "lns-cuda-kernels.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#define LNS_CUDA_CHECK(expr) ::lns_cuda::check((expr), #expr, __FILE__, __LINE__)

namespace lns_cuda {

[[noreturn]] void fail(cudaError_t err, const char * expr, const char * file, int line);

inline void check(cudaError_t err, const char * expr, const char * file, int line) {
    if (err != cudaSuccess) {
        fail(err, expr, file, line);
    }
}

// GPU index from GGML_LNS_CUDA_DEVICE (default 0), or -1 when that GPU does
// not exist or no CUDA driver is usable.
int selected_device();

// Device memory that only grows. reserve() keeps no contents.
class device_buffer {
public:
    device_buffer() = default;
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
    device_buffer(device_buffer && other) noexcept;
    device_buffer & operator=(device_buffer && other) noexcept;
    ~device_buffer();

    // Grows to at least n bytes; aborts with advice when the GPU is out of memory.
    void * reserve(size_t n);
    // Same, but reports an allocation failure by returning false.
    bool   try_reserve(size_t n);
    void * data() const { return ptr; }
    size_t size() const { return bytes; }

private:
    void * ptr   = nullptr;
    size_t bytes = 0;
};

// Pinned host memory that only grows: the staging side of uploads/downloads.
class host_buffer {
public:
    host_buffer() = default;
    host_buffer(const host_buffer &) = delete;
    host_buffer & operator=(const host_buffer &) = delete;
    ~host_buffer();

    void * reserve(size_t bytes);

private:
    void * ptr   = nullptr;
    size_t bytes = 0;
};

// Weights converted to LNS once and kept on the device. A weight is
// identified by its data pointer, type and layout; llama.cpp never writes
// weights in place, and the cache dies with the backend instance, before the
// model's buffers can be freed.
class weight_cache {
public:
    void set_budget(size_t bytes) { budget = bytes; }

    // nullptr when t is not cached
    const void * find(const ggml_tensor * t) const;

    // Device memory for a new entry, or nullptr when it would exceed the
    // budget or the GPU has no memory left.
    void * insert(const ggml_tensor * t, size_t bytes);

    size_t entries() const { return cache.size(); }
    size_t used()    const { return used_bytes; }

private:
    struct key {
        const void *                      data;
        ggml_type                         type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        std::array<size_t,  GGML_MAX_DIMS> nb;

        explicit key(const ggml_tensor * t);
        bool operator<(const key & other) const;
    };

    std::map<key, device_buffer> cache;
    size_t budget     = 0;
    size_t used_bytes = 0;
};

// Roles of the scratch buffers; each op uses each role at most once, so a
// role's buffer can be reused by the next op.
enum class role { src0, src1, dst, lns0, lns1, work, count };

class context {
public:
    context(const kernel_set & kernels, const char * name);
    ~context();

    context(const context &) = delete;
    context & operator=(const context &) = delete;

    const kernel_set & kernels() const { return arith; }

    // Monte mode: reserves the next `draws` draws of the process-wide Monte
    // sequence (owned by LNS16, ggml-lns.h) and returns their start state.
    // Ops reserve in graph order, so GPU and CPU-fallback ops draw exactly as
    // an all-CPU run would. Returns 0 in the other modes.
    uint16_t reserve_monte(uint64_t draws);

    // The stream every op runs on. The first call initializes the device:
    // nothing touches the GPU until an op actually runs, so a selected-out
    // LNSCUDA device costs no VRAM.
    cudaStream_t stream();

    float * device_f32(role r, int64_t n);
    void *  device_lns(role r, int64_t n);

    // t as a contiguous F32 array (ggml element order) on the device,
    // dequantizing on the host when t is not F32.
    const float * upload(const ggml_tensor * t, role r);

    // Copies n floats from the host to the device buffer of role r.
    const float * upload(const float * host, int64_t n, role r);

    // Writes a contiguous F32 device array into t (any strides) and waits
    // until it is there: the next op, on any backend, may read t.
    void download(const float * src, ggml_tensor * t);

    // A matrix operand of mul_mat (ne[0] = K, ne[1] = M) as LNS in the
    // k-major layout of mul_mat_args::w. Weights come from the cache, which
    // converts them on first use; other tensors are converted into scratch.
    const void * lns_matrix(const ggml_tensor * t);

private:
    void init_device();
    void convert_matrix(const ggml_tensor * t, void * dst);

    const kernel_set & arith;
    std::string        name;
    int                device      = -1;
    cudaStream_t       cuda_stream = nullptr;
    bool               cache_full_reported = false;

    weight_cache                                         weights;
    std::array<device_buffer, (size_t) role::count>      device_scratch;
    std::array<host_buffer,   (size_t) role::count>      host_scratch;
};

} // namespace lns_cuda
