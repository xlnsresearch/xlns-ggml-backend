#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// LNS arithmetic on a CUDA GPU, through xlnscuda. One implementation, two
// devices:
//   LNSCUDA16 — xlns16; results are bit-identical to the LNS16 CPU backend
//               built with the same GGML_LNS16_SUM (base, monte, lpvip or sum32)
//   LNSCUDA32 — xlns32; results match the LNS32 CPU backend
// Like the CPU LNS backends these are ACCEL devices on host buffers; select one
// with GGML_LNS_BACKEND=cuda16 or cuda32. The GPU is chosen with
// GGML_LNS_CUDA_DEVICE (default 0).

GGML_BACKEND_API ggml_backend_t     ggml_backend_lns_cuda16_init(void);
GGML_BACKEND_API ggml_backend_t     ggml_backend_lns_cuda32_init(void);
GGML_BACKEND_API bool               ggml_backend_is_lns_cuda(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_lns_cuda16_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_lns_cuda32_reg(void);

#ifdef __cplusplus
}
#endif
