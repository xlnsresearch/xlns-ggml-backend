#pragma once

#include "ggml.h"

// C-linkage conversion helpers (used by ggml.c type-traits hooks)
#ifdef __cplusplus
extern "C" {
#endif
void lns16_to_f32_row(const void * GGML_RESTRICT src, float * GGML_RESTRICT dst, int64_t n);
void f32_to_lns16_row(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t n);
#ifdef __cplusplus
}
#endif

// Name of the compile-time summation-variant selection
// ("base"|"monte"|"lpvip"|"sum32"), used for init logging, the device
// description, and the LNS_OP_AUDIT JSON records.
const char * lns16_sum_mode_name(void);

// "pure16" (xlns16monte.cpp) / "bylpvip" in a monte build, else nullptr
const char * lns16_monte_impl_name(void);

// Phase 1
void lns_mul_mat(struct ggml_tensor * dst);
void lns_mul_mat_id(struct ggml_tensor * dst);

// Phase 2 — element-wise
void lns_add(struct ggml_tensor * dst);
void lns_mul(struct ggml_tensor * dst);
void lns_div(struct ggml_tensor * dst);
void lns_scale(struct ggml_tensor * dst);
void lns_clamp(struct ggml_tensor * dst);

// Phase 2 — normalization / attention
void lns_soft_max(struct ggml_tensor * dst);
void lns_rms_norm(struct ggml_tensor * dst);
void lns_diag_mask_inf(struct ggml_tensor * dst);

// Phase 2 — activations
void lns_silu(struct ggml_tensor * dst);
void lns_swiglu(struct ggml_tensor * dst);
void lns_gelu(struct ggml_tensor * dst);
void lns_geglu(struct ggml_tensor * dst);
void lns_relu(struct ggml_tensor * dst);
void lns_tanh(struct ggml_tensor * dst);

// Phase 2 — data movement
void lns_get_rows(struct ggml_tensor * dst);
void lns_cpy(struct ggml_tensor * dst);
void lns_argsort(struct ggml_tensor * dst);

// Phase 2 — positional encoding
void lns_rope(struct ggml_tensor * dst);
