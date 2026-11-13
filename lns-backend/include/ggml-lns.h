#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t    ggml_backend_lns_init(void);
GGML_BACKEND_API bool              ggml_backend_is_lns(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_lns_reg(void);

// Monte summation state (GGML_LNS16_SUM=monte): one LFSR per process (seed 7),
// one draw per Monte addition, shared by LNS16 and LNSCUDA16. All return
// false in other modes; set_state rejects 0. Also reachable through
// ggml_backend_reg_get_proc_address on the LNS16 registry.
GGML_BACKEND_API bool ggml_backend_lns16_monte_get_state(uint16_t * state);
GGML_BACKEND_API bool ggml_backend_lns16_monte_set_state(uint16_t state);
// *start = current state, then advance it by `draws` steps
GGML_BACKEND_API bool ggml_backend_lns16_monte_reserve(uint64_t draws, uint16_t * start);

#ifdef __cplusplus
}
#endif
