// LNS16 backend: every supported op in xlns16 arithmetic on the CPU.
//
// The backend/device/registry plumbing is shared with the other LNS backends
// (lns-backend-framework.h); this file only says what LNS16 is and which
// kernel (lns-ops.cpp) computes which op.

#include "ggml-lns.h"

#include "lns-backend-framework.h"
#include "lns-ops.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

using lns::any_subop;
using lns::no_context;
using lns::op_kernel;
using lns::stateless;

const op_kernel<no_context> k_kernels[] = {
    { GGML_OP_MUL_MAT,       any_subop,          lns::check_mul_mat,               stateless<lns_mul_mat>       },
    { GGML_OP_MUL_MAT_ID,    any_subop,          lns::check_mul_mat_id,            stateless<lns_mul_mat_id>    },
    { GGML_OP_ADD,           any_subop,          lns::check_binary_f32_or_lns16,   stateless<lns_add>           },
    { GGML_OP_MUL,           any_subop,          lns::check_binary_f32_or_lns16,   stateless<lns_mul>           },
    { GGML_OP_DIV,           any_subop,          lns::check_binary_f32_or_lns16,   stateless<lns_div>           },
    { GGML_OP_SCALE,         any_subop,          lns::check_contiguous_f32,        stateless<lns_scale>         },
    { GGML_OP_CLAMP,         any_subop,          lns::check_contiguous_f32,        stateless<lns_clamp>         },
    { GGML_OP_SOFT_MAX,      any_subop,          lns::check_soft_max,              stateless<lns_soft_max>      },
    { GGML_OP_RMS_NORM,      any_subop,          lns::check_unary_like_f32,        stateless<lns_rms_norm>      },
    { GGML_OP_DIAG_MASK_INF, any_subop,          lns::check_contiguous_f32,        stateless<lns_diag_mask_inf> },
    { GGML_OP_UNARY,         GGML_UNARY_OP_SILU, lns::check_unary_activation,      stateless<lns_silu>          },
    { GGML_OP_UNARY,         GGML_UNARY_OP_GELU, lns::check_unary_activation,      stateless<lns_gelu>          },
    { GGML_OP_UNARY,         GGML_UNARY_OP_RELU, lns::check_unary_activation,      stateless<lns_relu>          },
    { GGML_OP_UNARY,         GGML_UNARY_OP_TANH, lns::check_unary_activation,      stateless<lns_tanh>          },
    { GGML_OP_GLU,           GGML_GLU_OP_SWIGLU, lns::check_glu,                   stateless<lns_swiglu>        },
    { GGML_OP_GLU,           GGML_GLU_OP_GEGLU,  lns::check_glu,                   stateless<lns_geglu>         },
    { GGML_OP_GET_ROWS,      any_subop,          lns::check_get_rows_f32_or_lns16, stateless<lns_get_rows>      },
    { GGML_OP_CPY,           any_subop,          lns::check_lns16_copy,            stateless<lns_cpy>           },
    { GGML_OP_CONT,          any_subop,          lns::check_lns16_copy,            stateless<lns_cpy>           },
    { GGML_OP_DUP,           any_subop,          lns::check_lns16_copy,            stateless<lns_cpy>           },
    { GGML_OP_ROPE,          any_subop,          lns::check_rope,                  stateless<lns_rope>          },
    { GGML_OP_ARGSORT,       any_subop,          lns::check_argsort,               stateless<lns_argsort>       },
};

struct lns16_traits : lns::backend_traits_base {
    static constexpr const char * name = "LNS16";

    // The compiled summation variant is part of the device identity: it
    // changes the numerics of every reduction, and test-lns-backend checks
    // its tolerance profile against this string.
    static const char * description() {
        static const std::string desc =
            std::string("Logarithmic Number System (xlns16, sum=") + lns16_sum_mode_name() + monte_suffix() + ")";
        return desc.c_str();
    }

    // ", monte=pure16" / ", monte=bylpvip" in a monte build
    static std::string monte_suffix() {
        const char * impl = lns16_monte_impl_name();
        return impl ? std::string(", monte=") + impl : std::string();
    }

    static ggml_guid_t guid() {
        static ggml_guid guid = {
            0x4c, 0x4e, 0x53, 0x31, 0x36, 0x2d, 0x62, 0x6b,
            0x65, 0x6e, 0x64, 0x2d, 0x67, 0x67, 0x6d, 0x6c
        };
        return &guid;
    }

    static lns::op_table<context> kernels() {
        return lns::make_op_table(k_kernels);
    }

    static void log_init() {
        std::fprintf(stderr, "ggml-lns: LNS16 backend initialized (sum mode: %s%s)\n", lns16_sum_mode_name(),
                     monte_suffix().c_str());
    }

    static const char * audit_fields() {
        static const std::string fields = std::string("\"sum_mode\":\"") + lns16_sum_mode_name() + "\",";
        return fields.c_str();
    }

    // the Monte state functions, for callers that only see the registry
    static void * proc_address(const char * name) {
        struct entry { const char * name; void * fn; };
        static const entry entries[] = {
            { "ggml_backend_lns16_monte_get_state", (void *) ggml_backend_lns16_monte_get_state },
            { "ggml_backend_lns16_monte_set_state", (void *) ggml_backend_lns16_monte_set_state },
            { "ggml_backend_lns16_monte_reserve",   (void *) ggml_backend_lns16_monte_reserve   },
        };
        for (const entry & e : entries) {
            if (std::strcmp(e.name, name) == 0) {
                return e.fn;
            }
        }
        return nullptr;
    }
};

using lns16_backend = lns::host_backend<lns16_traits>;

} // namespace

ggml_backend_t ggml_backend_lns_init(void) {
    return lns16_backend::init();
}

bool ggml_backend_is_lns(ggml_backend_t backend) {
    return lns16_backend::is(backend);
}

ggml_backend_reg_t ggml_backend_lns_reg(void) {
    return lns16_backend::reg();
}

GGML_BACKEND_DL_IMPL(ggml_backend_lns_reg)
