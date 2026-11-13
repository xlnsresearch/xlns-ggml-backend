#pragma once

// Framework for LNS backends that compute on ggml host (CPU) buffers.
//
// ggml asks a backend for three vtables (registry, device, backend) and an
// LNS backend would otherwise repeat about 400 lines of identical glue for
// them. Instead a backend describes itself with a Traits struct and gets the
// vtables from host_backend<Traits>:
//
//   struct my_traits : lns::backend_traits_base {
//       static constexpr const char * name = "LNS16";       // device, backend and registry name
//       static const char * description();                  // device description
//       static ggml_guid_t  guid();                         // unique per backend
//       static lns::op_table<context> kernels();            // what it computes, and how
//   };
//   using my_backend = lns::host_backend<my_traits>;
//
// Optional members (defaults in backend_traits_base) cover per-instance state
// (context / create_context), device presence (device_count), memory
// reporting, a one-time init log, extra audit fields and backend-specific
// functions exported through the registry (proc_address).
//
// The op table is the single source of truth: supports_op and graph_compute
// both look nodes up in it, so a backend cannot claim an op it does not
// implement, or implement one it never claims. Adding an op is one table row.
//
// Every LNS backend reports itself as an ACCEL device that uses the CPU buffer
// type: weights and activations stay in host memory, unsupported nodes run on
// the CPU backend without any copies, and GGML_LNS_BACKEND decides which LNS
// backend claims a node (lns-backend-config.h).

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "lns-backend-config.h"
#include "lns-op-support.h"

#include <cstddef>

namespace lns {

// Context of a backend that keeps no per-instance state.
struct no_context {};

// Wildcard for op_kernel::subop.
inline constexpr int any_subop = -1;

template <typename Ctx>
struct op_kernel {
    ggml_op op;
    int     subop;                                  // ggml_unary_op / ggml_glu_op, or any_subop
    op_support (*check)(const ggml_tensor * node);  // may this backend run the node?
    void       (*compute)(Ctx & ctx, ggml_tensor * node);
};

template <typename Ctx>
struct op_table {
    const op_kernel<Ctx> * first;
    size_t                 count;

    const op_kernel<Ctx> * begin() const { return first; }
    const op_kernel<Ctx> * end()   const { return first + count; }
};

template <typename Ctx, size_t N>
constexpr op_table<Ctx> make_op_table(const op_kernel<Ctx> (&kernels)[N]) {
    return { kernels, N };
}

// Adapts a context-free kernel, void kernel(ggml_tensor *), to an op_kernel.
template <void (*Kernel)(ggml_tensor *)>
void stateless(no_context &, ggml_tensor * node) {
    Kernel(node);
}

// Defaults for the optional Traits members.
struct backend_traits_base {
    using context = no_context;

    // Called by init(); returning nullptr makes init fail.
    static context * create_context() { return new context(); }

    // 0 hides the device, e.g. when no GPU is present.
    static size_t device_count() { return 1; }

    static void get_memory(size_t * free, size_t * total) {
        *free  = 0;
        *total = 0;
    }

    static const char * device_id() { return nullptr; }

    // Printed once, on the first init().
    static void log_init() {}

    // Extra members for the LNS_OP_AUDIT records (see audit_supports_op).
    static const char * audit_fields() { return ""; }

    // ggml_backend_reg_get_proc_address: backend-specific functions by name.
    static void * proc_address(const char *) { return nullptr; }
};

// Nodes that only reinterpret memory. Every LNS backend accepts them so that
// view chains do not split the graph, and none computes anything for them.
inline bool is_view_or_noop(ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

template <typename Traits>
class host_backend {
public:
    using context = typename Traits::context;

    static ggml_backend_reg_t reg() {
        static ggml_backend_reg r = {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ reg_iface,
            /* .context     = */ nullptr,
        };
        return &r;
    }

    static ggml_backend_t init() {
        if (Traits::device_count() == 0) {
            return nullptr;
        }
        context * ctx = Traits::create_context();
        if (ctx == nullptr) {
            return nullptr;
        }
        static bool logged = false;
        if (!logged) {
            logged = true;
            Traits::log_init();
        }
        return new ggml_backend {
            /* .guid    = */ Traits::guid(),
            /* .iface   = */ backend_iface,
            /* .device  = */ device(),
            /* .context = */ ctx,
        };
    }

    static bool is(ggml_backend_t backend) {
        return backend != nullptr && ggml_guid_matches(backend->guid, Traits::guid());
    }

    // What supports_op would answer, before the selection and LNS_DISABLE_OPS
    // filters. Exposed for tests and diagnostics.
    static op_support check(const ggml_tensor * node) {
        if (is_view_or_noop(node->op)) {
            return { true, "supported_view_or_noop" };
        }
        const op_kernel<context> * k = find(node);
        if (k == nullptr) {
            switch (node->op) {
                case GGML_OP_UNARY: return { false, "unsupported_unary_op" };
                case GGML_OP_GLU:   return { false, "unsupported_glu_op" };
                default:            return { false, "unsupported_op" };
            }
        }
        return k->check(node);
    }

private:
    static ggml_backend_dev_t device() {
        static ggml_backend_device dev = {
            /* .iface   = */ device_iface,
            /* .reg     = */ reg(),
            /* .context = */ nullptr,
        };
        return &dev;
    }

    static const op_kernel<context> * find(const ggml_tensor * node) {
        const int subop = node->op == GGML_OP_UNARY ? (int) ggml_get_unary_op(node)
                        : node->op == GGML_OP_GLU   ? (int) ggml_get_glu_op(node)
                        : any_subop;
        for (const op_kernel<context> & k : Traits::kernels()) {
            if (k.op == node->op && (k.subop == any_subop || k.subop == subop)) {
                return &k;
            }
        }
        return nullptr;
    }

    // ---- backend ----

    static const char * backend_get_name(ggml_backend_t) {
        return Traits::name;
    }

    static void backend_free(ggml_backend_t backend) {
        delete static_cast<context *>(backend->context);
        delete backend;
    }

    static ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
        context & ctx = *static_cast<context *>(backend->context);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || is_view_or_noop(node->op) || ggml_is_empty(node)) {
                continue;
            }
            const op_kernel<context> * k = find(node);
            if (k == nullptr) {
                GGML_ABORT("%s: unsupported op %s\n", Traits::name, ggml_op_desc(node));
            }
            k->compute(ctx, node);
        }
        return GGML_STATUS_SUCCESS;
    }

    static constexpr ggml_backend_i backend_iface = {
        /* .get_name            = */ backend_get_name,
        /* .free                = */ backend_free,
        /* .set_tensor_async    = */ nullptr,
        /* .get_tensor_async    = */ nullptr,
        /* .set_tensor_2d_async = */ nullptr,
        /* .get_tensor_2d_async = */ nullptr,
        /* .cpy_tensor_async    = */ nullptr,
        /* .synchronize         = */ nullptr,
        /* .graph_plan_create   = */ nullptr,
        /* .graph_plan_free     = */ nullptr,
        /* .graph_plan_update   = */ nullptr,
        /* .graph_plan_compute  = */ nullptr,
        /* .graph_compute       = */ backend_graph_compute,
        /* .event_record        = */ nullptr,
        /* .event_wait          = */ nullptr,
        /* .graph_optimize      = */ nullptr,
    };

    // ---- device ----

    static const char * dev_get_name(ggml_backend_dev_t) {
        return Traits::name;
    }

    static const char * dev_get_description(ggml_backend_dev_t) {
        return Traits::description();
    }

    static void dev_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
        Traits::get_memory(free, total);
    }

    // "enum" is required: ggml also declares a function ggml_backend_dev_type.
    static enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t) {
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }

    static void dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
        props->name        = dev_get_name(dev);
        props->description = dev_get_description(dev);
        props->type        = dev_get_type(dev);
        props->device_id   = Traits::device_id();
        dev_get_memory(dev, &props->memory_free, &props->memory_total);
        props->caps = {
            /* .async                = */ false,
            /* .host_buffer          = */ false,
            /* .buffer_from_host_ptr = */ true,
            /* .events               = */ false,
        };
    }

    static ggml_backend_t dev_init_backend(ggml_backend_dev_t, const char *) {
        return init();
    }

    static ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t) {
        return ggml_backend_cpu_buffer_type();
    }

    static ggml_backend_buffer_t dev_buffer_from_host_ptr(ggml_backend_dev_t, void * ptr, size_t size, size_t) {
        return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    }

    static bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor * node) {
        if (!backend_selected(Traits::name)) {
            audit_supports_op(Traits::name, Traits::audit_fields(), node, false, "disabled_by_env");
            return false;
        }
        const op_support support = check(node);
        audit_supports_op(Traits::name, Traits::audit_fields(), node, support.supported, support.reason);
        return support.supported && !op_disabled_by_env(node);
    }

    static bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
        return ggml_backend_buft_is_host(buft);
    }

    // Offload every supported op: this makes the scheduler prefer the LNS
    // backend over the CPU backend even though both use host buffers.
    static bool dev_offload_op(ggml_backend_dev_t dev, const ggml_tensor * node) {
        return dev_supports_op(dev, node);
    }

    static constexpr ggml_backend_device_i device_iface = {
        /* .get_name             = */ dev_get_name,
        /* .get_description      = */ dev_get_description,
        /* .get_memory           = */ dev_get_memory,
        /* .get_type             = */ dev_get_type,
        /* .get_props            = */ dev_get_props,
        /* .init_backend         = */ dev_init_backend,
        /* .get_buffer_type      = */ dev_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ dev_buffer_from_host_ptr,
        /* .supports_op          = */ dev_supports_op,
        /* .supports_buft        = */ dev_supports_buft,
        /* .offload_op           = */ dev_offload_op,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
    };

    // ---- registry ----

    static const char * reg_get_name(ggml_backend_reg_t) {
        return Traits::name;
    }

    static size_t reg_get_device_count(ggml_backend_reg_t) {
        return Traits::device_count();
    }

    static ggml_backend_dev_t reg_get_device(ggml_backend_reg_t, size_t index) {
        GGML_ASSERT(index < Traits::device_count());
        return device();
    }

    static void * reg_get_proc_address(ggml_backend_reg_t, const char * name) {
        return Traits::proc_address(name);
    }

    static constexpr ggml_backend_reg_i reg_iface = {
        /* .get_name         = */ reg_get_name,
        /* .get_device_count = */ reg_get_device_count,
        /* .get_device       = */ reg_get_device,
        /* .get_proc_address = */ reg_get_proc_address,
    };
};

} // namespace lns
