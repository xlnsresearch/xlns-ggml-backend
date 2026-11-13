#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Parity of the LNSCUDA backends with the CPU LNS backends.
//
// Usage: test-lns-cuda-parity [--only=cuda16|cuda32] [--expect-sum=MODE]
//
// LNSCUDA16 must be bit-identical to LNS16 built with the same GGML_LNS16_SUM
// (base, monte, lpvip or sum32); the test reads both modes from the device
// descriptions and refuses a pair whose modes differ (mixed libraries).
// For monte, every comparison starts CPU and GPU from the same LFSR state
// (LNS16's Monte state, via its registry), and the test also checks draw
// counts, runs without a reset in between, a second GPU backend instance and
// a scheduled graph with a CPU-fallback reduction (MUL_MAT_ID).
// LNSCUDA32 gets a 1e-5 scaled-error tolerance: its float activations may
// differ from glibc in the last ulp.
//
// Exit codes: 0 pass, 1 fail, 77 skipped (no CUDA device).

static uint32_t g_rng = 0x2545F491u;

static float rand_unit() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float) (g_rng >> 8) / 16777216.0f * 2.0f - 1.0f;
}

// ---- Monte state (LNS16 registry functions) ----------------------------------

struct monte_api {
    bool (*get)(uint16_t *)               = nullptr;
    bool (*set)(uint16_t)                 = nullptr;
    bool (*reserve)(uint64_t, uint16_t *) = nullptr;
    bool active                           = false; // a monte build

    uint16_t state() const {
        uint16_t s = 0;
        get(&s);
        return s;
    }
    // the state `draws` steps after s (does not change the current state)
    uint16_t advanced(uint16_t s, uint64_t draws) const {
        const uint16_t saved = state();
        uint16_t start = 0, end = 0;
        set(s);
        reserve(draws, &start);
        get(&end);
        set(saved);
        return end;
    }
};

static monte_api g_monte;

static void load_monte_api() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("LNS16");
    if (reg == nullptr) {
        return;
    }
    g_monte.get     = (bool (*)(uint16_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_lns16_monte_get_state");
    g_monte.set     = (bool (*)(uint16_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_lns16_monte_set_state");
    g_monte.reserve = (bool (*)(uint64_t, uint16_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_lns16_monte_reserve");
    uint16_t s = 0;
    g_monte.active = g_monte.get && g_monte.set && g_monte.reserve && g_monte.get(&s);
}

// draws an all-CPU run of the graph consumes (K-1 per dot, n-1 per softmax
// row, n per rms_norm row)
static uint64_t expected_draws(ggml_cgraph * graph) {
    uint64_t draws = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        switch (t->op) {
            case GGML_OP_MUL_MAT: {
                const int64_t K = t->src[0]->ne[0];
                if (K > 0) draws += (uint64_t) ggml_nelements(t) * (uint64_t) (K - 1);
                break;
            }
            case GGML_OP_MUL_MAT_ID: {
                const int64_t K = t->src[0]->ne[0];
                if (K > 0) draws += (uint64_t) ggml_nelements(t) * (uint64_t) (K - 1);
                break;
            }
            case GGML_OP_SOFT_MAX:
                if (t->ne[0] > 0) draws += (uint64_t) ggml_nrows(t) * (uint64_t) (t->ne[0] - 1);
                break;
            case GGML_OP_RMS_NORM:
                draws += (uint64_t) ggml_nelements(t);
                break;
            default:
                break;
        }
    }
    return draws;
}

// Inputs are filled by name: "pos" gets token positions, "ids" expert ids in
// [0, 4), "mask" a causal mask with -INFINITY above the diagonal,
// "freq_factors" RoPE frequency factors in [1, 8); everything else gets values
// in [-2, 2), quantized for quantized types.
static void fill_input(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v((size_t) n);
        const bool ids = std::strcmp(t->name, "ids") == 0;
        for (int64_t i = 0; i < n; i++) v[(size_t) i] = ids ? (int32_t) ((i * 3 + 1) % 4) : (int32_t) (i * 7 + 3);
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        return;
    }
    std::vector<float> v((size_t) n);
    if (std::strcmp(t->name, "mask") == 0) {
        for (int64_t r = 0; r < n / t->ne[0]; r++) {
            for (int64_t c = 0; c < t->ne[0]; c++) {
                v[(size_t) (r * t->ne[0] + c)] = c > (r % t->ne[1]) + 2 ? -INFINITY : 0.25f * rand_unit();
            }
        }
    } else if (std::strcmp(t->name, "freq_factors") == 0) {
        for (float & x : v) x = 4.5f + 3.5f * rand_unit();
    } else if (std::strcmp(t->name, "positive") == 0) {
        for (float & x : v) x = 0.05f + std::fabs(rand_unit());
    } else {
        for (float & x : v) x = 2.0f * rand_unit();
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, v.data(), q.data(), 0, n / t->ne[0], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

// A case builds its graph from inputs created in `weights` (allocated in a
// buffer marked as model weights, i.e. eligible for the GPU weight cache) and
// `inputs`, and returns the node whose output is compared.
using build_fn = std::function<ggml_tensor *(ggml_context * weights, ggml_context * inputs)>;

struct test_case {
    std::string name;
    build_fn    build;
};

struct comparison {
    size_t mismatches = 0;
    float  max_err    = 0.0f;
};

static comparison compare(const std::vector<float> & ref, const std::vector<float> & got) {
    comparison c;
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) == 0) {
            continue;
        }
        c.mismatches++;
        // scaled error, as in test-lns-backend: relative for |ref| >= 1,
        // absolute below, so values near zero do not inflate it
        const float err = std::fabs(ref[i] - got[i]) / std::fmax(std::fabs(ref[i]), 1.0f);
        c.max_err = std::isnan(err) ? INFINITY : std::fmax(c.max_err, err);
    }
    return c;
}

class graph_runner {
public:
    explicit graph_runner(const test_case & tc) {
        const ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        wctx = ggml_init(params);
        ictx = ggml_init(params);
        out  = tc.build(wctx, ictx);
        graph = ggml_new_graph(ictx);
        ggml_build_forward_expand(graph, out);

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (ggml_get_first_tensor(wctx) != nullptr) {
            wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
            ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, buft);
        for (ggml_context * c : { wctx, ictx }) {
            for (ggml_tensor * t = ggml_get_first_tensor(c); t != nullptr; t = ggml_get_next_tensor(c, t)) {
                if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
                    fill_input(t);
                }
            }
        }
    }

    ~graph_runner() {
        ggml_backend_buffer_free(ibuf);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        ggml_free(ictx);
        ggml_free(wctx);
    }

    // Names of the nodes `backend` would not run (it must run all of them).
    std::string unsupported(ggml_backend_t backend) const {
        std::string names;
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                names += std::string(names.empty() ? "" : ", ") + ggml_op_desc(node);
            }
        }
        return names;
    }

    std::vector<float> run(ggml_backend_t backend) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph_compute failed on %s\n", ggml_backend_name(backend));
            std::exit(1);
        }
        return result();
    }

    std::vector<float> run(ggml_backend_sched_t sched) {
        ggml_backend_sched_reset(sched);
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "sched graph_compute failed\n");
            std::exit(1);
        }
        return result();
    }

    ggml_cgraph * graph = nullptr;

private:
    std::vector<float> result() const {
        std::vector<float> v((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, v.data(), 0, ggml_nbytes(out));
        return v;
    }

    ggml_context *        wctx = nullptr;
    ggml_context *        ictx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_tensor *         out  = nullptr;
};

static ggml_tensor * named(ggml_tensor * t, const char * name) {
    ggml_set_name(t, name);
    return t;
}

static std::vector<test_case> make_cases() {
    std::vector<test_case> cases;
    auto add = [&](const std::string & name, build_fn fn) { cases.push_back({ name, std::move(fn) }); };

    for (ggml_type wtype : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K }) {
        add(std::string("mul_mat weights ") + ggml_type_name(wtype), [wtype](ggml_context * w, ggml_context * in) {
            ggml_tensor * a = ggml_new_tensor_2d(w, wtype, 256, 67);
            ggml_tensor * b = ggml_new_tensor_2d(in, GGML_TYPE_F32, 256, 5); // N not a multiple of 4
            return ggml_mul_mat(in, a, b);
        });
    }
    add("mul_mat long K=1536", [](ggml_context * w, ggml_context * in) {
        return ggml_mul_mat(in, ggml_new_tensor_2d(w, GGML_TYPE_F32, 1536, 40),
                                ggml_new_tensor_2d(in, GGML_TYPE_F32, 1536, 3));
    });
    add("mul_mat long K=4096, tail columns N=7", [](ggml_context * w, ggml_context * in) {
        return ggml_mul_mat(in, ggml_new_tensor_2d(w, GGML_TYPE_Q8_0, 4096, 33),
                                ggml_new_tensor_2d(in, GGML_TYPE_F32, 4096, 7));
    });
    add("mul_mat same-sign K=2048", [](ggml_context * w, ggml_context * in) {
        return ggml_mul_mat(in, named(ggml_new_tensor_2d(w, GGML_TYPE_F32, 2048, 9), "positive"),
                                named(ggml_new_tensor_2d(in, GGML_TYPE_F32, 2048, 2), "positive"));
    });
    add("mul_mat K=1 (no reduction)", [](ggml_context * w, ggml_context * in) {
        return ggml_mul_mat(in, ggml_new_tensor_2d(w, GGML_TYPE_F32, 1, 13), ggml_new_tensor_2d(in, GGML_TYPE_F32, 1, 6));
    });
    add("mul_mat generation shape N=1", [](ggml_context * w, ggml_context * in) {
        return ggml_mul_mat(in, ggml_new_tensor_2d(w, GGML_TYPE_Q4_K, 512, 300), ggml_new_tensor_2d(in, GGML_TYPE_F32, 512, 1));
    });
    add("mul_mat GQA broadcast, F16 KV view", [](ggml_context *, ggml_context * in) {
        // K cache [head_dim, n_kv, n_head_kv] viewed with permuted strides
        ggml_tensor * kv = ggml_new_tensor_3d(in, GGML_TYPE_F16, 64, 2, 24);
        ggml_tensor * k  = ggml_permute(in, kv, 0, 2, 1, 3); // [64, 24, 2]
        ggml_tensor * q  = ggml_new_tensor_3d(in, GGML_TYPE_F32, 64, 3, 6);
        return ggml_mul_mat(in, k, q);
    });
    add("mul_mat 4D batch broadcast (r2=2, r3=1)", [](ggml_context *, ggml_context * in) {
        ggml_tensor * a = ggml_new_tensor_4d(in, GGML_TYPE_F32, 96, 21, 2, 3);
        ggml_tensor * b = ggml_new_tensor_4d(in, GGML_TYPE_F32, 96, 6, 4, 3);
        return ggml_mul_mat(in, a, b);
    });
    add("add bias broadcast", [](ggml_context *, ggml_context * in) {
        return ggml_add(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 7), ggml_new_tensor_1d(in, GGML_TYPE_F32, 96));
    });
    add("mul broadcast rows", [](ggml_context *, ggml_context * in) {
        return ggml_mul(in, ggml_new_tensor_3d(in, GGML_TYPE_F32, 32, 5, 3), ggml_new_tensor_3d(in, GGML_TYPE_F32, 32, 1, 3));
    });
    add("div", [](ggml_context *, ggml_context * in) {
        return ggml_div(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 50, 4), ggml_new_tensor_2d(in, GGML_TYPE_F32, 50, 4));
    });
    add("scale", [](ggml_context *, ggml_context * in) {
        return ggml_scale(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 77, 3), 0.125f);
    });
    add("scale_bias", [](ggml_context *, ggml_context * in) {
        return ggml_scale_bias(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 77, 3), -1.5f, 0.75f);
    });
    add("silu", [](ggml_context *, ggml_context * in) { return ggml_silu(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("gelu", [](ggml_context *, ggml_context * in) { return ggml_gelu(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("relu", [](ggml_context *, ggml_context * in) { return ggml_relu(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("tanh", [](ggml_context *, ggml_context * in) { return ggml_tanh(in, ggml_new_tensor_1d(in, GGML_TYPE_F32, 1000)); });
    add("swiglu split", [](ggml_context *, ggml_context * in) {
        return ggml_swiglu_split(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 48, 5), ggml_new_tensor_2d(in, GGML_TYPE_F32, 48, 5));
    });
    add("swiglu fused", [](ggml_context *, ggml_context * in) {
        return ggml_swiglu(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 5));
    });
    add("swiglu fused swapped", [](ggml_context *, ggml_context * in) {
        return ggml_swiglu_swapped(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 5));
    });
    add("geglu fused", [](ggml_context *, ggml_context * in) {
        return ggml_geglu(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 96, 5));
    });
    add("rms_norm 576", [](ggml_context *, ggml_context * in) {
        return ggml_rms_norm(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 576, 4), 1e-5f);
    });
    add("rms_norm 2048 x 3 x 2", [](ggml_context *, ggml_context * in) {
        return ggml_rms_norm(in, ggml_new_tensor_3d(in, GGML_TYPE_F32, 2048, 3, 2), 1e-6f);
    });
    add("rms_norm singleton rows", [](ggml_context *, ggml_context * in) {
        return ggml_rms_norm(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 1, 9), 1e-5f);
    });
    add("soft_max", [](ggml_context *, ggml_context * in) {
        return ggml_soft_max(in, ggml_new_tensor_3d(in, GGML_TYPE_F32, 300, 3, 2));
    });
    add("soft_max masked, padded mask", [](ggml_context *, ggml_context * in) {
        ggml_tensor * kq   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 40, 5, 4);
        ggml_tensor * mask = named(ggml_new_tensor_2d(in, GGML_TYPE_F32, 40, 8), "mask");
        return ggml_soft_max_ext(in, kq, mask, 0.125f, 0.0f);
    });
    add("soft_max masked, long rows", [](ggml_context *, ggml_context * in) {
        ggml_tensor * kq   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 1100, 6, 2);
        ggml_tensor * mask = named(ggml_new_tensor_2d(in, GGML_TYPE_F32, 1100, 6), "mask");
        return ggml_soft_max_ext(in, kq, mask, 0.0883f, 0.0f);
    });
    add("soft_max singleton rows", [](ggml_context *, ggml_context * in) {
        return ggml_soft_max(in, ggml_new_tensor_2d(in, GGML_TYPE_F32, 1, 5));
    });
    for (int mode : { 0, GGML_ROPE_TYPE_NEOX }) {
        add(mode ? "rope neox, partial dims" : "rope normal, partial dims", [mode](ggml_context *, ggml_context * in) {
            ggml_tensor * x   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 64, 4, 6);
            ggml_tensor * pos = named(ggml_new_tensor_1d(in, GGML_TYPE_I32, 6), "pos");
            return ggml_rope_ext(in, x, pos, nullptr, 48, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        });
    }
    add("rope neox, frequency factors", [](ggml_context *, ggml_context * in) {
        ggml_tensor * x   = ggml_new_tensor_3d(in, GGML_TYPE_F32, 64, 4, 6);
        ggml_tensor * pos = named(ggml_new_tensor_1d(in, GGML_TYPE_I32, 6), "pos");
        ggml_tensor * ff  = named(ggml_new_tensor_1d(in, GGML_TYPE_F32, 32), "freq_factors");
        return ggml_rope_ext(in, x, pos, ff, 64, GGML_ROPE_TYPE_NEOX, 0, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    });
    return cases;
}

// A miniature attention + MoE-ish FFN block. CLAMP and MUL_MAT_ID are not GPU
// ops (and CONT is not an LNS op at all), so under the scheduler they must
// land on the CPU LNS / CPU backends: the chain [GPU LNS, CPU LNS, CPU] has to
// produce exactly what [CPU LNS, CPU] produces (Monte draws included).
static test_case make_chain_case() {
    return { "block with CPU-LNS fallback (sched)", [](ggml_context * w, ggml_context * in) {
        ggml_tensor * x     = ggml_new_tensor_2d(in, GGML_TYPE_F32, 128, 6);
        ggml_tensor * pos   = named(ggml_new_tensor_1d(in, GGML_TYPE_I32, 6), "pos");
        ggml_tensor * mask  = named(ggml_new_tensor_2d(in, GGML_TYPE_F32, 6, 8), "mask");
        ggml_tensor * ids   = named(ggml_new_tensor_2d(in, GGML_TYPE_I32, 2, 6), "ids");
        ggml_tensor * wq    = ggml_new_tensor_2d(w, GGML_TYPE_Q8_0, 128, 128);
        ggml_tensor * wup   = ggml_new_tensor_2d(w, GGML_TYPE_Q4_0, 128, 192);
        ggml_tensor * wexp  = ggml_new_tensor_3d(w, GGML_TYPE_Q8_0, 128, 64, 4);
        ggml_tensor * wdown = ggml_new_tensor_2d(w, GGML_TYPE_Q4_K, 256, 128);

        // attention: q = rope(Wq norm(x)), softmax(q q^T) q, per head
        ggml_tensor * h  = ggml_rms_norm(in, x, 1e-6f);
        ggml_tensor * q  = ggml_reshape_3d(in, ggml_mul_mat(in, wq, h), 32, 4, 6);
        q = ggml_rope_ext(in, q, pos, nullptr, 32, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qp  = ggml_cont(in, ggml_permute(in, q, 0, 2, 1, 3));       // [32, 6, 4]
        ggml_tensor * qt  = ggml_cont(in, ggml_transpose(in, qp));                // [6, 32, 4]
        ggml_tensor * kq  = ggml_soft_max_ext(in, ggml_mul_mat(in, qp, qp), mask, 0.17f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(in, qt, kq);                             // [32, 6, 4]
        ggml_tensor * att = ggml_cont_2d(in, ggml_permute(in, kqv, 0, 2, 1, 3), 128, 6);

        // experts on the CPU fallback: [64, 2, 6] -> [128, 6]
        ggml_tensor * h2  = ggml_rms_norm(in, ggml_add(in, att, x), 1e-6f);
        ggml_tensor * moe = ggml_mul_mat_id(in, wexp, ggml_reshape_3d(in, h2, 128, 1, 6), ids);
        ggml_tensor * moe2 = ggml_cont_2d(in, moe, 128, 6);

        // FFN: Wdown clamp(swiglu(Wup norm(x)))
        ggml_tensor * up  = ggml_clamp(in, ggml_swiglu(in, ggml_mul_mat(in, wup, h2)), -0.5f, 0.5f); // [96, 6]
        ggml_tensor * up2 = ggml_concat(in, up, up, 0);                           // [192, 6] -> pad to 256
        ggml_tensor * ffn = ggml_mul_mat(in, wdown, ggml_pad(in, up2, 64, 0, 0, 0));

        return ggml_scale(in, ggml_add(in, ggml_add(in, ggml_add(in, att, x), ffn), moe2), 0.5f);
    } };
}

struct pair_result {
    int run    = 0;
    int failed = 0;
};

// See the header comment for why LNSCUDA32 gets a tolerance.
static constexpr float k_lns32_tol = 1e-5f;

static bool check(const char * label, bool ok, const std::string & detail, pair_result & r) {
    r.run++;
    r.failed += ok ? 0 : 1;
    std::printf("  %-52s %s%s\n", label, ok ? "ok" : "FAIL", detail.c_str());
    return ok;
}

static bool grade(const char * label, const comparison & c, bool exact, pair_result & r) {
    const bool ok = exact ? c.mismatches == 0 : c.max_err <= k_lns32_tol;
    char detail[128];
    if (c.mismatches == 0) {
        std::snprintf(detail, sizeof(detail), " (bit-identical)");
    } else {
        std::snprintf(detail, sizeof(detail), " (%zu values differ, max scaled err %.3g)", c.mismatches, c.max_err);
    }
    return check(label, ok, detail, r);
}

static const uint16_t k_seed = 7;      // xlns16monte.cpp's initial state
static const uint16_t k_seed_alt = 0x1D2F;

static void run_pair(const char * gpu_name, const char * cpu_name, bool exact, bool monte, pair_result & r) {
    ggml_backend_t gpu  = ggml_backend_init_by_name(gpu_name, nullptr);
    ggml_backend_t gpu2 = ggml_backend_init_by_name(gpu_name, nullptr); // a second context
    ggml_backend_t cpu_lns = ggml_backend_init_by_name(cpu_name, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || gpu2 == nullptr || cpu_lns == nullptr || cpu == nullptr) {
        std::printf("ERROR: could not initialize %s / %s / CPU\n", gpu_name, cpu_name);
        r.run++;
        r.failed++;
        return;
    }
    std::printf("\n%s vs %s (%s%s)\n", gpu_name, cpu_name, exact ? "bit-exact" : "max scaled err 1e-5",
                monte ? ", Monte state snapshots" : "");

    // The GPU weight cache is keyed by tensor address and lives as long as the
    // backend; in llama.cpp the backend dies before the model's buffers do.
    // Keep every case's buffers alive until the backend is freed, so no case
    // can reuse a previous case's weight address.
    std::vector<std::unique_ptr<graph_runner>> runners;
    bool any_progress = false; // Monte: some run without reset changed the output
    for (const test_case & tc : make_cases()) {
        runners.push_back(std::make_unique<graph_runner>(tc));
        graph_runner & g = *runners.back();
        const std::string missing = g.unsupported(gpu) + g.unsupported(cpu_lns);
        if (!missing.empty()) {
            check(tc.name.c_str(), false, " (not claimed: " + missing + ")", r);
            continue;
        }
        if (!monte) {
            const std::vector<float> ref  = g.run(cpu_lns);
            const std::vector<float> cold = g.run(gpu); // converts (and caches) weights
            const std::vector<float> warm = g.run(gpu); // reuses the cached weights
            grade(tc.name.c_str(), compare(ref, cold), exact, r);
            if (compare(cold, warm).mismatches != 0) {
                check(tc.name.c_str(), false, " (cached weights changed the result)", r);
            }
            continue;
        }

        // Monte: CPU and GPU from the same state, twice each without a reset
        const uint64_t draws = expected_draws(g.graph);
        g_monte.set(k_seed);
        const std::vector<float> ref1 = g.run(cpu_lns);
        const uint16_t cpu_s1 = g_monte.state();
        const std::vector<float> ref2 = g.run(cpu_lns);
        const uint16_t cpu_s2 = g_monte.state();

        g_monte.set(k_seed);
        const std::vector<float> cold = g.run(gpu);  // converts (and caches) weights
        const uint16_t gpu_s1 = g_monte.state();
        const std::vector<float> next = g.run(gpu2); // other context, continues the sequence
        const uint16_t gpu_s2 = g_monte.state();
        g_monte.set(k_seed);
        const std::vector<float> warm = g.run(gpu);  // cached weights, same snapshot as cold
        const uint16_t gpu_s3 = g_monte.state();

        grade(tc.name.c_str(), compare(ref1, cold), true, r);
        std::string label = tc.name + ": next run, 2nd context";
        grade(label.c_str(), compare(ref2, next), true, r);
        bool ok = compare(cold, warm).mismatches == 0 && gpu_s3 == gpu_s1;
        if (!ok) check(tc.name.c_str(), false, " (cached weights / snapshot replay changed the result)", r);
        const uint16_t want_s1 = g_monte.advanced(k_seed, draws);
        const uint16_t want_s2 = g_monte.advanced(want_s1, draws);
        ok = cpu_s1 == want_s1 && cpu_s2 == want_s2 && gpu_s1 == want_s1 && gpu_s2 == want_s2;
        if (!ok) {
            char detail[160];
            std::snprintf(detail, sizeof(detail), " (draw accounting: %llu draws/run; states cpu %04x,%04x gpu %04x,%04x "
                          "want %04x,%04x)", (unsigned long long) draws, cpu_s1, cpu_s2, gpu_s1, gpu_s2, want_s1, want_s2);
            check(tc.name.c_str(), false, detail, r);
        }
        if (draws == 0 && (cpu_s1 != k_seed || gpu_s1 != k_seed)) {
            check(tc.name.c_str(), false, " (consumed draws without a reduction)", r);
        }
        if (draws > 0 && compare(ref1, ref2).mismatches > 0) {
            any_progress = true;
        }
    }
    if (monte) {
        check("Monte: consecutive runs keep advancing the sequence", any_progress,
              any_progress ? " (outputs change between runs without a reset)" : " (no run changed its output)", r);
    }

    {
        const test_case tc = make_chain_case();
        graph_runner g(tc);
        ggml_backend_t chain[] = { gpu, cpu_lns, cpu };
        ggml_backend_t plain[] = { cpu_lns, cpu };
        ggml_backend_sched_t s_chain = ggml_backend_sched_new(chain, nullptr, 3, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_t s_plain = ggml_backend_sched_new(plain, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        const uint16_t seed = monte ? k_seed_alt : 0;
        if (monte) g_monte.set(seed);
        const std::vector<float> ref = g.run(s_plain);
        const uint16_t ref_state = monte ? g_monte.state() : 0;
        if (monte) g_monte.set(seed);
        const std::vector<float> got = g.run(s_chain);
        const uint16_t got_state = monte ? g_monte.state() : 0;
        int on_gpu = 0, on_cpu_lns = 0, gpu_reductions = 0, cpu_reductions = 0;
        for (int i = 0; i < ggml_graph_n_nodes(g.graph); i++) {
            ggml_tensor * node = ggml_graph_node(g.graph, i);
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(s_chain, node);
            const bool reduction = node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID ||
                                   node->op == GGML_OP_SOFT_MAX || node->op == GGML_OP_RMS_NORM;
            on_gpu += b == gpu;
            on_cpu_lns += b == cpu_lns;
            gpu_reductions += reduction && b == gpu;
            cpu_reductions += reduction && b == cpu_lns;
        }
        std::printf("  (%d of %d nodes ran on %s, %d on %s; reductions: %d GPU, %d CPU fallback)\n", on_gpu,
                    ggml_graph_n_nodes(g.graph), gpu_name, on_cpu_lns, cpu_name, gpu_reductions, cpu_reductions);
        grade(tc.name.c_str(), compare(ref, got), exact, r);
        check("chain: GPU and CPU-fallback reductions both present", gpu_reductions > 0 && cpu_reductions > 0, "", r);
        if (monte) {
            const uint16_t want = g_monte.advanced(seed, expected_draws(g.graph));
            char detail[96];
            std::snprintf(detail, sizeof(detail), " (states: all-CPU %04x, mixed %04x, counted %04x)", ref_state,
                          got_state, want);
            check("chain: one Monte sequence across GPU and CPU fallback", ref_state == got_state && got_state == want,
                  detail, r);
        }
        ggml_backend_sched_free(s_chain);
        ggml_backend_sched_free(s_plain);
    }

    if (monte) {
        // explicit-state contract of the owner API
        const bool rejects_zero = !g_monte.set(0);
        g_monte.set(k_seed);
        uint16_t start = 0;
        g_monte.reserve(65535, &start);
        const bool wraps = start == k_seed && g_monte.state() == k_seed; // period 65535
        g_monte.reserve(1, &start);
        const bool steps = g_monte.state() == 0xB403;                     // 7 -> 0xB403
        check("Monte state API: rejects seed 0, period wrap, one step", rejects_zero && wraps && steps, "", r);
        g_monte.set(k_seed);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(gpu2);
    ggml_backend_free(cpu_lns);
    ggml_backend_free(cpu);
}

// "sum=<mode>" in a device description; "" when absent (LNS32 / LNSCUDA32)
static std::string sum_mode_of(const char * desc) {
    const char * p = desc ? std::strstr(desc, "sum=") : nullptr;
    if (p == nullptr) {
        return "";
    }
    p += 4;
    size_t n = std::strcspn(p, ",;) ");
    return std::string(p, n);
}

static std::string monte_impl_of(const char * desc) {
    const char * p = desc ? std::strstr(desc, "monte=") : nullptr;
    if (p == nullptr) {
        return "";
    }
    p += 6;
    return std::string(p, std::strcspn(p, ",;) "));
}

int main(int argc, char ** argv) {
    std::string only, expect_sum;
    for (int i = 1; i < argc; i++) {
        if (std::strncmp(argv[i], "--only=", 7) == 0) {
            only = argv[i] + 7;
        } else if (std::strncmp(argv[i], "--expect-sum=", 13) == 0) {
            expect_sum = argv[i] + 13;
        } else {
            std::fprintf(stderr, "usage: %s [--only=cuda16|cuda32] [--expect-sum=base|monte|lpvip|sum32]\n", argv[0]);
            return 1;
        }
    }
    if (!only.empty() && only != "cuda16" && only != "cuda32") {
        std::fprintf(stderr, "ERROR: --only must be cuda16 or cuda32\n");
        return 1;
    }

    // Every LNS device may claim ops; each comparison picks its backends
    // explicitly. Must be set before the first ggml call (the selection is
    // cached on first use).
    setenv("GGML_LNS_BACKEND", "LNSCUDA16,LNSCUDA32,LNS16,LNS32", 1);
    ggml_backend_load_all();

    if (ggml_backend_dev_by_name("LNSCUDA16") == nullptr) {
        std::printf("SKIP: no LNSCUDA device (no CUDA GPU, or built without -DGGML_LNSCUDA=ON)\n");
        return 77;
    }
    load_monte_api();

    pair_result r;
    if (only.empty() || only == "cuda16") {
        ggml_backend_dev_t lns16 = ggml_backend_dev_by_name("LNS16");
        ggml_backend_dev_t gpu16 = ggml_backend_dev_by_name("LNSCUDA16");
        const char * cpu_desc = lns16 ? ggml_backend_dev_description(lns16) : nullptr;
        const char * gpu_desc = ggml_backend_dev_description(gpu16);
        const std::string cpu_mode = sum_mode_of(cpu_desc), gpu_mode = sum_mode_of(gpu_desc);
        const std::string cpu_impl = monte_impl_of(cpu_desc), gpu_impl = monte_impl_of(gpu_desc);
        std::printf("LNSCUDA16: %s\nLNS16:     %s\n", gpu_desc, cpu_desc ? cpu_desc : "(missing)");
        if (lns16 == nullptr || cpu_mode.empty() || gpu_mode.empty()) {
            std::printf("ERROR: cannot read the summation modes of LNS16 / LNSCUDA16\n");
            return 1;
        }
        if (cpu_mode != gpu_mode || cpu_impl != gpu_impl) {
            std::printf("ERROR: LNSCUDA16 runs sum=%s%s%s but LNS16 runs sum=%s%s%s: no parity reference in this "
                        "build (LNS16 and LNSCUDA16 come from different builds; rebuild both with one GGML_LNS16_SUM)\n",
                        gpu_mode.c_str(), gpu_impl.empty() ? "" : ", monte=", gpu_impl.c_str(), cpu_mode.c_str(),
                        cpu_impl.empty() ? "" : ", monte=", cpu_impl.c_str());
            return 1;
        }
        if (!expect_sum.empty() && expect_sum != gpu_mode) {
            std::printf("ERROR: expected a GGML_LNS16_SUM=%s build, but this binary runs sum=%s\n",
                        expect_sum.c_str(), gpu_mode.c_str());
            return 1;
        }
        const bool monte = gpu_mode == "monte";
        if (monte != g_monte.active) {
            std::printf("ERROR: sum=%s, but the LNS16 Monte state API is %s\n", gpu_mode.c_str(),
                        g_monte.active ? "active" : "unavailable");
            return 1;
        }
        run_pair("LNSCUDA16", "LNS16", /*exact=*/true, monte, r);
    }
    if (only.empty() || only == "cuda32") {
        run_pair("LNSCUDA32", "LNS32", /*exact=*/false, /*monte=*/false, r);
    }

    std::printf("\n%s: %d of %d checks passed\n", r.failed ? "FAIL" : "PASS", r.run - r.failed, r.run);
    return r.failed ? 1 : 0;
}
