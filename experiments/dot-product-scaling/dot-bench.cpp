#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

typedef bool (*monte_set_state_fn)(uint16_t);

static ggml_backend_t g_backend = nullptr;
static monte_set_state_fn g_monte_set = nullptr;

static std::vector<float> run_graph(const std::vector<float> & a, int64_t K, int64_t M, const std::vector<float> * x,
                                    int64_t N) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * tx = x ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N) : nullptr;
    ggml_tensor * out = x ? ggml_mul_mat(ctx, ta, tx) : ggml_scale(ctx, ta, 1.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    ggml_backend_tensor_set(ta, a.data(), 0, a.size() * sizeof(float));
    if (x) ggml_backend_tensor_set(tx, x->data(), 0, x->size() * sizeof(float));
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        std::exit(1);
    }
    std::vector<float> r((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, r.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

static std::vector<float> lns_round(const std::vector<float> & v) {
    return run_graph(v, (int64_t) v.size(), 1, nullptr, 0);
}

static std::vector<float> lns_matmul(const std::vector<float> & a, const std::vector<float> & x, int64_t K, int64_t M,
                                     int64_t N) {
    return run_graph(a, K, M, &x, N);
}

struct exact_out {
    std::vector<long double> s, s1, s2;
};

static exact_out exact_matmul(const std::vector<float> & a, const std::vector<float> & x, int64_t K, int64_t M,
                              int64_t N) {
    exact_out e;
    e.s.resize(M * N);
    e.s1.resize(M * N);
    e.s2.resize(M * N);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t m = 0; m < M; m++) {
            long double s = 0, s1 = 0, s2 = 0;
            const float * ar = &a[m * K];
            const float * xr = &x[n * K];
            for (int64_t k = 0; k < K; k++) {
                const long double p = (long double) ar[k] * (long double) xr[k];
                s += p;
                s1 += fabsl(p);
                s2 += p * p;
            }
            e.s[n * M + m] = s;
            e.s1[n * M + m] = s1;
            e.s2[n * M + m] = sqrtl(s2);
        }
    }
    return e;
}

struct err_stats {
    double n = 0, sum = 0, sumsq = 0, abs_sum = 0, num = 0, den = 0;
    void add(long double got, long double ref, long double scale) {
        const double e = scale > 0 ? (double) ((got - ref) / scale) : 0.0;
        n++;
        sum += e;
        sumsq += e * e;
        abs_sum += std::fabs(e);
        num += (double) ((got - ref) * (got - ref));
        den += (double) (ref * ref);
    }
    double mean() const { return sum / n; }
    double rms() const { return std::sqrt(sumsq / n); }
    double sd() const { return std::sqrt(std::max(0.0, sumsq / n - mean() * mean())); }
    double rel_l2() const { return den > 0 ? std::sqrt(num / den) : 0.0; }
};

static void print_stats(FILE * f, const char * key, const err_stats & s) {
    std::fprintf(f, "\"%s\":{\"bias\":%.6g,\"sd\":%.6g,\"rms\":%.6g,\"mean_abs\":%.6g,\"rel_l2\":%.6g}", key, s.mean(),
                 s.sd(), s.rms(), s.abs_sum / s.n, s.rel_l2());
}

static void gen_inputs(const std::string & dist, int64_t K, int64_t M, int64_t N, std::mt19937_64 & rng,
                       std::vector<float> & a, std::vector<float> & x) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    a.resize(K * M);
    x.resize(K * N);
    for (auto & v : a) v = g(rng);
    if (dist == "signed") {
        for (auto & v : x) v = g(rng);
    } else if (dist == "positive") {
        for (auto & v : a) v = std::fabs(v);
        for (auto & v : x) v = std::fabs(g(rng));
    } else if (dist == "attn") {
        for (int64_t n = 0; n < N; n++) {
            float mx = -1e30f;
            for (int64_t k = 0; k < K; k++) {
                x[n * K + k] = 2.0f * g(rng);
                mx = std::max(mx, x[n * K + k]);
            }
            double s = 0;
            for (int64_t k = 0; k < K; k++) {
                x[n * K + k] = std::exp(x[n * K + k] - mx);
                s += x[n * K + k];
            }
            for (int64_t k = 0; k < K; k++) x[n * K + k] = (float) (x[n * K + k] / s);
        }
    } else {
        std::fprintf(stderr, "unknown dist %s\n", dist.c_str());
        std::exit(1);
    }
}

static int synth(const std::string & dist, int64_t kmin, int64_t kmax, int64_t outputs, int reps, uint64_t seed) {
    const int64_t N = 4;
    for (int64_t K = kmin; K <= kmax; K *= 2) {
        const int64_t M = std::max<int64_t>(1, outputs / N);
        std::mt19937_64 rng(seed + (uint64_t) K);
        std::vector<float> a0, x0;
        gen_inputs(dist, K, M, N, rng, a0, x0);
        const std::vector<float> a = lns_round(a0), x = lns_round(x0);
        const exact_out ex = exact_matmul(a, x, K, M, N);
        const exact_out ex0 = exact_matmul(a0, x0, K, M, N);
        std::vector<std::vector<float>> outs;
        for (int r = 0; r < reps; r++) {
            if (g_monte_set) g_monte_set((uint16_t) (1 + (seed * 7919 + (uint64_t) r * 104729 + (uint64_t) K) % 65535));
            outs.push_back(lns_matmul(a0, x0, K, M, N));
        }
        err_stats acc, acc2, total;
        for (int64_t i = 0; i < M * N; i++) {
            acc.add(outs[0][i], ex.s[i], ex.s1[i]);
            acc2.add(outs[0][i], ex.s[i], ex.s2[i]);
            total.add(outs[0][i], ex0.s[i], ex0.s1[i]);
        }
        std::printf("{\"dist\":\"%s\",\"K\":%lld,\"outputs\":%lld,", dist.c_str(), (long long) K, (long long) (M * N));
        print_stats(stdout, "acc", acc);
        std::printf(",");
        print_stats(stdout, "acc_l2scale", acc2);
        std::printf(",");
        print_stats(stdout, "total", total);
        if (reps > 1) {
            std::printf(",\"avg\":[");
            bool first = true;
            for (int r = 1; r <= reps; r *= 2) {
                err_stats s;
                for (int64_t i = 0; i < M * N; i++) {
                    long double m = 0;
                    for (int q = 0; q < r; q++) m += outs[q][i];
                    s.add(m / r, ex.s[i], ex.s1[i]);
                }
                std::printf("%s{\"reps\":%d,\"bias\":%.6g,\"sd\":%.6g,\"rms\":%.6g}", first ? "" : ",", r, s.mean(), s.sd(), s.rms());
                first = false;
            }
            std::printf("]");
        }
        std::printf("}\n");
        std::fflush(stdout);
    }
    return 0;
}

static int convert(int64_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> e(-20.0, 20.0);
    std::vector<float> v((size_t) n);
    for (auto & x : v) x = (float) ((rng() & 1 ? -1.0 : 1.0) * std::exp2(e(rng)));
    const std::vector<float> r1 = lns_round(v), r2 = lns_round(r1), r3 = lns_round(r2);
    double mean = 0, sq = 0, lg = 0, d12 = 0, d23 = 0;
    int64_t up = 0, down = 0, same = 0, ch12 = 0, ch23 = 0, cnt = 0;
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(r1[i]) || !std::isfinite(r3[i]) || r1[i] == 0 || r2[i] == 0 || r3[i] == 0) continue;
        const double rel = ((double) r1[i] - v[i]) / v[i];
        mean += rel;
        sq += rel * rel;
        lg += std::log2(std::fabs((double) r1[i] / v[i]));
        up += std::fabs(r1[i]) > std::fabs(v[i]);
        down += std::fabs(r1[i]) < std::fabs(v[i]);
        same += r1[i] == v[i];
        ch12 += r2[i] != r1[i];
        ch23 += r3[i] != r2[i];
        d12 += std::log2(std::fabs((double) r2[i] / r1[i]));
        d23 += std::log2(std::fabs((double) r3[i] / r2[i]));
        cnt++;
    }
    std::printf("{\"values\":%lld,\"mean_rel\":%.6g,\"rms_rel\":%.6g,\"mean_log2\":%.6g,\"frac_up\":%.6g,\"frac_down\":%.6g,"
                "\"frac_exact\":%.6g,\"frac_changed_round2\":%.6g,\"frac_changed_round3\":%.6g,\"drift_log2_round2\":%.6g,"
                "\"drift_log2_round3\":%.6g}\n", (long long) cnt, mean / cnt, std::sqrt(sq / cnt), lg / cnt,
                (double) up / cnt, (double) down / cnt, (double) same / cnt, (double) ch12 / cnt, (double) ch23 / cnt,
                d12 / cnt, d23 / cnt);
    return 0;
}

struct dump_entry {
    std::string file, name, op;
    int64_t ne[4];
};

static std::vector<dump_entry> read_index(const std::string & dir) {
    std::vector<dump_entry> v;
    std::ifstream in(dir + "/index.tsv");
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        dump_entry e;
        std::string ne[4];
        std::getline(ss, e.file, '\t');
        std::getline(ss, e.name, '\t');
        std::getline(ss, e.op, '\t');
        for (int i = 0; i < 4; i++) {
            std::getline(ss, ne[i], '\t');
            e.ne[i] = std::atoll(ne[i].c_str());
        }
        v.push_back(e);
    }
    return v;
}

static const dump_entry * find_dump(const std::vector<dump_entry> & idx, const std::string & base) {
    const dump_entry * best = nullptr;
    for (const auto & e : idx) {
        const std::string n = e.name.substr(0, e.name.find('#'));
        if (n == base && e.op != "RESHAPE" && e.op != "VIEW" && e.op != "PERMUTE" && e.op != "ROPE") best = &e;
    }
    if (!best) {
        std::fprintf(stderr, "tensor %s not in dump\n", base.c_str());
        std::exit(1);
    }
    return best;
}

static std::vector<float> load_dump(const std::string & dir, const dump_entry & e) {
    std::vector<float> v((size_t) (e.ne[0] * e.ne[1] * e.ne[2] * e.ne[3]));
    FILE * f = std::fopen((dir + "/" + e.file).c_str(), "rb");
    if (!f || std::fread(v.data(), sizeof(float), v.size(), f) != v.size()) {
        std::fprintf(stderr, "cannot read %s\n", e.file.c_str());
        std::exit(1);
    }
    std::fclose(f);
    return v;
}

static std::vector<float> load_weight(gguf_context * g, ggml_context * gctx, const std::string & name, int64_t & K,
                                      int64_t & M) {
    (void) g;
    ggml_tensor * t = ggml_get_tensor(gctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "weight %s not in model\n", name.c_str());
        std::exit(1);
    }
    K = t->ne[0];
    M = t->ne[1];
    std::vector<float> w((size_t) (K * M));
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(w.data(), t->data, w.size() * sizeof(float));
    } else {
        ggml_get_type_traits(t->type)->to_float(t->data, w.data(), (int64_t) w.size());
    }
    return w;
}

static void replay_report(const char * op, int layer, const std::vector<float> & out, const exact_out & ex,
                          const exact_out & ex0, const std::vector<int64_t> & keff, bool bucket) {
    std::map<int64_t, std::pair<err_stats, err_stats>> buckets;
    for (size_t i = 0; i < out.size(); i++) {
        int64_t b = 1;
        while (bucket && b * 2 <= keff[i]) b *= 2;
        if (!bucket) b = keff[i];
        buckets[b].first.add(out[i], ex.s[i], ex.s1[i]);
        buckets[b].second.add(out[i], ex0.s[i], ex0.s1[i]);
    }
    for (auto & kv : buckets) {
        std::printf("{\"op\":\"%s\",\"layer\":%d,\"K\":%lld,\"outputs\":%.0f,", op, layer, (long long) kv.first, kv.second.first.n);
        print_stats(stdout, "acc", kv.second.first);
        std::printf(",");
        print_stats(stdout, "total", kv.second.second);
        std::printf("}\n");
    }
    std::fflush(stdout);
}

static std::vector<float> run_norm(const std::vector<float> & x, int64_t n0, int64_t n1, int64_t n2, float eps,
                                   const std::vector<float> * mask, float scale) {
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2);
    ggml_tensor * tm = mask ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1) : nullptr;
    ggml_tensor * out = mask ? ggml_soft_max_ext(ctx, tx, tm, scale, 0.0f) : ggml_rms_norm(ctx, tx, eps);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    ggml_backend_tensor_set(tx, x.data(), 0, x.size() * sizeof(float));
    if (mask) ggml_backend_tensor_set(tm, mask->data(), 0, mask->size() * sizeof(float));
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed\n");
        std::exit(1);
    }
    std::vector<float> r((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, r.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

static void print_norm(const char * op, int layer, int64_t K, const err_stats & acc, const err_stats & total) {
    std::printf("{\"op\":\"%s\",\"layer\":%d,\"K\":%lld,\"outputs\":%.0f,", op, layer, (long long) K, acc.n);
    print_stats(stdout, "acc", acc);
    std::printf(",");
    print_stats(stdout, "total", total);
    std::printf("}\n");
}

static float rms_eps(gguf_context * g) {
    const int64_t ka = gguf_find_key(g, "general.architecture");
    if (ka < 0) return 1e-6f;
    const std::string key = std::string(gguf_get_val_str(g, ka)) + ".attention.layer_norm_rms_epsilon";
    const int64_t ke = gguf_find_key(g, key.c_str());
    return ke < 0 ? 1e-6f : gguf_get_val_f32(g, ke);
}

static int replay(const std::string & model, const std::string & dir, const std::string & ops, int layer) {
    ggml_context * gctx = nullptr;
    gguf_init_params gp = { false, &gctx };
    gguf_context * g = gguf_init_from_file(model.c_str(), gp);
    if (!g) {
        std::fprintf(stderr, "cannot load %s\n", model.c_str());
        return 1;
    }
    const std::vector<dump_entry> idx = read_index(dir);
    const std::string L = std::to_string(layer);
    std::stringstream ss(ops);
    std::string op;
    while (std::getline(ss, op, ',')) {
        if (op == "rms_norm") {
            const dump_entry * xe = find_dump(idx, "ffn_inp-" + L);
            const std::vector<float> x = load_dump(dir, *xe);
            const int64_t n = xe->ne[0], rows = xe->ne[1] * xe->ne[2] * xe->ne[3];
            const float eps = rms_eps(g);
            const std::vector<float> xr = lns_round(x);
            const std::vector<float> y = run_norm(x, n, rows, 1, eps, nullptr, 0.0f);
            err_stats acc, total;
            for (int64_t r = 0; r < rows; r++) {
                long double ss = 0, ss0 = 0, num = 0, den = 0;
                for (int64_t i = 0; i < n; i++) {
                    const long double a = xr[r * n + i], a0 = x[r * n + i];
                    ss += a * a;
                    ss0 += a0 * a0;
                    num += (long double) y[r * n + i] * a;
                    den += a * a;
                }
                if (den == 0) continue;
                const long double inv = num / den;
                const long double ex = 1.0L / sqrtl(ss / n + eps), ex0 = 1.0L / sqrtl(ss0 / n + eps);
                acc.add(inv, ex, ex);
                total.add(inv, ex0, ex0);
            }
            print_norm("rms_norm", layer, n, acc, total);
            continue;
        }
        if (op == "softmax") {
            const dump_entry * ke = find_dump(idx, "kq-" + L);
            const std::vector<float> kq = load_dump(dir, *ke);
            const int64_t n_kv = ke->ne[0], n_tok = ke->ne[1], n_head = ke->ne[2];
            std::vector<float> mask((size_t) (n_kv * n_tok));
            for (int64_t t = 0; t < n_tok; t++)
                for (int64_t j = 0; j < n_kv; j++) mask[t * n_kv + j] = j <= t ? 0.0f : -INFINITY;
            const float scale = 1.0f / sqrtf(64.0f);
            const std::vector<float> p = run_norm(kq, n_kv, n_tok, n_head, 0.0f, &mask, scale);
            std::map<int64_t, std::pair<err_stats, err_stats>> buckets;
            for (int64_t h = 0; h < n_head; h++) {
                for (int64_t t = 0; t < n_tok; t++) {
                    long double sum = 0;
                    for (int64_t j = 0; j <= t; j++) sum += p[(h * n_tok + t) * n_kv + j];
                    if (sum <= 0) continue;
                    int64_t b = 1;
                    while (b * 2 <= t + 1) b *= 2;
                    buckets[b].first.add(1.0L / sum, 1.0L, 1.0L);
                    buckets[b].second.add(1.0L / sum, 1.0L, 1.0L);
                }
            }
            for (auto & kv : buckets) print_norm("softmax", layer, kv.first, kv.second.first, kv.second.second);
            continue;
        }
        if (op == "kqv") {
            const dump_entry * pe = find_dump(idx, "kq_soft_max-" + L);
            const dump_entry * ve = find_dump(idx, "Vcur-" + L);
            const std::vector<float> P = load_dump(dir, *pe);
            std::vector<float> V = load_dump(dir, *ve);
            for (auto & v : V) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            const int64_t n_kv = pe->ne[0], n_tok = pe->ne[1], n_head = pe->ne[2];
            const int64_t n_embd_v = ve->ne[0], dh = 64, n_head_kv = n_embd_v / dh;
            std::vector<float> a, x;
            const int64_t K = n_tok;
            std::vector<float> outs;
            exact_out ex, ex0;
            std::vector<int64_t> keff;
            for (int64_t h = 0; h < n_head; h++) {
                const int64_t gkv = h / (n_head / n_head_kv);
                a.assign(dh * K, 0.0f);
                x.assign(K * n_tok, 0.0f);
                for (int64_t d = 0; d < dh; d++)
                    for (int64_t j = 0; j < K; j++) a[d * K + j] = V[j * n_embd_v + gkv * dh + d];
                for (int64_t t = 0; t < n_tok; t++)
                    for (int64_t j = 0; j < K; j++) x[t * K + j] = P[(h * n_tok + t) * n_kv + j];
                const std::vector<float> ar = lns_round(a), xr = lns_round(x);
                const std::vector<float> o = lns_matmul(a, x, K, dh, n_tok);
                const exact_out e = exact_matmul(ar, xr, K, dh, n_tok), e0 = exact_matmul(a, x, K, dh, n_tok);
                outs.insert(outs.end(), o.begin(), o.end());
                ex.s.insert(ex.s.end(), e.s.begin(), e.s.end());
                ex.s1.insert(ex.s1.end(), e.s1.begin(), e.s1.end());
                ex0.s.insert(ex0.s.end(), e0.s.begin(), e0.s.end());
                ex0.s1.insert(ex0.s1.end(), e0.s1.begin(), e0.s1.end());
                for (int64_t t = 0; t < n_tok; t++)
                    for (int64_t d = 0; d < dh; d++) keff.push_back(t + 1);
            }
            replay_report("kqv", layer, outs, ex, ex0, keff, true);
            continue;
        }
        std::string wname, xname;
        if (op == "q") { wname = "blk." + L + ".attn_q.weight"; xname = "attn_norm-" + L; }
        else if (op == "ffn_up") { wname = "blk." + L + ".ffn_up.weight"; xname = "ffn_norm-" + L; }
        else if (op == "ffn_down") { wname = "blk." + L + ".ffn_down.weight"; xname = "ffn_swiglu-" + L; }
        else if (op == "attn_out") { wname = "blk." + L + ".attn_output.weight"; xname = "kqv_out-" + L; }
        else { std::fprintf(stderr, "unknown op %s\n", op.c_str()); return 1; }
        int64_t K, M;
        const std::vector<float> w = load_weight(g, gctx, wname, K, M);
        const dump_entry * xe = find_dump(idx, xname);
        const std::vector<float> x = load_dump(dir, *xe);
        const int64_t N = xe->ne[1];
        if (xe->ne[0] != K) {
            std::fprintf(stderr, "shape mismatch for %s\n", op.c_str());
            return 1;
        }
        const std::vector<float> wr = lns_round(w), xr = lns_round(x);
        const std::vector<float> o = lns_matmul(w, x, K, M, N);
        const exact_out ex = exact_matmul(wr, xr, K, M, N), ex0 = exact_matmul(w, x, K, M, N);
        replay_report(op.c_str(), layer, o, ex, ex0, std::vector<int64_t>(o.size(), K), false);
    }
    gguf_free(g);
    ggml_free(gctx);
    return 0;
}

int main(int argc, char ** argv) {
    std::map<std::string, std::string> opt = {
        { "--device", "LNSCUDA16" }, { "--dist", "signed" }, { "--kmin", "16" }, { "--kmax", "16384" },
        { "--outputs", "4096" }, { "--reps", "1" }, { "--seed", "1" }, { "--model", "" }, { "--dump", "" },
        { "--ops", "q,ffn_up,ffn_down,attn_out,kqv,rms_norm,softmax" }, { "--layer", "0" },
    };
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s synth|replay|convert [--device LNSCUDA16|LNSCUDA32|LNS16|LNS32] [--dist signed|positive|attn]\n"
                             "          [--kmin K] [--kmax K] [--outputs N] [--reps R] [--seed S]\n"
                             "          [--model m.gguf --dump DIR --layer L --ops q,ffn_up,ffn_down,attn_out,kqv,rms_norm,softmax]\n", argv[0]);
        return 1;
    }
    const std::string cmd = argv[1];
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!opt.count(argv[i])) { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
        opt[argv[i]] = argv[i + 1];
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(opt["--device"].c_str());
    if (!dev) { std::fprintf(stderr, "device %s not found\n", opt["--device"].c_str()); return 1; }
    g_backend = ggml_backend_dev_init(dev, nullptr);
    if (ggml_backend_reg_t reg = ggml_backend_reg_by_name("LNS16")) {
        g_monte_set = (monte_set_state_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_lns16_monte_set_state");
        uint16_t probe;
        auto get = (bool (*)(uint16_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_lns16_monte_get_state");
        if (!get || !get(&probe) || opt["--device"].find("32") != std::string::npos) g_monte_set = nullptr;
    }
    int rc;
    if (cmd == "synth") {
        rc = synth(opt["--dist"], std::atoll(opt["--kmin"].c_str()), std::atoll(opt["--kmax"].c_str()),
                   std::atoll(opt["--outputs"].c_str()), std::atoi(opt["--reps"].c_str()), std::strtoull(opt["--seed"].c_str(), nullptr, 10));
    } else if (cmd == "convert") {
        rc = convert(std::atoll(opt["--outputs"].c_str()), std::strtoull(opt["--seed"].c_str(), nullptr, 10));
    } else if (cmd == "replay") {
        rc = replay(opt["--model"], opt["--dump"], opt["--ops"], std::atoi(opt["--layer"].c_str()));
    } else {
        std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
        rc = 1;
    }
    ggml_backend_free(g_backend);
    return rc;
}
