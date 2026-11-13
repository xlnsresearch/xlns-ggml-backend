// layer-dump: run ONE forward pass over a prompt and save the output of every
// named F32 tensor of the graph (per layer: norms, Q/K/V, attention scores and
// softmax, attention output, FFN, residual stream; then the final norm and the
// logits), for experiments/layer-by-layer/layer-compare.py.
//
// Build one binary per llama.cpp tree (like scripts/rc-logits.cpp):
//   T=build/llama.cpp/build-lns-base
//   g++ -O2 -std=c++17 -I build/llama.cpp/include -I build/llama.cpp/ggml/include \
//       experiments/layer-by-layer/layer-dump.cpp -o /tmp/layer-dump-base \
//       -L $T/bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/$T/bin
//
// Run (set GGML_LNS_BACKEND for the LNS trees):
//   GGML_LNS_BACKEND=cuda16 /tmp/layer-dump-base model.gguf "The capital of France is" out-dir/
//
// out-dir/index.tsv lists, in graph order: file, tensor name, op, ne0..ne3.
// A name that occurs more than once gets a #k suffix by occurrence, so two
// runs of the same model line up file by file.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct dump_state {
    std::string dir;
    FILE * index = nullptr;
    std::map<std::string, int> seen;
    int count = 0;
};

static bool wanted(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->name[0] != '\0' && std::strstr(t->name, " (") == nullptr &&
           ggml_is_contiguous(t);
}

static bool dump_cb(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return wanted(t);
    }
    if (!wanted(t)) {
        return true;
    }
    auto * st = static_cast<dump_state *>(user_data);
    std::string name = t->name;
    const int k = st->seen[name]++;
    if (k > 0) {
        name += "#" + std::to_string(k);
    }
    std::string file;
    for (char c : name) file += (c == '/' || c == ' ') ? '_' : c;
    file += ".bin";

    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    FILE * f = std::fopen((st->dir + "/" + file).c_str(), "wb");
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    std::fprintf(st->index, "%s\t%s\t%s\t%lld\t%lld\t%lld\t%lld\n", file.c_str(), name.c_str(), ggml_op_desc(t),
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    st->count++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <prompt> <out-dir>\n", argv[0]);
        return 1;
    }
    dump_state st;
    st.dir   = argv[3];
    st.index = std::fopen((st.dir + "/index.tsv").c_str(), "w");
    if (st.index == nullptr) {
        std::fprintf(stderr, "cannot write %s/index.tsv (create the directory first)\n", argv[3]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = dump_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "ctx failed\n"); return 3; }

    std::vector<llama_token> toks(512);
    const int n = llama_tokenize(vocab, argv[2], (int) std::strlen(argv[2]), toks.data(), 512, true, true);
    if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 4; }
    toks.resize(n);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) {
        std::fprintf(stderr, "decode failed\n");
        return 5;
    }
    std::fclose(st.index);
    std::fprintf(stderr, "[layer-dump] %d tensors, %d tokens\n", st.count, n);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
