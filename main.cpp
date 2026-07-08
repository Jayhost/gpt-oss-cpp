#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>

struct SamplerArgs {
    float temp = 1.0f;
    int top_k = 0;       // 0 = disabled
    float top_p = 1.0f;  // 1.0 = disabled
    float repeat_penalty = 1.0f;
};

int sample(float* logits, int vocab_size, const SamplerArgs& sa,
           const std::vector<int>& recent_tokens, float rng_val) {
    if (sa.repeat_penalty != 1.0f && !recent_tokens.empty()) {
        for (int t : recent_tokens) {
            if (t >= 0 && t < vocab_size) {
                if (logits[t] > 0) logits[t] /= sa.repeat_penalty;
                else logits[t] *= sa.repeat_penalty;
            }
        }
    }

    if (sa.temp != 1.0f) {
        for (int i = 0; i < vocab_size; i++) logits[i] /= sa.temp;
    }

    if (sa.top_k > 0 && sa.top_k < vocab_size) {
        std::vector<float> tmp(logits, logits + vocab_size);
        std::nth_element(tmp.begin(), tmp.begin() + vocab_size - sa.top_k, tmp.end());
        float threshold = tmp[vocab_size - sa.top_k];
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < threshold) logits[i] = -1e30f;
    }

    float mx = logits[0];
    for (int i = 1; i < vocab_size; i++) mx = std::max(mx, logits[i]);
    float sum = 0;
    std::vector<float> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - mx);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) probs[i] /= sum;

    if (sa.top_p < 1.0f) {
        std::vector<int> idx(vocab_size);
        for (int i = 0; i < vocab_size; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });
        float cumsum = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[idx[i]];
            if (cumsum > sa.top_p) {
                for (int j = i + 1; j < vocab_size; j++) probs[idx[j]] = 0;
                break;
            }
        }
        sum = 0;
        for (int i = 0; i < vocab_size; i++) sum += probs[i];
        for (int i = 0; i < vocab_size; i++) probs[i] /= sum;
    }

    float r = rng_val;
    float acc = 0;
    for (int i = 0; i < vocab_size; i++) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return vocab_size - 1;
}

uint64_t rng_state = 12345;
float rng_next() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 11) & 0xFFFFFF) / (float)0x1000000;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> \"prompt\" [max_tokens=128] [temp=1.0] [top_k=0] [top_p=1.0]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string raw_prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 128;
    SamplerArgs sa;
    if (argc > 4) sa.temp = atof(argv[4]);
    if (argc > 5) sa.top_k = atoi(argv[5]);
    if (argc > 6) sa.top_p = atof(argv[6]);

    GGUFFile gf;
    if (!gf.load(model_path)) return 1;
    printf("GGUF loaded: %zu tensors, %zu KV pairs\n", gf.tensors().size(), gf.kv().size());

    Model model;
    model.load(gf);

    Tokenizer tok;
    if (!tok.load(gf)) return 1;

    Inference infer;
    infer.init(&model);

    std::string prompt = "<|im_start|>user\n" + raw_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    
    std::vector<int> tokens = tok.encode(prompt);
    printf("Prompt tokens: %zu\n", tokens.size());
    printf("Tokens: ");
    for (int t : tokens) printf("%d ", t);
    printf("\n");

    std::vector<float> logits(model.cfg.vocab_size);

    auto t0 = std::chrono::steady_clock::now();
    int pos = 0;
    for (int t : tokens) {
        infer.forward(t, pos, logits.data());
        pos++;
    }
    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Prefill: %d tokens in %.0f ms (%.1f tok/s)\n",
           (int)tokens.size(), prefill_ms, tokens.size() / (prefill_ms/1000));

    // --- DEBUG DUMP ---
    FILE* f = fopen("logits_cpp.bin", "wb");
    if (f) {
        fwrite(logits.data(), sizeof(float), model.cfg.vocab_size, f);
        fclose(f);
        printf("Dumped %d logits to logits_cpp.bin\n", model.cfg.vocab_size);
    }
    // ------------------

    printf("\n--- Output ---\n");
    std::vector<int> generated;
    std::vector<int> recent;

    for (int i = 0; i < max_tokens; i++) {
        int next = sample(logits.data(), model.cfg.vocab_size, sa, recent, rng_next());

        if (next == model.cfg.eos_id) {
            printf("<EOS>\n");
            break;
        }

        std::string piece = tok.decode({next});
        printf("%s", piece.c_str());
        fflush(stdout);

        generated.push_back(next);
        recent.push_back(next);
        if (recent.size() > 64) recent.erase(recent.begin());

        infer.forward(next, pos, logits.data());
        pos++;
    }
    printf("\n");

    auto t2 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("\nGeneration: %d tokens in %.0f ms (%.1f tok/s)\n",
           (int)generated.size(), gen_ms, generated.size() / (gen_ms/1000));

    return 0;
}