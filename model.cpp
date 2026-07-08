#include "model.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400));
            mant &= 0x3ff;
            f = (sign<<31) | ((uint32_t)(127-15-e)<<23) | (mant<<13);
        }
    } else if (exp == 31) {
        f = (sign<<31) | (0xff<<23) | (mant<<13);
    } else {
        f = (sign<<31) | ((exp+127-15)<<23) | (mant<<13);
    }
    float r; memcpy(&r, &f, 4); return r;
}

static void rmsnorm(float* x, const float* w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i]*x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * ss * w[i];
}

static void matvec(const GGUFTensor* A, const float* B, float* C) {
    int K = A->ne[0], M = A->ne[1];
    if (A->type == GT_F32) {
        const float* a = (const float*)A->data;
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float s = 0;
            const float* row = a + (size_t)m*K;
            for (int k = 0; k < K; k++) s += row[k]*B[k];
            C[m] = s;
        }
    } else if (A->type == GT_Q4_0) {
        const block_q4_0* a = (const block_q4_0*)A->data;
        int nblk = K / 32;
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float s = 0;
            const block_q4_0* blk = a + (size_t)m*nblk;
            for (int b = 0; b < nblk; b++) {
                float d = fp16_to_fp32(blk[b].d);
                for (int i = 0; i < 16; i++) {
                    int q0 = (blk[b].qs[i] & 0xF) - 8;
                    int q1 = (blk[b].qs[i] >> 4) - 8;
                    s += d * q0 * B[b*32 + i];
                    s += d * q1 * B[b*32 + i + 16];
                }
            }
            C[m] = s;
        }
    } else if (A->type == GT_F16) {
        const uint16_t* a = (const uint16_t*)A->data;
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; m++) {
            float s = 0;
            const uint16_t* row = a + (size_t)m*K;
            for (int k = 0; k < K; k++) s += fp16_to_fp32(row[k])*B[k];
            C[m] = s;
        }
    }
}

static void matvec_expert(const GGUFTensor* A, int expert, int M_per_expert, const float* B, float* C) {
    int K = A->ne[0];
    size_t expert_stride;
    if (A->type == GT_Q4_0) expert_stride = (size_t)M_per_expert * (K/32) * sizeof(block_q4_0);
    else if (A->type == GT_F16) expert_stride = (size_t)M_per_expert * K * 2;
    else expert_stride = (size_t)M_per_expert * K * 4;

    GGUFTensor slice = *A;
    slice.ne = {A->ne[0], (uint64_t)M_per_expert};
    slice.data = A->data + expert * expert_stride;
    matvec(&slice, B, C);
}

void Model::load(GGUFFile& gf) {
    auto* arch = gf.get("general.architecture");
    cfg.arch = arch ? arch->s : "gpt-oss";
    std::string p = cfg.arch + ".";

    auto get = [&](const std::string& k, int def=0) -> int {
        auto* v = gf.get(k); return v ? v->i32() : def;
    };
    auto getf = [&](const std::string& k, float def=0) -> float {
        auto* v = gf.get(k); return v ? v->f32() : def;
    };
    auto getb = [&](const std::string& k, bool def=false) -> bool {
        auto* v = gf.get(k); return v ? v->b() : def;
    };

    cfg.n_layers   = get(p+"block_count");
    cfg.n_ctx      = 4096; // Hardcode to 4096 to prevent OOM on KV cache init
    cfg.n_embd     = get(p+"embedding_length");
    cfg.n_ff       = get(p+"feed_forward_length");
    cfg.n_head     = get(p+"attention.head_count");
    cfg.n_head_kv  = get(p+"attention.head_count_kv", cfg.n_head);
    cfg.head_dim   = get(p+"attention.key_length", 64);
    cfg.n_experts  = get(p+"expert_count", 0);
    cfg.n_experts_per_tok = get(p+"expert_used_count", 0);
    
    cfg.rope_freq_base = getf(p+"rope.freq_base", 150000.f); 
    cfg.rms_eps    = getf(p+"attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.sliding_window = get(p+"attention.sliding_window", 0);
    cfg.vocab_size = get(p+"vocab_size", 0);
    cfg.bos_id     = get("tokenizer.ggml.bos_token_id", 0);
    cfg.eos_id     = get("tokenizer.ggml.eos_token_id", 0);
    cfg.add_bos    = getb("tokenizer.ggml.add_bos_token", false);

    // FIX: Read YaRN scaling parameters from GGUF metadata
    cfg.rope_scaling_factor    = getf(p+"rope.scaling.factor", 32.0f);
    cfg.rope_ntk_alpha         = getf(p+"rope.ntk_alpha", 1.0f);
    cfg.rope_ntk_beta          = getf(p+"rope.ntk_beta", 32.0f);
    cfg.initial_context_length = get(p+"rope.scaling.original_context_length", 4096);

    token_embd  = gf.tensor("token_embd.weight");
    output_norm = gf.tensor("output_norm.weight");
    output      = gf.tensor("output.weight");

    if (cfg.n_layers == 0) {
        int max_blk = -1;
        for (const auto& t : gf.tensors()) {
            if (t.name.rfind("blk.", 0) == 0) {
                int idx = std::stoi(t.name.substr(4));
                if (idx > max_blk) max_blk = idx;
            }
        }
        cfg.n_layers = max_blk + 1;
    }
    if (cfg.n_embd == 0 && token_embd) cfg.n_embd = token_embd->ne[0];
    if (cfg.vocab_size == 0 && token_embd) cfg.vocab_size = token_embd->ne[1];

    const GGUFTensor* q_b = gf.tensor("blk.0.attn_q.bias");
    if (q_b) {
        if (cfg.n_head == 0) cfg.n_head = q_b->ne[0] / cfg.head_dim;
    }
    const GGUFTensor* k_b = gf.tensor("blk.0.attn_k.bias");
    if (k_b) {
        if (cfg.n_head_kv == 0) cfg.n_head_kv = k_b->ne[0] / cfg.head_dim;
    }
    const GGUFTensor* gate_inp = gf.tensor("blk.0.ffn_gate_inp.weight");
    if (gate_inp) {
        if (cfg.n_experts == 0) cfg.n_experts = gate_inp->ne[1];
        if (cfg.n_experts_per_tok == 0) cfg.n_experts_per_tok = cfg.n_experts;
    }
    const GGUFTensor* gate_exps = gf.tensor("blk.0.ffn_gate_exps.weight");
    if (gate_exps && cfg.n_ff == 0) cfg.n_ff = gate_exps->ne[1];

    if (cfg.n_ctx <= 0) cfg.n_ctx = 4096;
    if (cfg.n_embd == 0) cfg.n_embd = 2880;
    if (cfg.n_ff == 0) cfg.n_ff = 2880;
    if (cfg.n_layers == 0) cfg.n_layers = 24;
    if (cfg.head_dim == 0) cfg.head_dim = 64;
    if (cfg.n_head == 0) cfg.n_head = 64;
    if (cfg.n_head_kv == 0) cfg.n_head_kv = 8;
    if (cfg.n_experts == 0) cfg.n_experts = 4;
    if (cfg.n_experts_per_tok == 0) cfg.n_experts_per_tok = 4;

    layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        auto& L = layers[i];
        std::string b = "blk." + std::to_string(i) + ".";
        L.attn_norm = gf.tensor(b+"attn_norm.weight");
        L.post_attn_norm = gf.tensor(b+"post_attention_norm.weight");
        if (!L.post_attn_norm) L.post_attn_norm = gf.tensor(b+"ffn_norm.weight"); 
        
        L.q = gf.tensor(b+"attn_q.weight");
        L.k = gf.tensor(b+"attn_k.weight");
        L.v = gf.tensor(b+"attn_v.weight");
        L.o = gf.tensor(b+"attn_output.weight");
        
        L.q_bias = gf.tensor(b+"attn_q.bias");
        L.k_bias = gf.tensor(b+"attn_k.bias");
        L.v_bias = gf.tensor(b+"attn_v.bias");
        L.o_bias = gf.tensor(b+"attn_output.bias");
        
        L.attn_sinks = gf.tensor(b+"attn_sinks.weight");
        
        L.ffn_gate_inp = gf.tensor(b+"ffn_gate_inp.weight");
        L.ffn_gate_inp_bias = gf.tensor(b+"ffn_gate_inp.bias");
        L.gate_exps = gf.tensor(b+"ffn_gate_exps.weight");
        L.up_exps   = gf.tensor(b+"ffn_up_exps.weight");
        L.down_exps = gf.tensor(b+"ffn_down_exps.weight");
        L.gate_exps_bias = gf.tensor(b+"ffn_gate_exps.bias");
        L.up_exps_bias   = gf.tensor(b+"ffn_up_exps.bias");
        L.down_exps_bias = gf.tensor(b+"ffn_down_exps.bias");
    }

    is_moe = (cfg.n_experts > 0);
    
    printf("Model: %s | layers=%d embd=%d ff=%d heads=%d kv=%d hd=%d",
           cfg.arch.c_str(), cfg.n_layers, cfg.n_embd, cfg.n_ff,
           cfg.n_head, cfg.n_head_kv, cfg.head_dim);
    if (is_moe) printf(" experts=%d/%d", cfg.n_experts_per_tok, cfg.n_experts);
    printf(" vocab=%d ctx=%d\n", cfg.vocab_size, cfg.n_ctx);
    printf("  rope_base=%.0f sliding=%d\n",
           cfg.rope_freq_base, cfg.sliding_window);
}

void Inference::init(Model* m) {
    model = m;
    kv.resize(m->cfg.n_layers);
    for (int i = 0; i < m->cfg.n_layers; i++)
        kv[i].init(m->cfg.n_head_kv, m->cfg.head_dim, m->cfg.n_ctx);

    int hd = m->cfg.head_dim;
    int half = hd / 2;
    float d_half = (float)hd / 2.0f;
    
    // YaRN RoPE Computation
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; i++) {
        inv_freq[i] = 1.0f / powf(m->cfg.rope_freq_base, (float)(2*i) / hd);
    }

    float concentration = 1.0f;
    if (m->cfg.rope_scaling_factor > 1.0f) {
        concentration = 0.1f * logf(m->cfg.rope_scaling_factor) + 1.0f;
        float low = d_half * logf((float)m->cfg.initial_context_length / (m->cfg.rope_ntk_beta * 2.0f * (float)M_PI)) / logf(m->cfg.rope_freq_base);
        float high = d_half * logf((float)m->cfg.initial_context_length / (m->cfg.rope_ntk_alpha * 2.0f * (float)M_PI)) / logf(m->cfg.rope_freq_base);
        
        for (int i = 0; i < half; i++) {
            float freq_extra = 1.0f / powf(m->cfg.rope_freq_base, (float)(2*i) / hd);
            float freq_inter = 1.0f / (m->cfg.rope_scaling_factor * powf(m->cfg.rope_freq_base, (float)(2*i) / hd));
            float ramp = ((float)i - low) / (high - low);
            float mask = 1.0f - std::max(0.0f, std::min(1.0f, ramp));
            inv_freq[i] = freq_inter * (1.0f - mask) + freq_extra * mask;
        }
    }

    cos_table.resize((size_t)m->cfg.n_ctx * half);
    sin_table.resize((size_t)m->cfg.n_ctx * half);
    for (int pos = 0; pos < m->cfg.n_ctx; pos++) {
        for (int i = 0; i < half; i++) {
            float angle = pos * inv_freq[i];
            cos_table[(size_t)pos*half + i] = cosf(angle) * concentration;
            sin_table[(size_t)pos*half + i] = sinf(angle) * concentration;
        }
    }
}

static void apply_rope(float* x, const float* cos_t, const float* sin_t, int hd) {
    int half = hd / 2;
    // FIX: GPT-OSS uses Half-split RoPE (GPT-NeoX style)
    for (int i = 0; i < half; i++) {
        float x1 = x[i], x2 = x[i + half];
        x[i]       = x1 * cos_t[i] - x2 * sin_t[i];
        x[i+half]  = x2 * cos_t[i] + x1 * sin_t[i];
    }
}

void Inference::forward(int token, int pos, float* logits) {
    auto& cfg = model->cfg;
    int hd = cfg.head_dim;
    int half = hd / 2;

    auto add_bias = [](float* c, const GGUFTensor* b, int M) {
        if (b) {
            float* bd = (float*)b->data;
            for (int m = 0; m < M; m++) c[m] += bd[m];
        }
    };
    auto add_expert_bias = [](float* c, const GGUFTensor* b, int M, int e) {
        if (b) {
            float* bd = (float*)b->data + (size_t)e * M;
            for (int m = 0; m < M; m++) c[m] += bd[m];
        }
    };

    std::vector<float> h(cfg.n_embd);
    if (model->token_embd->type == GT_F32) {
        const float* emb = (const float*)model->token_embd->data;
        memcpy(h.data(), emb + (size_t)token*cfg.n_embd, cfg.n_embd*4);
    } else if (model->token_embd->type == GT_Q4_0) {
        const block_q4_0* blk = (const block_q4_0*)model->token_embd->data;
        int nblk = cfg.n_embd / 32;
        const block_q4_0* b = blk + (size_t)token * nblk;
        for (int i = 0; i < nblk; i++) {
            float d = fp16_to_fp32(b[i].d);
            for (int j = 0; j < 16; j++) {
                h[i*32 + j]      = d * ((b[i].qs[j] & 0xF) - 8);
                h[i*32 + j + 16] = d * ((b[i].qs[j] >> 4) - 8);
            }
        }
    }

    std::vector<float> qv(cfg.n_head * hd);
    std::vector<float> kv_(cfg.n_head_kv * hd);
    std::vector<float> vv(cfg.n_head_kv * hd);
    std::vector<float> attn_out(cfg.n_head * hd);
    std::vector<float> tmp;

    float scale = 1.0f / sqrtf((float)hd);

    for (int li = 0; li < cfg.n_layers; li++) {
        auto& L = model->layers[li];

        std::vector<float> normed(h);
        rmsnorm(normed.data(), (const float*)L.attn_norm->data, cfg.n_embd, cfg.rms_eps);

        matvec(L.q, normed.data(), qv.data());
        add_bias(qv.data(), L.q_bias, cfg.n_head * hd);
        matvec(L.k, normed.data(), kv_.data());
        add_bias(kv_.data(), L.k_bias, cfg.n_head_kv * hd);
        matvec(L.v, normed.data(), vv.data());
        add_bias(vv.data(), L.v_bias, cfg.n_head_kv * hd);

        for (int hh = 0; hh < cfg.n_head; hh++) {
            float* qh = qv.data() + hh*hd;
            apply_rope(qh, &cos_table[(size_t)pos*half], &sin_table[(size_t)pos*half], hd);
        }
        for (int hh = 0; hh < cfg.n_head_kv; hh++) {
            float* kh = kv_.data() + hh*hd;
            apply_rope(kh, &cos_table[(size_t)pos*half], &sin_table[(size_t)pos*half], hd);
        }

        memcpy(kv[li].kptr(pos), kv_.data(), cfg.n_head_kv*hd*4);
        memcpy(kv[li].vptr(pos), vv.data(), cfg.n_head_kv*hd*4);

        // FIX: Sliding window applied to EVEN layers (0, 2, 4...)
        bool is_sliding = (cfg.sliding_window > 0) && (li % 2 == 0);

        #pragma omp parallel for schedule(static)
        for (int hh = 0; hh < cfg.n_head; hh++) {
            int kvh = hh * cfg.n_head_kv / cfg.n_head;
            float* qh = qv.data() + hh*hd;

            int start = 0;
            if (is_sliding) start = std::max(0, pos - cfg.sliding_window + 1);
            
            // 1 extra slot for the attention sink
            std::vector<float> scores(pos - start + 2);
            for (int p2 = start; p2 <= pos; p2++) {
                float* kp = kv[li].kptr(p2) + kvh*hd;
                float dot = 0;
                for (int d = 0; d < hd; d++) dot += qh[d]*kp[d];
                dot *= scale;
                scores[p2 - start] = dot;
            }
            
            // Attention sink is added as an extra key column with constant logit
            scores[pos - start + 1] = L.attn_sinks ? ((float*)L.attn_sinks->data)[hh] : 0.0f;

            float mx = scores[0];
            for (auto s : scores) mx = std::max(mx, s);
            float sum = 0;
            for (auto& s : scores) { s = expf(s - mx); sum += s; }
            for (auto& s : scores) s /= sum;

            float* out_h = attn_out.data() + hh*hd;
            for (int d = 0; d < hd; d++) out_h[d] = 0;
            // Skip the last score (the sink weight) when accumulating V
            for (int p2 = start; p2 <= pos; p2++) {
                float w = scores[p2 - start];
                float* vp = kv[li].vptr(p2) + kvh*hd;
                for (int d = 0; d < hd; d++) out_h[d] += w * vp[d];
            }
        }

        tmp.resize(cfg.n_embd);
        matvec(L.o, attn_out.data(), tmp.data());
        add_bias(tmp.data(), L.o_bias, cfg.n_embd);
        for (int i = 0; i < cfg.n_embd; i++) h[i] += tmp[i];

        std::vector<float> ffn_normed(h);
        rmsnorm(ffn_normed.data(), (const float*)L.post_attn_norm->data, cfg.n_embd, cfg.rms_eps);

        std::vector<float> router_logits(cfg.n_experts);
        matvec(L.ffn_gate_inp, ffn_normed.data(), router_logits.data());
        add_bias(router_logits.data(), L.ffn_gate_inp_bias, cfg.n_experts);

        int topk = cfg.n_experts_per_tok;
        std::vector<int> idx(cfg.n_experts);
        for (int i = 0; i < cfg.n_experts; i++) idx[i] = i;
        
        // Top-k FIRST, then softmax ONLY over top-k
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
            [&](int a, int b){ return router_logits[a] > router_logits[b]; });

        std::vector<float> wts(topk);
        float rmx = router_logits[idx[0]];
        for (int i = 1; i < topk; i++) rmx = std::max(rmx, router_logits[idx[i]]);
        float rsum = 0;
        for (int i = 0; i < topk; i++) {
            wts[i] = expf(router_logits[idx[i]] - rmx);
            rsum += wts[i];
        }
        for (int i = 0; i < topk; i++) wts[i] /= rsum;

        std::vector<float> ffn_out(cfg.n_embd, 0);
        std::vector<float> gate_out(cfg.n_ff), up_out(cfg.n_ff), inter(cfg.n_ff), down_out(cfg.n_embd);
        for (int i = 0; i < topk; i++) {
            int e = idx[i];
            matvec_expert(L.gate_exps, e, cfg.n_ff, ffn_normed.data(), gate_out.data());
            add_expert_bias(gate_out.data(), L.gate_exps_bias, cfg.n_ff, e);
            matvec_expert(L.up_exps, e, cfg.n_ff, ffn_normed.data(), up_out.data());
            add_expert_bias(up_out.data(), L.up_exps_bias, cfg.n_ff, e);
            
            // FIX: Correct SwiGLU_OAI logic: clamp inputs, alpha=1.702 for sigmoid, +1 bias to linear branch
            for (int j = 0; j < cfg.n_ff; j++) {
                float x_glu = std::min(gate_out[j], 7.0f);
                float x_lin = std::max(-7.0f, std::min(up_out[j], 7.0f));
                float out_glu = x_glu / (1.0f + expf(-1.702f * x_glu));
                inter[j] = out_glu * (x_lin + 1.0f);
            }
                
            matvec_expert(L.down_exps, e, cfg.n_embd, inter.data(), down_out.data());
            add_expert_bias(down_out.data(), L.down_exps_bias, cfg.n_embd, e);
            for (int j = 0; j < cfg.n_embd; j++) ffn_out[j] += wts[i] * down_out[j];
        }
        for (int i = 0; i < cfg.n_embd; i++) h[i] += ffn_out[i];
    }

    rmsnorm(h.data(), (const float*)model->output_norm->data, cfg.n_embd, cfg.rms_eps);

    const GGUFTensor* lm = model->output ? model->output : model->token_embd;
    matvec(lm, h.data(), logits);
}