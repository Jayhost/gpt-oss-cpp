import gguf
import numpy as np
import struct

reader = gguf.GGUFReader("gpt-oss-20b.gguf")

def dequant_q4_0_fast(tensor):
    shape = tensor.shape
    count = np.prod(shape)
    blocks = count // 32
    
    # Read raw bytes directly to bypass gguf library structured array bugs
    raw = np.frombuffer(tensor.data.tobytes(), dtype=np.uint8).reshape(blocks, 18)
    
    # Extract d (first 2 bytes, little endian uint16)
    d_raw = raw[:, 0].astype(np.uint16) | (raw[:, 1].astype(np.uint16) << 8)
    sign = (d_raw & 0x8000).astype(np.uint32) << 16
    exponent = (d_raw & 0x7C00) >> 10
    mantissa = d_raw & 0x03FF
    f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
    f = np.where(exponent == 0, sign, f) # handle subnormals as 0
    d = f.view(np.float32)
    
    # Extract qs (next 16 bytes)
    qs = raw[:, 2:].astype(np.int16)
    q0 = (qs & 0x0F) - 8
    q1 = (qs >> 4) - 8
    
    out = np.empty((blocks, 32), dtype=np.float32)
    out[:, :16] = d[:, None] * q0
    out[:, 16:] = d[:, None] * q1
    
    np_shape = shape[::-1]
    return out.reshape(np_shape)

def dequant_f32_fast(tensor):
    shape = tensor.shape
    np_shape = shape[::-1]
    out = np.frombuffer(tensor.data.tobytes(), dtype=np.float32)
    return out.reshape(np_shape)

tensors = {t.name: t for t in reader.tensors}

# Token 27
emb = dequant_q4_0_fast(tensors["token_embd.weight"])
x = emb[27] 

w1 = dequant_f32_fast(tensors["blk.0.attn_norm.weight"]).flatten()
eps = 1e-5

def rms_norm(x, w, eps):
    ms = np.mean(x*x)
    return x * (1.0 / np.sqrt(ms + eps)) * w

xn = rms_norm(x, w1, eps)

qW = dequant_q4_0_fast(tensors["blk.0.attn_q.weight"])
qB = dequant_f32_fast(tensors["blk.0.attn_q.bias"]).flatten()
kW = dequant_q4_0_fast(tensors["blk.0.attn_k.weight"])
kB = dequant_f32_fast(tensors["blk.0.attn_k.bias"]).flatten()
vW = dequant_q4_0_fast(tensors["blk.0.attn_v.weight"])
vB = dequant_f32_fast(tensors["blk.0.attn_v.bias"]).flatten()

Q = xn @ qW.T + qB
K = xn @ kW.T + kB
V = xn @ vW.T + vB

nh, Dh, nkv = 64, 64, 8
Q = Q.reshape(nh, Dh)
K = K.reshape(nkv, Dh)
V = V.reshape(nkv, Dh)

scores = np.zeros(nh)
for h in range(nh):
    kvh = h // (nh // nkv)
    scores[h] = np.dot(Q[h], K[kvh]) / np.sqrt(Dh)

sinks = dequant_f32_fast(tensors["blk.0.attn_sinks.weight"]).flatten()
scores_with_sink = np.append(scores, sinks)
p = np.exp(scores_with_sink - np.max(scores_with_sink))
p = p / np.sum(p)
p_real = p[:-1] 

attn_out = np.zeros((nh, Dh), dtype=np.float32)
for h in range(nh):
    kvh = h // (nh // nkv)
    attn_out[h] = p_real[h] * V[kvh]

attn_out_flat = attn_out.flatten()
oW = dequant_q4_0_fast(tensors["blk.0.attn_output.weight"])
oB = dequant_f32_fast(tensors["blk.0.attn_output.bias"]).flatten()
attn_proj = attn_out_flat @ oW.T + oB

x = x + attn_proj

# MoE
w2 = dequant_f32_fast(tensors["blk.0.post_attention_norm.weight"]).flatten()
xn = rms_norm(x, w2, eps)

routerW = dequant_f32_fast(tensors["blk.0.ffn_gate_inp.weight"])
routerB = dequant_f32_fast(tensors["blk.0.ffn_gate_inp.bias"]).flatten()
router_logits = xn @ routerW.T + routerB

top_idx = np.argsort(router_logits)[::-1][:2]
top_logits = router_logits[top_idx]
top_weights = np.exp(top_logits - np.max(top_logits))
top_weights = top_weights / np.sum(top_weights)

gate_W = dequant_q4_0_fast(tensors["blk.0.ffn_gate_exps.weight"])
gate_B = dequant_f32_fast(tensors["blk.0.ffn_gate_exps.bias"]).reshape(4, 2880)
up_W = dequant_q4_0_fast(tensors["blk.0.ffn_up_exps.weight"])
up_B = dequant_f32_fast(tensors["blk.0.ffn_up_exps.bias"]).reshape(4, 2880)
down_W = dequant_q4_0_fast(tensors["blk.0.ffn_down_exps.weight"])
down_B = dequant_f32_fast(tensors["blk.0.ffn_down_exps.bias"]).reshape(4, 2880)

moe_out = np.zeros(2880, dtype=np.float32)
for i, e in enumerate(top_idx):
    g = xn @ gate_W[e].T + gate_B[e]
    u = xn @ up_W[e].T + up_B[e]
    
    # Standard SwiGLU logic exactly as dictated by the Python model.py
    g = np.clip(g, -7.0, 7.0)
    u = np.clip(u, -7.0, 7.0)
    mid = (g / (1.0 + np.exp(-g))) * u
    
    out = mid @ down_W[e].T + down_B[e]
    moe_out += out * top_weights[i]

x = x + moe_out

print("Python x after layer 0 (first 8 elements):", x[:8])