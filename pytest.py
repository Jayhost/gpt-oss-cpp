# reference_test.py
from llama_cpp import Llama
import numpy as np
import ctypes

model_path = "gpt-oss-20b.gguf"
prompt = "<|im_start|>user\nFrance capital?<|im_end|>\n<|im_start|>assistant\n"

print("Loading model in llama.cpp...")
llm = Llama(model_path=model_path, logits_all=True, verbose=False)

print("Tokenizing...")
tokens = llm.tokenize(prompt.encode('utf-8'), add_bos=False)
print(f"Tokens: {tokens}")

print("Evaluating prompt (prefill)...")
llm.eval(tokens)

n_vocab = llm.n_vocab()
n_tokens = len(tokens)

# FIX: Use ctypeslib to bridge the C pointer to a Numpy array safely
ptr = llm._ctx.get_logits()
logits_all = np.ctypeslib.as_array(ptr, shape=(n_tokens * n_vocab,))

# Extract the logits for the very last token processed
last_token_logits = np.copy(logits_all[-n_vocab:])

print(f"Saving {len(last_token_logits)} logits to logits_ref.npy")
np.save("logits_ref.npy", last_token_logits)

# See what llama.cpp generates
print("\nGenerating reference text...")
output = llm.create_completion(prompt, max_tokens=10, temperature=0.0)
print("Reference Output:", output['choices'][0]['text'])

# --- Comparison Block ---
import os
if os.path.exists("logits_cpp.bin"):
    print("\n--- Comparing Logits ---")
    ref = np.load("logits_ref.npy")
    cpp = np.fromfile("logits_cpp.bin", dtype=np.float32)
    
    if len(ref) != len(cpp):
        print(f"Vocab size mismatch! Ref: {len(ref)}, CPP: {len(cpp)}")
    else:
        diff = np.abs(ref - cpp)
        print(f"Max logit diff: {np.max(diff):.4f}")
        print(f"Mean logit diff: {np.mean(diff):.4f}")
        
        top5_ref = np.argsort(ref)[::-1][:5]
        top5_cpp = np.argsort(cpp)[::-1][:5]
        
        print(f"Top 5 Ref token IDs: {top5_ref}")
        print(f"Top 5 CPP token IDs: {top5_cpp}")
        
        # Decode predictions
        print("Ref predictions:", [(llm.detokenize([t]).decode('utf-8', errors='ignore'), t) for t in top5_ref])
        print("CPP predictions:", [(llm.detokenize([t]).decode('utf-8', errors='ignore'), t) for t in top5_cpp])
else:
    print("\nlogits_cpp.bin not found. Run your C++ engine with the debug dump to generate it.")