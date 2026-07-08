# analyze_gguf.py
from gguf import GGUFReader
import sys

def analyze_gguf(filepath):
    reader = GGUFReader(filepath)
    
    print("=== Architecture & Metadata ===")
    for key, val in reader.fields.items():
        # Print out the key architecture details
        if 'arch' in key or 'context' in key or 'embedding' in key or 'block_count' in key or 'head_count' in key or 'parallel_residual' in key:
            print(f"{key}: {val}")
            
    print("\n=== Tensor Names & Types ===")
    for tensor in reader.tensors:
        print(f"Name: {tensor.name:40s} | Shape: {str(tensor.shape):20s} | Type: {tensor.tensor_type}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_gguf.py <model.gguf>")
        sys.exit(1)
    analyze_gguf(sys.argv[1])