#!/usr/bin/env python3
"""analyze_gguf.py - Dump GGUF metadata and tensor info"""
import struct, sys

def read_string(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8')

def read_value(f, vtype):
    readers = {
        0: ('<B',1), 1: ('<b',1), 2: ('<H',2), 3: ('<h',2),
        4: ('<I',4), 5: ('<i',4), 6: ('<f',4), 7: ('<?',1),
        10: ('<Q',8), 11: ('<q',8), 12: ('<d',8)
    }
    if vtype == 8:
        return read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, arr_type) for _ in range(arr_len)]
    elif vtype in readers:
        fmt, sz = readers[vtype]
        return struct.unpack(fmt, f.read(sz))[0]
    raise ValueError(f"Unknown type {vtype}")

def main(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Bad magic: {magic}"
        
        # FIX: Use '<' to prevent struct alignment padding
        version, tensor_count, kv_count = struct.unpack('<IQQ', f.read(20))

        print(f"GGUF v{version}, {tensor_count} tensors, {kv_count} KV pairs\n")

        metadata = {}
        for _ in range(kv_count):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            metadata[key] = read_value(f, vtype)

        print("=== Architecture Metadata ===")
        arch_keys = sorted(k for k in metadata if k.startswith('gptoss') or k.startswith('general'))
        for k in arch_keys:
            v = metadata[k]
            if isinstance(v, list) and len(v) > 5:
                print(f"  {k}: [{len(v)} items] first={v[:5]}")
            else:
                print(f"  {k}: {v}")

        print("\n=== Tokenizer Metadata Keys ===")
        tok_keys = sorted(k for k in metadata if k.startswith('tokenizer'))
        for k in tok_keys:
            v = metadata[k]
            if isinstance(v, list) and len(v) > 5:
                print(f"  {k}: [{len(v)} items] first={v[:5]}")
            else:
                print(f"  {k}: {v}")

        # Read tensor info
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = list(struct.unpack(f'<{n_dims}Q', f.read(8*n_dims)))
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, dims, ttype, offset))

        type_names = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0'}
        
        print(f"\n=== Layer 0 Tensors (for reference) ===")
        for name, dims, ttype, offset in tensors:
            if 'blk.0.' in name:
                tn = type_names.get(ttype, f'type_{ttype}')
                print(f"  {name:55s} dims={dims} type={tn}")

        print(f"\n=== Non-layer Tensors ===")
        for name, dims, ttype, offset in tensors:
            if not name.startswith('blk.'):
                tn = type_names.get(ttype, f'type_{ttype}')
                print(f"  {name:55s} dims={dims} type={tn}")

        has_moe = any('ffn_gate_inp' in t[0] for t in tensors)
        has_qk_norm = any('attn_q_norm' in t[0] for t in tensors)
        print(f"\n=== Architecture Features ===")
        print(f"  MoE: {has_moe}")
        print(f"  QK-Norm: {has_qk_norm}")
        layer_indices = set()
        for name, *_ in tensors:
            if name.startswith('blk.'):
                idx = int(name.split('.')[1])
                layer_indices.add(idx)
        print(f"  Layer count: {len(layer_indices)}")

if __name__ == '__main__':
    main(sys.argv[1])