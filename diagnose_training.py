import torch
import numpy as np
import os

print("=== 768-dim Training Diagnosis ===\n")

# Check latest model
model_path = 'logs/naru_x_C768_3/naru_x_C768_3_50e_9400s.pth'
print(f"Analyzing: {model_path}")
print(f"File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB\n")

ckpt = torch.load(model_path, map_location='cpu', weights_only=True)

print("=== Model Metadata ===")
print(f"Embedder: {ckpt.get('embedder_model', 'NOT FOUND')}")
print(f"Text enc dim: {ckpt.get('text_enc_hidden_dim', 'NOT FOUND')}")
print(f"Hidden channels: {ckpt.get('hidden_channels', 'NOT FOUND')}")
print(f"Epoch: {ckpt.get('epoch', 'NOT FOUND')}")
print(f"Step: {ckpt.get('step', 'NOT FOUND')}")
print(f"Sample rate: {ckpt.get('sr', 'NOT FOUND')}")

print("\n=== Config Array ===")
config = ckpt.get('config', [])
if len(config) >= 18:
    print(f"[0] spec_channels: {config[0]}")
    print(f"[1] segment_size: {config[1]}")
    print(f"[2] inter_channels: {config[2]}")
    print(f"[3] hidden_channels: {config[3]}")
    print(f"[4] filter_channels: {config[4]}")
    print(f"[5] n_heads: {config[5]}")
    print(f"[6] n_layers: {config[6]}")
    print(f"[16] gin_channels: {config[16]}")
    print(f"[17] sample_rate: {config[17]}")

print("\n=== Critical Weights Check ===")
weights = ckpt.get('weight', {})
critical_keys = [
    ('enc_p.emb_phone.weight', 'TextEncoder phone embedding'),
    ('enc_p.emb_pitch.weight', 'TextEncoder pitch embedding'),
    ('dec.conv_pre.weight', 'Decoder input conv'),
    ('emb_g.weight', 'Speaker embedding'),
]

for key, desc in critical_keys:
    if key in weights:
        shape = list(weights[key].shape)
        print(f"{desc:35s}: {shape}")
    else:
        print(f"{desc:35s}: NOT FOUND ⚠")

# Check for NaN or inf in weights
print("\n=== Weight Statistics ===")
sample_keys = ['enc_p.emb_phone.weight', 'dec.conv_pre.weight']
for key in sample_keys:
    if key in weights:
        w = weights[key]
        print(f"\n{key}:")
        print(f"  Shape: {list(w.shape)}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  Min: {w.min().item():.6f}")
        print(f"  Max: {w.max().item():.6f}")
        print(f"  Has NaN: {torch.isnan(w).any().item()}")
        print(f"  Has Inf: {torch.isinf(w).any().item()}")

# Check FAISS index
print("\n=== FAISS Index Check ===")
index_path = 'logs/naru_x_C768_3/naru_x_C768_3.index'
if os.path.exists(index_path):
    import faiss
    index = faiss.read_index(index_path)
    print(f"Index dimension: {index.d}")
    print(f"Index total vectors: {index.ntotal}")
    print(f"Expected dimension: 768")

    if index.d == 768:
        print("✓ Index dimension matches model")

        # Sample a few vectors
        if index.ntotal > 0:
            big_npy = index.reconstruct_n(0, min(10, index.ntotal))
            print(f"\nSample vector statistics (first 10 vectors):")
            print(f"  Mean: {big_npy.mean():.6f}")
            print(f"  Std: {big_npy.std():.6f}")
            print(f"  Min: {big_npy.min():.6f}")
            print(f"  Max: {big_npy.max():.6f}")
            print(f"  Has NaN: {np.isnan(big_npy).any()}")
            print(f"  Has Inf: {np.isinf(big_npy).any()}")
    else:
        print(f"✗ Index dimension mismatch! Expected 768, got {index.d}")
else:
    print("✗ Index file not found")

# Check extracted embeddings
print("\n=== Extracted Embeddings Check ===")
import glob
emb_files = glob.glob('logs/naru_x_C768_3/extracted/*.npy')
if emb_files:
    print(f"Found {len(emb_files)} embedding files")
    # Sample first file
    sample_emb = np.load(emb_files[0])
    print(f"\nSample embedding (first file):")
    print(f"  Shape: {sample_emb.shape}")
    print(f"  Expected: (n_frames, 768)")
    print(f"  Mean: {sample_emb.mean():.6f}")
    print(f"  Std: {sample_emb.std():.6f}")
    print(f"  Has NaN: {np.isnan(sample_emb).any()}")
    print(f"  Has Inf: {np.isinf(sample_emb).any()}")
else:
    print("✗ No embedding files found")

print("\n=== Diagnosis Summary ===")
print("1. Check if model weights have NaN/Inf")
print("2. Check if FAISS index dimension matches (768)")
print("3. Check if extracted embeddings are valid")
print("4. Check training config (config.json)")
