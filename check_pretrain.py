import torch
import os

print("=== Checking Pretrain Models ===\n")

# Check G model
g_path = 'rvc/models/pretraineds/custom/G_2333333.pth'
if os.path.exists(g_path):
    print(f"G model: {g_path}")
    print(f"Size: {os.path.getsize(g_path) / (1024*1024):.1f} MB")

    g_ckpt = torch.load(g_path, map_location='cpu', weights_only=True)

    # Check if it's a state_dict or a full checkpoint
    if isinstance(g_ckpt, dict):
        if 'model' in g_ckpt:
            print("Format: Full checkpoint with 'model' key")
            state_dict = g_ckpt['model']
        else:
            print("Format: Direct state_dict")
            state_dict = g_ckpt

        print(f"Total keys: {len(state_dict.keys())}")
        print("\nFirst 20 keys with shapes:")
        for i, (k, v) in enumerate(list(state_dict.items())[:20]):
            if hasattr(v, 'shape'):
                print(f"  {k}: {list(v.shape)}")

        # Check critical dimensions
        print("\n=== Critical Dimensions ===")
        if 'enc_p.emb_phone.weight' in state_dict:
            shape = state_dict['enc_p.emb_phone.weight'].shape
            print(f"enc_p.emb_phone.weight: {list(shape)}")
            print(f"  → hidden_channels = {shape[0]}")

        if 'dec.conv_pre.weight' in state_dict:
            shape = state_dict['dec.conv_pre.weight'].shape
            print(f"dec.conv_pre.weight: {list(shape)}")
            print(f"  → inter_channels = {shape[1]}")

        if 'emb_g.weight' in state_dict:
            shape = state_dict['emb_g.weight'].shape
            print(f"emb_g.weight: {list(shape)}")
            print(f"  → gin_channels = {shape[1]}")
    else:
        print(f"Unexpected format: {type(g_ckpt)}")

print("\n" + "="*50 + "\n")

# Check D model
d_path = 'rvc/models/pretraineds/custom/D_2333333.pth'
if os.path.exists(d_path):
    print(f"D model: {d_path}")
    print(f"Size: {os.path.getsize(d_path) / (1024*1024):.1f} MB")

    d_ckpt = torch.load(d_path, map_location='cpu', weights_only=True)

    if isinstance(d_ckpt, dict):
        if 'model' in d_ckpt:
            state_dict = d_ckpt['model']
        else:
            state_dict = d_ckpt

        print(f"Total keys: {len(state_dict.keys())}")
        print("\nFirst 10 keys with shapes:")
        for i, (k, v) in enumerate(list(state_dict.items())[:10]):
            if hasattr(v, 'shape'):
                print(f"  {k}: {list(v.shape)}")

print("\n" + "="*50)
print("\n=== Comparison ===")
print("Expected for 192-dim: G ~100-150 MB, enc_p.emb_phone=[192,768], dec.conv_pre=[512,192,7]")
print("Expected for 768-dim: G ~400-500 MB, enc_p.emb_phone=[768,768], dec.conv_pre=[1024,768,7]")
