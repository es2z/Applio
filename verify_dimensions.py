import torch
import json

print("=== Verifying 768-dim model ===")
ckpt = torch.load('logs/naru_x_C768_2/naru_x_C768_2_8e_1720s.pth', map_location='cpu', weights_only=True)

print("\n1. Config array check:")
print(f"   inter_channels (index 2): {ckpt['config'][2]}")
print(f"   hidden_channels (index 3): {ckpt['config'][3]}")
print(f"   filter_channels (index 4): {ckpt['config'][4]}")
print(f"   gin_channels (index 16): {ckpt['config'][16]}")
print(f"   upsample_initial_channel (index 13): {ckpt['config'][13]}")

print("\n2. Critical weight dimensions:")
critical_weights = [
    ('enc_p.emb_phone.weight', 'TextEncoder input projection'),
    ('enc_p.emb_phone.bias', 'TextEncoder input bias'),
    ('enc_p.encoder.attn_layers.0.conv_q.weight', 'Attention Q projection'),
    ('enc_p.proj.weight', 'TextEncoder output projection'),
    ('flow.flows.0.pre.weight', 'Flow pre-conv'),
    ('flow.flows.0.post.weight', 'Flow post-conv'),
    ('dec.conv_pre.weight', 'Decoder input'),
    ('emb_g.weight', 'Speaker embedding'),
]

for key, desc in critical_weights:
    if key in ckpt['weight']:
        shape = list(ckpt['weight'][key].shape)
        print(f"   {desc:35s} : {shape}")
    else:
        print(f"   {desc:35s} : NOT FOUND")

print("\n3. Expected dimensions for 768-dim architecture:")
print("   TextEncoder:")
print("     - emb_phone: [hidden_channels, text_enc_hidden_dim] = [768, 768]")
print("     - emb_pitch: [256, hidden_channels] = [256, 768]")
print("     - proj: [inter_channels*2, hidden_channels] = [1536, 768]")
print("   Flow:")
print("     - pre: [inter_channels//2, hidden_channels] = [384, 768]")
print("     - post: [inter_channels//2, hidden_channels] = [384, 768]")
print("   Decoder:")
print("     - conv_pre: [upsample_initial_channel, inter_channels] = [1024, 768]")
print("   Speaker:")
print("     - emb_g: [n_speakers, gin_channels] = [1, 512]")

print("\n4. Verification:")
emb_phone_shape = ckpt['weight']['enc_p.emb_phone.weight'].shape
proj_shape = ckpt['weight']['enc_p.proj.weight'].shape
dec_pre_shape = ckpt['weight']['dec.conv_pre.weight'].shape
emb_g_shape = ckpt['weight']['emb_g.weight'].shape

hidden_channels_actual = emb_phone_shape[0]
inter_channels_actual = dec_pre_shape[1]
gin_channels_actual = emb_g_shape[1]

print(f"   Actual hidden_channels (from emb_phone): {hidden_channels_actual}")
print(f"   Actual inter_channels (from dec.conv_pre): {inter_channels_actual}")
print(f"   Actual gin_channels (from emb_g): {gin_channels_actual}")

if hidden_channels_actual == 768 and inter_channels_actual == 768 and gin_channels_actual == 512:
    print("\n   ✓ MODEL IS GENUINELY 768-DIM HIGH-CAPACITY ARCHITECTURE")
elif hidden_channels_actual == 192 and inter_channels_actual == 192 and gin_channels_actual == 256:
    print("\n   ✗ MODEL IS ACTUALLY 192-DIM (metadata is wrong!)")
else:
    print(f"\n   ? MODEL HAS UNEXPECTED DIMENSIONS")

print("\n5. Model file size:")
import os
size_mb = os.path.getsize('logs/naru_x_C768_2/naru_x_C768_2_8e_1720s.pth') / (1024 * 1024)
print(f"   Model size: {size_mb:.1f} MB")
print(f"   Expected for 768-dim: ~500-600 MB")
print(f"   Expected for 192-dim: ~100-150 MB")
