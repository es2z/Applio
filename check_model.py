import torch

ckpt = torch.load('logs/naru_x_C768_2/naru_x_C768_2_8e_1720s.pth', map_location='cpu', weights_only=True)
print('Total keys in checkpoint:', len(ckpt['weight'].keys()))
print('\nFirst 30 keys:')
for i, k in enumerate(list(ckpt['weight'].keys())[:30]):
    shape = list(ckpt['weight'][k].shape)
    print(f'  {k}: shape={shape}')

print('\nChecking important dimensions:')
print('  enc_p.emb_phone.weight:', list(ckpt['weight']['enc_p.emb_phone.weight'].shape))
print('  enc_p.emb_phone.bias:', list(ckpt['weight']['enc_p.emb_phone.bias'].shape))
if 'enc_p.emb_pitch.weight' in ckpt['weight']:
    print('  enc_p.emb_pitch.weight:', list(ckpt['weight']['enc_p.emb_pitch.weight'].shape))
print('  emb_g.weight:', list(ckpt['weight']['emb_g.weight'].shape))
print('  dec.conv_pre.weight:', list(ckpt['weight']['dec.conv_pre.weight'].shape))
print('  dec.conv_post.weight:', list(ckpt['weight']['dec.conv_post.weight'].shape))
