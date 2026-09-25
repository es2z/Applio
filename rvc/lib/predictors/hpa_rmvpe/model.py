"""The HPA-RMVPE network, ported verbatim from the upstream repository.

Source: https://github.com/PhamHuynhAnh16/HPA-RMVPE at commit
0cdb7db22b381f0ee053bd540e8cdc90180443b7 (src/model.py, src/deepunet.py, src/yolo.py),
weights at https://huggingface.co/AnhP/HPA-RMVPE (MIT, see docs/licenses/HPA-RMVPE-LICENSE).

Module and attribute names are kept exactly as upstream: the published state_dict
keys are derived from them. Only the network lives here. The mel front end, the
chunking and the decoder are RMVPE's and are inherited from RMVPE0Predictor, because
upstream's inference.py is RMVPE's code with nothing changed but the network.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rvc.lib.predictors.RMVPE import N_CLASS, N_MELS, BiGRU


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.pwconv(self.dwconv(x))))


class DS_Bottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, shortcut=True):
        super().__init__()
        c_ = c1
        self.dsconv1 = DSConv(c1, c_, k=3, s=1)
        self.dsconv2 = DSConv(c_, c2, k=k, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        if self.shortcut:
            return x + self.dsconv2(self.dsconv1(x))
        return self.dsconv2(self.dsconv1(x))


class DS_C3k(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(
            *[DS_Bottleneck(c_, c_, k=k, shortcut=True) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class DS_C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = DS_C3k(c_, c_, n=n, k=k, e=1.0)
        self.cv2 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        return self.cv2(self.m(self.cv1(x)))


class AdaptiveHyperedgeGeneration(nn.Module):
    def __init__(self, in_channels, num_hyperedges, num_heads):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = max(1, in_channels // num_heads)

        self.global_proto = nn.Parameter(torch.randn(num_hyperedges, in_channels))
        self.context_mapper = nn.Linear(
            2 * in_channels, num_hyperedges * in_channels, bias=False
        )
        self.query_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        B, N, C = x.shape
        f_avg = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_max = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        f_ctx = torch.cat((f_avg, f_max), dim=1)

        delta_P = self.context_mapper(f_ctx).view(B, self.num_hyperedges, C)
        P = self.global_proto.unsqueeze(0) + delta_P

        z = self.query_proj(x)
        z = z.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        P = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(
            0, 2, 3, 1
        )

        sim = (z @ P) * self.scale
        s_bar = sim.mean(dim=1)
        return F.softmax(s_bar.permute(0, 2, 1), dim=-1)


class HypergraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W_e = nn.Linear(in_channels, in_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, A):
        f_m = torch.bmm(A, x)
        f_m = self.act(self.W_e(f_m))
        x_out = torch.bmm(A.transpose(1, 2), f_m)
        x_out = self.act(self.W_v(x_out))
        return x + x_out


class AdaptiveHypergraphComputation(nn.Module):
    def __init__(self, in_channels, out_channels, num_hyperedges, num_heads):
        super().__init__()
        self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(
            in_channels, num_hyperedges, num_heads
        )
        self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        A = self.adaptive_hyperedge_gen(x_flat)
        x_out_flat = self.hypergraph_conv(x_flat, A)
        return x_out_flat.permute(0, 2, 1).view(B, -1, H, W)


class C3AH(nn.Module):
    def __init__(self, c1, c2, num_hyperedges, num_heads, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads)
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x_lateral = self.cv1(x)
        x_ahc = self.ahc(self.cv2(x))
        return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))


class HyperACE(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_hyperedges=16,
        num_heads=8,
        k=2,
        l=1,
        c_h=0.5,
        c_l=0.25,
    ):
        super().__init__()

        c2, c3, c4, c5 = in_channels
        c_mid = c4
        self.fuse_conv = Conv(c2 + c3 + c4 + c5, c_mid, 1, 1)

        self.c_h = int(c_mid * c_h)
        self.c_l = int(c_mid * c_l)
        self.c_s = c_mid - self.c_h - self.c_l

        self.high_order_branch = nn.ModuleList(
            [
                C3AH(
                    self.c_h,
                    self.c_h,
                    num_hyperedges=num_hyperedges,
                    num_heads=num_heads,
                    e=1.0,
                )
                for _ in range(k)
            ]
        )
        self.high_order_fuse = Conv(self.c_h * k, self.c_h, 1, 1)

        self.low_order_branch = nn.Sequential(
            *[DS_C3k(self.c_l, self.c_l, n=1, k=3, e=1.0) for _ in range(l)]
        )

        self.final_fuse = Conv(self.c_h + self.c_l + self.c_s, out_channels, 1, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        B2, B3, B4, B5 = x
        H4, W4 = B4.shape[2:]
        B2_resized = F.interpolate(
            B2, size=(H4, W4), mode="bilinear", align_corners=False
        )
        B3_resized = F.interpolate(
            B3, size=(H4, W4), mode="bilinear", align_corners=False
        )
        B5_resized = F.interpolate(
            B5, size=(H4, W4), mode="bilinear", align_corners=False
        )

        x_b = self.fuse_conv(torch.cat((B2_resized, B3_resized, B4, B5_resized), dim=1))
        x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

        x_h_outs = [m(x_h) for m in self.high_order_branch]
        x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))
        x_l_out = self.low_order_branch(x_l)

        return self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, f_in, h):
        return f_in + self.gamma * h


class YOLO13Encoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.stem = DSConv(in_channels, base_channels, k=3, s=1)
        self.p2 = nn.Sequential(
            DSConv(base_channels, base_channels * 2, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 2, base_channels * 2, n=1),
        )
        self.p3 = nn.Sequential(
            DSConv(base_channels * 2, base_channels * 4, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 4, base_channels * 4, n=2),
        )
        self.p4 = nn.Sequential(
            DSConv(base_channels * 4, base_channels * 8, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 8, base_channels * 8, n=2),
        )
        self.p5 = nn.Sequential(
            DSConv(base_channels * 8, base_channels * 16, k=3, s=(2, 2)),
            DS_C3k2(base_channels * 16, base_channels * 16, n=1),
        )
        self.out_channels = [
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

    def forward(self, x):
        x = self.stem(x)
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]


class YOLO13FullPADDecoder(nn.Module):
    def __init__(self, encoder_channels, hyperace_out_c, out_channels_final):
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = encoder_channels
        c_d5, c_d4, c_d3, c_d2 = c_p5, c_p4, c_p3, c_p2

        self.h_to_d5 = Conv(hyperace_out_c, c_d5, 1, 1)
        self.h_to_d4 = Conv(hyperace_out_c, c_d4, 1, 1)
        self.h_to_d3 = Conv(hyperace_out_c, c_d3, 1, 1)
        self.h_to_d2 = Conv(hyperace_out_c, c_d2, 1, 1)

        self.fusion_d5 = GatedFusion(c_d5)
        self.fusion_d4 = GatedFusion(c_d4)
        self.fusion_d3 = GatedFusion(c_d3)
        self.fusion_d2 = GatedFusion(c_d2)

        self.skip_p5 = Conv(c_p5, c_d5, 1, 1)
        self.skip_p4 = Conv(c_p4, c_d4, 1, 1)
        self.skip_p3 = Conv(c_p3, c_d3, 1, 1)
        self.skip_p2 = Conv(c_p2, c_d2, 1, 1)

        self.up_d5 = DS_C3k2(c_d5, c_d4, n=1)
        self.up_d4 = DS_C3k2(c_d4, c_d3, n=1)
        self.up_d3 = DS_C3k2(c_d3, c_d2, n=1)

        self.final_d2 = DS_C3k2(c_d2, c_d2, n=1)
        self.final_conv = Conv(c_d2, out_channels_final, 1, 1)

    def _hace(self, h_ace, like, project):
        return project(
            F.interpolate(h_ace, size=like.shape[2:], mode="bilinear", align_corners=False)
        )

    def forward(self, enc_feats, h_ace):
        p2, p3, p4, p5 = enc_feats

        d5 = self.skip_p5(p5)
        d5 = self.fusion_d5(d5, self._hace(h_ace, d5, self.h_to_d5))

        d5_up = F.interpolate(d5, size=p4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.up_d5(d5_up) + self.skip_p4(p4)
        d4 = self.fusion_d4(d4, self._hace(h_ace, d4, self.h_to_d4))

        d4_up = F.interpolate(d4, size=p3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.up_d4(d4_up) + self.skip_p3(p3)
        d3 = self.fusion_d3(d3, self._hace(h_ace, d3, self.h_to_d3))

        d3_up = F.interpolate(d3, size=p2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.up_d3(d3_up) + self.skip_p2(p2)
        d2 = self.fusion_d2(d2, self._hace(h_ace, d2, self.h_to_d2))

        d2 = self.final_d2(d2)
        return self.final_conv(d2)


class DeepUnet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        en_out_channels=16,
        base_channels=64,
        hyperace_k=2,
        hyperace_l=1,
        num_hyperedges=16,
        num_heads=8,
    ):
        super().__init__()
        self.encoder = YOLO13Encoder(in_channels, base_channels)
        enc_ch = self.encoder.out_channels
        self.hyperace = HyperACE(
            in_channels=enc_ch,
            out_channels=enc_ch[-1],
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            k=hyperace_k,
            l=hyperace_l,
        )
        self.decoder = YOLO13FullPADDecoder(
            encoder_channels=enc_ch,
            hyperace_out_c=enc_ch[-1],
            out_channels_final=en_out_channels,
        )

    def forward(self, x):
        original_size = x.shape[2:]
        features = self.encoder(x)
        h_ace = self.hyperace(features)
        x_dec = self.decoder(features, h_ace)
        return F.interpolate(
            x_dec, size=original_size, mode="bilinear", align_corners=False
        )


class E2E0(nn.Module):
    """Upstream's E2E0 with the hyperparameters it was trained with.

    num_heads=4 is not in the weights (it only splits query_proj's output), so a
    wrong value would load without complaint and give different pitch; it is fixed
    here rather than exposed.
    """

    def __init__(self, n_gru=1, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(
            in_channels=in_channels,
            en_out_channels=en_out_channels,
            base_channels=64,
            hyperace_k=2,
            hyperace_l=1,
            num_hyperedges=16,
            num_heads=4,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)
