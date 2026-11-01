"""SPIN on a hand-installed HuBERT Large: the truncated embedder, the head and the loss.

The embedder is kushinada-hubert-large cut to its first `layers` transformer layers
(num_hidden_layers in the config, so the rest are never even loaded). Because the model
is do_stable_layer_norm, its last_hidden_state is encoder.layer_norm applied to the last
kept layer, which is exactly what rvc.lib.utils.embedder_forward returns for the full
model at output_layer == layers. So the cut model is a drop-in embedder whose default
output is the layer chosen in the layer comparison.

Only the top `train_layers` transformer layers learn. Everything below - CNN, feature
projection, positional convolution, lower layers - is frozen and run in eval mode, so no
dropout or layerdrop touches it, and the top layers keep their own dropout (0.1, the
config's) in train mode. encoder.layer_norm is frozen too, unlike the official SPIN, which
trains a pre-norm model's final norm with its last layer: RVC is sensitive to the
features' magnitude (see the k2 notes in CLAUDE.md), and a frozen norm keeps it where the
rest of the pipeline expects it.

The head and the loss are SPIN's (src/nn/dnn.py with hid_dims [256], which is a single
linear layer, and src/nn/swav_vq_dis.py; MIT, docs/licenses/spin-LICENSE): project to 256,
L2-normalise, score against a 256-entry codebook at temperature 0.1, and predict each
view's Sinkhorn-balanced assignment (epsilon 0.02, 3 iterations) from the other view.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rvc.lib.utils import HubertModelWithFinalProj, _attach_input_preprocessing


def load_truncated(model_dir, layers):
    """The embedder in model_dir cut to `layers` transformer layers, ready for embedder_forward."""
    model = HubertModelWithFinalProj.from_pretrained(model_dir, num_hidden_layers=layers)
    _attach_input_preprocessing(model, model_dir, None, default_do_normalize=True)
    model.output_layer = None
    model.feature_scale = 1.0
    return model


def trainable_layers(model, train_layers):
    return list(model.encoder.layers[-train_layers:])


def set_trainable(model, train_layers):
    """Freeze everything but the top train_layers layers; returns their parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    params = []
    for layer in trainable_layers(model, train_layers):
        layer.train()
        for p in layer.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


@torch.no_grad()
def compute_sinkhorn(out, epsilon, sinkhorn_iterations):
    # out: (B, K). Single process, so the distributed reductions of the original are gone.
    B, K = out.shape
    Q = out.div(epsilon).exp().t()
    Q.div_(Q.sum())
    for _ in range(sinkhorn_iterations):
        Q.div_(Q.sum(dim=1, keepdim=True) * K)
        Q.div_(Q.sum(dim=0, keepdim=True) * B)
    Q.mul_(B)
    return Q.t()


class SpinHead(nn.Module):
    def __init__(self, dim=1024, proj_dim=256, num_vars=256, epsilon=0.02,
                 sinkhorn_iters=3, temp=0.1):
        super().__init__()
        self.pred = nn.Linear(dim, proj_dim)
        self.codebook = nn.Linear(proj_dim, num_vars, bias=False)
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.temp = temp

    @torch.no_grad()
    def normalize_codebook(self):
        self.codebook.weight.copy_(F.normalize(self.codebook.weight, dim=1, p=2))

    def forward(self, feat_1, feat_2):
        """Swapped prediction loss between two views' frames, both (N, dim)."""
        z_1 = F.normalize(self.pred(feat_1).float(), dim=1)
        z_2 = F.normalize(self.pred(feat_2).float(), dim=1)
        logits_1 = F.linear(z_1, self.codebook.weight.float())
        logits_2 = F.linear(z_2, self.codebook.weight.float())
        with torch.no_grad():
            tgt_1 = compute_sinkhorn(logits_1.detach(), self.epsilon, self.sinkhorn_iters)
            tgt_2 = compute_sinkhorn(logits_2.detach(), self.epsilon, self.sinkhorn_iters)
        log_prob_1 = (logits_1 / self.temp).log_softmax(1)
        log_prob_2 = (logits_2 / self.temp).log_softmax(1)
        loss = -0.5 * ((tgt_1 * log_prob_2).sum(1).mean() + (tgt_2 * log_prob_1).sum(1).mean())
        with torch.no_grad():
            logits = torch.cat([logits_1, logits_2], dim=0) / self.temp
            hard = F.one_hot(logits.argmax(-1), logits.shape[1]).float().mean(0)
            code_ppl = torch.exp(-torch.sum(hard * torch.log(hard + 1e-7)))
            avg = logits.softmax(-1).mean(0)
            prob_ppl = torch.exp(-torch.sum(avg * torch.log(avg + 1e-7)))
            acc = 0.5 * ((logits_1.argmax(1) == tgt_2.argmax(1)).float().mean()
                         + (logits_2.argmax(1) == tgt_1.argmax(1)).float().mean())
        return loss, {"code_ppl": code_ppl.item(), "prob_ppl": prob_ppl.item(),
                      "acc": acc.item()}


def warmup_decay(step, warmup=2500, max_step=5000, base_lr=1e-4, init_lr=1e-6, final_lr=1e-6):
    """SPIN's linear_warmup_decay as a LambdaLR factor."""
    rate_i, rate_f = init_lr / base_lr, final_lr / base_lr
    if step <= warmup:
        return rate_i + (1.0 - rate_i) * step / warmup
    return max(rate_f, 1.0 - (1.0 - rate_f) * (step - warmup) / (max_step - warmup - 1))
