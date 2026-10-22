"""The generator call realtime uses when one of the opt-in options needs more than
`Synthesizer.infer`.

`Synthesizer.infer` renders the whole conversion window with fresh noise every call.
The opt-in options in rvc/realtime/options.py need two things it does not offer:

* render only the newest part of the window (output position "trim"), keeping
  `FLOW_MARGIN_FRAMES` frames in front of it so the flow still sees context. That is
  what RVC's own realtime does (`flow_head = max(head - 24, 0)`,
  RVC-Project/Retrieval-based-Voice-Conversion-WebUI infer/module/models.py);
* pass APEX-GAN's decoder a fixed noise envelope reference.

With none of those, realtime keeps calling `Synthesizer.infer` exactly as before.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from rvc.lib.algorithm.generators.apex_gan import APEXGANGenerator

# The flow's coupling layers each see a few frames on either side; 24 frames (240 ms at
# 100 Hz) is the margin RVC's realtime keeps in front of the rendered region.
FLOW_MARGIN_FRAMES = 24
# Synthesizer.infer's latent temperature.
LATENT_NOISE_SCALE = 0.66666


@dataclass(frozen=True)
class SynthesisPlan:
    """How the generator is called for one session. Static, so torch.compile can
    specialise on it."""

    skip_head: Optional[int] = None  # frames of past context not rendered
    return_length: Optional[int] = None  # frames rendered after skip_head
    flow_margin: int = FLOW_MARGIN_FRAMES
    envelope_reference: Optional[float] = None  # APEX-GAN only

    @property
    def flow_head(self):
        if self.skip_head is None:
            return 0
        return max(int(self.skip_head) - self.flow_margin, 0)


def supports_excitation_control(net_g):
    """True when the decoder accepts the envelope override."""
    return isinstance(getattr(net_g, "dec", None), APEXGANGenerator)


def realtime_infer(
    net_g,
    phone,
    phone_lengths,
    pitch,
    nsff0,
    sid,
    plan: SynthesisPlan,
):
    """Synthesizer.infer with the realtime options. Returns the waveform [B, 1, T]."""
    g = net_g.emb_g(sid).unsqueeze(-1)
    m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, phone_lengths)

    flow_head = plan.flow_head
    if plan.skip_head is not None:
        m_p = m_p[:, :, flow_head:]
        logs_p = logs_p[:, :, flow_head:]
        x_mask = x_mask[:, :, flow_head:]

    z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * LATENT_NOISE_SCALE) * x_mask
    z = net_g.flow(z_p, x_mask, g=g, reverse=True)

    if plan.skip_head is not None:
        head = int(plan.skip_head)
        length = int(plan.return_length)
        dec_head = head - flow_head
        z = z[:, :, dec_head : dec_head + length]
        x_mask = x_mask[:, :, dec_head : dec_head + length]
        if nsff0 is not None:
            nsff0 = nsff0[:, head : head + length]

    if not net_g.use_f0:
        output = net_g.dec(z * x_mask, g=g)
    elif supports_excitation_control(net_g):
        output = net_g.dec(
            z * x_mask,
            nsff0,
            g=g,
            envelope_reference=plan.envelope_reference,
        )
    else:
        output = net_g.dec(z * x_mask, nsff0, g=g)
    if net_g.dec_extra is not None:
        output = output[0]
    return output
