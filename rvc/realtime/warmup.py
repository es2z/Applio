"""The realtime warm-up, shared by AudioCallbacks and the offline stream harness.

Moved out of AudioCallbacks.__init__ unchanged, so that tools/realtime_stream_eval.py
starts a session exactly the way the Realtime tab does without importing the audio
device layer.
"""

import numpy as np
import torch


def warm_up(
    vc,
    f0_up_key=0,
    index_rate=0.5,
    protect=0.5,
    volume_envelope=1,
    f0_autotune=False,
    f0_autotune_strength=1,
    proposed_pitch=False,
    proposed_pitch_threshold=155.0,
):
    """Run the conversion a few times before any audio device opens.

    Same shapes and grad mode as the audio callback, fed a tone, since silence can
    bypass F0. This runs even without compilation: it is what builds the lazily created
    F0 model, whose weight initialisation consumes RNG. Doing that before the seed is
    applied is what makes two sessions actually line up - otherwise the second session
    reuses torchcrepe's cache, draws less randomness than the first, and diverges.
    """
    runtime = vc.vc_model
    compiling = runtime.pipeline.compile_session.enabled
    if compiling:
        print("[Realtime] Preparing compilation. The first start may take time.")
    devices = (
        [torch.device(runtime.device)]
        if torch.device(runtime.device).type == "cuda"
        else []
    )
    tone = (
        0.01 * np.sin(2 * np.pi * 220 * np.arange(vc.block_frame) / 48000)
    ).astype(np.float32)
    # Do not feed synthetic audio into the stateful speech detector.
    vad = runtime.vad
    runtime.vad = None
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            warmup_blocks = 3 if compiling else 1
            if getattr(runtime, "fcn_session", None) is not None:
                delay48 = runtime.fcn_session.stream.holdback_samples * 3
                warmup_blocks = max(
                    warmup_blocks,
                    (delay48 + vc.block_frame - 1) // vc.block_frame + 2,
                )
            for _ in range(warmup_blocks):
                runtime.inference(
                    tone,
                    f0_up_key,
                    index_rate,
                    protect,
                    volume_envelope,
                    f0_autotune,
                    f0_autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                )
            if devices:
                torch.cuda.synchronize(runtime.device)
    finally:
        runtime.vad = vad
        runtime.flush_buffers()
        if hasattr(runtime, "reset_fcn_stream"):
            runtime.reset_fcn_stream()
        if hasattr(runtime, "reset_stream_state"):
            runtime.reset_stream_state()
        runtime.consecutive_silence_frames = 0
