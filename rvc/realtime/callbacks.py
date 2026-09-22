import os
import sys
import threading
import numpy as np
import torch

sys.path.append(os.getcwd())

from rvc.realtime.rng import apply_seed
from rvc.realtime.audio import Audio
from rvc.realtime.core import VoiceChanger


class AudioCallbacks:
    def __init__(
        self,
        pass_through: bool = False,
        read_chunk_size: int = 192,
        cross_fade_overlap_size: float = 0.1,
        extra_convert_size: float = 0.5,
        model_path: str = None,
        index_path: str = None,
        f0_method: str = "rmvpe",
        embedder_model: str = None,
        embedder_model_custom: str = None,
        embedder_precision: str = "fp32",
        silent_threshold: int = -90,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
        vad_enabled: bool = False,
        vad_sensitivity: int = 3,
        vad_frame_ms: int = 30,
        sid: int = 0,
        hybrid_blend_ratio: float = 0.5,
        fcn_profile=None,
        # device: str = "cuda",
    ):
        self.pass_through = pass_through
        self.lock = threading.Lock()
        self.vc = VoiceChanger(
            read_chunk_size=read_chunk_size,
            cross_fade_overlap_size=cross_fade_overlap_size,
            extra_convert_size=extra_convert_size,
            model_path=model_path,
            index_path=index_path,
            f0_method=f0_method,
            embedder_model=embedder_model,
            embedder_model_custom=embedder_model_custom,
            embedder_precision=embedder_precision,
            silent_threshold=silent_threshold,
            vad_enabled=vad_enabled,
            vad_sensitivity=vad_sensitivity,
            vad_frame_ms=vad_frame_ms,
            sid=sid,
            hybrid_blend_ratio=hybrid_blend_ratio,
            fcn_profile=fcn_profile,
        )
        self.audio = Audio(
            self,
            f0_up_key,
            index_rate,
            protect,
            volume_envelope,
            f0_autotune,
            f0_autotune_strength,
            proposed_pitch,
            proposed_pitch_threshold,
            input_audio_gain,
            output_audio_gain,
            monitor_audio_gain,
            monitor,
        )

        # Warm up with the same shapes and grad mode as the audio callback, before
        # opening any audio device. Use a tone, since silence can bypass F0.
        # This runs even without compilation: it is what builds the lazily created F0
        # model, whose weight initialisation consumes RNG. Doing that before the seed is
        # applied is what makes two sessions actually line up - otherwise the second
        # session reuses torchcrepe's cached model, draws less randomness than the
        # first, and diverges.
        runtime = self.vc.vc_model
        if not pass_through:
            compiling = runtime.pipeline.compile_session.enabled
            if compiling:
                print("[Realtime] Preparing compilation. The first start may take time.")
            devices = [torch.device(runtime.device)] if torch.device(runtime.device).type == "cuda" else []
            tone = (0.01 * np.sin(
                2 * np.pi * 220 * np.arange(self.vc.block_frame) / 48000
            )).astype(np.float32)
            # Do not feed synthetic audio into the stateful speech detector.
            vad = runtime.vad
            runtime.vad = None
            try:
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    warmup_blocks = 3 if compiling else 1
                    if getattr(runtime, "fcn_session", None) is not None:
                        delay48 = runtime.fcn_session.stream.holdback_samples * 3
                        warmup_blocks = max(warmup_blocks, (delay48 + self.vc.block_frame - 1) // self.vc.block_frame + 2)
                    for _ in range(warmup_blocks):
                        runtime.inference(
                            tone, f0_up_key, index_rate, protect, volume_envelope,
                            f0_autotune, f0_autotune_strength, proposed_pitch,
                            proposed_pitch_threshold,
                        )
                    torch.cuda.synchronize(runtime.device)
            finally:
                runtime.vad = vad
                runtime.flush_buffers()
                if hasattr(runtime, "reset_fcn_stream"):
                    runtime.reset_fcn_stream()
                runtime.consecutive_silence_frames = 0

        # Last, so the stream starts from the same RNG state however much randomness
        # model and kernel setup happened to consume above.
        self.seed = apply_seed()
        if self.seed is not None:
            print(f"[Realtime] Fixed RNG seed {self.seed}.")

    def reset_stream(self):
        """Reset FCN state after dropped capture samples or reconnection."""
        runtime = self.vc.vc_model
        if getattr(runtime, "fcn_session", None) is not None:
            with self.lock:
                runtime.flush_buffers()
                runtime.reset_fcn_stream()
                runtime.consecutive_silence_frames = 0
                if self.vc.sola_buffer is not None:
                    self.vc.sola_buffer.zero_()

    def change_voice(
        self,
        received_data: np.ndarray,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        if self.pass_through:  # through
            vol = float(np.sqrt(np.square(received_data).mean(dtype=np.float32)))
            return received_data, vol, [0, 0, 0], None

        try:
            with self.lock:
                audio, vol, perf = self.vc.on_request(
                    received_data,
                    f0_up_key,
                    index_rate,
                    protect,
                    volume_envelope,
                    f0_autotune,
                    f0_autotune_strength,
                    proposed_pitch,
                    proposed_pitch_threshold,
                )

            return audio, vol, perf, None
        except Exception as error:
            import traceback

            # Track consecutive errors
            if not hasattr(self, '_error_count'):
                self._error_count = 0
                self._last_error_type = None

            error_type = type(error).__name__
            self._error_count += 1

            # Only log every 10th occurrence of the same error to avoid spam
            if self._last_error_type != error_type or self._error_count % 10 == 1:
                print(f"[Voice Conversion Error] {error_type}: {error} (count: {self._error_count})")
                print(traceback.format_exc())
                self._last_error_type = error_type

            # Return silence with same length as input
            return np.zeros(len(received_data), dtype=np.float32), 0, [0, 0, 0], None
