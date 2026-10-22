import os
import sys
import threading
import numpy as np
import torch

sys.path.append(os.getcwd())

from rvc.realtime.rng import apply_seed
from rvc.realtime.audio import Audio
from rvc.realtime.core import VoiceChanger
from rvc.realtime.options import RealtimeOptions
from rvc.realtime.warmup import warm_up


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
        options: RealtimeOptions = None,
        # device: str = "cuda",
    ):
        self.pass_through = pass_through
        self.options = options or RealtimeOptions()
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
            options=self.options,
        )
        ring = self.options.transport == "ring"

        # Warm up with the same shapes and grad mode as the audio callback, before
        # opening any audio device (rvc/realtime/warmup.py says why it always runs).
        # The ring transport also times a few blocks for its initial output reserve.
        warmup_timings = []
        if not pass_through:
            warmup_timings = warm_up(
                self.vc, f0_up_key, index_rate, protect, volume_envelope,
                f0_autotune, f0_autotune_strength, proposed_pitch,
                proposed_pitch_threshold,
                timing_blocks=5 if ring else 0,
            )

        audio_args = (
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
        if ring:
            from rvc.realtime.ring_transport import RingAudio

            self.audio = RingAudio(
                *audio_args,
                block_frame=self.vc.block_frame,
                stream_mode=self.options.ring_stream_mode,
                warmup_timings_ms=warmup_timings,
                algorithmic_delay_ms=getattr(self.vc.vc_model, "algorithmic_delay_ms", 0.0),
            )
        else:
            self.audio = Audio(*audio_args)

        # Last, so the stream starts from the same RNG state however much randomness
        # model and kernel setup happened to consume above.
        self.seed = apply_seed()
        if self.seed is not None:
            print(f"[Realtime] Fixed RNG seed {self.seed}.")
        # Draw the time-locked noise from the seeded generator, so a fixed seed fixes
        # it too. Without those options this draws nothing.
        reset = getattr(self.vc.vc_model, "reset_stream_state", None)
        if reset is not None and not pass_through:
            reset()

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
