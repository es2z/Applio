import os
import sys
import threading
import numpy as np

sys.path.append(os.getcwd())

from rvc.realtime.audio import Audio
from rvc.realtime.core import VoiceChanger
from rvc.realtime.runtime import RuntimeAudioShape
from tabs.settings.sections.torch_compile import RealtimeCompileSettings


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
        runtime_shape: RuntimeAudioShape | None = None,
        compile_settings: RealtimeCompileSettings | None = None,
        # device: str = "cuda",
    ):
        self.pass_through = pass_through
        self.lock = threading.Lock()
        self.vc = VoiceChanger(
            read_chunk_size,
            cross_fade_overlap_size,
            extra_convert_size,
            model_path,
            index_path,
            f0_method,
            embedder_model,
            embedder_model_custom,
            silent_threshold,
            vad_enabled,
            vad_sensitivity,
            vad_frame_ms,
            sid,
            hybrid_blend_ratio,
            runtime_shape,
            compile_settings,
            # device,
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
            runtime_shape=self.vc.runtime_shape,
        )

    def warmup(self):
        with self.lock:
            return self.vc.warmup()

    def configure_runtime_shape(self, runtime_shape: RuntimeAudioShape):
        with self.lock:
            self.vc.configure_runtime_shape(runtime_shape)
            self.audio.runtime_shape = runtime_shape

    def compile_statuses(self):
        statuses = []
        pipeline = getattr(self.vc.vc_model, "pipeline", None)
        converter = getattr(pipeline, "vc", None)
        rvc_session = getattr(converter, "compile_session", None)
        if rvc_session is not None:
            statuses.append(rvc_session.status)
        f0_model = getattr(pipeline, "f0_model", None)
        if f0_model is not None:
            statuses.extend(
                session.status for session in getattr(f0_model, "_sessions", {}).values()
            )
        return statuses

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
