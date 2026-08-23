import os
import sys
from pathlib import Path

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from torchfcpe import spawn_bundled_infer_model
import torchcrepe
from swift_f0 import SwiftF0
import numpy as np
import onnxruntime as ort
from rvc.lib.predictors import onnxcrepe
from tabs.settings.sections.torch_compile import get_torch_compile_settings
from tabs.settings.sections.torch_compile import (
    RealtimeCompileSettings,
)


class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03):
        f0 = self.model.infer_from_audio(x, thred=filter_radius)
        return f0


class CREPE:
    def __init__(
        self,
        device,
        sample_rate=16000,
        hop_size=160,
        compile_settings: RealtimeCompileSettings | None = None,
        compile_signature: str = "offline",
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.compile_settings = compile_settings
        self.compile_signature = compile_signature
        self._sessions = {}

    def _predict(self, model, *args, **kwargs):
        if self.compile_settings is None:
            compile_enabled, compile_mode = get_torch_compile_settings()
            kwargs["compile_model"] = compile_enabled
            kwargs["compile_mode"] = compile_mode
            return torchcrepe.predict(*args, model=model, **kwargs)
        from rvc.realtime.compile_session import CrepeSession

        session = self._sessions.get(model)
        if session is None:
            session = CrepeSession(
                self.device,
                model,
                self.compile_settings,
                self.compile_signature,
            )
            self._sessions[model] = session
        return session.predict(*args, model=model, **kwargs)

    def finish_compile_warmup(self):
        for session in self._sessions.values():
            session.finish_warmup()

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        batch_size = 512

        f0, pd = self._predict(
            model,
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
            decoder=torchcrepe.decode.weighted_argmax,
        )
        # Apply median filter to both f0 and periodicity (matching reference implementation)
        f0 = torchcrepe.filter.median(f0, 3)
        pd = torchcrepe.filter.median(pd, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()

        return f0


class MANGIO_CREPE(CREPE):
    def __init__(
        self,
        device,
        sample_rate=16000,
        hop_size=160,
        compile_settings: RealtimeCompileSettings | None = None,
        compile_signature: str = "offline",
    ):
        super().__init__(
            device,
            sample_rate,
            hop_size,
            compile_settings,
            compile_signature,
        )

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

        # Normalize audio (mangio-crepe specific)
        x = x.astype(np.float32)
        scale = float(np.quantile(np.abs(x), 0.999))
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            return np.zeros(p_len, dtype=np.float32)
        x /= scale

        # Convert to tensor and move to device
        audio = torch.from_numpy(x).to(self.device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)

        # Handle multi-channel audio
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()

        # Predict using torchcrepe with periodicity (Applio improvement)
        pitch, pd = self._predict(
            model,
            audio,
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            batch_size=self.hop_size * 2,
            device=self.device,
            pad=True,
            return_periodicity=True,
        )

        # Apply periodicity filter (Applio improvement for noise reduction)
        pd = torchcrepe.filter.median(pd, 3)
        pitch = torchcrepe.filter.median(pitch, 3)
        pitch[pd < 0.1] = 0

        # Resize the pitch for final f0 (mangio-crepe specific)
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source
        )
        f0 = np.nan_to_num(target)

        return f0


class FCNF0PP:
    """Official PENN FCNF0++ inference with RVC-compatible framing."""

    HOP_SECONDS = 0.01
    CENTER = "half-hop"
    DECODER = "viterbi"
    BATCH_SIZE = 2048
    # PERIODICITY_THRESHOLD = 0.065
    PERIODICITY_THRESHOLD = 0.04

    def __init__(
        self,
        device,
        sample_rate=16000,
        hop_size=160,
        compile_settings: RealtimeCompileSettings | None = None,
        compile_signature: str = "offline",
    ):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.compile_settings = compile_settings
        self.compile_signature = compile_signature
        self._sessions = {}
        self.checkpoint = Path(
            now_dir, "rvc", "models", "predictors", "fcnf0++.pt"
        )
        try:
            import penn
        except Exception as error:
            raise RuntimeError(
                "FCNF0++ requires the official 'penn' package and a torbi "
                "binary compatible with the installed PyTorch/CUDA build."
            ) from error
        self.penn = penn

    def _predict(self, *args, **kwargs):
        if self.compile_settings is None:
            return self.penn.from_audio(*args, **kwargs)
        from rvc.realtime.compile_session import PennSession

        session = self._sessions.get("fcnf0++")
        if session is None:
            session = PennSession(
                self.penn,
                self.device,
                self.compile_settings,
                self.compile_signature,
            )
            self._sessions["fcnf0++"] = session
        return session.predict(self.penn.from_audio, *args, **kwargs)

    def finish_compile_warmup(self):
        for session in self._sessions.values():
            session.finish_warmup()

    def _gpu_index(self):
        if self.device.type != "cuda":
            return None
        if self.device.index is not None:
            return self.device.index
        return torch.cuda.current_device()

    @staticmethod
    def _match_length(values, p_len):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] >= p_len:
            return values[:p_len]
        return np.pad(values, (0, p_len - values.shape[0]))

    def _voiced_mask(self, periodicity, f0_min=None, f0_max=None):
        """Return RVC's voiced/unvoiced mask without changing pitch timing."""
        return periodicity >= self.PERIODICITY_THRESHOLD

    def _postprocess_pitch(self, pitch):
        """Hook for variants; the official FCNF0++ contour is unchanged."""
        return pitch

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None):
        if torch.is_tensor(x):
            audio = x.detach().float().cpu()
        else:
            audio = torch.as_tensor(x, dtype=torch.float32)

        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        audio = audio.reshape(1, -1).contiguous()

        if p_len is None:
            hop_samples = round(self.sample_rate * self.HOP_SECONDS)
            p_len = audio.shape[-1] // hop_samples
        p_len = int(p_len)
        if p_len <= 0:
            return np.zeros(max(0, p_len), dtype=np.float32)

        pitch, periodicity = self._predict(
            audio,
            sample_rate=self.sample_rate,
            hopsize=self.HOP_SECONDS,
            fmin=f0_min,
            fmax=f0_max,
            checkpoint=self.checkpoint if self.checkpoint.is_file() else None,
            batch_size=self.BATCH_SIZE,
            center=self.CENTER,
            decoder=self.DECODER,
            interp_unvoiced_at=None,
            gpu=self._gpu_index(),
        )

        pitch = self._match_length(pitch.detach().float().cpu().numpy(), p_len)
        periodicity = self._match_length(
            periodicity.detach().float().cpu().numpy(), p_len
        )
        pitch = np.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0)
        periodicity = np.nan_to_num(
            periodicity, nan=0.0, posinf=0.0, neginf=0.0
        )
        pitch = self._postprocess_pitch(pitch)
        pitch[~self._voiced_mask(periodicity, f0_min, f0_max)] = 0.0
        pitch[(pitch < f0_min) | (pitch > f0_max)] = 0.0
        return pitch.astype(np.float32, copy=False)


class FCNF0PP_SPEECH(FCNF0PP):
    """FCNF0++ with speech-oriented voiced/unvoiced stabilization.

    It keeps the official PENN model and framing, then applies small RVC-side
    filters to reduce audible pitch jitter and voiced/unvoiced chatter.
    """

    # ======================================================================
    # fcnf0++-speech の音質調整はここです
    # ======================================================================
    # 1. UIのF0 methodで「fcnf0++-speech」を選びます。
    # 2. 下の値を一項目だけ変更して保存します。
    # 3. 実行中のRVCを完全に終了してから起動し直します。
    #    （このファイルは起動時に読み込まれるため、画面の変換停止／再開だけでは
    #     変更が反映されない場合があります。）
    # 4. 同じ入力・モデル・index rate・pitch shiftで録音して比較します。
    #
    # おすすめの試験順:
    # A. 声がかすれる／細い／小声が途切れる
    #      VOICING_ON_THRESHOLD  = 0.020
    #      VOICING_OFF_THRESHOLD = 0.010
    #    まだ途切れる場合だけ 0.010 / 0.003 を試します。低くしすぎると、
    #    息・摩擦音・環境音まで音程として拾います。ON/OFFは必ず対で変更します。
    #
    # B. Mangioより透明感・広がり・ビブラートが弱い／丸く聞こえる
    #      PITCH_MEDIAN_FILTER_SIZE = 1
    #    1はF0平滑化なし、3は現在の推奨値です。1でザラつきや音程飛びが増えたら
    #    3へ戻してください。5はさらに滑らかですが、通常は鈍くなるので非推奨です。
    #
    # C. 子音境界がにじむ／発音の輪郭が弱い
    #      PERIODICITY_MEDIAN_FILTER_SIZE = 1
    #    1は有声判定の平滑化なし、3は10 msだけ落ちる判定穴を抑えます。
    #
    # D. coarse F0修正そのものの音色差を比較する
    #      USE_MEL_SCALED_REALTIME_COARSE = False
    #    Trueが正しいmel量子化、Falseがこのforkの従来リアルタイム式です。
    #    高域のpitch embeddingが変わるため、響きの差を感じる場合があります。
    #
    # 最初はA、次にBを別々に試してください。同時に変更すると、どちらが効いたか
    # 判断できません。CENTER/HOP_SECONDS/f0_min/f0_maxはこの比較中は変更しません。

    # decoderはviterbiを推奨します。argmaxは速い一方、実測でF0ジッターと
    # octave errorが増えたため、音質比較中は変更しないでください。
    DECODER = "viterbi"

    # Bの調整箇所: 1=平滑化なし、3=現在値、5=強い平滑化（奇数のみ）。
    PITCH_MEDIAN_FILTER_SIZE = 5

    # Cの調整箇所: 1=無効、3=現在値（奇数のみ）。
    PERIODICITY_MEDIAN_FILTER_SIZE = 5

    # Aの調整箇所（上から Stable/current、Balanced、Open）:
    #   0.025 / 0.015 = 誤検出を抑える現在値。小声では細くなる場合があります。
    #   0.020 / 0.010 = 最初に試す推奨値。響きと誤検出の中間です。
    #   0.010 / 0.003 = 最も開いた設定。息や背景音を拾う可能性があります。
    VOICING_ON_THRESHOLD = 0.010
    VOICING_OFF_THRESHOLD = 0.003

    # Dの調整箇所: True=正しいmel量子化、False=従来リアルタイム式。
    # このフラグが影響するのはfcnf0++-speechだけです。
    USE_MEL_SCALED_REALTIME_COARSE = False

    @staticmethod
    def _median_filter(values, size):
        size = int(size)
        if size <= 1 or values.size < 2:
            return values.copy()
        if size % 2 == 0:
            raise ValueError("FCNF0++ median filter size must be odd")
        radius = size // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, size)
        return np.median(windows, axis=-1).astype(values.dtype, copy=False)

    def _postprocess_pitch(self, pitch):
        return self._median_filter(pitch, self.PITCH_MEDIAN_FILTER_SIZE)

    def _normalize_periodicity(self, periodicity, f0_min, f0_max):
        """Remove entropy's frequency-range-dependent uniform floor."""
        pitch_bins = int(getattr(self.penn, "PITCH_BINS", 1440))
        penn_fmin = float(getattr(self.penn, "FMIN", 31.0))
        cents_per_bin = float(getattr(self.penn, "CENTS_PER_BIN", 5.0))

        def frequency_to_bin(frequency, quantize):
            cents = 1200.0 * np.log2(float(frequency) / penn_fmin)
            index = int(quantize(cents / cents_per_bin))
            return min(pitch_bins - 1, max(0, index))

        min_index = frequency_to_bin(f0_min, np.floor)
        max_index = frequency_to_bin(f0_max, np.ceil)
        allowed_bins = max(2, max_index - min_index)
        uniform_floor = 1.0 - np.log(allowed_bins) / np.log(pitch_bins)
        return np.clip(
            (periodicity - uniform_floor) / (1.0 - uniform_floor),
            0.0,
            1.0,
        )

    def _voiced_mask(self, periodicity, f0_min=None, f0_max=None):
        periodicity = self._normalize_periodicity(
            periodicity, f0_min, f0_max
        )
        periodicity = self._median_filter(
            periodicity, self.PERIODICITY_MEDIAN_FILTER_SIZE
        )
        voiced = np.zeros(periodicity.shape, dtype=bool)
        active = False
        for index, confidence in enumerate(periodicity):
            threshold = (
                self.VOICING_OFF_THRESHOLD
                if active
                else self.VOICING_ON_THRESHOLD
            )
            active = bool(confidence >= threshold)
            voiced[index] = active
        return voiced


class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_bundled_infer_model(self.device)

    def get_f0(self, x, p_len=None, filter_radius=0.006):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0


class SWIFT:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = "cpu"
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, confidence_threshold=0.9):
        if torch.is_tensor(x):
            x = x.cpu().numpy()

        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        f0_min = max(f0_min, 46.875)
        f0_max = min(f0_max, 2093.75)

        detector = SwiftF0(
            fmin=f0_min, fmax=f0_max, confidence_threshold=confidence_threshold
        )
        result = detector.detect_from_array(x, self.sample_rate)
        if len(result.timestamps) == 0:
            return np.zeros(p_len)
        target_time = (
            np.arange(p_len) * self.hop_size + self.hop_size / 2
        ) / self.sample_rate
        pitch = np.nan_to_num(result.pitch_hz, nan=0.0)
        pitch[~result.voicing] = 0.0
        f0 = np.interp(target_time, result.timestamps, pitch, left=0.0, right=0.0)

        return f0


class CREPE_ONNX:
    def __init__(self, device, model_path, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model_path = model_path

        # Setup ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

        # Ensure audio is float32
        x = x.astype(np.float32)

        # Calculate precision in milliseconds based on hop_size
        # hop_size is in samples at sample_rate
        precision = (self.hop_size / self.sample_rate) * 1000  # convert to ms

        # Use onnxcrepe.predict for proper CREPE ONNX inference
        # Use weighted_argmax decoder (same as reference RVC project)
        f0, pd = onnxcrepe.predict(
            self.session,
            x,
            self.sample_rate,
            precision=precision,
            fmin=f0_min,
            fmax=f0_max,
            batch_size=256,
            return_periodicity=True,
            decoder=onnxcrepe.decode.weighted_argmax,
        )

        # Apply filtering (matching reference RVC project):
        # 1. Apply median filter on f0
        f0 = onnxcrepe.filter.median(f0, 3)

        # 2. Apply median filter on periodicity
        pd = onnxcrepe.filter.median(pd, 3)

        # 3. Zero out low confidence predictions
        f0[pd < 0.1] = 0
        f0 = f0.squeeze()

        # Ensure f0 is the correct length
        if len(f0) != p_len:
            if len(f0) > 0:
                # Use numpy's interp which is simpler and handles edges better
                x_old = np.linspace(0, len(f0) - 1, len(f0))
                x_new = np.linspace(0, len(f0) - 1, p_len)
                f0 = np.interp(x_new, x_old, f0)
            else:
                f0 = np.zeros(p_len, dtype=np.float32)

        return f0.astype(np.float32)
