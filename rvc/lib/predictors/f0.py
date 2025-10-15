import os
import torch

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from torchfcpe import spawn_bundled_infer_model
import torchcrepe
from swift_f0 import SwiftF0
import numpy as np
import onnxruntime as ort
from rvc.lib.predictors import onnxcrepe


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
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        batch_size = 512

        f0, pd = torchcrepe.predict(
            x.float().to(self.device).unsqueeze(dim=0),
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
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


class MANGIO_CREPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, x, f0_min=50, f0_max=1100, p_len=None, model="full"):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

        # Normalize audio (mangio-crepe specific)
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)

        # Convert to tensor and move to device
        audio = torch.from_numpy(x).to(self.device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)

        # Handle multi-channel audio
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()

        # Predict using torchcrepe with periodicity (Applio improvement)
        pitch, pd = torchcrepe.predict(
            audio,
            self.sample_rate,
            self.hop_size,
            f0_min,
            f0_max,
            model=model,
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
