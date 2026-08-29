import dataclasses
import pathlib
import librosa
import numpy as np
import resampy
import torch
import torchcrepe
import torchfcpe
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

# from tools.anyf0.rmvpe import RMVPE
from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from rvc.lib.predictors.crepe_models import (
    CREPE_METHOD_TO_MODEL,
    MANGIO_CREPE_METHOD_TO_MODEL,
    resolve_crepe_model,
)
from rvc.lib.predictors.f0 import MANGIO_CREPE
from rvc.configs.config import Config
from tabs.settings.sections.torch_compile import get_torch_compile_settings

config = Config()


@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    sample_rate: int = 44100
    hop_length: int = 512
    f0_min: int = 50
    f0_max: int = 1600
    method: str = "rmvpe"
    x: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.x, self.sample_rate = librosa.load(self.wav_path, sr=self.sample_rate)

    @property
    def hop_size(self):
        return self.hop_length / self.sample_rate

    @property
    def wav16k(self):
        return resampy.resample(self.x, self.sample_rate, 16000)

    def extract_f0(self):
        f0 = None
        method = self.method
        if method in CREPE_METHOD_TO_MODEL:
            wav16k_torch = torch.FloatTensor(self.wav16k).unsqueeze(0).to(config.device)
            # Get torch.compile settings
            compile_enabled, compile_mode = get_torch_compile_settings()
            f0 = torchcrepe.predict(
                wav16k_torch,
                sample_rate=16000,
                hop_length=160,
                batch_size=512,
                fmin=self.f0_min,
                fmax=self.f0_max,
                model=resolve_crepe_model(method),
                device=config.device,
                compile_model=compile_enabled,
                compile_mode=compile_mode,
            )
            f0 = f0[0].cpu().numpy()
        elif method in MANGIO_CREPE_METHOD_TO_MODEL:
            wav16k = self.wav16k
            model = MANGIO_CREPE(
                device=config.device, sample_rate=16000, hop_size=160
            )
            f0 = model.get_f0(
                wav16k,
                self.f0_min,
                self.f0_max,
                len(wav16k) // 160,
                resolve_crepe_model(method),
            )
        elif method == "fcpe":
            audio = librosa.to_mono(self.x)
            audio_length = len(audio)
            f0_target_length = (audio_length // self.hop_length) + 1
            audio = (
                torch.from_numpy(audio)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(config.device)
            )
            model = torchfcpe.spawn_bundled_infer_model(device=config.device)

            f0 = model.infer(
                audio,
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                interp_uv=False,
                output_interp_target_length=f0_target_length,
            )
            f0 = f0.squeeze().cpu().numpy()
        elif method == "rmvpe":
            model_rmvpe = RMVPE0Predictor(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                device=config.device,
                # hop_length=80
            )
            f0 = model_rmvpe.infer_from_audio(self.wav16k, thred=0.03)

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self.hz_to_cents(f0, librosa.midi_to_hz(0))

    def plot_f0(self, f0):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(f0)
        plt.title(self.method)
        plt.xlabel("Time (frames)")
        plt.ylabel("F0 (cents)")
        plt.show()

    @staticmethod
    def hz_to_cents(F, F_ref=55.0):
        F_temp = np.array(F).astype(float)
        F_temp[F_temp == 0] = np.nan
        F_cents = 1200 * np.log2(F_temp / F_ref)
        return F_cents
