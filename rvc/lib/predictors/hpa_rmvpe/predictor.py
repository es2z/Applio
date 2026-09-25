"""HPA-RMVPE as an RMVPE0Predictor with a different network.

Upstream's HPARMVPE class is RMVPE's predictor with the network swapped: the same
16 kHz / hop 160 / 128 bin mel front end, the same padding to 32 frames, the same
32000 frame chunks and the same local average cents decoder with thred 0.03. So only
__init__ is replaced here and everything numeric is inherited, which also keeps the
output on exactly RMVPE's frame grid (len(audio) // 160 + 1 frames).

On speech the network reports the pitch of ~20 ms before each frame (measured 16-24 ms
against both RMVPE and CREPE on four recordings, both checkpoints; only 3.5-6 ms on
synthetic harmonic tones - docs/hpa-rmvpe.md). With aligned=True the audio gets
HPA_RMVPE_LAG_FRAMES hops of reflect padding at the end and the first that many frames
are dropped, so frame i is read from the window centred 20 ms later and the grid is
unchanged. The network is untouched either way.

torch.compile follows the F0 TorchCompile setting ("torch_compile_enabled", the one
CREPE uses) on every path. Offline callers (conversion, batch, TTS, training
extraction, the F0 curve tool) see a different length on every call, so they compile
with dynamic shapes and no CUDA graphs; realtime runs one fixed window per block, so it
compiles static shapes and lets the mode decide on CUDA graphs.
"""

import threading

import numpy as np

from rvc.lib.predictors.f0_methods import HPA_RMVPE_LAG_FRAMES
from rvc.lib.predictors.RMVPE import N_CLASS, N_MELS, MelSpectrogram, RMVPE0Predictor
from rvc.lib.predictors.hpa_rmvpe.model import E2E0
from rvc.lib.predictors.hpa_rmvpe.weights import load_state_dict

COMPILE_PROFILES = ("offline", "realtime")


def compile_settings():
    """(enabled, mode) of the F0 TorchCompile setting."""
    from tabs.settings.sections.torch_compile import get_torch_compile_settings

    return get_torch_compile_settings()


class HPARMVPEPredictor(RMVPE0Predictor):
    def __init__(self, variant, device="cpu", compile_profile=None):
        """compile_profile is None (never compile), "offline" or "realtime"; with
        either of the latter the network is compiled only if the setting is on."""
        if compile_profile is not None and compile_profile not in COMPILE_PROFILES:
            raise ValueError(f"Unknown compile profile {compile_profile!r}")
        state, self.weight_sha256 = load_state_dict(variant)
        model = E2E0(1, 1, 16)
        model.load_state_dict(state, strict=True)
        model.eval()
        self.variant = variant
        self.device = device
        self.eager_model = model.to(device)
        self.compile_path = None
        self.model = self.eager_model
        self.compile_enabled, self.compile_mode = False, None
        if compile_profile is not None:
            enabled, mode = compile_settings()
            if enabled:
                self.compile_enabled, self.compile_mode = True, mode
                self._compile(compile_profile)
        # What RMVPE0Predictor.__init__ sets besides the model, built the same way.
        self.resample_kernel = {}
        self.mel_extractor = MelSpectrogram(N_MELS, 16000, 1024, 160, None, 30, 8000).to(device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def _compile(self, profile):
        from rvc.realtime.compile_session import CompiledPath

        realtime = profile == "realtime"
        self.compile_path = CompiledPath(
            "HPA-RMVPE", self.eager_model, True, self.compile_mode, self.device,
            dynamic=not realtime, cudagraphs=None if realtime else False,
        )
        if self.compile_path.compiled is not None:
            # mel2hidden only ever calls self.model(mel_chunk).
            self.model = self.compile_path
            print(f"[HPA-RMVPE] Compiling ({self.compile_mode}, {profile}). The first call may take a while.")

    def get_f0(self, x, filter_radius=0.03, aligned=False):
        if not aligned:
            return self.infer_from_audio(x, thred=filter_radius)
        shift = HPA_RMVPE_LAG_FRAMES
        padded = np.pad(np.asarray(x, dtype=np.float32), (0, shift * 160), mode="reflect")
        return self.infer_from_audio(padded, thred=filter_radius)[shift:]


_offline_cache = {}
_offline_lock = threading.Lock()


def get_offline_predictor(variant, device):
    """One predictor per (variant, device), rebuilt when the compile setting changes.

    Offline conversion used to build RMVPE on every call; doing that here would
    recompile on every call, so the network is kept for the life of the process.
    """
    enabled, mode = compile_settings()
    key = (variant, str(device))
    with _offline_lock:
        cached = _offline_cache.get(key)
        wanted = (bool(enabled), mode if enabled else None)
        if cached is not None and (cached.compile_enabled, cached.compile_mode) == wanted:
            return cached
        _offline_cache.pop(key, None)
        predictor = HPARMVPEPredictor(variant, device, compile_profile="offline")
        _offline_cache[key] = predictor
        return predictor
