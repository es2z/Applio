import os
import sys
import faiss
import numpy as np
import torch
import torch.nn.utils.parametrize
import torch.nn.functional as F
import torchaudio.transforms as tat
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.utils.torch import circular_write
from rvc.realtime.compile_session import CompileSession, load_settings
from rvc.realtime.options import RealtimeOptions
from rvc.realtime.synthesis import (
    SynthesisPlan,
    realtime_infer,
    supports_excitation_control,
)
from rvc.configs.config import Config
from rvc.infer.pipeline import Autotune, AudioProcessor
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.predictors.f0_quantization import (
    COARSE_MIN,
    align_profile_coarse,
    quantize_f0,
    recorded_coarse_max,
)
from rvc.lib.predictors.crepe_models import (
    CREPE_METHOD_TO_MODEL,
    MANGIO_CREPE_METHOD_TO_MODEL,
    resolve_crepe_model,
)
from rvc.lib.predictors.f0 import (
    CREPE,
    FCPE,
    MANGIO_CREPE,
    RMVPE,
    SWIFT,
)
from rvc.lib.utils import (
    checkpoint_gen_istft,
    checkpoint_text_enc_hidden_dim,
    embedder_forward,
    load_embedding,
    warn_on_feature_scale_mismatch,
    HubertModelWithFinalProj,
)


class RealtimeVoiceConverter:
    """
    A class for performing realtime voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self, weight_root):
        """
        Initializes the RealtimeVoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load configuration
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.use_f0 = None  # Whether the model uses F0
        # None: Synthesizer.infer exactly as always. A SynthesisPlan: the realtime call
        # in rvc/realtime/synthesis.py. Set once per session, before any compilation.
        self.synthesis = None
        # Clip the generator output to +-1 here (as always), or leave it to the final
        # output (clip_stage "output"). Set once per session by configure_stream.
        self.clip_output = True
        self.last_peak = None
        # load weights and setup model network.
        self.load_model(weight_root)
        self.setup_network()

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.

        Args:
            weight_root (str): Path to the model weights.
        """
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=True)
            if os.path.isfile(weight_root)
            else None
        )

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = checkpoint_text_enc_hidden_dim(self.cpt)
            self.vocoder = self.cpt.get("vocoder", "HiFi-GAN")
            # Only SiFi-GAN reads this; it decides the shape of the filter network's
            # residual blocks, so the model cannot be rebuilt without it.
            self.sifigan_filter_resblock = self.cpt.get(
                "sifigan_filter_resblock", "rvc"
            )
            # Only CodenameRingFormer and APEX-GAN read these; (None, None) for anything else
            # leaves Synthesizer's defaults in place. They decide the width of conv_post
            # and noise_convs, so that decoder cannot be rebuilt without them.
            gen_istft_n_fft, gen_istft_hop_size = checkpoint_gen_istft(self.cpt)
            print(f"[Realtime] Loading model with vocoder: {self.vocoder}")
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                vocoder=self.vocoder,
                sifigan_filter_resblock=self.sifigan_filter_resblock,
                gen_istft_n_fft=gen_istft_n_fft,
                gen_istft_hop_size=gen_istft_hop_size,
            )

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            strip_parametrizations(self.net_g)
            self.net_g = self.net_g.to(self.config.device).float()
            self.net_g.eval()
            # self.net_g.remove_weight_norm()

    def inference(
        self,
        feats: Tensor,
        p_len: Tensor,
        sid: Tensor,
        pitch: Tensor,
        pitchf: Tensor,
        infer_audio=None,
        noise=(None, None, None),
    ):
        infer_audio = infer_audio or self.infer_audio
        if self.synthesis is None:
            output = infer_audio(feats, p_len, pitch, pitchf, sid)
        else:
            output = infer_audio(feats, p_len, pitch, pitchf, sid, *noise)

        # The generator's own peak, before anything clips it, for the level readout.
        # A device tensor: reading it is left to the status line, off the audio thread.
        self.last_peak = output.detach().abs().amax()
        if not getattr(self, "clip_output", True):
            return output
        return torch.clip(output, -1.0, 1.0, out=output)

    def infer_audio(
        self, feats, p_len, pitch, pitchf, sid, z_noise=None, dec_noise=None, dec_phase0=None
    ):
        if self.synthesis is None:
            return self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
        return realtime_infer(
            self.net_g,
            feats,
            p_len,
            pitch,
            pitchf,
            sid,
            self.synthesis,
            z_noise=z_noise,
            dec_noise=dec_noise,
            dec_phase0=dec_phase0,
        )[0, 0]


class Realtime_Pipeline:
    def __init__(
        self,
        vc: RealtimeVoiceConverter,
        hubert_model: HubertModelWithFinalProj = None,
        index=None,
        big_npy=None,
        f0_method: str = "rmvpe",
        sid: int = 0,
        hybrid_blend_ratio: float = 0.5,
        fcn_profile=None,
        options: RealtimeOptions = None,
    ):
        self.vc = vc
        self.options = options or RealtimeOptions()
        # Per-session stream layout and the time-locked state behind the opt-in options;
        # configure_stream fills these in once the window size is known.
        self.stream = None
        self.voiced_buffer = None
        self.z_noise = None
        self.dec_noise = None
        self.dec_phase0 = None
        self.f0_cache_frames = None
        self._last_voiced = None
        self.hubert_model = hubert_model
        self.index = index
        self.big_npy = big_npy
        self.use_f0 = vc.use_f0
        self.version = vc.version
        self.f0_method = f0_method
        self.hybrid_blend_ratio = hybrid_blend_ratio
        self.sample_rate = 16000
        self.tgt_sr = vc.tgt_sr
        self.window = 160
        self.model_window = self.tgt_sr // 100
        # The model's coarse F0 range: what the pitch embedding is indexed by, and the
        # search range of the predictors that take one, exactly as during extraction.
        self.f0_min = COARSE_MIN
        self.f0_max = recorded_coarse_max(vc.cpt)
        self.device = vc.config.device
        self.sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
        self.autotune = Autotune()
        self.resamplers = {}
        self.f0_model = None
        self.f0_model_secondary = None
        self.fcn_pitch = None
        from rvc.lib.predictors.f0_methods import (
            FCN_METHODS,
            FCNF0PP_METHODS,
            HPA_RMVPE_METHODS,
            gap_filled_base,
            hpa_rmvpe_variant,
        )

        self.hpa_rmvpe_aligned = False
        if f0_method in FCN_METHODS:
            from rvc.lib.predictors.fcn import FCNPredictor, resolve_profile

            if torch.device(self.device).type != "cuda":
                raise ValueError("FCN realtime requires CUDA")
            checkpoint = (vc.cpt or {}).get("f0_extraction", {})
            profile = align_profile_coarse(
                resolve_profile(f0_method, fcn_profile, checkpoint.get("profile")),
                self.f0_max,
            )
            if checkpoint.get("method") and checkpoint["method"] != f0_method:
                print(f"FCN method differs from checkpoint: {checkpoint['method']} -> {f0_method}")
            self.f0_model = FCNPredictor(self.device, f0_method, profile)
        elif f0_method in FCNF0PP_METHODS:
            from rvc.lib.predictors.fcnf0pp import FCNF0PPPredictor, resolve_profile

            # Built eagerly so a bad profile or a missing weight fails before audio
            # starts, and so the model is warm for the first block.
            checkpoint = (vc.cpt or {}).get("f0_extraction", {})
            profile = align_profile_coarse(
                resolve_profile(f0_method, fcn_profile, checkpoint.get("profile")),
                self.f0_max,
            )
            if checkpoint.get("method") and checkpoint["method"] != f0_method:
                print(f"F0 method differs from checkpoint: {checkpoint['method']} -> {f0_method}")
            self.f0_model = FCNF0PPPredictor(self.device, f0_method, profile)
        elif gap_filled_base(f0_method) in HPA_RMVPE_METHODS:
            from rvc.lib.predictors.hpa_rmvpe import HPARMVPEPredictor

            # Built eagerly, like FCNF0++, so a first-use download or a bad weight
            # happens before audio starts; its own compile path joins the session so
            # the warm-up compiles it and the status reports it.
            variant, self.hpa_rmvpe_aligned = hpa_rmvpe_variant(gap_filled_base(f0_method))
            self.f0_model = HPARMVPEPredictor(
                variant, self.device, compile_profile="realtime"
            )
        self.compile_session = CompileSession(
            load_settings(),
            lambda feats: embedder_forward(hubert_model, feats),
            self.vc.infer_audio,
            self.device,
            f0=getattr(self.f0_model, "compile_path", None),
        )

    def configure_stream(
        self,
        p_len: int,
        skip_head: int,
        return_length: int,
        shift_frames,
        trim: bool,
    ):
        """Fix the window layout for this session and set up the opt-in state.

        p_len: frames (100 Hz) in the conversion window. skip_head / return_length: the
        frames the "trim" output position leaves as context / renders. shift_frames: how
        many frames the window moves per block, or None when a block is not a whole
        number of 10 ms frames (the time-locked options need it to be).
        """
        options = self.options
        hop = self.model_window
        self.stream = {
            "p_len": int(p_len),
            "skip_head": int(skip_head),
            "return_length": int(return_length),
            "shift": shift_frames,
            "trim": bool(trim),
        }
        net_g = self.vc.net_g
        excitation = supports_excitation_control(net_g) and bool(self.use_f0)
        needs_shift = options.time_locked_noise or options.phase_continuity or options.f0_cache
        if needs_shift and shift_frames is None:
            print(
                "[Realtime] Time-locked noise, phase continuity and the F0 cache need a "
                "Chunk Size that is a whole number of 10 ms frames (a multiple of 40 ms "
                "works); they are off for this session."
            )
        locked = shift_frames is not None
        if (options.phase_continuity or options.apex_noise_envelope != "window") and not excitation:
            print(
                "[Realtime] Phase continuity and the APEX-GAN noise envelope apply to "
                "APEX-GAN models only; this model ignores them."
            )

        envelope_reference = None
        if excitation and options.apex_noise_envelope == "fixed":
            envelope_reference = net_g.dec.noise_envelope_reference(
                options.apex_noise_voiced_ratio
            )
            print(
                f"[Realtime] APEX-GAN noise envelope reference {envelope_reference:.6g} "
                f"(voiced ratio {options.apex_noise_voiced_ratio:.2f})."
            )

        if options.needs_realtime_synthesis:
            self.vc.synthesis = SynthesisPlan(
                skip_head=int(skip_head) if trim else None,
                return_length=int(return_length) if trim else None,
                envelope_reference=envelope_reference,
                latent_scale=float(options.latent_noise_scale),
            )
        else:
            self.vc.synthesis = None
        self.vc.clip_output = options.clip_stage == "generator"
        plan = self.vc.synthesis

        rendered = int(return_length) if trim else int(p_len)
        self.z_noise = self.dec_noise = self.dec_phase0 = None
        if plan is not None and options.time_locked_noise and locked:
            channels = net_g.enc_p.out_channels
            self.z_noise = torch.randn(1, channels, int(p_len), device=self.device)
            if excitation:
                self.dec_noise = torch.randn(1, int(p_len) * hop, 1, device=self.device)
        if plan is not None and options.phase_continuity and locked and excitation:
            # float64: the decoder accumulates this phase in float64 too, so that two
            # overlapping windows reach the same samples (apex_gan.py says why).
            self.dec_phase0 = torch.zeros(1, device=self.device, dtype=torch.float64)
        self._rendered_frames = rendered

        self.f0_cache_frames = None
        if options.f0_cache and locked:
            margin = int(round(options.f0_cache_margin_ms / 10.0))
            self.f0_cache_frames = (int(shift_frames), int(shift_frames) + margin)
        self.voiced_buffer = None

    def reset_stream_state(self):
        """Forget the time-locked state, e.g. after the warm-up's synthetic tone."""
        if self.z_noise is not None:
            self.z_noise.normal_()
        if self.dec_noise is not None:
            self.dec_noise.normal_()
        if self.dec_phase0 is not None:
            self.dec_phase0.zero_()
        if self.voiced_buffer is not None:
            self.voiced_buffer.zero_()

    def _advance_noise(self):
        """Move the time-locked noise by one block and draw only the new frames."""
        shift = self.stream["shift"] if self.stream else None
        if not shift:
            return
        for buffer, step in ((self.z_noise, shift), (self.dec_noise, shift * self.model_window)):
            if buffer is None:
                continue
            dim = 2 if buffer is self.z_noise else 1
            length = buffer.shape[dim]
            if step >= length:
                buffer.normal_()
                continue
            if dim == 2:
                buffer[:, :, :-step] = buffer[:, :, step:].clone()
                buffer[:, :, -step:].normal_()
            else:
                buffer[:, :-step] = buffer[:, step:].clone()
                buffer[:, -step:].normal_()

    def _noise_inputs(self):
        """(z_noise, dec_noise, dec_phase0) for this block's generator call."""
        plan = self.vc.synthesis
        if plan is None:
            return (None, None, None)
        z_noise = dec_noise = None
        if self.z_noise is not None:
            z_noise = self.z_noise[:, :, plan.flow_head :]
        if self.dec_noise is not None:
            hop = self.model_window
            if plan.skip_head is not None:
                start = int(plan.skip_head) * hop
                dec_noise = self.dec_noise[:, start : start + int(plan.return_length) * hop]
            else:
                dec_noise = self.dec_noise
        return (z_noise, dec_noise, self.dec_phase0)

    def _advance_phase(self, pitchf_rendered):
        """Carry the sine phase to where the next window's rendered region starts."""
        if self.dec_phase0 is None or not self.stream or not self.stream["shift"]:
            return
        shift = self.stream["shift"]
        # The same float64 arithmetic SineGenerator uses for one frame's advance.
        increments = (pitchf_rendered[0, :shift].double() / float(self.tgt_sr)) * float(
            self.model_window
        )
        self.dec_phase0.add_(torch.remainder(increments, 1.0).sum()).remainder_(1.0)

    def _write_pitch(self, new, buffer):
        """Write a window's F0 into its buffer: whole (as always) or, with the F0 cache,
        only the newest frames after shifting the settled ones along."""
        if self.f0_cache_frames is None:
            circular_write(new, buffer)
            return
        shift, recompute = self.f0_cache_frames
        length = buffer.shape[0]
        if shift >= length:
            circular_write(new, buffer)
            return
        buffer[:-shift] = buffer[shift:].clone()
        count = min(recompute, new.shape[0], length)
        buffer[-count:] = new[-count:]

    def get_f0(
        self,
        x: Tensor,
        pitch: Tensor = None,
        pitchf: Tensor = None,
        f0_up_key: int = 0,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1.0,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.
        """

        from rvc.lib.predictors.f0_gap_fill import fill_unvoiced_gaps
        from rvc.lib.predictors.f0_methods import (
            FCN_METHODS,
            FCNF0PP_METHODS,
            GAP_FILLED_METHODS,
            HPA_RMVPE_METHODS,
            gap_filled_base,
        )

        if self.f0_method in FCN_METHODS:
            if self.fcn_pitch is None:
                raise RuntimeError("FCN realtime requires a synchronized stream window")
            return self._fcn_adjusted_pitch(f0_up_key, f0_autotune, f0_autotune_strength, proposed_pitch, proposed_pitch_threshold)
        if torch.is_tensor(x):
            # If the input is a tensor, it will need to be converted to numpy array to calculate with RMVPE and FCPE.
            x = x.cpu().numpy()

        # Voicing from the predictor, kept through pitch correction (FCNF0++ only).
        voiced = None
        if self.f0_method in FCNF0PP_METHODS:
            # Stateless: the whole window is recomputed each block, as for RMVPE/CREPE.
            f0 = self.f0_model.get_f0(x, x.shape[0] // self.window)
            voiced = f0 > 0
        elif gap_filled_base(self.f0_method) == "rmvpe":
            if self.f0_model is None:
                self.f0_model = RMVPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(x, filter_radius=0.03)
        elif gap_filled_base(self.f0_method) in HPA_RMVPE_METHODS:
            f0 = self.f0_model.get_f0(
                x, filter_radius=0.03, aligned=self.hpa_rmvpe_aligned
            )
        elif self.f0_method == "fcpe":
            if self.f0_model is None:
                self.f0_model = FCPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(x, x.shape[0] // self.window, filter_radius=0.006)
        elif self.f0_method == "swift":
            if self.f0_model is None:
                self.f0_model = SWIFT(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(
                x,
                self.f0_min,
                self.f0_max,
                x.shape[0] // self.window,
                confidence_threshold=0.887,
            )
        elif self.f0_method in CREPE_METHOD_TO_MODEL:
            model_size = resolve_crepe_model(self.f0_method)

            if self.f0_model is None:
                self.f0_model = CREPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    # hop_size=self.window,
                    # hop_size=164,
                    hop_size=160,
                )
            f0 = self.f0_model.get_f0(
                x,
                self.f0_min,
                self.f0_max,
                x.shape[0] // self.window,
                model=model_size,
            )
        elif self.f0_method in MANGIO_CREPE_METHOD_TO_MODEL:
            model_size = resolve_crepe_model(self.f0_method)

            if self.f0_model is None:
                self.f0_model = MANGIO_CREPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.f0_model.get_f0(
                x,
                self.f0_min,
                self.f0_max,
                x.shape[0] // self.window,
                model=model_size,
            )
        # The predictor's own voicing, before any gap fill: what "protect" can use.
        self._last_voiced = (voiced if voiced is not None else f0 > 0).copy()
        if self.f0_method in GAP_FILLED_METHODS:
            # The whole window is recomputed each block, so this fills the window.
            f0 = fill_unvoiced_gaps(f0)

        # f0 adjustments
        if f0_autotune is True:
            f0 = self.autotune.autotune_f0(f0, f0_autotune_strength)
        elif proposed_pitch is True:
            limit = 12
            # calculate median f0 of the audio
            valid_f0 = np.where(f0 > 0)[0]
            if len(valid_f0) < 2:
                # no valid f0 detected
                up_key = 0
            else:
                median_f0 = float(
                    np.median(np.interp(np.arange(len(f0)), valid_f0, f0[valid_f0]))
                )
                if median_f0 <= 0 or np.isnan(median_f0):
                    up_key = 0
                else:
                    # calculate proposed shift
                    up_key = max(
                        -limit,
                        min(
                            limit,
                            int(
                                np.round(
                                    12 * np.log2(proposed_pitch_threshold / median_f0)
                                )
                            ),
                        ),
                    )
            print(
                "calculated pitch offset:", up_key
            )  # Might need to hide so terminal output doesn't become a mess
            f0 *= pow(2, (f0_up_key + up_key) / 12)
        else:
            f0 *= pow(2, f0_up_key / 12)

        if voiced is not None:
            f0[~voiced] = 0
        # The same mel quantization as training extraction and offline conversion, over
        # the model's range. (This used to subtract the Hz bounds from mel values for
        # every method but FCNF0++, which put 200 Hz on bin 37 instead of the 41 it was
        # trained on at 1680 Hz.)
        f0_coarse = torch.from_numpy(quantize_f0(f0, self.f0_min, self.f0_max)).to(
            self.device
        )
        f0 = torch.from_numpy(f0).to(self.device).float()

        if pitch is not None and pitchf is not None:
            self._write_pitch(f0_coarse, pitch)
            self._write_pitch(f0, pitchf)
            if self.options.protect_voicing == "unfilled":
                if self.voiced_buffer is None or self.voiced_buffer.shape != pitchf.shape:
                    self.voiced_buffer = torch.zeros_like(pitchf)
                voiced_now = torch.from_numpy(
                    self._last_voiced.astype(np.float32)
                ).to(self.device)
                self._write_pitch(voiced_now, self.voiced_buffer)
        else:
            pitch = f0_coarse
            pitchf = f0

        return pitch.unsqueeze(0), pitchf.unsqueeze(0)

    def _fcn_adjusted_pitch(self, shift, autotune, strength, proposed, target):
        # Only the finalized 100 Hz vector crosses to NumPy for existing pitch
        # correction. Capture, normalization, network and stream stay on CUDA.
        f0 = self.fcn_pitch.cpu().numpy().copy()
        voiced = f0 > 0
        if autotune:
            f0 = self.autotune.autotune_f0(f0, strength)
        elif proposed:
            valid = np.flatnonzero(voiced)
            offset = 0
            if len(valid) >= 2:
                median = np.median(np.interp(np.arange(len(f0)), valid, f0[valid]))
                if median > 0:
                    offset = max(-12, min(12, int(np.round(12 * np.log2(target / median)))))
            f0 *= 2 ** ((shift + offset) / 12)
        else:
            f0 *= 2 ** (shift / 12)
        f0[~voiced] = 0
        coarse = quantize_f0(f0, self.f0_min, self.f0_max)
        return torch.from_numpy(coarse).to(self.device)[None], torch.from_numpy(f0).to(self.device)[None]

    def voice_conversion(
        self,
        audio: Tensor,
        pitch: Tensor = None,
        pitchf: Tensor = None,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        p_len: int = 0,
        silence_front: int = 0,
        skip_head: int = None,
        return_length: int = None,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
    ):
        """
        Performs realtime voice conversion on a given audio segment.
        """
        assert audio.dim() == 1, audio.dim()
        feats = audio.view(1, -1).to(self.device)

        formant_length = int(np.ceil(return_length * 1.0))
        plan = self.vc.synthesis
        if plan is not None:
            self._advance_noise()

        pitch, pitchf = (
            self.get_f0(
                audio[silence_front:],
                pitch,
                pitchf,
                f0_up_key,
                f0_autotune,
                f0_autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
            )
            if self.use_f0
            else (None, None)
        )

        # extract features
        feats = self.compile_session.embedder(feats).float()
        feats = (
            self.hubert_model.final_proj(feats[0]).unsqueeze(0)
            if self.version == "v1"
            else feats
        )

        feats = torch.cat((feats, feats[:, -1:, :]), 1)
        # make a copy for pitch guidance and protection
        feats0 = feats.detach().clone() if self.use_f0 else None

        if (
            self.index
        ):  # set by parent function, only true if index is available, loaded, and index rate > 0
            # Legacy plays the oldest part of the window yet blends the index only from
            # skip_head on - kept as it was. "trim" blends every frame the rendered
            # region is computed from, flow margin included.
            index_head = plan.flow_head if plan is not None and plan.skip_head is not None else skip_head
            feats = self._retrieve_speaker_embeddings(
                index_head, feats, self.index, self.big_npy, index_rate
            )
        # feature upsampling
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[
            :, :p_len, :
        ]

        if self.use_f0:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )[:, :p_len, :]
            pitch, pitchf = pitch[:, -p_len:], pitchf[:, -p_len:] * (
                formant_length / return_length
            )

            # Pitch protection blending
            if protect < 0.5:
                if (
                    self.options.protect_voicing == "unfilled"
                    and self.voiced_buffer is not None
                ):
                    # The predictor's voicing before any gap fill, so protect still
                    # finds consonants and breaths when the F0 itself is filled.
                    voiced = self.voiced_buffer[-p_len:].unsqueeze(0)
                    pitchff = torch.where(
                        voiced > 0,
                        torch.ones_like(voiced),
                        torch.full_like(voiced, protect),
                    )
                else:
                    pitchff = pitchf.detach().clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                    1 - pitchff.unsqueeze(-1)
                )
                feats = feats.to(feats0.dtype)
        else:
            pitch, pitchf = None, None

        p_len = torch.tensor([p_len], device=self.device, dtype=torch.int64)
        out_audio = self.vc.inference(
            feats, p_len, self.sid, pitch, pitchf,
            infer_audio=self.compile_session.rvc,
            noise=self._noise_inputs(),
        ).float()
        if plan is not None and pitchf is not None:
            rendered = (
                pitchf[:, plan.skip_head : plan.skip_head + plan.return_length]
                if plan.skip_head is not None
                else pitchf
            )
            self._advance_phase(rendered)
        if volume_envelope != 1:
            # The rendered region only, when "trim" leaves the past context unrendered.
            source = (
                audio[plan.skip_head * self.window :]
                if plan is not None and plan.skip_head is not None
                else audio
            )
            out_audio = AudioProcessor.change_rms(
                source, self.sample_rate, out_audio, self.tgt_sr, volume_envelope
            )

        scaled_window = int(np.floor(1.0 * self.model_window))

        if scaled_window != self.model_window:
            if scaled_window not in self.resamplers:
                self.resamplers[scaled_window] = tat.Resample(
                    orig_freq=scaled_window,
                    new_freq=self.model_window,
                    dtype=torch.float32,
                ).to(self.device)
            out_audio = self.resamplers[scaled_window](
                out_audio[: return_length * scaled_window]
            )

        return out_audio

    def _retrieve_speaker_embeddings(
        self, skip_head, feats, index, big_npy, index_rate
    ):
        if index.d != feats.shape[-1]:
            print(
                f"Skipping the index: it holds {index.d} wide vectors but the embedder "
                f"produced {feats.shape[-1]} wide ones. Rebuild the index with the same "
                "embedder the model was trained on."
            )
            return feats
        skip_offset = skip_head // 2
        npy = feats[0][skip_offset:].cpu().numpy()
        score, ix = index.search(npy, k=8)
        weight = np.square(1 / score)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats[0][skip_offset:] = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats[0][skip_offset:]
        )
        return feats


def load_faiss_index(file_index):
    if file_index != "" and os.path.exists(file_index):
        try:
            index = faiss.read_index(file_index)
            big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as error:
            print(f"An error occurred reading the FAISS index: {error}")
            index = big_npy = None
    else:
        index = big_npy = None

    return index, big_npy


EMBEDDER_PRECISIONS = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def create_pipeline(
    model_path: str = None,
    index_path: str = None,
    f0_method: str = "rmvpe",
    embedder_model: str = None,
    embedder_model_custom: str = None,
    # device: str = "cuda",
    sid: int = 0,
    hybrid_blend_ratio: float = 0.5,
    embedder_precision: str = "fp32",
    fcn_profile=None,
    options: RealtimeOptions = None,
):
    """
    Initialize real-time voice conversion pipeline.
    """

    vc = RealtimeVoiceConverter(model_path)
    index, big_npy = load_faiss_index(
        index_path.strip()
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip()
        .replace("trained", "added")
    )

    # The output layer travels with the model, so realtime reads the same layer the
    # features were extracted from without the user having to set it twice.
    hubert_model = load_embedding(
        embedder_model,
        embedder_model_custom,
        (vc.cpt or {}).get("embedder_output_layer"),
    )
    dtype = EMBEDDER_PRECISIONS.get(embedder_precision, torch.float32)
    hubert_model = hubert_model.to(device=vc.config.device, dtype=dtype)
    hubert_model.eval()
    warn_on_feature_scale_mismatch(hubert_model, vc.cpt)

    pipeline = Realtime_Pipeline(
        vc,
        hubert_model,
        index,
        big_npy,
        f0_method,
        sid,
        hybrid_blend_ratio,
        fcn_profile,
        options,
    )

    return pipeline


def strip_parametrizations(module: torch.nn.Module):
    """
    Remove all parametrizations (e.g., weight norm) from a module and log each removal.
    """
    for name, submodule in module.named_modules():
        if hasattr(submodule, "parametrizations"):
            for pname, plist in list(submodule.parametrizations.items()):
                # print(f"Removing parametrizations from {name}.{pname}: {[p.__class__.__name__ for p in plist]}")
                torch.nn.utils.parametrize.remove_parametrizations(
                    submodule, pname, leave_parametrized=True
                )
