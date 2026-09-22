"""FCN session capture clock and synchronized delayed audio/pitch windows."""

import torch
import torch.nn.functional as F

from rvc.lib.predictors.fcn.streaming import FCNStream


class CaptureResampler:
    """Stateful 48→16 kHz wrapper preserving torchaudio's FIR and phase."""

    def __init__(self, resampler, device):
        self.resampler = resampler
        self.device = device
        self.context = ((resampler.kernel.shape[-1] + 2) // 3) * 3
        self.lookahead = resampler.kernel.shape[-1] - resampler.width
        self.reset()

    def reset(self):
        self.buffer = torch.empty(0, device=self.device)
        self.start = self.count = self.next_output = 0

    def push(self, audio, final=False):
        self.buffer = torch.cat((self.buffer, audio))
        self.count += len(audio)
        stop = (
            (self.count + 2) // 3
            if final
            else max(0, (self.count - self.lookahead) // 3)
        )
        if stop <= self.next_output:
            return audio[:0]
        waveform = self.resampler(self.buffer)
        start = self.next_output - self.start // 3
        result = waveform[start : stop - self.start // 3]
        self.next_output = stop
        retain = max(0, (stop * 3 - self.context) // 3 * 3)
        drop = retain - self.start
        if drop > 0:
            self.buffer = self.buffer[drop:].clone()
            self.start = retain
        return result


class FCNRealtimeSession:
    def __init__(self, predictor, capture_resampler, window_samples):
        self.stream = FCNStream(predictor)
        self.capture = CaptureResampler(capture_resampler, predictor.device)
        self.device = predictor.device
        self.window_samples = window_samples
        self.window_frames = window_samples // 160
        self.reset()

    @property
    def holdback_ms(self):
        return self.stream.holdback_samples / 16 + self.capture.lookahead / 48

    def reset(self):
        self.stream.reset()
        self.capture.reset()
        self.audio = torch.empty(0, device=self.device)
        self.pitch = torch.empty(0, device=self.device)
        self.audio_start = 0
        self.window_end = 0

    @torch.inference_mode()
    def push(self, audio48):
        raw = torch.as_tensor(audio48, device=self.device, dtype=torch.float32)
        if raw.ndim != 1 or not torch.isfinite(raw).all():
            raise ValueError("FCN capture requires finite mono audio")
        samples = self.capture.push(raw)
        self.audio = torch.cat((self.audio, samples))
        track = self.stream.push(samples)
        self.pitch = torch.cat((self.pitch, track.pitch_hz))[-self.window_frames :]
        self.window_end = self.stream.next_frame * 160
        start = max(0, self.window_end - self.window_samples)
        waveform = self.audio[
            start - self.audio_start : self.window_end - self.audio_start
        ]
        waveform = F.pad(waveform, (self.window_samples - len(waveform), 0))
        pitch = F.pad(self.pitch, (self.window_frames - len(self.pitch), 0))
        drop = start - self.audio_start
        if drop > 0:
            self.audio = self.audio[drop:].clone()
            self.audio_start = start
        return waveform, pitch
