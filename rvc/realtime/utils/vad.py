import webrtcvad
import numpy as np


class VADProcessor:
    def __init__(self, sensitivity_mode=3, sample_rate=16000, frame_duration_ms=30):
        """
        Initializes the VADProcessor.

        Args:
            sensitivity_mode (int): VAD sensitivity (0-3). 3 is most aggressive.
            sample_rate (int): Sample rate of the audio. Must be 8000, 16000, 32000, or 48000 Hz.
                               WebRTC VAD internally works best with 16000 Hz.
            frame_duration_ms (int): Duration of each audio frame in ms. Must be 10, 20, or 30.
        """

        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("VAD sample rate must be 8000, 16000, 32000, or 48000 Hz")
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")

        self.sensitivity_mode = sensitivity_mode
        # A fresh detector needs a few frames before its verdicts mean anything: on
        # stationary room tone it calls exactly the first three frames speech and
        # nothing after them. Discarding those 90 ms is what separates room tone
        # (0 speech frames left) from real speech (2 to 29 of the remaining 29).
        self.warmup_frames = 3
        # Built per call, not kept. webrtcvad adapts to what it has heard, and on a
        # realtime stream that alternates loud speech with room tone it ends up calling
        # the room tone speech: measured on 0.96 s blocks of -50 dBFS room tone, a
        # detector carried through the session reported 27-31 of 32 frames as speech,
        # while a fresh one reported 3. Speech itself reads 14-28 of 32 either way, so
        # discarding the state is what makes the two separable at all.
        self.vad = webrtcvad.Vad(sensitivity_mode)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        # print(f"VAD Initialized: SR={sample_rate}, Frame Duration={frame_duration_ms}ms, Frame Length={self.frame_length} samples")

    def is_speech(self, audio_chunk_float32, min_ratio=0.0):
        """
        Detects if the given audio chunk contains speech.

        Args:
            audio_chunk_float32 (np.ndarray): A chunk of audio data in float32 format, mono.
                                              The sample rate must match the one VAD was initialized with.
            min_ratio (float): Fraction of frames that must read as speech. The default
                               of 0.0 keeps the original "any single frame" behaviour.

        Returns:
            bool: True if speech is detected in the chunk, False otherwise.
        """
        return self.speech_ratio(audio_chunk_float32) > min_ratio

    def speech_ratio(self, audio_chunk_float32):
        """Fraction of the chunk's frames that read as speech, 0.0 for an empty chunk.

        The first `warmup_frames` frames only prime the detector and are not scored.
        """

        if audio_chunk_float32.ndim > 1 and audio_chunk_float32.shape[1] == 1:
            audio_chunk_float32 = audio_chunk_float32.flatten()
        elif audio_chunk_float32.ndim > 1:
            # If stereo, average to mono. This is a simple approach.
            # For better results, ensure mono input from the source.
            print("VAD Warning: Received stereo audio, averaging to mono.")
            audio_chunk_float32 = np.mean(audio_chunk_float32, axis=1)

        # Convert float32 audio to int16 PCM
        # WebRTC VAD expects 16-bit linear PCM audio.
        if np.max(np.abs(audio_chunk_float32)) > 1.0:
            # print(
            #     f"VAD Warning: Input audio chunk has values outside [-1.0, 1.0]: min={np.min(audio_chunk_float32)}, max={np.max(audio_chunk_float32)}. Clipping."
            # )
            audio_chunk_float32 = np.clip(audio_chunk_float32, -1.0, 1.0)

        audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)

        num_frames = len(audio_chunk_int16) // self.frame_length
        if num_frames == 0 and len(audio_chunk_int16) > 0:
            # If the chunk is smaller than one frame, pad it for VAD analysis
            # This might not be ideal but handles small initial chunks
            padding = np.zeros(
                self.frame_length - len(audio_chunk_int16), dtype=np.int16
            )
            audio_chunk_int16 = np.concatenate((audio_chunk_int16, padding))
            num_frames = 1
        elif num_frames == 0 and len(audio_chunk_int16) == 0:
            return 0.0  # Empty chunk

        try:
            vad = webrtcvad.Vad(self.sensitivity_mode)
            hits = 0
            scored = 0
            for i in range(num_frames):
                start = i * self.frame_length
                end = start + self.frame_length
                frame = audio_chunk_int16[start:end]
                # The VAD expects bytes, not a NumPy array.
                speech = bool(vad.is_speech(frame.tobytes(), self.sample_rate))
                if i < self.warmup_frames:
                    continue  # primes the detector; its verdict here is noise
                scored += 1
                hits += speech
            return hits / scored if scored else 0.0
        except Exception as e:
            # webrtcvad can sometimes throw "Error talking to VAD" or similar
            # if frame length is not perfect.
            print(
                f"VAD processing error: {e}. Chunk length: {len(audio_chunk_int16)}, Frame length: {self.frame_length}"
            )
            # Fallback: assume speech on error, so a detector problem can never gate
            # audio that is really there.
            return 1.0
