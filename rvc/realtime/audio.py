import os
import sys
import librosa
import traceback
import numpy as np
import sounddevice as sd
from queue import Queue
from dataclasses import dataclass

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import AUDIO_SAMPLE_RATE


@dataclass
class ServerAudioDevice:
    index: int = 0
    name: str = ""
    host_api: str = ""
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: int = 0


def check_the_device(device, type: str = "input", hostapis=None):
    """
    Check if an audio device is available and working.

    For WDM-KS devices: Uses a lenient test since they only support callback-based
    (non-blocking) mode and may fail standard blocking tests. WDM-KS devices are
    accepted if the error indicates they're valid but just don't support blocking mode.
    """
    # Get host API name if available
    host_api_name = ""
    if hostapis and "hostapi" in device:
        try:
            host_api_name = hostapis[device["hostapi"]]["name"]
        except (IndexError, KeyError):
            pass

    stream_cls = sd.InputStream if type == "input" else sd.OutputStream

    # For WDM-KS devices, use a more lenient test with explicit device specification
    if "WDM-KS" in host_api_name or "Windows WDM-KS" in host_api_name:
        try:
            # Test with a smaller blocksize and explicit latency for WDM-KS
            with stream_cls(
                device=device["index"],
                dtype=np.float32,
                samplerate=device["default_samplerate"],
                channels=1,  # Test with mono
                blocksize=512,
                latency='low',
            ):
                return True
        except Exception as e:
            # WDM-KS devices might fail to open if already in use, but still valid
            # Only reject if it's clearly not a valid device
            if "Invalid device" in str(e) or "Device unavailable" in str(e):
                return False
            # For other errors (like "Blocking API not supported"), assume device is valid
            # WDM-KS only supports callback-based (non-blocking) mode, so this is expected
            return True

    # Standard test for non-WDM-KS devices
    try:
        with stream_cls(
            device=device["index"],
            dtype=np.float32,
            samplerate=device["default_samplerate"],
        ):
            return True
    except Exception:
        return False


def list_audio_device():
    """
    Function to query audio devices and host api.
    """
    try:
        audio_device_list = sd.query_devices()
    except Exception as e:
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []

    try:
        hostapis = sd.query_hostapis()
    except Exception as e:
        print("An error occurred while querying the host api:", e)
        hostapis = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the host api:", e)
        hostapis = []

    input_audio_device_list = [
        d
        for d in audio_device_list
        if d["max_input_channels"] > 0 and check_the_device(d, "input", hostapis)
    ]
    output_audio_device_list = [
        d
        for d in audio_device_list
        if d["max_output_channels"] > 0 and check_the_device(d, "output", hostapis)
    ]

    audio_input_device = []
    audio_output_device = []

    for d in input_audio_device_list:
        input_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_input_device.append(input_audio_device)

    for d in output_audio_device_list:
        output_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_output_device.append(output_audio_device)

    return audio_input_device, audio_output_device


class Audio:
    def __init__(
        self,
        callbacks,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch=False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
    ):
        self.callbacks = callbacks
        self.mon_queue = Queue()  # Queue for monitor audio
        self.io_queue = Queue()  # Queue for separate input/output streams (WDM-KS support)

        # Stream objects - either use duplex stream OR separate input/output streams
        self.stream = None  # Duplex stream (used for WASAPI, ASIO, etc.)
        self.input_stream = None  # Separate input stream (used for WDM-KS compatibility)
        self.output_stream = None  # Separate output stream (used for WDM-KS compatibility)
        self.monitor = None  # Optional monitor stream

        self.running = False
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.volume_envelope = volume_envelope
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposed_pitch = proposed_pitch
        self.proposed_pitch_threshold = proposed_pitch_threshold

    def get_input_audio_device(self, index: int):
        audioinput, _ = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def get_output_audio_device(self, index: int):
        _, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def process_data(self, indata: np.ndarray):
        indata = indata * self.input_audio_gain
        unpacked_data = librosa.to_mono(indata.T)

        return self.callbacks.change_voice(
            unpacked_data,
            self.f0_up_key,
            self.index_rate,
            self.protect,
            self.volume_envelope,
            self.f0_autotune,
            self.f0_autotune_strength,
            self.proposed_pitch,
            self.proposed_pitch_threshold,
        )

    def process_data_with_time(self, indata: np.ndarray):
        out_wav, _, perf, _ = self.process_data(indata)
        performance_ms = perf[1]
        # print(f"real-time voice conversion performance: {performance_ms:.2f} ms")
        self.latency = performance_ms  # latency to display on the application interface

        return out_wav

    def audio_stream_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        try:
            out_wav = self.process_data_with_time(indata)

            output_channels = outdata.shape[1]
            if self.use_monitor:
                self.mon_queue.put(out_wav)

            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())

    def audio_queue(self, outdata: np.ndarray, frames, times, status):
        try:
            mon_wav = self.mon_queue.get()

            while self.mon_queue.qsize() > 0:
                self.mon_queue.get()

            output_channels = outdata.shape[1]
            outdata[:] = (
                np.repeat(mon_wav, output_channels).reshape(-1, output_channels)
                * self.monitor_audio_gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio queue: {error}")
            print(traceback.format_exc())

    def input_callback(self, indata: np.ndarray, frames, times, status):
        """
        Callback for input-only stream (used when input and output use separate streams).
        Processes the input audio and puts it in the queue for the output stream.
        """
        try:
            out_wav = self.process_data_with_time(indata)

            if self.use_monitor:
                self.mon_queue.put(out_wav)

            # Put processed audio into queue for output stream
            self.io_queue.put(out_wav)
        except Exception as error:
            print(f"An error occurred in input callback: {error}")
            print(traceback.format_exc())

    def output_callback(self, outdata: np.ndarray, frames, times, status):
        """
        Callback for output-only stream (used when input and output use separate streams).
        Gets processed audio from the queue and outputs it.
        """
        try:
            # Get processed audio from queue with timeout to avoid blocking indefinitely
            try:
                out_wav = self.io_queue.get(timeout=0.1)
            except:
                # Queue is empty, output silence
                outdata[:] = 0
                return

            # Clear old data from queue to reduce latency
            while self.io_queue.qsize() > 0:
                out_wav = self.io_queue.get()

            output_channels = outdata.shape[1]
            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )
        except Exception as error:
            print(f"An error occurred in output callback: {error}")
            print(traceback.format_exc())

    def run_audio_stream(
        self,
        block_frame: int,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int,
        input_max_channel: int,
        output_max_channel: int,
        output_monitor_max_channel: int,
        input_extra_setting,
        output_extra_setting,
        output_monitor_extra_setting,
    ):
        self.stream = sd.Stream(
            callback=self.audio_stream_callback,
            latency="low",
            dtype=np.float32,
            device=(input_device_id, output_device_id),
            blocksize=block_frame,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=(input_max_channel, output_max_channel),
            extra_settings=(input_extra_setting, output_extra_setting),
        )
        self.stream.start()

        if self.use_monitor:
            self.monitor = sd.OutputStream(
                callback=self.audio_queue,
                dtype=np.float32,
                device=output_monitor_id,
                blocksize=block_frame,
                samplerate=AUDIO_SAMPLE_RATE,
                channels=output_monitor_max_channel,
                extra_settings=output_monitor_extra_setting,
            )
            self.monitor.start()

    def run_audio_stream_separate(
        self,
        block_frame: int,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int,
        input_max_channel: int,
        output_max_channel: int,
        output_monitor_max_channel: int,
        input_extra_setting,
        output_extra_setting,
        output_monitor_extra_setting,
    ):
        """
        Run audio with separate input and output streams.

        This method is used when WDM-KS devices are involved or when input and output
        use different host APIs. Separate streams avoid compatibility issues that can
        occur with duplex streams across different audio APIs.

        The input stream captures audio, processes it, and puts it in a queue.
        The output stream retrieves processed audio from the queue and outputs it.
        """
        # Create input stream
        self.input_stream = sd.InputStream(
            callback=self.input_callback,
            latency="low",
            dtype=np.float32,
            device=input_device_id,
            blocksize=block_frame,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=input_max_channel,
            extra_settings=input_extra_setting,
        )

        # Create output stream
        self.output_stream = sd.OutputStream(
            callback=self.output_callback,
            latency="low",
            dtype=np.float32,
            device=output_device_id,
            blocksize=block_frame,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=output_max_channel,
            extra_settings=output_extra_setting,
        )

        # Start streams
        self.input_stream.start()
        self.output_stream.start()

        if self.use_monitor:
            self.monitor = sd.OutputStream(
                callback=self.audio_queue,
                dtype=np.float32,
                device=output_monitor_id,
                blocksize=block_frame,
                samplerate=AUDIO_SAMPLE_RATE,
                channels=output_monitor_max_channel,
                extra_settings=output_monitor_extra_setting,
            )
            self.monitor.start()

    def stop(self):
        self.running = False

        if self.stream is not None:
            self.stream.close()
            self.stream = None

        if self.input_stream is not None:
            self.input_stream.close()
            self.input_stream = None

        if self.output_stream is not None:
            self.output_stream.close()
            self.output_stream = None

        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None

    def start(
        self,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int = None,
        exclusive_mode: bool = False,
        asio_input_channel: int = -1,
        asio_output_channel: int = -1,
        asio_output_monitor_channel: int = -1,
        read_chunk_size: int = 192,
    ):
        """
        Start the realtime audio processing with the specified devices.

        Supports WDM-KS devices by automatically detecting them and using separate
        input/output streams instead of duplex mode when necessary.
        """
        self.stop()

<<<<<<< HEAD
=======
        # NOTE: Not calling sd._terminate() and sd._initialize() here.
        # Re-initialization can invalidate device indices obtained before calling start(),
        # and self.stop() already properly closes all streams.

>>>>>>> 98f810ef (input WASAPI-> output WDM-KSに対応めっちゃ遅延減ったやったー)
        input_audio_device, output_audio_device = self.get_input_audio_device(
            input_device_id
        ), self.get_output_audio_device(output_device_id)
        input_channels, output_channels = (
            input_audio_device.max_input_channels,
            output_audio_device.max_output_channels,
        )

        (
            input_extra_setting,
            output_extra_setting,
            output_monitor_extra_setting,
            monitor_channels,
        ) = (None, None, None, None)
        wasapi_exclusive_mode = bool(exclusive_mode)

        if input_audio_device and "WASAPI" in input_audio_device.host_api:
            input_extra_setting = sd.WasapiSettings(
                exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode
            )
        elif (
            input_audio_device
            and "ASIO" in input_audio_device.host_api
            and asio_input_channel != -1
        ):
            input_extra_setting = sd.AsioSettings(
                channel_selectors=[asio_input_channel]
            )
            input_channels = 1

        if output_audio_device and "WASAPI" in output_audio_device.host_api:
            output_extra_setting = sd.WasapiSettings(
                exclusive=wasapi_exclusive_mode, auto_convert=not wasapi_exclusive_mode
            )
        elif (
            input_audio_device
            and "ASIO" in input_audio_device.host_api
            and asio_output_channel != -1
        ):
            output_extra_setting = sd.AsioSettings(
                channel_selectors=[asio_output_channel]
            )
            output_channels = 1

        if self.use_monitor:
            output_monitor_device = self.get_output_audio_device(output_monitor_id)
            monitor_channels = output_monitor_device.max_output_channels

            if output_monitor_device and "WASAPI" in output_monitor_device.host_api:
                output_monitor_extra_setting = sd.WasapiSettings(
                    exclusive=wasapi_exclusive_mode,
                    auto_convert=not wasapi_exclusive_mode,
                )
            elif (
                output_monitor_device
                and "ASIO" in output_monitor_device.host_api
                and asio_output_monitor_channel != -1
            ):
                output_monitor_extra_setting = sd.AsioSettings(
                    channel_selectors=[asio_output_monitor_channel]
                )
                monitor_channels = 1

        block_frame = int((read_chunk_size * 128 / 48000) * AUDIO_SAMPLE_RATE)

        # WDM-KS Support: Check if we need to use separate input/output streams
        # WDM-KS (Windows Driver Model - Kernel Streaming) provides lower latency but has
        # compatibility limitations:
        # 1. Cannot be used in duplex mode (combined input/output stream)
        # 2. May conflict with WASAPI exclusive mode when used in mixed configurations
        # 3. Only supports callback-based (non-blocking) operation
        use_separate_streams = False
        if input_audio_device and output_audio_device:
            input_host = input_audio_device.host_api
            output_host = output_audio_device.host_api

            # Use separate streams if:
            # 1. Either device uses WDM-KS (known to have compatibility issues with duplex streams)
            # 2. Input and output use different host APIs (may not be compatible)
            if "WDM-KS" in input_host or "WDM-KS" in output_host:
                use_separate_streams = True
                print(f"[WDM-KS detected] Using separate input/output streams for compatibility")

                # When using separate streams with WDM-KS, disable WASAPI exclusive mode
                # Exclusive mode can cause device conflicts when mixing WDM-KS and WASAPI
                # auto_convert=True allows automatic sample rate conversion for compatibility
                if "WASAPI" in input_host and input_extra_setting:
                    print(f"[WDM-KS] Disabling WASAPI exclusive mode for compatibility")
                    input_extra_setting = sd.WasapiSettings(exclusive=False, auto_convert=True)
                if "WASAPI" in output_host and output_extra_setting:
                    output_extra_setting = sd.WasapiSettings(exclusive=False, auto_convert=True)
            elif input_host != output_host:
                use_separate_streams = True
                print(f"[Different host APIs] Using separate input/output streams: {input_host} -> {output_host}")

        try:
            if use_separate_streams:
                self.run_audio_stream_separate(
                    block_frame,
                    input_device_id,
                    output_device_id,
                    output_monitor_id,
                    input_channels,
                    output_channels,
                    monitor_channels,
                    input_extra_setting,
                    output_extra_setting,
                    output_monitor_extra_setting,
                )
            else:
                self.run_audio_stream(
                    block_frame,
                    input_device_id,
                    output_device_id,
                    output_monitor_id,
                    input_channels,
                    output_channels,
                    monitor_channels,
                    input_extra_setting,
                    output_extra_setting,
                    output_monitor_extra_setting,
                )
            self.running = True
        except Exception as error:
            print(f"An error occurred while streaming audio: {error}")
            print(traceback.format_exc())
