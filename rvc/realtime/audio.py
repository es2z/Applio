import os
import sys
import librosa
import traceback
import numpy as np
import sounddevice as sd
from queue import Queue

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import AUDIO_SAMPLE_RATE
from rvc.realtime.devices import AudioDeviceRef, list_audio_device


# Compatibility name retained for imports outside the Realtime tab.
ServerAudioDevice = AudioDeviceRef


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

        # Auto-reconnect settings
        self.auto_reconnect_enabled = True
        self.consecutive_errors = 0
        self.max_consecutive_errors = 150  # Trigger reconnect after 150 consecutive errors (increased from 50 for stability with large extra_infer_size)
        self.reconnect_in_progress = False
        self.last_stream_params = None  # Store parameters for reconnection
        self.reconnect_success = False  # Flag to indicate successful reconnection
        self.last_error = None  # Store last error message for UI display

        # Track consecutive silent outputs for queue management
        # Note: silence detection is done in core.py, this just tracks output
        self.consecutive_silent_outputs = 0
        self.silent_output_threshold_to_stop_adding = 3  # Stop adding to queue after 3 silent outputs
        self.silent_output_threshold_to_clear_queue = 5  # Clear queue after 5 silent outputs (fast cleanup)

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
            # Check for buffer overflow/underflow
            if status:
                print(f"[Audio Stream Warning] {status}")
                self.consecutive_errors += 1
            else:
                # Reset error count on successful processing
                self.consecutive_errors = 0

            out_wav = self.process_data_with_time(indata)

            output_channels = outdata.shape[1]
            if self.use_monitor:
                self.mon_queue.put(out_wav)

            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )

            # Check if we need to reconnect
            self._check_and_reconnect()

        except Exception as error:
            self.consecutive_errors += 1
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())
            # Output silence on error to avoid glitches
            outdata[:] = 0
            # Check if we need to reconnect
            self._check_and_reconnect()

    def audio_queue(self, outdata: np.ndarray, frames, times, status):
        try:
            # Check for buffer overflow/underflow
            if status:
                print(f"[Monitor Stream Warning] {status}")

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
            # Output silence on error
            outdata[:] = 0

    def input_callback(self, indata: np.ndarray, frames, times, status):
        """
        Callback for input-only stream (used when input and output use separate streams).
        Processes the input audio and puts it in the queue for the output stream.
        """
        try:
            # Check for buffer overflow/underflow
            if status:
                print(f"[Input Stream Warning] {status}")
                self.consecutive_errors += 1
            else:
                # Reset error count on successful processing
                self.consecutive_errors = 0

            out_wav = self.process_data_with_time(indata)

            # Check if the output is silence (all values near zero)
            # Silence detection is done in core.py, this just checks output
            is_silent_output = np.abs(out_wav).max() < 1e-5

            if is_silent_output:
                self.consecutive_silent_outputs += 1
            else:
                self.consecutive_silent_outputs = 0

            if self.use_monitor:
                self.mon_queue.put(out_wav)

            # Put processed audio into queue for output stream
            # IMPORTANT: Keep queue small to minimize latency
            # max_queue_size = 1 means we only keep the latest frame
            # This prevents queue buildup during startup and reduces latency
            max_queue_size = 1
            min_queue_size = 0

            # Silent output handling strategy:
            # 1. Stop adding silence after threshold to let queue drain naturally
            # 2. Clear queue only after extended silent output (audio completely stopped)
            if is_silent_output and self.consecutive_silent_outputs > self.silent_output_threshold_to_stop_adding:
                # Stop adding silence to queue to allow natural draining
                # Output callback will continue to output remaining data

                # Reset output queue empty counter during silence (this is normal, not an error)
                if hasattr(self, '_queue_empty_count'):
                    self._queue_empty_count = 0

                # If silent output continues for a very long time, clear the queue
                # This prevents stale data from playing when new audio starts
                if self.consecutive_silent_outputs > self.silent_output_threshold_to_clear_queue:
                    # Clear queue only after extended silent output (audio completely finished)
                    while not self.io_queue.empty():
                        try:
                            self.io_queue.get_nowait()
                        except:
                            break
                # Don't add silence to queue
                pass
            else:
                # Normal operation: add data to queue and manage queue size
                current_size = self.io_queue.qsize()
                if current_size > max_queue_size:
                    print(f"[Input] Queue size too large ({current_size}), clearing to {min_queue_size}")
                    cleared = 0
                    while self.io_queue.qsize() > min_queue_size:
                        try:
                            self.io_queue.get_nowait()
                            cleared += 1
                        except:
                            break
                    print(f"[Input] Cleared {cleared} items from queue")

                self.io_queue.put(out_wav)

            # Check if we need to reconnect
            self._check_and_reconnect()

        except Exception as error:
            self.consecutive_errors += 1
            print(f"An error occurred in input callback: {error}")
            print(traceback.format_exc())
            # Put silence in queue to maintain synchronization
            try:
                self.io_queue.put(np.zeros(indata.shape[0], dtype=np.float32))
            except:
                pass
            # Check if we need to reconnect
            self._check_and_reconnect()

    def output_callback(self, outdata: np.ndarray, frames, times, status):
        """
        Callback for output-only stream (used when input and output use separate streams).
        Gets processed audio from the queue and outputs it.
        """
        try:
            # Check for buffer overflow/underflow
            if status:
                print(f"[Output Stream Warning] {status}")
                self.consecutive_errors += 1

            # Get processed audio from queue with timeout to avoid blocking indefinitely
            try:
                out_wav = self.io_queue.get(timeout=0.1)
            except:
                # Queue is empty
                # This is normal during startup or when input processing is slower than output
                if not hasattr(self, '_queue_empty_count'):
                    self._queue_empty_count = 0
                self._queue_empty_count += 1

                # Only log if queue has been empty for an extended period (potential problem)
                # First 100 empties are ignored (startup period)
                # After that, log every 50th occurrence to detect persistent issues
                if self._queue_empty_count > 100 and self._queue_empty_count % 50 == 1:
                    print(f"[Output] Queue empty (count: {self._queue_empty_count}), outputting silence - this may indicate processing is too slow")

                # Increment error count for empty queue
                self.consecutive_errors += 1
                outdata[:] = 0
                return

            # Reset empty count and error count on successful get
            if hasattr(self, '_queue_empty_count'):
                self._queue_empty_count = 0
            self.consecutive_errors = 0

            # Latency reduction: Use only the latest frame if queue has accumulated
            # This ensures minimal latency by discarding old buffered data
            queue_size = self.io_queue.qsize()
            if queue_size > 1:
                # Discard old frames, keep only the latest
                skipped = 0
                while self.io_queue.qsize() > 1:
                    try:
                        self.io_queue.get_nowait()
                        skipped += 1
                    except:
                        break
                # Now get the latest frame
                out_wav = self.io_queue.get_nowait()
                if skipped > 0:
                    print(f"[Output] Skipped {skipped} old frames to reduce latency (queue had {queue_size} items)")

            output_channels = outdata.shape[1]
            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )
        except Exception as error:
            self.consecutive_errors += 1
            print(f"An error occurred in output callback: {error}")
            print(traceback.format_exc())
            # Output silence on error
            outdata[:] = 0

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
        latency_mode: str = "low",
    ):
        # Use latency_mode to balance between responsiveness and stability
        self.stream = sd.Stream(
            callback=self.audio_stream_callback,
            latency=latency_mode,
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
        latency_mode: str = "low",
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
        # Use latency_mode to balance between responsiveness and stability
        self.input_stream = sd.InputStream(
            callback=self.input_callback,
            latency=latency_mode,
            dtype=np.float32,
            device=input_device_id,
            blocksize=block_frame,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=input_max_channel,
            extra_settings=input_extra_setting,
        )

        # Create output stream
        # Use latency_mode to balance between responsiveness and stability
        self.output_stream = sd.OutputStream(
            callback=self.output_callback,
            latency=latency_mode,
            dtype=np.float32,
            device=output_device_id,
            blocksize=block_frame,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=output_max_channel,
            extra_settings=output_extra_setting,
        )

        # Start streams - IMPORTANT: Start output first to prevent queue buildup
        # If input starts first, it will process and queue data before output can consume it
        # This causes initial latency as the queue fills up
        self.output_stream.start()
        self.input_stream.start()

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

    def _check_and_reconnect(self):
        """
        Check if we need to trigger auto-reconnect based on consecutive errors.
        """
        if not self.auto_reconnect_enabled or self.reconnect_in_progress:
            return

        if self.consecutive_errors >= self.max_consecutive_errors:
            print(f"\n[Auto-Reconnect] Detected {self.consecutive_errors} consecutive errors. Attempting to reconnect...")
            self.reconnect_in_progress = True
            self._attempt_reconnect()

    def _attempt_reconnect(self):
        """
        Attempt to reconnect the audio streams.
        """
        import threading
        import time

        def reconnect_worker():
            try:
                print("[Auto-Reconnect] Stopping current streams...")
                # Close current streams (but keep last_stream_params)
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except:
                        pass
                    self.stream = None

                if self.input_stream is not None:
                    try:
                        self.input_stream.close()
                    except:
                        pass
                    self.input_stream = None

                if self.output_stream is not None:
                    try:
                        self.output_stream.close()
                    except:
                        pass
                    self.output_stream = None

                if self.monitor is not None:
                    try:
                        self.monitor.close()
                    except:
                        pass
                    self.monitor = None

                # Clear queues
                while not self.io_queue.empty():
                    try:
                        self.io_queue.get_nowait()
                    except:
                        break

                while not self.mon_queue.empty():
                    try:
                        self.mon_queue.get_nowait()
                    except:
                        break

                # Wait a bit before reconnecting
                time.sleep(0.5)

                # Attempt to restart streams with saved parameters
                if self.last_stream_params is not None:
                    print("[Auto-Reconnect] Restarting streams with saved parameters...")
                    params = self.last_stream_params

                    # Determine if we should use separate streams
                    use_separate_streams = params.get('use_separate_streams', False)
                    latency_mode = params.get('latency_mode', 'low')

                    try:
                        if use_separate_streams:
                            self.run_audio_stream_separate(
                                params['block_frame'],
                                params['input_device_id'],
                                params['output_device_id'],
                                params['output_monitor_id'],
                                params['input_max_channel'],
                                params['output_max_channel'],
                                params['output_monitor_max_channel'],
                                params['input_extra_setting'],
                                params['output_extra_setting'],
                                params['output_monitor_extra_setting'],
                                latency_mode,
                            )
                        else:
                            self.run_audio_stream(
                                params['block_frame'],
                                params['input_device_id'],
                                params['output_device_id'],
                                params['output_monitor_id'],
                                params['input_max_channel'],
                                params['output_max_channel'],
                                params['output_monitor_max_channel'],
                                params['input_extra_setting'],
                                params['output_extra_setting'],
                                params['output_monitor_extra_setting'],
                                latency_mode,
                            )

                        # Reset error counter on successful reconnect
                        self.consecutive_errors = 0
                        self.reconnect_success = True  # Set flag for UI to detect
                        self.last_error = None  # Clear error
                        print("[Auto-Reconnect] Successfully reconnected streams!")

                    except Exception as e:
                        self.last_error = f"Reconnect failed: {e}"
                        print(f"[Auto-Reconnect] Failed to reconnect: {e}")
                        print(traceback.format_exc())
                else:
                    print("[Auto-Reconnect] No saved parameters found. Cannot reconnect.")

            except Exception as e:
                print(f"[Auto-Reconnect] Error during reconnection: {e}")
                print(traceback.format_exc())
            finally:
                self.reconnect_in_progress = False

        # Run reconnection in a separate thread to avoid blocking the audio callback
        reconnect_thread = threading.Thread(target=reconnect_worker, daemon=True)
        reconnect_thread.start()

    def stop(self):
        self.running = False
        self.auto_reconnect_enabled = False  # Disable auto-reconnect when explicitly stopping
        self.reconnect_success = False  # Reset reconnect flag
        self.last_error = None  # Clear error

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

        # Clear all queues and reset counters to ensure clean state
        while not self.io_queue.empty():
            try:
                self.io_queue.get_nowait()
            except:
                break

        if self.use_monitor:
            while not self.mon_queue.empty():
                try:
                    self.mon_queue.get_nowait()
                except:
                    break

        # Reset counters
        self.consecutive_silent_outputs = 0
        if hasattr(self, '_queue_empty_count'):
            self._queue_empty_count = 0
        self.consecutive_errors = 0

    def clear_buffers(self):
        """
        Clear all buffers and reset state for clean startup.
        This reduces initial latency when connecting by ensuring no old data remains.
        """
        # Clear queues (already done in stop(), but ensure they're empty)
        while not self.io_queue.empty():
            try:
                self.io_queue.get_nowait()
            except:
                break

        if self.use_monitor:
            while not self.mon_queue.empty():
                try:
                    self.mon_queue.get_nowait()
                except:
                    break

        # Reset counters
        self.consecutive_silent_outputs = 0
        if hasattr(self, '_queue_empty_count'):
            self._queue_empty_count = 0
        self.consecutive_errors = 0

        # Clear VoiceChanger buffers
        if hasattr(self, 'callbacks') and hasattr(self.callbacks, 'vc'):
            vc = self.callbacks.vc

            # Clear sola_buffer
            if vc.sola_buffer is not None:
                vc.sola_buffer.zero_()

            # Clear Realtime model buffers
            if hasattr(vc, 'vc_model') and vc.vc_model is not None:
                vc.vc_model.flush_buffers()
                vc.vc_model.consecutive_silence_frames = 0

    def start(
        self,
        input_device: AudioDeviceRef | int,
        output_device: AudioDeviceRef | int,
        output_monitor: AudioDeviceRef | int | None = None,
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

        # Clear all buffers and reset state for clean startup
        # This reduces initial latency by ensuring no old data remains
        self.clear_buffers()

        # NOTE: Not calling sd._terminate() and sd._initialize() here.
        # Re-initialization can invalidate device indices obtained before calling start(),
        # and self.stop() already properly closes all streams.

        input_audio_device = (
            input_device
            if isinstance(input_device, AudioDeviceRef)
            else self.get_input_audio_device(input_device)
        )
        output_audio_device = (
            output_device
            if isinstance(output_device, AudioDeviceRef)
            else self.get_output_audio_device(output_device)
        )
        if input_audio_device is None or output_audio_device is None:
            raise ValueError("The selected input/output device is no longer present")
        if (
            input_audio_device.direction != "input"
            or output_audio_device.direction != "output"
        ):
            raise ValueError("Audio device direction does not match the selected role")
        input_device_id = input_audio_device.index
        output_device_id = output_audio_device.index
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
        output_monitor_id = None
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
            output_audio_device
            and "ASIO" in output_audio_device.host_api
            and asio_output_channel != -1
        ):
            output_extra_setting = sd.AsioSettings(
                channel_selectors=[asio_output_channel]
            )
            output_channels = 1

        if self.use_monitor:
            output_monitor_device = (
                output_monitor
                if isinstance(output_monitor, AudioDeviceRef)
                else self.get_output_audio_device(output_monitor)
            )
            if output_monitor_device is None:
                raise ValueError("The selected monitor device is no longer present")
            output_monitor_id = output_monitor_device.index
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

        # Use low latency mode for minimal delay
        latency_mode = "low"

        try:
            # Save parameters for auto-reconnect
            self.last_stream_params = {
                'block_frame': block_frame,
                'input_device_id': input_device_id,
                'output_device_id': output_device_id,
                'output_monitor_id': output_monitor_id,
                'input_max_channel': input_channels,
                'output_max_channel': output_channels,
                'output_monitor_max_channel': monitor_channels,
                'input_extra_setting': input_extra_setting,
                'output_extra_setting': output_extra_setting,
                'output_monitor_extra_setting': output_monitor_extra_setting,
                'use_separate_streams': use_separate_streams,
                'latency_mode': latency_mode,
            }

            # Enable auto-reconnect and reset status flags
            self.auto_reconnect_enabled = True
            self.consecutive_errors = 0
            self.reconnect_success = False
            self.last_error = None

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
                    latency_mode,
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
                    latency_mode,
                )
            self.running = True
        except Exception as error:
            print(f"An error occurred while streaming audio: {error}")
            print(traceback.format_exc())
            raise
