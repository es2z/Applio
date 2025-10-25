import gradio as gr
import sounddevice as sd
import os
import sys
import time
import json
import regex as re
import shutil
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.callbacks import AudioCallbacks
from rvc.realtime.audio import list_audio_device
from rvc.realtime.core import AUDIO_SAMPLE_RATE
from rvc.configs.config_utils import load_config, save_config, update_nested_config

from assets.i18n.i18n import I18nAuto
from tabs.realtime.template import RealtimeTemplateManager

i18n = I18nAuto()

model_root = os.path.join(now_dir, "logs")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)

def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/").lower()

MODEL_FOLDER = re.compile(r"^(?:model.{0,4}|mdl(?:s)?|weight.{0,4}|zip(?:s)?)$")
INDEX_FOLDER = re.compile(r"^(?:ind.{0,4}|idx(?:s)?)$")


def is_mdl_alias(name: str) -> bool:
    return bool(MODEL_FOLDER.match(name))


def is_idx_alias(name: str) -> bool:
    return bool(INDEX_FOLDER.match(name))


def alias_score(path: str, want_model: bool) -> int:
    parts = normalize_path(os.path.dirname(path)).split("/")
    has_mdl = any(is_mdl_alias(p) for p in parts)
    has_idx = any(is_idx_alias(p) for p in parts)
    if want_model:
        return 2 if has_mdl else (1 if has_idx else 0)
    else:
        return 2 if has_idx else (1 if has_mdl else 0)


def get_files(type="model"):
    assert type in ("model", "index"), "Invalid type for get_files (models or index)"
    is_model = type == "model"
    exts = (".pth", ".onnx") if is_model else (".index",)
    exclude_prefixes = ("G_", "D_") if is_model else ()
    exclude_substr = None if is_model else "trained"

    best = {}
    order = 0

    for root, _, files in os.walk(model_root_relative, followlinks=True):
        for file in files:
            if not file.endswith(exts):
                continue
            if any(file.startswith(p) for p in exclude_prefixes):
                continue
            if exclude_substr and exclude_substr in file:
                continue

            full = os.path.join(root, file)
            real = os.path.realpath(full)
            score = alias_score(full, is_model)

            prev = best.get(real)
            if (
                prev is None
            ):  # Prefer higher score; if equal score, use first encountered
                best[real] = (score, order, full)
            else:
                prev_score, prev_order, _ = prev
                if score > prev_score:
                    best[real] = (score, prev_order, full)
            order += 1

    return [t[2] for t in sorted(best.values(), key=lambda x: x[1])]


def folders_same(
    a: str, b: str
) -> bool:  # Used to "pair" index and model folders based on path names
    """
    True if:
      1) The two normalized paths are totally identical..OR
      2) One lives under a MODEL_FOLDER and the other lives
         under an INDEX_FOLDER, at the same relative subpath
         i.e.  logs/models/miku  and  logs/index/miku  =  "SAME FOLDER"
    """
    a = normalize_path(a)
    b = normalize_path(b)
    if a == b:
        return True

    def split_after_alias(p):
        parts = p.split("/")
        for i, part in enumerate(parts):
            if is_mdl_alias(part) or is_idx_alias(part):
                base = part
                rel = "/".join(parts[i + 1 :])
                return base, rel
        return None, None

    base_a, rel_a = split_after_alias(a)
    base_b, rel_b = split_after_alias(b)

    if rel_a is None or rel_b is None:
        return False

    if rel_a == rel_b and (
        (is_mdl_alias(base_a) and is_idx_alias(base_b))
        or (is_idx_alias(base_a) and is_mdl_alias(base_b))
    ):
        return True
    return False


def match_index(model_file_value):
    if not model_file_value:
        return ""

    # Derive the information about the model's name and path for index matching
    model_folder = normalize_path(os.path.dirname(model_file_value))
    model_name = os.path.basename(model_file_value)
    base_name = os.path.splitext(model_name)[0]
    common = re.sub(r"[_\-\.\+](?:e|s|v|V)\d.*$", "", base_name)
    prefix_match = re.match(r"^(.*?)[_\-\.\+]", base_name)
    prefix = prefix_match.group(1) if prefix_match else None

    same_count = 0
    last_same = None
    same_substr = None
    same_prefixed = None
    external_exact = None
    external_substr = None
    external_pref = None

    for idx in get_files("index"):
        idx_folder = os.path.dirname(idx)
        idx_folder_n = normalize_path(idx_folder)
        idx_name = os.path.basename(idx)
        idx_base = os.path.splitext(idx_name)[0]

        in_same = folders_same(model_folder, idx_folder_n)
        if in_same:
            same_count += 1
            last_same = idx

            # 1) EXACT match to loaded model name and folders_same = True
            if idx_base == base_name:
                return idx

            # 2) Substring match to model name and folders_same
            if common in idx_base and same_substr is None:
                same_substr = idx

            # 3) Prefix match to model name and folders_same
            if prefix and idx_base.startswith(prefix) and same_prefixed is None:
                same_prefixed = idx

        # If it's NOT in a paired folder (folders_same = False) we look elseware:
        else:
            # 4) EXACT match to model name in external directory
            if idx_base == base_name and external_exact is None:
                external_exact = idx

            # 5) Substring match to model name in ED
            if common in idx_base and external_substr is None:
                external_substr = idx

            # 6) Prefix match to model name in ED
            if prefix and idx_base.startswith(prefix) and external_pref is None:
                external_pref = idx

    # Fallback: If there is exactly one index file in the same (or paired) folder,
    # we should assume that's the intended index file even if the name doesnt match
    if same_count == 1:
        return last_same

    # Then by remaining priority queue:
    if same_substr:
        return same_substr
    if same_prefixed:
        return same_prefixed
    if external_exact:
        return external_exact
    if external_substr:
        return external_substr
    if external_pref:
        return external_pref

    return ""

def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def get_speakers_id(model):
    if model:
        try:
            model_data = torch.load(
                os.path.join(now_dir, model), map_location="cpu", weights_only=True
            )
            speakers_id = model_data.get("speakers_id")
            if speakers_id:
                return list(range(speakers_id))
            else:
                return [0]
        except Exception as e:
            return [0]
    else:
        return [0]


def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)

    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))
    if config_file:
        shutil.copy(
            config_file, os.path.join(target_folder, os.path.basename(config_file))
        )

    return f"Files moved to folder {target_folder}"


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders

names = get_files("model")
default_weight = names[0] if names else None

PASS_THROUGH = False
interactive_true = gr.update(interactive=True)
interactive_false = gr.update(interactive=False)
running, callbacks, audio_manager = False, None, None

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")

def save_realtime_settings(
    input_device, output_device, monitor_device, model_file, index_file
):
    """Save realtime settings to config.json (thread-safe)"""
    try:
        updates = {}

        # Only save non-None values, preserve existing values for None inputs
        if input_device is not None:
            updates["input_device"] = input_device or ""
        if output_device is not None:
            updates["output_device"] = output_device or ""
        if monitor_device is not None:
            updates["monitor_device"] = monitor_device or ""
        if model_file is not None:
            updates["model_file"] = model_file or ""
        if index_file is not None:
            updates["index_file"] = index_file or ""

        if updates:
            update_nested_config(CONFIG_PATH, "realtime", updates)
    except Exception as e:
        print(f"Error saving realtime settings: {e}")


def load_realtime_settings():
    """Load realtime settings from config.json (thread-safe)"""
    try:
        config = load_config(CONFIG_PATH)
        realtime_config = config.get("realtime", {})
        return {
            "input_device": realtime_config.get("input_device", ""),
            "output_device": realtime_config.get("output_device", ""),
            "monitor_device": realtime_config.get("monitor_device", ""),
            "model_file": realtime_config.get("model_file", ""),
            "index_file": realtime_config.get("index_file", ""),
        }
    except Exception as e:
        print(f"Error loading realtime settings: {e}")
        return {
            "input_device": "",
            "output_device": "",
            "monitor_device": "",
            "model_file": "",
            "index_file": "",
        }


def get_safe_dropdown_value(saved_value, choices, fallback_value=None):
    """Safely get a dropdown value, ensuring it exists in choices"""
    if saved_value and saved_value in choices:
        return saved_value
    elif fallback_value and fallback_value in choices:
        return fallback_value
    elif choices:
        return choices[0]
    else:
        return None


def get_safe_index_value(saved_value, choices, fallback_value=None):
    """Safely get an index file value, handling file path matching"""
    # Handle empty string, None, or whitespace-only values
    if not saved_value or (isinstance(saved_value, str) and not saved_value.strip()):
        if fallback_value and fallback_value in choices:
            return fallback_value
        elif choices:
            return choices[0]
        else:
            return None

    # Check exact match first
    if saved_value in choices:
        return saved_value

    # Check if saved value is a filename that matches any choice
    saved_filename = os.path.basename(saved_value)
    for choice in choices:
        if os.path.basename(choice) == saved_filename:
            return choice

    # Fallback to default or first choice
    if fallback_value and fallback_value in choices:
        return fallback_value
    elif choices:
        return choices[0]
    else:
        return None


def start_realtime(
    input_audio_device: str,
    input_audio_gain: int,
    input_asio_channels: int,
    output_audio_device: str,
    output_audio_gain: int,
    output_asio_channels: int,
    monitor_output_device: str,
    monitor_audio_gain: int,
    monitor_asio_channels: int,
    use_monitor_device: bool,
    exclusive_mode: bool,
    vad_enabled: bool,
    chunk_size: float,
    cross_fade_overlap_size: float,
    extra_convert_size: float,
    silent_threshold: int,
    pitch: int,
    index_rate: float,
    volume_envelope: float,
    protect: float,
    f0_method: str,
    hybrid_blend_ratio: float,
    pth_path: str,
    index_path: str,
    sid: int,
    f0_autotune: bool,
    f0_autotune_strength: float,
    proposed_pitch: bool,
    proposed_pitch_threshold: float,
    embedder_model: str,
    embedder_model_custom: str = None,
):
    global running, callbacks, audio_manager
    running = True

    if not input_audio_device or not output_audio_device:
        yield (
            "Please select valid input/output devices!",
            interactive_true,
            interactive_false,
        )
        return
    if use_monitor_device and not monitor_output_device:
        yield (
            "Please select a valid monitor device!",
            interactive_true,
            interactive_false,
        )
        return
    if not pth_path:
        yield (
            "Model path not provided. Aborting conversion.",
            interactive_true,
            interactive_false,
        )
        return

    yield "Starting Realtime...", interactive_false, interactive_true

    read_chunk_size = int(chunk_size * AUDIO_SAMPLE_RATE / 1000 / 128)

    sid = int(sid) if sid is not None else 0

    input_audio_gain /= 100.0
    output_audio_gain /= 100.0
    monitor_audio_gain /= 100.0

    try:
        input_devices, output_devices = get_audio_devices_formatted()
        input_device_id = input_devices[input_audio_device]
        output_device_id = output_devices[output_audio_device]
        output_monitor_id = (
            output_devices[monitor_output_device] if use_monitor_device else None
        )
    except (ValueError, IndexError):
        yield "Incorrectly formatted audio device. Stopping.", interactive_true, interactive_false
        return

    callbacks = AudioCallbacks(
        pass_through=PASS_THROUGH,
        read_chunk_size=read_chunk_size,
        cross_fade_overlap_size=cross_fade_overlap_size,
        extra_convert_size=extra_convert_size,
        model_path=pth_path,
        index_path=str(index_path),
        f0_method=f0_method,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        silent_threshold=silent_threshold,
        f0_up_key=pitch,
        index_rate=index_rate,
        protect=protect,
        volume_envelope=volume_envelope,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        proposed_pitch=proposed_pitch,
        proposed_pitch_threshold=proposed_pitch_threshold,
        input_audio_gain=input_audio_gain,
        output_audio_gain=output_audio_gain,
        monitor_audio_gain=monitor_audio_gain,
        monitor=use_monitor_device,
        vad_enabled=vad_enabled,
        vad_sensitivity=3,
        vad_frame_ms=30,
        sid=sid,
        hybrid_blend_ratio=hybrid_blend_ratio,
    )

    audio_manager = callbacks.audio
    audio_manager.start(
        input_device_id=input_device_id,
        output_device_id=output_device_id,
        output_monitor_id=output_monitor_id,
        exclusive_mode=exclusive_mode,
        asio_input_channel=input_asio_channels,
        asio_output_channel=output_asio_channels,
        asio_output_monitor_channel=monitor_asio_channels,
        read_chunk_size=read_chunk_size,
    )

    yield "Realtime is ready!", interactive_false, interactive_true

    while running and callbacks is not None and audio_manager is not None:
        time.sleep(0.1)

        # Check for reconnection status
        if hasattr(audio_manager, "reconnect_success") and audio_manager.reconnect_success:
            # Reconnection succeeded - update UI and reset flag
            audio_manager.reconnect_success = False
            yield "Reconnected successfully! Realtime is ready!", interactive_false, interactive_true
            time.sleep(0.5)  # Show message briefly
        elif hasattr(audio_manager, "last_error") and audio_manager.last_error:
            # Reconnection failed - show error but keep stop button enabled
            error_msg = audio_manager.last_error
            audio_manager.last_error = None  # Clear error after displaying once
            yield f"Error: {error_msg}", interactive_false, interactive_true
            time.sleep(1.0)  # Show error message for 1 second
        elif hasattr(audio_manager, "reconnect_in_progress") and audio_manager.reconnect_in_progress:
            # Reconnection in progress
            yield "Reconnecting...", interactive_false, interactive_true
        elif hasattr(audio_manager, "latency"):
            # Normal operation - show latency
            yield f"Latency: {audio_manager.latency:.2f} ms", interactive_false, interactive_true

    return gr.update(), gr.update(), gr.update()


def stop_realtime():
    global running, callbacks, audio_manager

    # Always attempt to stop if running flag is True, regardless of other states
    if running:
        running = False  # Set flag first to stop the main loop

        # Stop audio manager if it exists
        if audio_manager is not None:
            try:
                audio_manager.stop()
            except Exception as e:
                print(f"Error stopping audio manager: {e}")

            if hasattr(audio_manager, "latency"):
                try:
                    del audio_manager.latency
                except:
                    pass

        # Clean up references
        audio_manager = callbacks = None

        return gr.update(value="Stopping..."), interactive_true, interactive_false
    else:
        return "Realtime pipeline not found!", interactive_true, interactive_false


def get_audio_devices_formatted():
    try:
        input_devices, output_devices = list_audio_device()

        def priority(name: str) -> int:
            n = name.lower()
            if "virtual" in n:
                return 0
            if "vb" in n:
                return 1
            return 2

        output_sorted = sorted(output_devices, key=lambda d: priority(d.name))
        input_sorted = sorted(
            input_devices, key=lambda d: priority(d.name), reverse=True
        )

        input_device_list = {
            f"{input_sorted.index(d)+1}: {d.name} ({d.host_api})": d.index
            for d in input_sorted
        }
        output_device_list = {
            f"{output_sorted.index(d)+1}: {d.name} ({d.host_api})": d.index
            for d in output_sorted
        }

        return input_device_list, output_device_list
    except Exception:
        return [], []


def realtime_tab():
    input_devices, output_devices = get_audio_devices_formatted()
    input_devices, output_devices = list(input_devices.keys()), list(
        output_devices.keys()
    )

    # Load saved settings
    saved_settings = load_realtime_settings()

    # Initialize template manager
    template_manager = RealtimeTemplateManager()

    with gr.Blocks() as ui:
        with gr.Row():
            start_button = gr.Button(i18n("Start"), variant="primary")
            stop_button = gr.Button(i18n("Stop"), interactive=False)
        latency_info = gr.Label(label=i18n("Status"), value="Realtime not started.")
        terms_checkbox = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            info=i18n(
                "Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your realtime."
            ),
            value=False,
            interactive=True,
        )

        # Template Section
        with gr.Accordion(i18n("Templates"), open=False):
            with gr.Row():
                template_dropdown = gr.Dropdown(
                    label=i18n("Template"),
                    choices=template_manager.get_template_list(),
                    value=None,
                    interactive=True,
                    scale=3,
                )
                apply_template_button = gr.Button(i18n("Apply"), scale=1)
                overwrite_template_button = gr.Button(i18n("Overwrite"), scale=1)
                delete_template_button = gr.Button(i18n("Delete"), scale=1)
                add_template_button = gr.Button(i18n("Add"), scale=1)

            # Hidden components for template name input
            with gr.Row(visible=False) as template_name_input_row:
                template_name_input = gr.Textbox(
                    label=i18n("Template Name"),
                    placeholder=i18n("Enter template name"),
                    interactive=True,
                    scale=3,
                )
                template_save_button = gr.Button(i18n("OK"), scale=1, variant="primary")
                template_cancel_button = gr.Button(i18n("Cancel"), scale=1)

            # Hidden state to track current operation
            template_operation_state = gr.State(value="")

        with gr.Tabs():
            with gr.TabItem(i18n("Audio Settings")):
                with gr.Row():
                    refresh_devices_button = gr.Button(i18n("Refresh Audio Devices"))
                with gr.Row():
                    with gr.Accordion(i18n("Input Device"), open=True):
                        with gr.Column():
                            input_audio_device = gr.Dropdown(
                                label=i18n("Input Device"),
                                info=i18n(
                                    "Select the microphone or audio interface you will be speaking into."
                                ),
                                choices=input_devices,
                                value=get_safe_dropdown_value(
                                    saved_settings["input_device"], input_devices
                                ),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            input_audio_gain = gr.Slider(
                                minimum=0,
                                maximum=200,
                                value=100,
                                label=i18n("Input Gain (%)"),
                                info=i18n(
                                    "Adjusts the input volume before processing. Prevents clipping or boosts a quiet mic."
                                ),
                                interactive=True,
                            )
                            input_asio_channels = gr.Slider(
                                minimum=-1,
                                maximum=16,
                                value=-1,
                                step=1,
                                label=i18n("Input ASIO Channel"),
                                info=i18n(
                                    "For ASIO drivers, selects a specific input channel. Leave at -1 for default."
                                ),
                                interactive=True,
                            )
                    with gr.Accordion("Output Device", open=True):
                        with gr.Column():
                            output_audio_device = gr.Dropdown(
                                label=i18n("Output Device"),
                                info=i18n(
                                    "Select the device where the final converted voice will be sent (e.g., a virtual cable)."
                                ),
                                choices=output_devices,
                                value=get_safe_dropdown_value(
                                    saved_settings["output_device"], output_devices
                                ),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            output_audio_gain = gr.Slider(
                                minimum=0,
                                maximum=200,
                                value=100,
                                label=i18n("Output Gain (%)"),
                                info=i18n(
                                    "Adjusts the final volume of the converted voice after processing."
                                ),
                                interactive=True,
                            )
                            output_asio_channels = gr.Slider(
                                minimum=-1,
                                maximum=16,
                                value=-1,
                                step=1,
                                label=i18n("Output ASIO Channel"),
                                info=i18n(
                                    "For ASIO drivers, selects a specific output channel. Leave at -1 for default."
                                ),
                                interactive=True,
                            )
                with gr.Accordion("Monitor Device (Optional)", open=False):
                    with gr.Column():
                        use_monitor_device = gr.Checkbox(
                            label=i18n("Use Monitor Device"),
                            value=False,
                            interactive=True,
                        )
                        monitor_output_device = gr.Dropdown(
                            label=i18n("Monitor Device"),
                            info=i18n(
                                "Select the device for monitoring your voice (e.g., your headphones)."
                            ),
                            choices=output_devices,
                            value=get_safe_dropdown_value(
                                saved_settings["monitor_device"], output_devices
                            ),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        monitor_audio_gain = gr.Slider(
                            minimum=0,
                            maximum=200,
                            value=100,
                            label=i18n("Monitor Gain (%)"),
                            info=i18n(
                                "Adjusts the volume of the monitor feed, independent of the main output."
                            ),
                            interactive=True,
                        )
                        monitor_asio_channels = gr.Slider(
                            minimum=-1,
                            maximum=16,
                            value=-1,
                            step=1,
                            label=i18n("Monitor ASIO Channel"),
                            info=i18n(
                                "For ASIO drivers, selects a specific monitor output channel. Leave at -1 for default."
                            ),
                            interactive=True,
                        )
                with gr.Row():
                    exclusive_mode = gr.Checkbox(
                        label=i18n("Exclusive Mode (WASAPI)"),
                        info=i18n(
                            "For WASAPI (Windows), gives the app exclusive control for potentially lower latency."
                        ),
                        value=True,
                        interactive=True,
                    )
                    vad_enabled = gr.Checkbox(
                        label=i18n("Enable VAD"),
                        info=i18n(
                            "Enables Voice Activity Detection to only process audio when you are speaking, saving CPU."
                        ),
                        value=True,
                        interactive=True,
                    )

            with gr.TabItem(i18n("Model Settings")):
                with gr.Row():
                    model_choices = (
                        sorted(names, key=extract_model_and_epoch) if names else []
                    )
                    model_file = gr.Dropdown(
                        label=i18n("Voice Model"),
                        choices=model_choices,
                        interactive=True,
                        value=get_safe_dropdown_value(
                            saved_settings["model_file"], model_choices, default_weight
                        ),
                        allow_custom_value=True,
                    )
                    index_choices = get_files("index")
                    index_file = gr.Dropdown(
                        label=i18n("Index File"),
                        choices=index_choices,
                        value=get_safe_index_value(
                            saved_settings["index_file"],
                            index_choices,
                            match_index(default_weight) if default_weight else None,
                        ),
                        interactive=True,
                        allow_custom_value=True,
                    )

                with gr.Row():
                    unload_button = gr.Button(i18n("Unload Voice"))
                    refresh_button = gr.Button(i18n("Refresh"))
                with gr.Column():
                    autotune = gr.Checkbox(
                        label=i18n("Autotune"),
                        info=i18n(
                            "Apply a soft autotune to your inferences, recommended for singing conversions."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )
                    autotune_strength = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Autotune Strength"),
                        info=i18n(
                            "Set the autotune strength - the more you increase it the more it will snap to the chromatic grid."
                        ),
                        visible=False,
                        value=1,
                        interactive=True,
                    )
                    proposed_pitch = gr.Checkbox(
                        label=i18n("Proposed Pitch"),
                        info=i18n(
                            "Adjust the input audio pitch to match the voice model range."
                        ),
                        visible=True,
                        value=False,
                        interactive=True,
                    )
                    proposed_pitch_threshold = gr.Slider(
                        minimum=50.0,
                        maximum=1200.0,
                        label=i18n("Proposed Pitch Threshold"),
                        info=i18n(
                            "Male voice models typically use 155.0 and female voice models typically use 255.0."
                        ),
                        visible=False,
                        value=155.0,
                        interactive=True,
                    )
                    sid = gr.Dropdown(
                        label=i18n("Speaker ID"),
                        choices=(
                            get_speakers_id(default_weight) if default_weight else [0]
                        ),
                        value=0,
                        interactive=True,
                    )
                    pitch = gr.Slider(
                        minimum=-24,
                        maximum=24,
                        step=0.01,
                        label=i18n("Pitch"),
                        info=i18n(
                            "Set the pitch of the audio, the higher the value, the higher the pitch."
                        ),
                        value=0,
                        interactive=True,
                    )
                    index_rate = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Search Feature Ratio"),
                        info=i18n(
                            "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                        ),
                        value=0.75,
                        interactive=True,
                    )
                    volume_envelope = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=1,
                        label=i18n("Volume Envelope"),
                        info=i18n(
                            "Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed."
                        ),
                        interactive=True,
                    )
                    protect = gr.Slider(
                        minimum=0,
                        maximum=0.5,
                        value=0.5,
                        label=i18n("Protect Voiceless Consonants"),
                        info=i18n(
                            "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                        ),
                        interactive=True,
                    )
                    f0_method = gr.Radio(
                        choices=[
                            "rmvpe",
                            "fcpe",
                            "swift",
                            "crepe-tiny",
                            "crepe-small",
                            "crepe-medium",
                            "crepe-large",
                            "crepe-full",
                            "mangio-crepe-tiny",
                            "mangio-crepe-small",
                            "mangio-crepe-medium",
                            "mangio-crepe-large",
                            "mangio-crepe-full",
                        ],
                        value="swift",
                        label=i18n("Pitch extraction algorithm"),
                        info=i18n(
                            "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases. CREPE: tiny (fast) or full (accurate)."
                        ),
                        interactive=True,
                    )
                    hybrid_blend_ratio = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label=i18n("Hybrid Blend Ratio"),
                        info=i18n(
                            "Blend ratio for hybrid mode. 0 = full rmvpe, 1 = full CREPE. Only used when hybrid mode is selected."
                        ),
                        visible=False,
                        interactive=True,
                    )
                    embedder_model = gr.Radio(
                        choices=[
                            "contentvec",
                            "spin",
                            "chinese-hubert-base",
                            "japanese-hubert-base",
                            "korean-hubert-base",
                            "custom",
                        ],
                        value="contentvec",
                        label=i18n("Embedder Model"),
                        info=i18n("Model used for learning speaker embedding."),
                        interactive=True,
                    )
                    with gr.Column(visible=False) as embedder_custom:
                        with gr.Accordion(i18n("Custom Embedder"), open=True):
                            with gr.Row():
                                embedder_model_custom = gr.Dropdown(
                                    label=i18n("Select Custom Embedder"),
                                    choices=refresh_embedders_folders(),
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                refresh_embedders_button = gr.Button(
                                    i18n("Refresh embedders")
                                )
                            folder_name_input = gr.Textbox(
                                label=i18n("Folder Name"), interactive=True
                            )
                            with gr.Row():
                                bin_file_upload = gr.File(
                                    label=i18n("Upload .bin"),
                                    type="filepath",
                                    interactive=True,
                                )
                                config_file_upload = gr.File(
                                    label=i18n("Upload .json"),
                                    type="filepath",
                                    interactive=True,
                                )
                            move_files_button = gr.Button(
                                i18n("Move files to custom embedder folder")
                            )

            with gr.TabItem(i18n("Performance Settings")):
                chunk_size = gr.Slider(
                    minimum=2.7,
                    maximum=2730.7,
                    value=512,
                    step=1,
                    label=i18n("Chunk Size (ms)"),
                    info=i18n(
                        "Audio buffer size in milliseconds. Lower values may reduce latency but increase CPU load."
                    ),
                    interactive=True,
                )
                cross_fade_overlap_size = gr.Slider(
                    minimum=0.05,
                    maximum=0.2,
                    value=0.05,
                    step=0.01,
                    label=i18n("Crossfade Overlap Size (s)"),
                    info=i18n(
                        "Duration of the fade between audio chunks to prevent clicks. Higher values create smoother transitions but may increase latency."
                    ),
                    interactive=True,
                )
                extra_convert_size = gr.Slider(
                    minimum=0.1,
                    maximum=5,
                    value=0.5,
                    step=0.1,
                    label=i18n("Extra Conversion Size (s)"),
                    info=i18n(
                        "Amount of extra audio processed to provide context to the model. Improves conversion quality at the cost of higher CPU usage."
                    ),
                    interactive=True,
                )
                silent_threshold = gr.Slider(
                    minimum=-90,
                    maximum=-60,
                    value=-90,
                    step=1,
                    label=i18n("Silence Threshold (dB)"),
                    info=i18n(
                        "Volume level below which audio is treated as silence and not processed. Helps to save CPU resources and reduce background noise."
                    ),
                    interactive=True,
                )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                yield message, interactive_true, interactive_false
                return
            yield from start_realtime(*args)

        def update_on_model_change(model_path):
            new_index = match_index(model_path)
            new_sids = get_speakers_id(model_path)

            # Get updated index choices
            new_index_choices = get_files("index")
            # Use the matched index as fallback, but handle empty strings
            fallback_index = new_index if new_index and new_index.strip() else None
            safe_index_value = get_safe_index_value(
                "", new_index_choices, fallback_index
            )

            return gr.update(
                choices=new_index_choices, value=safe_index_value
            ), gr.update(choices=new_sids, value=0 if new_sids else None)

        def refresh_devices():
            sd._terminate()
            sd._initialize()

            input_choices, output_choices = get_audio_devices_formatted()
            input_choices, output_choices = list(input_choices.keys()), list(
                output_choices.keys()
            )
            return (
                gr.update(choices=input_choices),
                gr.update(choices=output_choices),
                gr.update(choices=output_choices),
            )

        def toggle_visible(checkbox):
            return {"visible": checkbox, "__type__": "update"}

        def toggle_visible_embedder_custom(embedder_model):
            if embedder_model == "custom":
                return {"visible": True, "__type__": "update"}
            return {"visible": False, "__type__": "update"}

        def toggle_hybrid_blend_slider(f0_method):
            if f0_method.startswith("hybrid("):
                return {"visible": True, "__type__": "update"}
            return {"visible": False, "__type__": "update"}

        refresh_devices_button.click(
            fn=refresh_devices,
            outputs=[input_audio_device, output_audio_device, monitor_output_device],
        )

        autotune.change(
            fn=toggle_visible,
            inputs=[autotune],
            outputs=[autotune_strength],
        )

        proposed_pitch.change(
            fn=toggle_visible,
            inputs=[proposed_pitch],
            outputs=[proposed_pitch_threshold],
        )

        embedder_model.change(
            fn=toggle_visible_embedder_custom,
            inputs=[embedder_model],
            outputs=[embedder_custom],
        )

        f0_method.change(
            fn=toggle_hybrid_blend_slider,
            inputs=[f0_method],
            outputs=[hybrid_blend_ratio],
        )

        move_files_button.click(
            fn=create_folder_and_move_files,
            inputs=[folder_name_input, bin_file_upload, config_file_upload],
            outputs=[],
        )
        refresh_embedders_button.click(
            fn=lambda: gr.update(choices=refresh_embedders_folders()),
            inputs=[],
            outputs=[embedder_model_custom],
        )

        start_button.click(
            fn=enforce_terms,
            inputs=[
                terms_checkbox,
                input_audio_device,
                input_audio_gain,
                input_asio_channels,
                output_audio_device,
                output_audio_gain,
                output_asio_channels,
                monitor_output_device,
                monitor_audio_gain,
                monitor_asio_channels,
                use_monitor_device,
                exclusive_mode,
                vad_enabled,
                chunk_size,
                cross_fade_overlap_size,
                extra_convert_size,
                silent_threshold,
                pitch,
                index_rate,
                volume_envelope,
                protect,
                f0_method,
                hybrid_blend_ratio,
                model_file,
                index_file,
                sid,
                autotune,
                autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
                embedder_model,
                embedder_model_custom,
            ],
            outputs=[latency_info, start_button, stop_button],
        )

        stop_button.click(
            fn=stop_realtime, outputs=[latency_info, start_button, stop_button]
        ).then(
            fn=lambda: (
                yield gr.update(value="Stopped"),
                interactive_true,
                interactive_false,
            ),
            inputs=None,
            outputs=[latency_info, start_button, stop_button],
        )
        unload_button.click(
            fn=lambda: (
                {"value": "", "__type__": "update"},
                {"value": "", "__type__": "update"},
            ),
            inputs=[],
            outputs=[model_file, index_file],
        )
        model_file.select(
            fn=update_on_model_change, inputs=[model_file], outputs=[index_file, sid]
        )

        # Save settings when devices or model change
        def save_input_device(input_device):
            if input_device:
                save_realtime_settings(input_device, None, None, None, None)

        def save_output_device(output_device):
            if output_device:
                save_realtime_settings(None, output_device, None, None, None)

        def save_monitor_device(monitor_device):
            if monitor_device:
                save_realtime_settings(None, None, monitor_device, None, None)

        def save_model_file(model_file):
            if model_file:
                save_realtime_settings(None, None, None, model_file, None)

        def save_index_file(index_file):
            # Only save if index_file is not None and not empty
            if index_file:
                save_realtime_settings(None, None, None, None, index_file)

        # Add event handlers to save settings
        input_audio_device.change(
            fn=save_input_device, inputs=[input_audio_device], outputs=[]
        )

        output_audio_device.change(
            fn=save_output_device, inputs=[output_audio_device], outputs=[]
        )

        monitor_output_device.change(
            fn=save_monitor_device, inputs=[monitor_output_device], outputs=[]
        )

        def refresh_all():
            new_names = get_files("model")
            new_indexes = get_files("index")
            input_choices, output_choices = get_audio_devices_formatted()
            input_choices, output_choices = list(input_choices.keys()), list(
                output_choices.keys()
            )
            return (
                gr.update(choices=sorted(new_names, key=extract_model_and_epoch)),
                gr.update(choices=new_indexes),
                gr.update(choices=input_choices),
                gr.update(choices=output_choices),
                gr.update(choices=output_choices),
            )

        # Template functions
        def apply_template_settings(template_name):
            """Load and apply template settings"""
            if not template_name:
                gr.Warning("Please select a template first.")
                return [gr.update()] * 31

            template_data = template_manager.load_template(template_name)
            if not template_data:
                gr.Warning(f"Template '{template_name}' not found.")
                return [gr.update()] * 31

            # Check if devices exist in current device list
            audio_tab = template_data.get("audioTab", {})
            model_tab = template_data.get("modelTab", {})

            input_dev = audio_tab.get("input", {}).get("device", "")
            output_dev = audio_tab.get("output", {}).get("device", "")
            monitor_dev = audio_tab.get("monitor", {}).get("device", "")
            model_file = model_tab.get("voice", {}).get("model_path", "")
            index_file = model_tab.get("voice", {}).get("index_path", "")

            missing_devices = []
            if input_dev and input_dev not in input_devices:
                missing_devices.append(f"Input: {input_dev}")
            if output_dev and output_dev not in output_devices:
                missing_devices.append(f"Output: {output_dev}")
            if monitor_dev and monitor_dev not in output_devices:
                missing_devices.append(f"Monitor: {monitor_dev}")

            if missing_devices:
                gr.Info(f"Template loaded. Note: Some devices are not currently available:\n" + "\n".join(missing_devices))

            # Save all realtime settings at once (prevents race condition from individual .change() events)
            update_nested_config(CONFIG_PATH, "realtime", {
                "input_device": input_dev or "",
                "output_device": output_dev or "",
                "monitor_device": monitor_dev or "",
                "model_file": model_file or "",
                "index_file": index_file or "",
            })

            return template_manager.extract_settings_to_gradio_updates(template_data)

        def on_apply_template(template_name):
            """Apply template without confirmation"""
            if not template_name:
                gr.Warning("Please select a template first.")
                return [gr.update()] * 31

            return apply_template_settings(template_name)

        def on_overwrite_template(template_name):
            """Show modal for overwrite with current template name"""
            if not template_name:
                gr.Warning("Please select a template first.")
                return gr.update(visible=False), gr.update(value=""), "overwrite"

            # Show modal with current template name pre-filled
            return (
                gr.update(visible=True),
                gr.update(value=template_name),
                f"overwrite:{template_name}",
            )

        def on_add_template():
            """Show modal for adding new template"""
            return gr.update(visible=True), gr.update(value=""), "add"

        def on_template_save(
            operation_state, new_name,
            inp_device, inp_gain, inp_asio, out_device, out_gain, out_asio,
            use_mon, mon_device, mon_gain, mon_asio, excl_mode, vad_en,
            mdl_file, idx_file, atune, atune_str, prop_pitch, prop_pitch_thresh,
            speaker_id, ptch, idx_rate, vol_env, prot, f0_meth, hybrid_ratio,
            emb_model, emb_custom, chnk_size, cross_fade, extra_conv, silent_thresh,
            stable_md
        ):
            """Handle save button in modal"""
            if not operation_state:
                return (
                    gr.update(visible=False),
                    gr.update(),
                    "",
                )

            new_name = new_name.strip() if new_name else ""
            existing_templates = template_manager.get_template_list()

            if operation_state.startswith("overwrite:"):
                # Overwrite existing template
                current_name = operation_state.split(":", 1)[1]

                if not new_name:
                    new_name = current_name

                error = template_manager.validate_template_name(new_name, existing_templates, current_name)
                if error:
                    gr.Warning(error)
                    return gr.update(), gr.update(), operation_state

                # If name changed, delete old template
                if new_name != current_name:
                    template_manager.delete_template(current_name)

                # Save current settings
                settings = template_manager.create_settings_dict(
                    inp_device, inp_gain, inp_asio, out_device, out_gain, out_asio,
                    use_mon, mon_device, mon_gain, mon_asio, excl_mode, vad_en,
                    mdl_file, idx_file, atune, atune_str, prop_pitch, prop_pitch_thresh,
                    speaker_id, ptch, idx_rate, vol_env, prot, f0_meth, hybrid_ratio,
                    emb_model, emb_custom, chnk_size, cross_fade, extra_conv, silent_thresh,
                    stable_md
                )
                template_manager.save_template(new_name, settings)
                gr.Info(f"Template '{new_name}' saved successfully.")

                # Hide modal and refresh dropdown
                return (
                    gr.update(visible=False),
                    gr.update(choices=template_manager.get_template_list(), value=new_name),
                    "",
                )

            elif operation_state == "add":
                # Add new template
                error = template_manager.validate_template_name(new_name, existing_templates)
                if error:
                    gr.Warning(error)
                    return gr.update(), gr.update(), operation_state

                # Save current settings
                settings = template_manager.create_settings_dict(
                    inp_device, inp_gain, inp_asio, out_device, out_gain, out_asio,
                    use_mon, mon_device, mon_gain, mon_asio, excl_mode, vad_en,
                    mdl_file, idx_file, atune, atune_str, prop_pitch, prop_pitch_thresh,
                    speaker_id, ptch, idx_rate, vol_env, prot, f0_meth, hybrid_ratio,
                    emb_model, emb_custom, chnk_size, cross_fade, extra_conv, silent_thresh,
                    stable_md
                )
                template_manager.save_template(new_name, settings)
                gr.Info(f"Template '{new_name}' created successfully.")

                # Hide modal and refresh dropdown
                return (
                    gr.update(visible=False),
                    gr.update(choices=template_manager.get_template_list(), value=new_name),
                    "",
                )

            return gr.update(visible=False), gr.update(), ""

        def on_template_cancel():
            """Handle cancel button in modal"""
            return gr.update(visible=False), ""

        def on_delete_template(template_name):
            """Delete template (using JavaScript confirm would be better, but using direct delete for now)"""
            if not template_name:
                gr.Warning("Please select a template first.")
                return gr.update()

            template_manager.delete_template(template_name)
            gr.Info(f"Template '{template_name}' deleted successfully.")

            # Refresh template list
            new_list = template_manager.get_template_list()
            return gr.update(choices=new_list, value=new_list[0] if new_list else None)

        model_file.change(fn=save_model_file, inputs=[model_file], outputs=[])

        index_file.change(fn=save_index_file, inputs=[index_file], outputs=[])

        refresh_button.click(
            fn=refresh_all,
            outputs=[
                model_file,
                index_file,
                input_audio_device,
                output_audio_device,
                monitor_output_device,
            ],
        )

        # Template button event handlers
        apply_template_button.click(
            fn=on_apply_template,
            inputs=[template_dropdown],
            outputs=[
                input_audio_device,
                input_audio_gain,
                input_asio_channels,
                output_audio_device,
                output_audio_gain,
                output_asio_channels,
                use_monitor_device,
                monitor_output_device,
                monitor_audio_gain,
                monitor_asio_channels,
                exclusive_mode,
                vad_enabled,
                model_file,
                index_file,
                autotune,
                autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
                sid,
                pitch,
                index_rate,
                volume_envelope,
                protect,
                f0_method,
                hybrid_blend_ratio,
                embedder_model,
                embedder_model_custom,
                chunk_size,
                cross_fade_overlap_size,
                extra_convert_size,
                silent_threshold,
            ],
        )

        overwrite_template_button.click(
            fn=on_overwrite_template,
            inputs=[template_dropdown],
            outputs=[template_name_input_row, template_name_input, template_operation_state],
        )

        add_template_button.click(
            fn=on_add_template,
            inputs=[],
            outputs=[template_name_input_row, template_name_input, template_operation_state],
        )

        delete_template_button.click(
            fn=on_delete_template,
            inputs=[template_dropdown],
            outputs=[template_dropdown],
        )

        template_save_button.click(
            fn=on_template_save,
            inputs=[
                template_operation_state,
                template_name_input,
                input_audio_device,
                input_audio_gain,
                input_asio_channels,
                output_audio_device,
                output_audio_gain,
                output_asio_channels,
                use_monitor_device,
                monitor_output_device,
                monitor_audio_gain,
                monitor_asio_channels,
                exclusive_mode,
                vad_enabled,
                model_file,
                index_file,
                autotune,
                autotune_strength,
                proposed_pitch,
                proposed_pitch_threshold,
                sid,
                pitch,
                index_rate,
                volume_envelope,
                protect,
                f0_method,
                hybrid_blend_ratio,
                embedder_model,
                embedder_model_custom,
                chunk_size,
                cross_fade_overlap_size,
                extra_convert_size,
                silent_threshold,
            ],
            outputs=[template_name_input_row, template_dropdown, template_operation_state],
        )

        template_cancel_button.click(
            fn=on_template_cancel,
            inputs=[],
            outputs=[template_name_input_row, template_operation_state],
        )