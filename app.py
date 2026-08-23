import gradio as gr
import sys
import os
import logging

from tabs.settings.sections.torch_compile import bootstrap_torch_compile_environment

bootstrap_torch_compile_environment()

import torch

from typing import Any

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 6969
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add current directory to sys.path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Zluda hijack
import rvc.lib.zluda

# Import Tabs
from tabs.inference.inference import inference_tab
from tabs.train.train import train_tab
from tabs.extra.extra import extra_tab
from tabs.report.report import report_tab
from tabs.download.download import download_tab
from tabs.tts.tts import tts_tab
from tabs.voice_blender.voice_blender import voice_blender_tab
from tabs.plugins.plugins import plugins_tab
from tabs.settings.settings import settings_tab
from tabs.realtime.realtime import realtime_tab
from tabs.settings.sections.torch_compile import (
    load_torch_compile_enabled,
    load_torch_compile_fcnf0pp_enabled,
    load_torch_compile_rvc_enabled,
    load_torch_compile_mode,
    save_realtime_compile_settings,
    is_torch_compile_available,
    get_triton_status,
    clear_inactive_compile_caches,
    TORCH_COMPILE_MODES,
)

# Run prerequisites
from core import run_prerequisites_script

run_prerequisites_script(
    pretraineds_hifigan=True,
    models=True,
    exe=True,
)

# Initialize i18n
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

# Start Discord presence if enabled
from tabs.settings.sections.presence import load_config_presence

if load_config_presence():
    from assets.discord_presence import RPCManager

    RPCManager.start_presence()

# Check installation
import assets.installation_checker as installation_checker

installation_checker.check_installation()

# Load theme
import assets.themes.loadThemes as loadThemes

my_applio = loadThemes.load_theme() or "ParityError/Interstellar"

# Define Gradio interface
with gr.Blocks(
    theme=my_applio, title="Applio", css="footer{display:none !important}"
) as Applio:
    gr.Markdown("# Applio")
    gr.Markdown(
        i18n(
            "A simple, high-quality voice conversion tool focused on ease of use and performance."
        )
    )
    gr.Markdown(
        i18n(
            "[Support](https://discord.gg/urxFjYmYYh) — [GitHub](https://github.com/IAHispano/Applio)"
        )
    )

    # TorchCompile Settings (collapsible, initially collapsed)
    torch_compile_available = is_torch_compile_available()
    triton_available, triton_status = get_triton_status()
    torch_compile_initial_enabled = load_torch_compile_enabled()
    torch_compile_fcnf0pp_initial_enabled = load_torch_compile_fcnf0pp_enabled()
    torch_compile_rvc_initial_enabled = load_torch_compile_rvc_enabled()
    with gr.Accordion(i18n("TorchCompile Settings"), open=False):
        if not torch_compile_available:
            if not torch.cuda.is_available():
                gr.Markdown(
                    i18n(
                        "Note: CUDA is not available. TorchCompile requires CUDA."
                    )
                )
        gr.Markdown(
            i18n(
                f"Backend: {triton_status}. Changes are saved now and applied only on the next Realtime Start."
            )
        )
        with gr.Row():
            torch_compile_crepe_checkbox = gr.Checkbox(
                label=i18n("Compile CREPE / Mangio-CREPE"),
                info=i18n(
                    "Compile the dominant CREPE pitch model before audio streams start."
                ),
                value=torch_compile_initial_enabled,
                interactive=True,
            )
            torch_compile_rvc_checkbox = gr.Checkbox(
                label=i18n("Compile RVC generator (experimental)"),
                info=i18n(
                    "Compile the RVC generator independently; failures fall back to eager without disabling CREPE."
                ),
                value=torch_compile_rvc_initial_enabled,
                interactive=True,
            )
            torch_compile_fcnf0pp_checkbox = gr.Checkbox(
                label=i18n("Compile FCNF0++ (experimental)"),
                info=i18n(
                    "Compile FCNF0++ only for fixed-shape realtime inference. Eager is usually faster and remains the default."
                ),
                value=torch_compile_fcnf0pp_initial_enabled,
                interactive=True,
            )
            torch_compile_mode_dropdown = gr.Dropdown(
                label=i18n("TorchCompile Mode"),
                info=i18n(
                    "default is the stable baseline. reduce-overhead enables CUDA Graphs and may use more VRAM. max-autotune-no-cudagraphs tunes kernels without CUDA Graphs."
                ),
                choices=TORCH_COMPILE_MODES,
                value=load_torch_compile_mode(),
                interactive=True,
                visible=(
                    torch_compile_initial_enabled
                    or torch_compile_rvc_initial_enabled
                    or torch_compile_fcnf0pp_initial_enabled
                ),
            )
            clear_compile_cache_button = gr.Button(
                i18n("Clear inactive compile caches")
            )
        compile_cache_status = gr.Markdown("")

        def on_torch_compile_change(
            crepe_enabled, rvc_enabled, fcnf0pp_enabled, mode
        ):
            save_realtime_compile_settings(
                crepe_enabled, rvc_enabled, mode, fcnf0pp_enabled
            )
            return gr.update(
                visible=bool(crepe_enabled or rvc_enabled or fcnf0pp_enabled)
            )

        def clear_compile_cache_when_stopped():
            import importlib

            realtime_state = importlib.import_module("tabs.realtime.realtime")
            if realtime_state.running:
                return "Stop Realtime before clearing compile caches."
            return clear_inactive_compile_caches()

        torch_compile_crepe_checkbox.change(
            fn=on_torch_compile_change,
            inputs=[
                torch_compile_crepe_checkbox,
                torch_compile_rvc_checkbox,
                torch_compile_fcnf0pp_checkbox,
                torch_compile_mode_dropdown,
            ],
            outputs=[torch_compile_mode_dropdown],
        )
        torch_compile_rvc_checkbox.change(
            fn=on_torch_compile_change,
            inputs=[
                torch_compile_crepe_checkbox,
                torch_compile_rvc_checkbox,
                torch_compile_fcnf0pp_checkbox,
                torch_compile_mode_dropdown,
            ],
            outputs=[torch_compile_mode_dropdown],
        )
        torch_compile_fcnf0pp_checkbox.change(
            fn=on_torch_compile_change,
            inputs=[
                torch_compile_crepe_checkbox,
                torch_compile_rvc_checkbox,
                torch_compile_fcnf0pp_checkbox,
                torch_compile_mode_dropdown,
            ],
            outputs=[torch_compile_mode_dropdown],
        )
        torch_compile_mode_dropdown.change(
            fn=on_torch_compile_change,
            inputs=[
                torch_compile_crepe_checkbox,
                torch_compile_rvc_checkbox,
                torch_compile_fcnf0pp_checkbox,
                torch_compile_mode_dropdown,
            ],
            outputs=[torch_compile_mode_dropdown],
        )
        clear_compile_cache_button.click(
            fn=clear_compile_cache_when_stopped,
            inputs=[],
            outputs=[compile_cache_status],
        )

    with gr.Tab(i18n("Inference")):
        inference_tab()

    with gr.Tab(i18n("Training")):
        train_tab()

    with gr.Tab(i18n("TTS")):
        tts_tab()

    with gr.Tab(i18n("Voice Blender")):
        voice_blender_tab()

    with gr.Tab(i18n("Realtime")):
        realtime_tab()

    with gr.Tab(i18n("Plugins")):
        plugins_tab()

    with gr.Tab(i18n("Download")):
        download_tab()

    with gr.Tab(i18n("Report a Bug")):
        report_tab()

    with gr.Tab(i18n("Extra")):
        extra_tab()

    with gr.Tab(i18n("Settings")):
        settings_tab()

    gr.Markdown(
        """
    <div style="text-align: center; font-size: 0.9em; text-color: a3a3a3;">
    By using Applio, you agree to comply with ethical and legal standards, respect intellectual property and privacy rights, avoid harmful or prohibited uses, and accept full responsibility for any outcomes, while Applio disclaims liability and reserves the right to amend these terms.
    </div>
    """
    )


def launch_gradio(server_name: str, server_port: int) -> None:
    Applio.launch(
        favicon_path="assets/ICON.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
    )


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(
                f"Failed to launch on port {port}, trying again on port {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"An error occurred launching Gradio: {error}")
            break
