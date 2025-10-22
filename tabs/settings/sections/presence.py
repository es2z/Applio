import os
import sys
import gradio as gr
from assets.i18n.i18n import I18nAuto
from assets.discord_presence import RPCManager
from rvc.configs.config_utils import load_config, update_config

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()
config_file = os.path.join(now_dir, "assets", "config.json")


def load_config_presence():
    config = load_config(config_file)
    return config.get("discord_presence", True)


def save_config(value):
    update_config(config_file, {"discord_presence": value})


def presence_tab():
    with gr.Row():
        with gr.Column():
            presence = gr.Checkbox(
                label=i18n("Enable Applio integration with Discord presence"),
                info=i18n(
                    "It will activate the possibility of displaying the current Applio activity in Discord."
                ),
                interactive=True,
                value=load_config_presence(),
            )
            presence.change(
                fn=toggle,
                inputs=[presence],
                outputs=[],
            )


def toggle(checkbox):
    save_config(bool(checkbox))
    if load_config_presence() == True:
        try:
            RPCManager.start_presence()
        except KeyboardInterrupt:
            RPCManager.stop_presence()
    else:
        RPCManager.stop_presence()
