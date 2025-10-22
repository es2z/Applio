import os
import sys
import gradio as gr
from assets.i18n.i18n import I18nAuto
from rvc.configs.config_utils import load_config, update_config

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()
config_file = os.path.join(now_dir, "assets", "config.json")

filter_trigger = None


def get_filter_trigger():
    global filter_trigger
    if filter_trigger is None:
        filter_trigger = gr.Textbox(visible=False)
    return filter_trigger


def load_config_filter():
    cfg = load_config(config_file)
    return bool(cfg.get("model_index_filter", False))


def save_config_filter(val: bool):
    update_config(config_file, {"model_index_filter": bool(val)})


def filter_tab():
    checkbox = gr.Checkbox(
        label=i18n("Enable model/index list filter"),
        info=i18n(
            "Adds a keyword filter for the model/index selection lists in the Inference and TTS tabs."
        ),
        value=load_config_filter(),
        interactive=True,
    )
    checkbox.change(fn=save_config_filter, inputs=[checkbox], outputs=[])
    return checkbox
