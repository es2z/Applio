import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import gradio as gr
from assets.i18n.i18n import I18nAuto
from rvc.configs.config_utils import load_config, update_config

i18n = I18nAuto()

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")


def set_precision(precision: str):
    update_config(CONFIG_PATH, {"precision": precision})
    print(f"Precision set to {precision}.")
    return f"Precision set to {precision}."


def get_precision():
    config = load_config(CONFIG_PATH)
    return config.get("precision", None)


def precision_tab():
    precision = gr.Radio(
        label=i18n("Precision"),
        info=i18n("Select the precision you want to use for training and inference."),
        value=get_precision(),
        choices=["fp32", "fp16", "bf16"],
        interactive=True,
    )
    precision_info = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
        value="",
        max_lines=1,
    )
    button = gr.Button(i18n("Update precision"))

    button.click(
        fn=set_precision,
        inputs=[precision],
        outputs=[precision_info],
    )
