import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

import gradio as gr
from assets.i18n.i18n import I18nAuto
from rvc.configs.config_utils import load_config, update_config

i18n = I18nAuto()

CONFIG_PATH = os.path.join(now_dir, "assets", "config.json")


def set_model_author(model_author: str):
    update_config(CONFIG_PATH, {"model_author": model_author})
    print(f"Model author set to {model_author}.")
    return f"Model author set to {model_author}."


def get_model_author():
    config = load_config(CONFIG_PATH)
    return config.get("model_author", None)


def model_author_tab():
    model_author_name = gr.Textbox(
        label=i18n("Model Author Name"),
        info=i18n("The name that will appear in the model information."),
        value=get_model_author(),
        placeholder=i18n("Enter your nickname"),
        interactive=True,
    )
    model_author_output_info = gr.Textbox(
        label=i18n("Output Information"),
        info=i18n("The output information will be displayed here."),
        value="",
        max_lines=1,
    )
    button = gr.Button(i18n("Set name"))

    button.click(
        fn=set_model_author,
        inputs=[model_author_name],
        outputs=[model_author_output_info],
    )
