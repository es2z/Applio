import os, sys
import gradio as gr
from assets.i18n.i18n import I18nAuto
from rvc.configs.config_utils import load_config, update_nested_config

now_dir = os.getcwd()
sys.path.append(now_dir)

i18n = I18nAuto()

config_file = os.path.join(now_dir, "assets", "config.json")


def get_language_settings():
    config = load_config(config_file)
    lang_config = config.get("lang", {})

    if not lang_config.get("override", False):
        return "Language automatically detected in the system"
    else:
        return lang_config.get("selected_lang", "en_US")


def save_lang_settings(selected_language):
    if selected_language == "Language automatically detected in the system":
        updates = {"override": False}
    else:
        updates = {"override": True, "selected_lang": selected_language}

    update_nested_config(config_file, "lang", updates)
    gr.Info("Language have been saved. Restart Applio to apply the changes.")


def lang_tab():
    with gr.Column():
        selected_language = gr.Dropdown(
            label=i18n("Language"),
            info=i18n(
                "Select the language you want to use. (Requires restarting Applio)"
            ),
            value=get_language_settings(),
            choices=["Language automatically detected in the system"]
            + i18n._get_available_languages(),
            interactive=True,
        )

        selected_language.change(
            fn=save_lang_settings,
            inputs=[selected_language],
            outputs=[],
        )
