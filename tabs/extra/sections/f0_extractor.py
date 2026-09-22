from tabs.components import mangio_crepe_decoder, fcn_profile_controls
import os
import librosa
import gradio as gr
from matplotlib import pyplot as plt

from rvc.lib.predictors.F0Extractor import F0Extractor
from rvc.lib.predictors.crepe_models import CREPE_UI_METHODS
from rvc.lib.predictors.f0_methods import FCN_UI_METHODS

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()


def extract_f0_curve(audio_path: str, method: str, fcn_profile=None):
    print("Extracting F0 Curve...")
    image_path = os.path.join("logs", "f0_plot.png")
    txt_path = os.path.join("logs", "f0_curve.csv")
    os.makedirs("logs", exist_ok=True)
    sr = librosa.get_samplerate(audio_path)

    librosa.note_to_hz("C1")
    librosa.note_to_hz("C8")

    f0_extractor = F0Extractor(audio_path, sample_rate=sr, method=method, fcn_profile=fcn_profile)
    track = f0_extractor.extract_track()
    f0 = track["pitch_hz"]

    plt.figure(figsize=(10, 4))
    plt.plot(track["timestamps"], f0)
    plt.title(method)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as txtfile:
        txtfile.write("seconds,pitch_hz,confidence,voiced\n")
        for time, pitch, confidence, voiced in zip(track["timestamps"], f0, track["confidence"], track["voiced"]):
            txtfile.write(f"{time},{pitch},{confidence},{int(voiced)}\n")

    print("F0 Curve extracted successfully!")
    return image_path, txt_path


def f0_extractor_tab():
    audio = gr.Audio(label=i18n("Upload Audio"), type="filepath")
    f0_method = gr.Radio(
        label=i18n("Pitch extraction algorithm"),
        info=i18n(
            "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
        ),
        choices=[*CREPE_UI_METHODS, *FCN_UI_METHODS, "fcpe", "rmvpe"],
        value="rmvpe",
    )
    mangio_crepe_decoder(f0_method)
    fcn_profile = fcn_profile_controls(f0_method)
    button = gr.Button(i18n("Extract F0 Curve"))

    with gr.Row():
        txt_output = gr.File(label=i18n("F0 Curve"), type="filepath")
        image_output = gr.Image(type="filepath", interactive=False)

    button.click(
        fn=extract_f0_curve,
        inputs=[
            audio,
            f0_method,
            fcn_profile,
        ],
        outputs=[image_output, txt_output],
    )
