"""UI pieces shared by more than one tab."""

import gradio as gr

from assets.i18n.i18n import I18nAuto
from rvc.lib.predictors.crepe_decoder import (
    DECODERS,
    load_decoder,
    save_decoder,
    uses_mangio_crepe,
)

i18n = I18nAuto()


def fcn_profile_controls(f0_method, realtime=False):
    from rvc.lib.predictors.f0_methods import FCN_METHODS

    with gr.Group(visible=f0_method.value in FCN_METHODS) as group:
        gr.Markdown(
            "**FCN (CUDA)** — FCN-993 reproduces the original predictor without a voicing threshold. "
            "FCN-993-RVC is experimental and requires an explicit threshold profile; no calibrated default is available. "
            "Settings are fixed when processing starts. Optional network compilation: add `\"compile_model\": true` to the profile (default OFF)."
        )
        if realtime:
            gr.Markdown("FCN delays audio and F0 together by 140–150 ms, plus capture filtering and up to 10 ms of grid alignment. Device, chunk and queue latency are additional.")
        profile = gr.Textbox(
            label="FCN profile JSON or local JSON path",
            info="Leave blank for baseline defaults or a matching checkpoint profile. coarse_max may be 1680 (default) or 1100 Hz.",
            value="", lines=3,
        )
    f0_method.change(
        fn=lambda method: gr.update(visible=method in FCN_METHODS),
        inputs=[f0_method], outputs=[group], show_progress=False,
    )
    return profile


def mangio_crepe_decoder(f0_method):
    """A decoder picker that shows itself while a mangio-crepe method is selected.

    The setting is global rather than per tab or per template, so the picker reloads its
    value whenever the pitch algorithm changes and every tab stays in step.
    """
    decoder = gr.Dropdown(
        label=i18n("Mangio-CREPE Decoder"),
        info=i18n(
            "How mangio-crepe turns the network's output into a pitch. 'viterbi' is what it has always used, but it is not repeatable on CUDA: argmax breaks ties arbitrarily and CREPE's bins are 20 cents apart, so the same audio drifts up to 26 cents between runs. 'weighted_argmax' is bit-identical run to run, at the cost of a different pitch estimate, so the voice changes with it."
        ),
        choices=list(DECODERS),
        value=load_decoder(),
        visible=uses_mangio_crepe(f0_method.value),
        interactive=True,
    )

    f0_method.change(
        fn=lambda method: gr.update(
            visible=uses_mangio_crepe(method), value=load_decoder()
        ),
        inputs=[f0_method],
        outputs=[decoder],
        show_progress=False,
    )
    decoder.change(
        fn=save_decoder, inputs=[decoder], outputs=[], show_progress=False
    )
    return decoder
