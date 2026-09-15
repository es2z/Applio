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
