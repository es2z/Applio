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


FCN_PROFILE_HELP = (
    "**FCN (CUDA)** — FCN-993 reproduces the original predictor without a voicing threshold. "
    "For voice conversion with FCN, use **FCN-993-RVC**. No JSON is required: leave the profile blank to use a matching checkpoint profile or the bundled **Balanced v1** settings (enter 0.50, exit 0.40, median 5 ms). "
    'Settings are fixed when processing starts. Optional network compilation: add `"compile_model": true` to the profile (default OFF).'
)
FCN_REALTIME_HELP = "FCN delays audio and F0 together by 140–150 ms, plus capture filtering and up to 10 ms of grid alignment. Device, chunk and queue latency are additional."
FCNF0PP_PROFILE_HELP = (
    "**FCNF0++ (PENN)** — FCNF0++ passes PENN's pitch through unchanged, with every frame voiced: use it to judge the model itself. "
    "**FCNF0++-RVC** is the same pitch with frames whose PENN periodicity is at or below `periodicity_threshold` set to unvoiced, and nothing else. "
    "The **(aligned)** variants centre each analysis window 11 ms later, cancelling the ~11 ms the model lags on speech-range voices; on high voices (above ~300 Hz) they run that much early instead. "
    'Leave the profile blank to use a matching checkpoint profile or the bundled defaults (Viterbi decoding; periodicity threshold 0.035 for FCNF0++-RVC and 0.0425 for FCNF0++-RVC (aligned), each chosen by the voiced-F1 criterion PENN itself uses). `"decoder": "argmax"` is also available. Details: docs/fcnf0pp.md'
)
FCNF0PP_REALTIME_HELP = "FCNF0++ keeps no stream state and adds no holdback: every block recomputes the whole conversion window, like RMVPE and CREPE."


def _f0_profile_help(method, realtime):
    from rvc.lib.predictors.f0_methods import FCN_METHODS, FCNF0PP_METHODS

    if method in FCNF0PP_METHODS:
        return FCNF0PP_PROFILE_HELP + ("\n\n" + FCNF0PP_REALTIME_HELP if realtime else "")
    if method in FCN_METHODS:
        return FCN_PROFILE_HELP + ("\n\n" + FCN_REALTIME_HELP if realtime else "")
    return ""


def recommended_f0_profile_json(method):
    from rvc.lib.predictors.f0_methods import FCNF0PP_METHODS

    if method in FCNF0PP_METHODS:
        from rvc.lib.predictors.fcnf0pp.profiles import recommended_profile_json
    else:
        from rvc.lib.predictors.fcn.profiles import recommended_profile_json
    return recommended_profile_json(method)


def fcn_profile_controls(f0_method, realtime=False):
    """The profile box shared by FCN-993 and FCNF0++; the JSON's method picks the family."""
    from rvc.lib.predictors.f0_methods import PROFILE_METHODS

    with gr.Group(visible=f0_method.value in PROFILE_METHODS) as group:
        help_text = gr.Markdown(_f0_profile_help(f0_method.value, realtime))
        profile = gr.Textbox(
            label="F0 profile JSON or local JSON path (optional)",
            info="Normally leave blank. To explicitly use the recommended settings instead of checkpoint settings, press the button below. Edit the JSON only to customize. Details: docs/fcn-993.md, docs/fcnf0pp.md",
            value="",
            lines=3,
        )
        recommended = gr.Button("Load recommended F0 settings")
        recommended.click(
            fn=recommended_f0_profile_json,
            inputs=[f0_method],
            outputs=[profile],
            show_progress=False,
        )
    f0_method.change(
        fn=lambda method: (
            gr.update(visible=method in PROFILE_METHODS),
            gr.update(value=_f0_profile_help(method, realtime)),
        ),
        inputs=[f0_method],
        outputs=[group, help_text],
        show_progress=False,
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
    decoder.change(fn=save_decoder, inputs=[decoder], outputs=[], show_progress=False)
    return decoder
