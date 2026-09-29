"""F0 method registration without importing any inference runtime."""

# Two of ardaillon/FCN-f0's networks, each as the original predictor and as an -rvc
# variant that adds voicing (docs/fcn-993.md, docs/fcn-929.md).
FCN_METHODS = ("fcn-993", "fcn-993-rvc", "fcn-929", "fcn-929-rvc")
FCN_UI_METHODS = [
    ("FCN-993", "fcn-993"),
    ("FCN-993-RVC", "fcn-993-rvc"),
    ("FCN-929", "fcn-929"),
    ("FCN-929-RVC", "fcn-929-rvc"),
]

# The -aligned methods place each window 11 ms later to cancel the model's own lag on
# speech (docs/fcnf0pp.md); the others are PENN's framing unchanged.
FCNF0PP_METHODS = ("fcnf0++", "fcnf0++-rvc", "fcnf0++-aligned", "fcnf0++-rvc-aligned")
FCNF0PP_UI_METHODS = [
    ("FCNF0++", "fcnf0++"),
    ("FCNF0++-RVC", "fcnf0++-rvc"),
    ("FCNF0++ (aligned)", "fcnf0++-aligned"),
    ("FCNF0++-RVC (aligned)", "fcnf0++-rvc-aligned"),
]

# Methods whose extraction settings are a versioned profile, recorded in model_info.json
# and carried into the exported checkpoint as "f0_extraction".
PROFILE_METHODS = FCN_METHODS + FCNF0PP_METHODS

# HPA-RMVPE: RMVPE's front end and decoder with a hypergraph-attention network, at two
# of its published checkpoints (docs/hpa-rmvpe.md). No profile: it quantizes like rmvpe.
# On speech its pitch runs ~20 ms late against RMVPE and CREPE; the -aligned methods
# frame each window HPA_RMVPE_LAG_FRAMES later to cancel that, the others are upstream's.
HPA_RMVPE_METHODS = (
    "hpa-rmvpe-76000",
    "hpa-rmvpe-76000-aligned",
    "hpa-rmvpe-112000",
    "hpa-rmvpe-112000-aligned",
)
HPA_RMVPE_UI_METHODS = [
    ("HPA-RMVPE (76000)", "hpa-rmvpe-76000"),
    ("HPA-RMVPE (76000, aligned)", "hpa-rmvpe-76000-aligned"),
    ("HPA-RMVPE (112000)", "hpa-rmvpe-112000"),
    ("HPA-RMVPE (112000, aligned)", "hpa-rmvpe-112000-aligned"),
]
HPA_RMVPE_LAG_FRAMES = 2

# A method's pitch with every unvoiced frame filled from its voiced neighbours
# (fill_unvoiced_gaps), so the model never sees F0 = 0 mid-speech. In realtime listening
# tests this was clearer than the unfilled method on every model tried, including one
# trained on unfilled F0 (docs/f0-benchmarks.md, section 5). Maps method -> base method.
GAP_FILLED_METHODS = {
    "rmvpe-filled": "rmvpe",
    "hpa-rmvpe-76000-aligned-filled": "hpa-rmvpe-76000-aligned",
    "hpa-rmvpe-112000-aligned-filled": "hpa-rmvpe-112000-aligned",
}
GAP_FILLED_UI_METHODS = [
    ("RMVPE (gaps filled)", "rmvpe-filled"),
    ("HPA-RMVPE (76000, aligned, gaps filled)", "hpa-rmvpe-76000-aligned-filled"),
    ("HPA-RMVPE (112000, aligned, gaps filled)", "hpa-rmvpe-112000-aligned-filled"),
]


def gap_filled_base(method):
    """The method a gap-filled method runs before filling, else the method itself."""
    return GAP_FILLED_METHODS.get(method, method)


def fcn_variant(method):
    """(architecture id, is_rvc) of an FCN method, e.g. ("fcn-929", True)."""
    if method not in FCN_METHODS:
        raise ValueError(f"Not an FCN method: {method!r}")
    return method.removesuffix("-rvc"), method.endswith("-rvc")


def hpa_rmvpe_variant(method):
    """(checkpoint name, aligned) of an HPA-RMVPE method, e.g. ("76000", True)."""
    if method not in HPA_RMVPE_METHODS:
        raise ValueError(f"Not an HPA-RMVPE method: {method!r}")
    name = method[len("hpa-rmvpe-"):]
    aligned = name.endswith("-aligned")
    return name.removesuffix("-aligned"), aligned
