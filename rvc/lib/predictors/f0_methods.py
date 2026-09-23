"""F0 method registration without importing any inference runtime."""

FCN_METHODS = ("fcn-993", "fcn-993-rvc")
FCN_UI_METHODS = [("FCN-993", "fcn-993"), ("FCN-993-RVC", "fcn-993-rvc")]

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
