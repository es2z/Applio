CREPE_METHOD_TO_MODEL = {
    "crepe": "full",
    "crepe-tiny": "tiny",
    "crepe-small": "small",
    "crepe-medium": "medium",
    "crepe-large": "large",
    "crepe-full": "full",
    "crepe-full-speech": "full_speech",
}

MANGIO_CREPE_METHOD_TO_MODEL = {
    "mangio-crepe": "full",
    "mangio-crepe-tiny": "tiny",
    "mangio-crepe-small": "small",
    "mangio-crepe-medium": "medium",
    "mangio-crepe-large": "large",
    "mangio-crepe-full": "full",
    "mangio-crepe-full-speech": "full_speech",
}

CREPE_UI_METHODS = [
    "crepe-tiny",
    "mangio-crepe-tiny",
    "crepe-small",
    "mangio-crepe-small",
    "crepe-medium",
    "mangio-crepe-medium",
    "crepe-large",
    "mangio-crepe-large",
    "crepe-full",
    "mangio-crepe-full",
    "crepe-full-speech",
    "mangio-crepe-full-speech",
]

CREPE_CLI_METHODS = [*CREPE_UI_METHODS, "crepe", "mangio-crepe"]


def resolve_crepe_model(method: str) -> str:
    """Return the torchcrepe model name for an application F0 method."""
    if method in CREPE_METHOD_TO_MODEL:
        return CREPE_METHOD_TO_MODEL[method]
    if method in MANGIO_CREPE_METHOD_TO_MODEL:
        return MANGIO_CREPE_METHOD_TO_MODEL[method]
    raise ValueError(f"Unknown CREPE method: {method}")
