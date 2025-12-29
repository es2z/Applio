import os


# Map vocoder names to folder names
VOCODER_FOLDER_MAP = {
    "hifi-gan": "hifi-gan",
    "mrf hifi-gan": "mrf-hifi-gan",
    "refinegan": "refinegan",
}


def pretrained_selector(vocoder, sample_rate):
    """
    Select pretrained models based on vocoder type and sample rate.

    Args:
        vocoder: Vocoder name (e.g., "HiFi-GAN", "MRF HiFi-GAN", "RefineGAN")
        sample_rate: Sample rate as int (e.g., 32000, 40000, 48000)

    Returns:
        Tuple of (generator_path, discriminator_path) or ("", "") if not found
    """
    vocoder_lower = vocoder.lower()
    folder_name = VOCODER_FOLDER_MAP.get(vocoder_lower, vocoder_lower.replace(" ", "-"))
    base_path = os.path.join("rvc", "models", "pretraineds", folder_name)

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""
