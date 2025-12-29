import os


# Map vocoder names to folder names
VOCODER_FOLDER_MAP = {
    "hifi-gan": "hifi-gan",
    "mrf hifi-gan": "mrf-hifi-gan",
    "refinegan": "refinegan",
}

# RefineGAN variant configurations (32kHz only)
# Each variant has (G_filename, D_filename)
# All RefineGAN models are 32kHz - sample_rate parameter is ignored for RefineGAN
REFINEGAN_VARIANTS = {
    "RFGv3_CV (ContentVec)": ("RFGv3_CV_G_1771500.pth", "RFGv3_CV_D_1771500.pth"),
    "RFGv3_SPv2 (SPIN v2)": ("RFGv3_SPv2_G_1771500.pth", "RFGv3_SPv2_D_1771500.pth"),
}


def get_refinegan_variants():
    """
    Get list of available RefineGAN variants.

    Returns:
        List of variant names
    """
    return list(REFINEGAN_VARIANTS.keys())


def pretrained_selector(vocoder, sample_rate, refinegan_variant="RFGv3_CV (ContentVec)"):
    """
    Select pretrained models based on vocoder type and sample rate.

    Args:
        vocoder: Vocoder name (e.g., "HiFi-GAN", "MRF HiFi-GAN", "RefineGAN")
        sample_rate: Sample rate as int (e.g., 32000, 40000, 48000)
        refinegan_variant: RefineGAN variant name (only used when vocoder is RefineGAN)

    Returns:
        Tuple of (generator_path, discriminator_path) or ("", "") if not found
    """
    vocoder_lower = vocoder.lower()
    folder_name = VOCODER_FOLDER_MAP.get(vocoder_lower, vocoder_lower.replace(" ", "-"))
    base_path = os.path.join("rvc", "models", "pretraineds", folder_name)

    sr_short = str(sample_rate)[:2]  # "32000" -> "32"

    if vocoder_lower == "refinegan":
        # RefineGAN: use variant-specific filenames (all 32kHz)
        if refinegan_variant in REFINEGAN_VARIANTS:
            g_filename, d_filename = REFINEGAN_VARIANTS[refinegan_variant]
        else:
            # Default to RFGv3_CV if unknown variant
            g_filename, d_filename = REFINEGAN_VARIANTS["RFGv3_CV (ContentVec)"]
        path_g = os.path.join(base_path, g_filename)
        path_d = os.path.join(base_path, d_filename)
    else:
        # Standard naming for HiFi-GAN and MRF HiFi-GAN
        path_g = os.path.join(base_path, f"f0G{sr_short}k.pth")
        path_d = os.path.join(base_path, f"f0D{sr_short}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""
