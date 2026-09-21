import os


def _pretrained_paths(vocoder, sample_rate):
    base_path = os.path.join("rvc", "models", "pretraineds", f"{vocoder.lower()}")
    return (
        os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth"),
        os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth"),
    )


def pretrained_selector(vocoder, sample_rate):
    path_g, path_d = _pretrained_paths(vocoder, sample_rate)
    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d

    # There are no stock RefineGAN pretrains above 32k. The HiFi-GAN pretrain at the same
    # sample rate is the next best start: rvc/train/warm_start.py inherits its encoders,
    # flow, speaker embedding, residual blocks and matching sub-discriminators.
    if vocoder != "HiFi-GAN":
        fallback_g, fallback_d = _pretrained_paths("HiFi-GAN", sample_rate)
        if os.path.exists(fallback_g) and os.path.exists(fallback_d):
            print(
                f"No {vocoder} pretrained model for {sample_rate} Hz ({path_g}); warm "
                "starting from the HiFi-GAN one instead."
            )
            return fallback_g, fallback_d

    print(
        f"No pretrained model found for {vocoder} at {sample_rate} Hz ({path_g}); "
        "training from scratch."
    )
    return "", ""
