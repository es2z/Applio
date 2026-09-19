"""Start a separate training history from an existing pair of weights."""

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from rvc.lib.utils import (
    describe_embedder_mismatch,
    embedder_identity_from_model_info,
)
from rvc.train.warm_start import EMBEDDER_PROJECTION_PREFIX


def _pid_running(pid):
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
            capture_output=True, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # tasklist uses the Windows code page, which can differ from Python's
        # UTF-8 mode. CSV punctuation and numeric PIDs are ASCII in either case;
        # inspect raw bytes without decoding localized names or status messages.
        return f'","{int(pid)}","'.encode("ascii") in result.stdout
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def _target_embedder(root):
    """The embedder the features in this folder were extracted with, or None.

    None means the folder records no embedder at all, which predates any of this being
    tracked and cannot be compared against.
    """
    path = root / "model_info.json"
    if not path.is_file():
        return None
    identity = embedder_identity_from_model_info(
        json.loads(path.read_text(encoding="utf-8"))
    )
    return identity if identity["embedder_model"] else None


def _retarget_embedder(checkpoint, target_embedder, text_enc_hidden_dim):
    """Rebuild enc_p.emb_phone.weight when the source was fitted to other features.

    Re-extracting a folder with a different embedder leaves its G_*.pth behind, and that
    weight is the one tensor in the generator that reads the embedder's feature space.
    A different width says so in the shapes; a different embedder at the same width does
    not, and is exactly as stale. Everything else - the encoder, flow, decoder and
    speaker embedding - means the same thing under any embedder and is kept.

    The bias is an offset in the encoder's own hidden space, which is kept, so it is kept
    with it. Returns what it did, or None when there was nothing to do.
    """
    if not target_embedder:
        return None
    reason = describe_embedder_mismatch(
        checkpoint, target_embedder, "the source checkpoint"
    )
    if reason is None:
        checkpoint.update(target_embedder)
        return None

    weight_key = EMBEDDER_PROJECTION_PREFIX + "weight"
    weight = checkpoint["model"].get(weight_key)
    if weight is None:
        raise ValueError(
            f"{reason}, and it has no {weight_key} to rebuild. "
            "Reset expects a generator checkpoint here."
        )
    if text_enc_hidden_dim is None:
        text_enc_hidden_dim = weight.shape[1]
    fresh = torch.nn.Linear(text_enc_hidden_dim, weight.shape[0]).weight
    checkpoint["model"][weight_key] = fresh.detach().to(weight.dtype).clone()
    checkpoint.update(target_embedder)
    message = (
        f"Reset (G): {reason}; {weight_key} starts from scratch at "
        f"{tuple(checkpoint['model'][weight_key].shape)} "
        f"(was {tuple(weight.shape)}). The rest of the generator is kept."
    )
    print(message)
    return message


def reset_training_run(experiment_dir):
    """Archive local outputs and install epoch-zero checkpoints, with rollback.

    Dataset paths are never followed. The archive is a separate TensorBoard run
    outside the experiment, including when the old cleanup option is used later.
    Source epochs may differ. If G records a discriminator version, D is warm
    started into that layout with fresh optimizer groups before installation.
    If the folder has since been re-extracted with a different embedder, G's
    enc_p.emb_phone.weight is rebuilt and both files are re-stamped, so the run that
    follows is not refused by assert_resumable for weights it no longer carries.
    """
    root = Path(experiment_dir).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for pid in config.get("process_pids", []):
        if _pid_running(pid):
            raise ValueError("Stop the existing training process before resetting.")

    def latest(prefix):
        paths = list(root.glob(f"{prefix}_*.pth"))
        return max(paths, key=lambda p: int("".join(re.findall(r"\d", p.name))), default=None)

    sources = [latest("G"), latest("D")]
    if not all(sources):
        raise ValueError("Reset requires both G_*.pth and D_*.pth in this model folder.")
    if any(p.is_symlink() for p in sources):
        raise ValueError("Reset requires local checkpoints, not symbolic links.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    archive = root.parent / "_training_history" / root.name / run_id
    archive.mkdir(parents=True, exist_ok=False)
    staged = []
    moved = []
    installed = []
    epochs = []
    target_disc_version = None
    discriminator_transfer = None
    target_embedder = _target_embedder(root)
    embedder_transfer = None
    try:
        # Validate and stage both checkpoints before touching any existing output.
        for tag, source in zip(("G", "D"), sources):
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            if not checkpoint.get("model"):
                raise ValueError(f"No model weights in {source}")
            epochs.append(checkpoint["iteration"])
            if tag == "G":
                target_disc_version = checkpoint.get("disc_version")
                embedder_transfer = _retarget_embedder(
                    checkpoint,
                    target_embedder,
                    config.get("model", {}).get("text_enc_hidden_dim"),
                )
            elif target_disc_version is not None:
                # A new run may combine a G and D from different architectures.
                # Match D components by meaning, and create optimizer groups for
                # the target layout (the source may have fewer parameters).
                from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
                from rvc.train.warm_start import warm_start

                discriminator = MultiPeriodDiscriminator(
                    config["model"].get("use_spectral_norm", False),
                    version=target_disc_version,
                )
                transfer = warm_start(discriminator, source, "D")
                discriminator_transfer = transfer.summary(str(source))
                checkpoint["model"] = discriminator.state_dict()
                checkpoint["optimizer"] = torch.optim.AdamW(
                    discriminator.parameters(), lr=config["train"]["learning_rate"]
                ).state_dict()
                checkpoint["disc_version"] = target_disc_version
                del discriminator, transfer
            if tag == "D" and target_embedder:
                # Nothing in the discriminator depends on the embedder, so its stamp is
                # metadata only - but leaving the previous embedder's name on it would
                # misdescribe what the next run trains it on.
                checkpoint.update(target_embedder)
            checkpoint["iteration"] = 0
            checkpoint["scaler"] = {}
            checkpoint["learning_rate"] = config["train"]["learning_rate"]
            optimizer = checkpoint["optimizer"]
            optimizer["state"] = {}
            for group in optimizer["param_groups"]:
                group["lr"] = group["initial_lr"] = config["train"]["learning_rate"]
                group["betas"] = tuple(config["train"]["betas"])
                group["eps"] = config["train"]["eps"]
            stage = archive / f"{tag}_0.staged"
            torch.save(checkpoint, stage)
            staged.append(stage)
            check = torch.load(stage, map_location="cpu", weights_only=True)
            for name, tensor in checkpoint["model"].items():
                if not torch.equal(tensor, check["model"][name]):
                    raise ValueError(f"Weight verification failed: {name}")
            del checkpoint, check
        # Snapshot provenance, including copied filelist references and GUI settings.
        for name in ("config.json", "model_info.json", "filelist.txt"):
            if (root / name).is_file():
                shutil.copy2(root / name, archive / name)
        (archive / "reset.json").write_text(json.dumps({
            "source_checkpoints": [p.name for p in sources],
            "source_epoch": epochs[0] if epochs[0] == epochs[1] else None,
            "source_epochs": dict(zip(("G", "D"), epochs)),
            "discriminator_transfer": discriminator_transfer,
            "embedder_transfer": embedder_transfer,
            "new_epoch": 1,
            "new_global_step": 0,
            "optimizer": "G and D reset",
            "settings": config["train"],
        }, indent=2), encoding="utf-8")
        outputs = [p for p in root.iterdir() if (
            p.suffix == ".pth" or p.name == "training_data.json"
            or p.name == "eval" or p.name.startswith("events.out.tfevents.")
        )]
        for source in outputs:
            if source.is_symlink() or (hasattr(source, "is_junction") and source.is_junction()):
                raise ValueError(f"Cannot archive linked output: {source}")
        for source in outputs:
            destination = archive / source.name
            source.rename(destination)
            moved.append((source, destination))
        for tag, stage in zip(("G", "D"), staged):
            destination = root / f"{tag}_0.pth"
            stage.rename(destination)
            installed.append(destination)
    except BaseException:
        for path in installed:
            path.unlink()
        for original, saved in reversed(moved):
            saved.rename(original)
        raise
    return archive
