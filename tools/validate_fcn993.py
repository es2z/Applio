"""Layer parity with an isolated TensorFlow oracle, without adding TF to runtime.

Run `oracle weights.h5 inputs.npy oracle.npz` in the TF environment, then
`compare fcn-993.pt inputs.npy oracle.npz` in the application environment.
The oracle preserves upstream Conv2D layout/order; only the obsolete optimizer
and training-only reshape/flatten are omitted.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def oracle(weights, inputs, output, precision="float32"):
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import h5py
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    x = tf.keras.Input(shape=(None, 1, 1), dtype=precision)
    y = x
    layers, outputs = [], {}
    for i, channels in enumerate((256, 32, 32, 128, 256, 512), 1):
        conv = tf.keras.layers.Conv2D(
            channels, (32, 1), activation="relu", name=f"conv{i}", dtype=precision
        )
        y = conv(y)
        layers.append(conv)
        outputs[f"conv{i}"] = y
        if i < 4:
            y = tf.keras.layers.MaxPool2D((2, 1), dtype=precision)(y)
            outputs[f"pool{i}"] = y
        bn = tf.keras.layers.BatchNormalization(
            epsilon=0.001, momentum=0.99, name=f"conv{i}-BN", dtype=precision
        )
        y = bn(y, training=False)
        layers.append(bn)
        outputs[f"bn{i}"] = y
    classifier = tf.keras.layers.Conv2D(
        486, (4, 1), activation=None, name="classifier", dtype=precision
    )
    y = classifier(y)
    layers.append(classifier)
    outputs["classifier"] = y
    outputs["sigmoid"] = tf.keras.layers.Activation("sigmoid", dtype=precision)(y)
    model = tf.keras.Model(x, outputs)
    with h5py.File(weights) as f:
        for layer in layers:
            names = (
                ("gamma", "beta", "moving_mean", "moving_variance")
                if layer.name.endswith("-BN")
                else ("kernel", "bias")
            )
            layer.set_weights(
                [f[f"{layer.name}/{layer.name}/{name}:0"][()] for name in names]
            )
    values = model(np.load(inputs)[:, :, None, None], training=False)
    np.savez(
        output, **{key: value.numpy()[:, :, 0, :] for key, value in values.items()}
    )


def compare(
    weights,
    inputs,
    reference,
    device="cpu",
    report=None,
    precision="float32",
    output_gate=False,
):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rvc.lib.predictors.fcn.decoder import FCNDecoder
    from rvc.lib.predictors.fcn.model import FCNModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    dtype = torch.float64 if precision == "float64" else torch.float32
    model = FCNModel().to(device=device, dtype=dtype)
    model.load_state_dict(
        torch.load(weights, map_location=device, weights_only=True)["state_dict"],
        strict=True,
    )
    ref = np.load(reference)
    x = torch.from_numpy(np.load(inputs)).to(device=device, dtype=dtype)[:, None, :]
    values = {}
    with torch.inference_mode():
        for i in range(1, 7):
            x = torch.relu(getattr(model, f"conv{i}")(x))
            values[f"conv{i}"] = x
            if i < 4:
                x = torch.nn.functional.max_pool1d(x, 2, 2)
                values[f"pool{i}"] = x
            x = getattr(model, f"bn{i}")(x)
            values[f"bn{i}"] = x
        values["classifier"] = model.classifier(x)
        values["sigmoid"] = torch.sigmoid(values["classifier"])
    failures = []
    metrics = {
        "device": device,
        "torch": torch.__version__,
        "precision": precision,
        "layers": {},
    }
    atol, rtol = (1e-9, 1e-9) if precision == "float64" else (1e-5, 1e-4)
    for key, value in values.items():
        actual = value.transpose(1, 2).cpu().numpy()
        error = np.max(np.abs(actual - ref[key]))
        passed = np.allclose(actual, ref[key], atol=atol, rtol=rtol)
        print(f"{key}: max_abs={error:.8g}, pass={passed}")
        metrics["layers"][key] = {"max_abs": float(error), "pass": bool(passed)}
        if not passed:
            failures.append(key)
            # Isolate accumulated FP32 roundoff from parameter/layout mistakes:
            # evaluate this layer using the oracle's preceding activation.
            if key.startswith("conv"):
                i = int(key[4:])
                prior = np.load(inputs)[:, :, None] if i == 1 else ref[f"bn{i - 1}"]
                prior = (
                    torch.from_numpy(prior)
                    .transpose(1, 2)
                    .to(device=device, dtype=dtype)
                )
                isolated = (
                    torch.relu(getattr(model, key)(prior)).transpose(1, 2).cpu().numpy()
                )
                print(f"  isolated max_abs={np.max(np.abs(isolated - ref[key])):.8g}")
    decoder = FCNDecoder()
    actual_cents, _, _ = decoder(values["sigmoid"].transpose(1, 2))
    ref_cents, _, _ = decoder(torch.from_numpy(ref["sigmoid"]).to(device))
    difference = (actual_cents - ref_cents).abs().max().item()
    print(f"decoded max cents difference: {difference:.8g}")
    metrics["max_cents_difference"] = difference
    metrics["layerwise_passed"] = not failures
    output_passed = metrics["layers"]["sigmoid"]["pass"] and difference <= 0.1
    metrics["output_passed"] = output_passed
    metrics["gate"] = "output" if output_gate else "all_layers"
    metrics["passed"] = output_passed and (output_gate or not failures)
    if report:
        Path(report).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if not metrics["passed"]:
        raise AssertionError(f"Parity failed: {failures}; cents={difference}")


def compare_waveforms(weights, inputs, reference, report=None):
    """Compare the entire CUDA preprocessing/blocking/decoder path to the oracle."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rvc.lib.predictors.fcn import FCNPredictor

    predictor = FCNPredictor("cuda", weight_path=weights)
    ref = np.load(reference)["sigmoid"]
    mapping = np.linspace(1200 * np.log2(3), 1200 * np.log2(100), 486)
    results = []
    for audio, expected in zip(np.load(inputs), ref, strict=True):
        cents, _, confidence, activation = predictor.native(
            audio, return_activation=True
        )
        expected_cents = []
        for frame in expected:
            center = int(np.argmax(frame))
            start, stop = max(0, center - 4), min(486, center + 5)
            expected_cents.append(
                np.sum(frame[start:stop] * mapping[start:stop])
                / np.sum(frame[start:stop])
            )
        cents_error = float(np.max(np.abs(cents - expected_cents)))
        activation_error = float(np.max(np.abs(activation - expected)))
        confidence_error = float(np.max(np.abs(confidence - expected.max(-1))))
        passed = bool(
            np.allclose(activation, expected, atol=1e-5, rtol=1e-4)
            and cents_error <= 0.1
        )
        results.append(
            {
                "max_activation_error": activation_error,
                "max_confidence_error": confidence_error,
                "max_cents_error": cents_error,
                "passed": passed,
            }
        )
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "waveforms": results,
        "passed": all(item["passed"] for item in results),
    }
    if report:
        Path(report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise AssertionError("CUDA waveform parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    generate = subparsers.add_parser(
        "fixtures", help="Generate seven deterministic network inputs"
    )
    generate.add_argument("output")
    generate.add_argument("--samples", type=int, default=1001)
    for mode in ("oracle", "compare", "waveform"):
        command = subparsers.add_parser(mode)
        command.add_argument("weights")
        command.add_argument("inputs")
        command.add_argument("output")
        command.add_argument("--device", default="cpu")
        command.add_argument("--report")
        command.add_argument(
            "--precision", choices=("float32", "float64"), default="float32"
        )
        command.add_argument(
            "--output-gate",
            action="store_true",
            help="Keep hidden-layer discrepancies as diagnostics; require original output tolerances. Use only after float64 layer audit.",
        )
    args = parser.parse_args()
    if args.mode == "fixtures":
        if args.samples < 993:
            parser.error("Network inputs require at least 993 samples")
        t = np.arange(args.samples) / 8000
        rng = np.random.default_rng(993)
        sine = np.sin(2 * np.pi * 220 * t)
        inputs = np.stack(
            [
                sine,
                sine + 0.3 * np.sin(2 * np.pi * 440 * t),
                rng.normal(size=len(t)),
                np.eye(1, len(t), len(t) // 2)[0],
                np.ones(len(t)),
                np.zeros(len(t)),
                sine * 1e-12,
            ]
        ).astype(np.float32)
        np.save(args.output, inputs)
    elif args.mode == "oracle":
        oracle(args.weights, args.inputs, args.output, args.precision)
    elif args.mode == "waveform":
        compare_waveforms(args.weights, args.inputs, args.output, args.report)
    else:
        compare(
            args.weights,
            args.inputs,
            args.output,
            args.device,
            args.report,
            args.precision,
            args.output_gate,
        )
