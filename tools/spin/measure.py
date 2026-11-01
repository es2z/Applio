"""Does an embedder keep the content and drop the speaker? Measured on the ITA corpus.

    env314\\Scripts\\python.exe -X utf8 tools\\spin\\measure.py --out logs\\_spin\\measure.json \\
        kushinada-hubert-large@22 logs\\_spin\\run1\\ckpt_1000.pt ...

The ITA corpus has every speaker read the same sentences, which allows three numbers
without any labels:
- same / diff: the DTW-aligned cosine distance between two speakers reading the same
  sentence, against two speakers reading different sentences. Speaker information inflates
  the first; content is what keeps the second apart. same/diff falling means the speaker
  matters less relative to the content.
- spk_nn: how often an utterance's nearest neighbour (cosine of time-averaged features)
  is the same speaker. Chance is 1/speakers. Lower means less speaker identity.
- norm: the mean per-frame L2 norm, which RVC is sensitive to.

Each embedder is either a name from rvc.lib.utils.EMBEDDERS with an optional @layer, run
through load_embedding exactly as extraction does, or a SPIN checkpoint (ckpt_*.pt).
"""

import argparse
import itertools
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rvc.lib.utils import embedder_forward, load_audio, load_embedding  # noqa: E402
from tools.spin.export import load_spin_checkpoint  # noqa: E402

DATASET = os.path.join(REPO, "temp4")
# temp4 numbers ITA F1..F25 as 9..33 and M1..M25 as 34..58.
SPEAKERS = [9, 10, 11, 12, 34, 35, 36, 37]
SENTENCES = list(range(1, 100, 5))


def load_model(spec, device):
    if spec.endswith(".pt"):
        model, _ = load_spin_checkpoint(spec)
        model.eval()
    else:
        name, _, layer = spec.partition("@")
        model = load_embedding(name, None, int(layer) if layer else None)
    return model.to(device)


def features(model, paths, device):
    out = []
    with torch.no_grad():
        for path in paths:
            wav = torch.from_numpy(load_audio(path, 16000)).float().unsqueeze(0).to(device)
            out.append(embedder_forward(model, wav)[0].float().cpu().numpy())
    return out


def dtw_cost(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    cost = 1.0 - a @ b.T
    acc, path = librosa.sequence.dtw(C=cost)
    return float(acc[-1, -1] / len(path))


def measure(feats, keys):
    same, diff = [], []
    rng = np.random.default_rng(0)
    by_key = dict(zip(keys, feats))
    for sentence in SENTENCES:
        for s1, s2 in itertools.combinations(SPEAKERS, 2):
            same.append(dtw_cost(by_key[(s1, sentence)], by_key[(s2, sentence)]))
    for _ in range(len(same)):
        s1, s2 = rng.choice(SPEAKERS, 2, replace=False)
        n1, n2 = rng.choice(SENTENCES, 2, replace=False)
        diff.append(dtw_cost(by_key[(int(s1), int(n1))], by_key[(int(s2), int(n2))]))
    means = np.stack([f.mean(0) for f in feats])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    sim = means @ means.T
    np.fill_diagonal(sim, -np.inf)
    nearest = sim.argmax(1)
    spk_nn = float(np.mean([keys[i][0] == keys[j][0] for i, j in enumerate(nearest)]))
    norm = float(np.mean([np.linalg.norm(f, axis=1).mean() for f in feats]))
    return {"same": float(np.mean(same)), "diff": float(np.mean(diff)),
            "same_over_diff": float(np.mean(same) / np.mean(diff)), "spk_nn": spk_nn,
            "norm": norm, "pairs": len(same), "utterances": len(feats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("embedders", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = [(s, n) for s in SPEAKERS for n in SENTENCES]
    paths = [os.path.join(DATASET, str(s), f"ITA#{n}.wav") for s, n in keys]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        sys.exit(f"missing ITA files: {missing[:3]}")
    results = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
    for spec in args.embedders:
        model = load_model(spec, device)
        results[spec] = measure(features(model, paths, device), keys)
        print(spec, json.dumps(results[spec]), flush=True)
        del model
        torch.cuda.empty_cache()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
