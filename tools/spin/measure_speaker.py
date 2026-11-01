"""How recognisable is the user's voice in an embedder's features? Nearest speaker centroid.

    env314\\Scripts\\python.exe -X utf8 tools\\spin\\measure_speaker.py \\
        --out logs\\_spin\\measure_speaker.json kushinada-hubert-large@22 logs\\_spin\\run1\\ckpt_5000.pt

Every clip is cut into 3 s windows - about the longest slice RVC's training extraction
feeds the embedder - and a window is its time-averaged, L2-normalised feature. A speaker's
centroid is the normalised mean of up to 40 windows from 20 of its temp4 files.
- user_top1: how often a window of the user's held-out recordings (my_voice/heldout, never
  trained on by RVC or SPIN) has the user (temp4 speaker 59) as its nearest centroid.
  Chance is 1/60. Lower means less of the user's voice survives in the features.
- user_rank: the user's mean rank among the 60 centroids (1 = nearest).
- user_margin: cosine to the user's centroid minus the nearest other speaker's; positive
  means still recognisable.
- others_top1: the same top-1 for every other speaker, on 5 files per speaker that are
  not in its centroid, for the general level of speaker information.

Embedder specs are those of measure.py.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rvc.lib.utils import embedder_forward, load_audio  # noqa: E402
from tools.spin.measure import load_model  # noqa: E402

DATASET = os.path.join(REPO, "temp4")
USER = "59"
HELDOUT = os.path.join(REPO, "my_voice", "heldout")
SR = 16000
WINDOW = 3 * SR
CENTROID_FILES, CENTROID_WINDOWS, QUERY_FILES = 20, 40, 5


def windows(path):
    wav = load_audio(path, SR).astype(np.float32)
    out = [wav[i:i + WINDOW] for i in range(0, len(wav), WINDOW)]
    # Keep windows of at least 1.5 s that are not near silence.
    return [w for w in out if len(w) >= WINDOW // 2 and np.sqrt(np.mean(w ** 2)) > 10 ** (-45 / 20)]


def clip_sets():
    centroid, queries = {}, {}
    for speaker in sorted(os.listdir(DATASET), key=int):
        names = sorted(n for n in os.listdir(os.path.join(DATASET, speaker)) if n.endswith(".wav"))
        picks = np.linspace(0, len(names) - 1, CENTROID_FILES + QUERY_FILES).round().astype(int)
        picks = list(dict.fromkeys(picks.tolist()))
        paths = [os.path.join(DATASET, speaker, names[i]) for i in picks]
        centroid[speaker] = paths[:CENTROID_FILES]
        queries[speaker] = paths[CENTROID_FILES:] if speaker != USER else \
            sorted(os.path.join(HELDOUT, n) for n in os.listdir(HELDOUT) if n.endswith(".wav"))
    return centroid, queries


def embed(model, wavs, device):
    out = []
    with torch.no_grad():
        for w in wavs:
            feats = embedder_forward(model, torch.from_numpy(w).unsqueeze(0).to(device))[0]
            v = feats.float().mean(0).cpu().numpy()
            out.append(v / np.linalg.norm(v))
    return np.stack(out)


def measure(model, device, centroid_paths, query_paths):
    speakers = list(centroid_paths)
    cents = []
    for s in speakers:
        wins = [w for p in centroid_paths[s] for w in windows(p)][:CENTROID_WINDOWS]
        c = embed(model, wins, device).mean(0)
        cents.append(c / np.linalg.norm(c))
    cents = np.stack(cents)
    user = speakers.index(USER)
    result, other_hits, other_n = {}, 0, 0
    for s in speakers:
        wins = [w for p in query_paths[s] for w in windows(p)]
        if not wins:
            continue
        sims = embed(model, wins, device) @ cents.T
        me = speakers.index(s)
        top = sims.argmax(1)
        if s == USER:
            ranks = (sims > sims[:, [user]]).sum(1) + 1
            others = np.delete(sims, user, axis=1).max(1)
            result.update(user_top1=float(np.mean(top == user)), user_rank=float(ranks.mean()),
                          user_margin=float(np.mean(sims[:, user] - others)),
                          user_windows=len(wins))
        else:
            other_hits += int(np.sum(top == me))
            other_n += len(wins)
    result.update(others_top1=other_hits / other_n, others_windows=other_n,
                  speakers=len(speakers), chance=1 / len(speakers))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("embedders", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.gpu_memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,
                                                   torch.cuda.current_device())
    centroid_paths, query_paths = clip_sets()
    results = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
    for spec in args.embedders:
        model = load_model(spec, device)
        results[spec] = measure(model, device, centroid_paths, query_paths)
        print(spec, json.dumps(results[spec]), flush=True)
        del model
        torch.cuda.empty_cache()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
