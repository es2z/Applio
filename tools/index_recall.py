"""How many of the truly nearest index vectors the stored FAISS search actually finds.

    env\\python.exe -X utf8 tools\\index_recall.py <model.index> <queries> [--frames 2000]
        [--top-k 8] [--nprobe 1 4 16 64]

<queries> is a folder of extracted feature files (.npy, frames x dim) - for example
`logs/<a model of your own voice>/extracted`, made with the same embedder as the index.
Realtime looks up the *source* speaker's features in the *target* speaker's index, so
use features of the voice you convert from, not the index's own training features
(those sit on the index vectors themselves and every search finds them).

Applio writes `IVF{N // 39},Flat` with nprobe 1: a query is compared with the ~39
vectors of one cell only. This prints, per nprobe, how often the nearest vector and the
top-k set are the true ones, how much further away the returned nearest vector is, how
far the blended feature (what the model is given at index rate 1) is from the exact
search's, and the CPU time per 100 frames. The exact row is what the Realtime tab's
"Index search: exact" option uses (rvc/realtime/index_search.py).

Nothing is written; the index is only read.
"""

import argparse
import glob
import os
import sys
import time

import faiss
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rvc.realtime.index_search import ExactIndex  # noqa: E402


def load_queries(folder, frames, dim, seed):
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        raise SystemExit(f"No .npy feature files in {folder}")
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    rows, total = [], 0
    for path in files:
        feature = np.load(path)
        if feature.ndim != 2 or feature.shape[1] != dim:
            raise SystemExit(
                f"{path} is {feature.shape}, but the index holds {dim} wide vectors. "
                "Use features made with the embedder the index was built with."
            )
        rows.append(feature.astype(np.float32))
        total += len(feature)
        if total >= frames:
            break
    return np.concatenate(rows)[:frames]


def blend(vectors, score, ix):
    weight = np.square(1 / np.maximum(score, 1e-12))
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(vectors[ix] * np.expand_dims(weight, axis=2), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("index")
    parser.add_argument("queries")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    index = faiss.read_index(args.index)
    vectors = index.reconstruct_n(0, index.ntotal)
    queries = load_queries(args.queries, args.frames, index.d, args.seed)
    ivf = faiss.extract_index_ivf(index)
    print(
        f"{index.ntotal} vectors x {index.d}, {ivf.nlist} cells "
        f"(~{index.ntotal / ivf.nlist:.0f} vectors each), stored nprobe {ivf.nprobe}; "
        f"{len(queries)} query frames, top {args.top_k}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exact = ExactIndex(vectors, device, args.top_k)
    query_tensor = torch.from_numpy(queries).to(device)
    exact.search(query_tensor[:8])  # first-call set-up is not the search's cost
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    exact_distance, exact_ix = exact.search(query_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    exact_ms = (time.perf_counter() - start) * 1e3 / len(queries) * 100
    exact_distance = exact_distance.cpu().numpy()
    exact_ix = exact_ix.cpu().numpy()
    exact_blend = blend(vectors, exact_distance, exact_ix)
    scale = np.linalg.norm(queries - exact_blend, axis=1).mean()

    print(
        f"{'search':>12} {'nearest':>8} {'top-k':>6} {'dist x':>7} "
        f"{'blend diff':>10} {'ms/100 frames':>14}"
    )
    stored = ivf.nprobe
    for nprobe in args.nprobe:
        ivf.nprobe = min(nprobe, ivf.nlist)
        start = time.perf_counter()
        score, ix = index.search(queries, args.top_k)
        ms = (time.perf_counter() - start) * 1e3 / len(queries) * 100
        nearest = np.mean(ix[:, 0] == exact_ix[:, 0])
        top = np.mean(
            [len(set(a) & set(b)) / args.top_k for a, b in zip(ix, exact_ix)]
        )
        ratio = np.mean(np.sqrt(score[:, 0] / np.maximum(exact_distance[:, 0], 1e-12)))
        # Relative to how far the exact blend moves a frame in the first place.
        diff = np.linalg.norm(blend(vectors, score, ix) - exact_blend, axis=1).mean()
        print(
            f"{'nprobe ' + str(ivf.nprobe):>12} {nearest:8.1%} {top:6.1%} "
            f"{ratio:7.3f} {diff / scale:10.1%} {ms:14.2f}"
        )
    ivf.nprobe = stored
    print(
        f"{'exact/' + device:>12} {1:8.1%} {1:6.1%} {1:7.3f} {0:10.1%} {exact_ms:14.2f}"
    )
    print(
        "nearest / top-k: share of frames (of neighbours) that are the true ones. "
        "dist x: distance of the returned nearest vector over the true nearest's. "
        "blend diff: distance between the blended feature and the exact search's, "
        "relative to the distance from the query to the exact blend."
    )


if __name__ == "__main__":
    main()
