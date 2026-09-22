"""Original 486-state categorical Viterbi objective, for evaluation only.

No UV state. Lowest-index ties are deterministic here; equally optimal paths
can differ from hmmlearn's floating-point tie breaking. Compare log probability
as well as pitch paths when observations are degenerate.
"""

import numpy as np


def parameters():
    bins = np.arange(486)
    transition = np.maximum(12 - np.abs(bins[:, None] - bins[None, :]), 0).astype(
        np.float64
    )
    transition /= transition.sum(1, keepdims=True)
    emission = np.eye(486) * 0.1 + np.full((486, 486), 0.9 / 486)
    return np.full(486, 1 / 486), transition, emission


def viterbi(activation):
    activation = np.asarray(activation)
    if (
        activation.ndim != 2
        or activation.shape[1] != 486
        or not np.isfinite(activation).all()
    ):
        raise ValueError("Expected finite [native frames, 486] activation")
    if not len(activation):
        return np.empty(0, np.int64), np.empty(0)
    prior, transition, emission = parameters()
    observation = activation.argmax(1)
    with np.errstate(divide="ignore"):
        log_transition = np.log(transition)
    log_emission = np.log(emission)
    score = np.log(prior) + log_emission[:, observation[0]]
    back = np.zeros((len(activation), 486), dtype=np.int64)
    for t in range(1, len(activation)):
        candidates = score[:, None] + log_transition
        back[t] = candidates.argmax(0)
        score = candidates[back[t], np.arange(486)] + log_emission[:, observation[t]]
    path = np.empty(len(activation), np.int64)
    path[-1] = score.argmax()
    for t in range(len(path) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    mapping = np.linspace(1200 * np.log2(3), 1200 * np.log2(100), 486)
    cents = []
    for row, center in zip(activation, path):
        start, stop = max(0, center - 4), min(486, center + 5)
        weights = row[start:stop]
        cents.append(
            np.sum(weights * mapping[start:stop]) / weights.sum()
            if weights.sum()
            else np.nan
        )
    return path, np.asarray(cents)
