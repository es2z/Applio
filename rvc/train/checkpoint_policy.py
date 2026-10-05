"""Which checkpoints a run keeps beyond the latest one, and the record of what it kept.

The latest resume checkpoints, G_2333333.pth / D_2333333.pth in the model folder, are
always saved and overwritten at the save interval (Save Every Epoch, which is also the
interval exported RVC models are written at). On top of that a run can keep:

- every: weights-only G/D every `save_every_epoch * gd_every` epochs, for good.
- best:  the `best_export_n` best exported RVC models and the `best_gd_n` best
         weights-only G/D, ranked by an evaluation score (rvc/train/evaluation.py).
- last:  the newest exported RVC model, whenever either of the above is in use.

Why: quality swings from one checkpoint to the next - measured on RingFormer+, a quiet
60e between noisier 50e and 55e by 20-25% - so the checkpoint worth fine-tuning from is
often not the latest, and keeping every resume checkpoint to be sure costs 1.5 GB a save.
A kept G/D is weights only (about a third of that): it is a warm start source for a new
folder or a training reset, never something a run resumes from.

Everything kept lives under logs/<model>/checkpoints/, never next to the resume
checkpoints: latest_checkpoint_path() and reset_run.py's latest() both take the root
G_*.pth with the largest number in its name, so a kept G/D in the root would be resumed
from.

The policy is stored as config.json["checkpoint_policy"] by core.py and read by
train.py, the way learning_rate is, so train.py's positional arguments do not grow and a
folder without the key trains exactly as before.
"""

import json
import os
import shutil
import time

POLICY_KEY = "checkpoint_policy"
CHECKPOINTS_DIR = "checkpoints"
STATE_FILE = "state.json"
CRITERIA = ("mel", "noise", "combined")

DEFAULT_POLICY = {
    # Keep weights-only G/D every save_every_epoch * N epochs. 0 = off.
    "gd_every": 0,
    # Keep the N best exported RVC models. 0 = off.
    "best_export_n": 0,
    # Keep the N best weights-only G/D. 0 = off.
    "best_gd_n": 0,
    # Evaluate every epoch (1) or at every save (0).
    "eval_every": 0,
    # What "best" means; see score_of().
    "best_criterion": "combined",
    # assets/eval_sets/<name> to rank checkpoints with; required by best.
    "eval_set": "",
}

_INT_KEYS = ("gd_every", "best_export_n", "best_gd_n")


def normalize_policy(raw):
    """A complete, validated policy from whatever config.json or a caller holds.

    Missing keys take their defaults and unknown ones are ignored; values train.py could
    not use raise ValueError.
    """
    policy = dict(DEFAULT_POLICY)
    for key, value in dict(raw or {}).items():
        if key in DEFAULT_POLICY and value is not None:
            policy[key] = value
    for key in _INT_KEYS:
        value = policy[key]
        if isinstance(value, bool) or int(value) != float(value) or int(value) < 0:
            raise ValueError(f"{key} must be a whole number of 0 or more, got {value!r}.")
        policy[key] = int(value)
    if policy["eval_every"] not in (0, 1):
        raise ValueError(f"eval_every must be 0 (at every save) or 1 (every epoch), got {policy['eval_every']!r}.")
    policy["eval_every"] = int(policy["eval_every"])
    policy["eval_set"] = str(policy["eval_set"] or "").strip()
    if policy["best_criterion"] not in CRITERIA:
        raise ValueError(
            f"best_criterion must be one of {', '.join(CRITERIA)}, got "
            f"{policy['best_criterion']!r}."
        )
    if best_is_active(policy) and not policy["eval_set"]:
        raise ValueError("Keeping the best checkpoints needs an evaluation set to rank them.")
    return policy


def best_is_active(policy):
    return bool(policy["best_export_n"] or policy["best_gd_n"])


def policy_is_active(policy):
    """Whether this policy asks for anything beyond the latest resume checkpoints."""
    return bool(policy["gd_every"]) or best_is_active(policy)


def policy_from_ui(
    gd_keep,
    gd_n,
    best_enabled,
    best_keep_export,
    best_keep_gd,
    best_n,
    best_split,
    best_export_n,
    best_gd_n,
    eval_timing,
    best_criterion,
    eval_set,
):
    """The policy the Training tab's save settings describe.

    Every value a hidden control holds is ignored: an unticked "keep G/D" means no G/D,
    whatever its slider says, and so on down the best section.
    """
    keep_best = bool(best_enabled)
    split = bool(best_split)
    shared = int(best_n or 0)
    best_exports = int(best_export_n or 0) if split else shared
    best_gds = int(best_gd_n or 0) if split else shared
    return {
        "gd_every": int(gd_n_or_zero(gd_keep, gd_n)),
        "best_export_n": best_exports if keep_best and best_keep_export else 0,
        "best_gd_n": best_gds if keep_best and best_keep_gd else 0,
        "eval_every": 1 if eval_timing == "epoch" else 0,
        "best_criterion": best_criterion or DEFAULT_POLICY["best_criterion"],
        "eval_set": (eval_set or "") if keep_best and (best_keep_export or best_keep_gd) else "",
    }


def gd_n_or_zero(gd_keep, gd_n):
    return max(1, int(gd_n or 1)) if gd_keep else 0


def ui_from_policy(policy, default_best_n=3):
    """What the Training tab's save settings should show for a policy (policy_from_ui
    inverted). Counts that are off show a sensible default, so ticking a box offers one."""
    export_n, gd_n = policy["best_export_n"], policy["best_gd_n"]
    split = bool(export_n and gd_n and export_n != gd_n)
    return {
        "gd_keep": policy["gd_every"] > 0,
        "gd_n": policy["gd_every"] or 1,
        "best_enabled": best_is_active(policy),
        "best_keep_export": export_n > 0,
        "best_keep_gd": gd_n > 0,
        "best_n": max(export_n, gd_n) or default_best_n,
        "best_split": split,
        "best_export_n": export_n or default_best_n,
        "best_gd_n": gd_n or default_best_n,
        "eval_timing": "epoch" if policy["eval_every"] == 1 else "save",
        "best_criterion": policy["best_criterion"],
        "eval_set": policy["eval_set"],
    }


def read_policy(model_path):
    """The policy recorded in a run's config.json, or the defaults."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return dict(DEFAULT_POLICY)
    with open(config_path, "r", encoding="utf-8") as f:
        return normalize_policy(json.load(f).get(POLICY_KEY))


def apply_checkpoint_policy(model_path, **values):
    """Write the policy into a run's config.json, leaving everything else be.

    Values left at None are not changed. A config that has never had a policy is left
    untouched when every value given is the default, so training with nothing ticked
    changes nothing, the same rule as the Initial Generator LR Boost.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    given = {key: value for key, value in values.items() if key in DEFAULT_POLICY and value is not None}
    current = config.get(POLICY_KEY)
    merged = normalize_policy({**(current or {}), **given})
    if (current is None and merged == DEFAULT_POLICY) or current == merged:
        return merged
    config[POLICY_KEY] = merged
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Set checkpoint policy in {config_path}: {merged}")
    return merged


def wants_eval(policy, epoch, save_every_epoch):
    if not best_is_active(policy):
        return False
    interval = 1 if policy["eval_every"] == 1 else int(save_every_epoch)
    return epoch % interval == 0


def wants_gd_every(policy, epoch, save_every_epoch):
    every = int(save_every_epoch) * policy["gd_every"]
    return every > 0 and epoch % every == 0


def score_of(scores, criterion, baseline):
    """One number per evaluation, lower is better; None when it cannot be computed.

    - mel:      held-out (B) mel L1, or the trained clips' (A) when there is no B.
    - noise:    mean of the B and C noise ratios (generated / reference HF flatness in
                voiced frames), or A's when neither exists.
    - combined: mel and noise, each divided by this run's first evaluation, summed - so
                both count equally whatever their units.
    """

    def pick(*keys):
        return [scores[key] for key in keys if scores.get(key) is not None]

    mel = pick("B_mel") or pick("A_mel")
    noise = pick("B_noise", "C_noise") or pick("A_noise")
    mel = mel[0] if mel else None
    noise = sum(noise) / len(noise) if noise else None
    if criterion == "mel":
        return mel
    if criterion == "noise":
        return noise
    if mel is None or noise is None or not baseline:
        return None
    base_mel, base_noise = baseline.get("mel"), baseline.get("noise")
    if not base_mel or not base_noise:
        return None
    return mel / base_mel + noise / base_noise


def baseline_of(scores):
    """The values a combined score is relative to, from one evaluation."""
    mel = score_of(scores, "mel", None)
    noise = score_of(scores, "noise", None)
    if mel is None or noise is None:
        return None
    return {"mel": mel, "noise": noise}


def rank_best(best, candidate, n):
    """Insert candidate into the ranking, keeping the n lowest scores.

    Returns (new ranking best first, entries that left it). An existing entry for the
    same epoch is replaced (a resume re-running an epoch). Ties keep the earlier epoch.
    """
    others = [entry for entry in best if entry["epoch"] != candidate["epoch"]]
    replaced = [entry for entry in best if entry["epoch"] == candidate["epoch"]]
    ranked = sorted(others + [candidate], key=lambda entry: (entry["score"], entry["epoch"]))
    return ranked[:n], replaced + [entry for entry in ranked[n:] if entry is not candidate]


def _remove(paths):
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class CheckpointKeeper:
    """Carries out a policy for one run and records it in checkpoints/state.json.

    The file writing itself is passed in (export(path), save_gd(g_path, d_path)), so the
    keeping logic can be tested without a model.
    """

    def __init__(self, experiment_dir, model_name, policy):
        self.policy = policy
        self.model_name = model_name
        self.root = os.path.join(experiment_dir, CHECKPOINTS_DIR)
        self.state_path = os.path.join(self.root, STATE_FILE)
        self.state = self._load_state()

    # state

    def _empty_state(self):
        return {
            "version": 2,
            "policy": self.policy,
            "evaluations": [],
            "baseline": None,
            "best": [],
            "every": [],
            "last": None,
            "auto_trained": [],
        }

    def _load_state(self):
        if not os.path.isfile(self.state_path):
            return self._empty_state()
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict) or state.get("version") != 2:
                raise ValueError("unknown state layout")
        except (OSError, ValueError) as error:
            # Keep the unreadable file for inspection and start a fresh record rather
            # than stopping a run over bookkeeping.
            aside = f"{self.state_path}.unreadable-{time.strftime('%Y%m%d%H%M%S')}"
            os.replace(self.state_path, aside)
            print(f"Could not read {self.state_path} ({error}); moved it to {aside}.")
            return self._empty_state()
        fresh = self._empty_state()
        fresh.update(state)
        # Files deleted by hand no longer count; an entry left with nothing is dropped.
        for entry in fresh["best"]:
            for key in ("export", "G", "D"):
                if entry.get(key) and not os.path.isfile(entry[key]):
                    entry.pop(key)
        fresh["best"] = [e for e in fresh["best"] if e.get("export") or e.get("G")]
        fresh["policy"] = self.policy
        return fresh

    def save_state(self):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_path)

    # paths

    def _dir(self, kind):
        path = os.path.join(self.root, kind)
        os.makedirs(path, exist_ok=True)
        return path

    def export_path(self, kind, epoch, step):
        return os.path.join(self._dir(kind), f"{self.model_name}_{epoch}e_{step}s.pth")

    def gd_paths(self, kind, epoch):
        directory = self._dir(kind)
        return os.path.join(directory, f"G_{epoch}e.pth"), os.path.join(directory, f"D_{epoch}e.pth")

    # actions

    def save_last(self, epoch, step, export):
        """Replace the newest exported RVC model."""
        path = self.export_path("last", epoch, step)
        previous = self.state.get("last")
        export(path)
        if previous and previous.get("export") != path:
            _remove([previous["export"]])
        self.state["last"] = {"epoch": epoch, "step": step, "export": path}
        self.save_state()
        return path

    def save_gd_every(self, epoch, step, save_gd):
        """Keep weights-only G/D for good."""
        entry = {"epoch": epoch, "step": step}
        entry["G"], entry["D"] = self.gd_paths("every", epoch)
        save_gd(entry["G"], entry["D"])
        self.state["every"] = [e for e in self.state["every"] if e["epoch"] != epoch] + [entry]
        self.save_state()
        return entry

    def record_evaluation(self, epoch, step, scores, export, save_gd):
        """Record an evaluation and keep what the best counts ask for.

        Returns (score, rank) - rank is 1-based in the ranking, None when this epoch did
        not make it or nothing is being kept.
        """
        if self.state["baseline"] is None:
            self.state["baseline"] = baseline_of(scores)
        score = score_of(scores, self.policy["best_criterion"], self.state["baseline"])
        self.state["evaluations"] = [
            e for e in self.state["evaluations"] if e["epoch"] != epoch
        ] + [{"epoch": epoch, "step": step, "scores": scores, "score": score}]
        rank = None
        export_n, gd_n = self.policy["best_export_n"], self.policy["best_gd_n"]
        if best_is_active(self.policy) and score is not None:
            candidate = {"epoch": epoch, "step": step, "score": score}
            ranking, left = rank_best(self.state["best"], candidate, max(export_n, gd_n))
            _remove([entry.get(key) for entry in left for key in ("export", "G", "D")])
            # Entries only ever move down the ranking, so files are only ever deleted -
            # never something that would have to be recreated from older weights.
            for position, entry in enumerate(ranking, start=1):
                if entry is candidate:
                    rank = position
                    if position <= export_n:
                        entry["export"] = self.export_path("best", epoch, step)
                        export(entry["export"])
                    if position <= gd_n:
                        entry["G"], entry["D"] = self.gd_paths("best", epoch)
                        save_gd(entry["G"], entry["D"])
                    continue
                if position > export_n and entry.get("export"):
                    _remove([entry.pop("export")])
                if position > gd_n and entry.get("G"):
                    _remove([entry.pop("G"), entry.pop("D", None)])
            self.state["best"] = ranking
        self.save_state()
        return score, rank


def remove_checkpoints(experiment_dir):
    """What "Fresh Training" does to this module's files."""
    path = os.path.join(experiment_dir, CHECKPOINTS_DIR)
    if os.path.isdir(path):
        shutil.rmtree(path)
