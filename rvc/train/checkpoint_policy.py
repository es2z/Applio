"""Which checkpoints a run keeps beyond the last one, and the record of what it kept.

Until this existed a run could only keep the last G/D (save_only_latest) or every one of
them, and exported models were written at the same moments. But quality swings from one
checkpoint to the next - measured on RingFormer+, a quiet 60e between noisier 50e and
55e, by 20-25% - so the checkpoint worth fine-tuning from is often not the last one, and
keeping every G/D to be sure costs 1.5 GB per save.

So a run can now also keep:
- every:  an exported model every `export_every` epochs, and optionally weights-only G/D
          every `export_every * gd_every` epochs.
- best:   the `best_n` best exported models by an evaluation score (rvc/train/
          evaluation.py), optionally with weights-only G/D for each.
- last:   the newest exported model, replaced on every save, alongside the root
          G_2333333.pth / D_2333333.pth that resume and the training reset use.

Everything lives under logs/<model>/checkpoints/, never next to the resume checkpoints:
latest_checkpoint_path() and reset_run.py's latest() both pick the root G_*.pth with the
largest number in its name, so a kept G/D in the root would be resumed from.

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
    # Keep an exported model every N epochs. 0 = off.
    "export_every": 0,
    # Also keep weights-only G/D every export_every * N epochs. 0 = off.
    "gd_every": 0,
    # assets/eval_sets/<name> to score checkpoints with. "" = no evaluation.
    "eval_set": "",
    # Evaluate every N epochs. 0 = at every save (save_every_epoch).
    "eval_every": 0,
    # Keep the N best exported models by score. 0 = off.
    "best_n": 0,
    # What "best" means; see score_of().
    "best_criterion": "combined",
    # Keep weights-only G/D with each best exported model.
    "best_save_gd": True,
}

_INT_KEYS = ("export_every", "gd_every", "eval_every", "best_n")


def normalize_policy(raw):
    """A complete, validated policy from whatever config.json or a caller holds.

    Missing keys take their defaults; values train.py could not use raise ValueError.
    """
    policy = dict(DEFAULT_POLICY)
    for key, value in dict(raw or {}).items():
        if key not in DEFAULT_POLICY or value is None:
            continue
        policy[key] = value
    for key in _INT_KEYS:
        value = policy[key]
        if isinstance(value, bool) or int(value) != float(value) or int(value) < 0:
            raise ValueError(f"{key} must be a whole number of 0 or more, got {value!r}.")
        policy[key] = int(value)
    policy["eval_set"] = str(policy["eval_set"] or "").strip()
    if policy["best_criterion"] not in CRITERIA:
        raise ValueError(
            f"best_criterion must be one of {', '.join(CRITERIA)}, got "
            f"{policy['best_criterion']!r}."
        )
    policy["best_save_gd"] = bool(policy["best_save_gd"])
    if policy["best_n"] and not policy["eval_set"]:
        raise ValueError("Keeping the best checkpoints needs an evaluation set to score them.")
    return policy


def policy_is_active(policy):
    """Whether this policy asks for anything beyond what train.py always did."""
    return bool(policy["export_every"] or policy["best_n"] or policy["eval_set"])


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
    untouched when every value given is the default, so training with the section
    untouched changes nothing, the same rule as the Initial Generator LR Boost.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    given = {key: value for key, value in values.items() if key in DEFAULT_POLICY and value is not None}
    current = config.get(POLICY_KEY)
    merged = normalize_policy({**(current or {}), **given})
    if current is None and merged == DEFAULT_POLICY:
        return merged
    if current == merged:
        return merged
    config[POLICY_KEY] = merged
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Set checkpoint policy in {config_path}: {merged}")
    return merged


def eval_interval(policy, save_every_epoch):
    return policy["eval_every"] or int(save_every_epoch)


def wants_eval(policy, epoch, save_every_epoch):
    return bool(policy["eval_set"]) and epoch % eval_interval(policy, save_every_epoch) == 0


def wants_export_every(policy, epoch):
    return policy["export_every"] > 0 and epoch % policy["export_every"] == 0


def wants_gd_every(policy, epoch):
    every = policy["export_every"] * policy["gd_every"]
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
        values = [scores.get(key) for key in keys]
        values = [value for value in values if value is not None]
        return values

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


def update_best(best, candidate, n):
    """Insert candidate into the best list, keeping the n lowest scores.

    Returns (new list sorted best first, entries that dropped out, whether candidate
    made it). An existing entry for the same epoch is replaced (a resume re-running an
    epoch). Ties keep the earlier epoch.
    """
    others = [entry for entry in best if entry["epoch"] != candidate["epoch"]]
    replaced = [entry for entry in best if entry["epoch"] == candidate["epoch"]]
    ranked = sorted(others + [candidate], key=lambda entry: (entry["score"], entry["epoch"]))
    kept, evicted = ranked[:n], ranked[n:]
    accepted = candidate in kept
    return kept, replaced + [entry for entry in evicted if entry is not candidate], accepted


def _remove(paths):
    for path in paths:
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
            "version": 1,
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
            if not isinstance(state, dict) or state.get("version") != 1:
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
        # A best entry whose exported model was deleted by hand no longer competes.
        fresh["best"] = [entry for entry in fresh["best"] if os.path.isfile(entry["export"])]
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
        """Replace the newest exported model."""
        path = self.export_path("last", epoch, step)
        previous = self.state.get("last")
        export(path)
        if previous and previous.get("export") != path:
            _remove([previous["export"]])
        self.state["last"] = {"epoch": epoch, "step": step, "export": path}
        self.save_state()
        return path

    def save_every(self, epoch, step, export, save_gd):
        """Keep an exported model, and weights-only G/D on the multiple, for good."""
        entry = {"epoch": epoch, "step": step, "export": self.export_path("every", epoch, step)}
        export(entry["export"])
        if wants_gd_every(self.policy, epoch):
            entry["G"], entry["D"] = self.gd_paths("every", epoch)
            save_gd(entry["G"], entry["D"])
        self.state["every"] = [e for e in self.state["every"] if e["epoch"] != epoch] + [entry]
        self.save_state()
        return entry

    def record_evaluation(self, epoch, step, scores, export, save_gd):
        """Record an evaluation and keep the checkpoint if it is among the best.

        Returns (score, rank) - rank is 1-based within the kept best, None when not kept
        or when no best is being kept.
        """
        if self.state["baseline"] is None:
            self.state["baseline"] = baseline_of(scores)
        score = score_of(scores, self.policy["best_criterion"], self.state["baseline"])
        self.state["evaluations"] = [
            e for e in self.state["evaluations"] if e["epoch"] != epoch
        ] + [{"epoch": epoch, "step": step, "scores": scores, "score": score}]
        rank = None
        if self.policy["best_n"] and score is not None:
            candidate = {"epoch": epoch, "step": step, "score": score,
                         "export": self.export_path("best", epoch, step)}
            if self.policy["best_save_gd"]:
                candidate["G"], candidate["D"] = self.gd_paths("best", epoch)
            kept, dropped, accepted = update_best(self.state["best"], candidate, self.policy["best_n"])
            for entry in dropped:
                _remove([entry.get(key) for key in ("export", "G", "D") if entry.get(key)])
            if accepted:
                export(candidate["export"])
                if self.policy["best_save_gd"]:
                    save_gd(candidate["G"], candidate["D"])
                rank = kept.index(candidate) + 1
            self.state["best"] = kept
        self.save_state()
        return score, rank


def remove_checkpoints(experiment_dir):
    """What "Fresh Training" does to this module's files."""
    path = os.path.join(experiment_dir, CHECKPOINTS_DIR)
    if os.path.isdir(path):
        shutil.rmtree(path)
