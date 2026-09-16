"""学習中 / 学習済みモデルの進捗を TensorBoard のログから表にしてターミナルに出す。

使い方（どこから実行してもよい）:
    env\\python.exe -X utf8 tools\\training_status.py <run> [<run> ...]

<run> は logs/ 以下のフォルダ名。一意に決まるなら一部だけでもよい（例: large_4）。
  - 1 つ: その学習の進捗
  - 2 つ以上: 並べて表示し、1 つ目を基準にした差も出す

表示内容（epoch の刻み、平均の幅、指標、レイアウトなど）は下の「設定」を直接編集する。
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================== 設定（ここを直接編集） ==============================

# 引数なしで実行したときに表示する run（フォルダ名）。空なら使い方を表示して終了する。
DEFAULT_RUNS = []

# --- 表示する epoch 行 ------------------------------------------------------------
# 行 = EARLY_EPOCHS + EPOCH_SPAN 刻みの epoch + EXTRA_EPOCHS + 各 run の最新 epoch
# 例: EARLY_EPOCHS = [1, 5, 10]、EPOCH_SPAN = 50 なら 1, 5, 10, 50, 100, 150, ...
EARLY_EPOCHS = [1, 2, 3, 5, 10, 15]
# EARLY_EPOCHS の最後より後ろを何 epoch 刻みで出すか。0 なら刻みの行は出さない。
EPOCH_SPAN = 20
# 刻みとは別に必ず出したい epoch（例: [250, 500]）
EXTRA_EPOCHS = []
# 各 run の最新 epoch の行を足すか
SHOW_LATEST = True
# 刻みの行がこの数を超えたら、刻みを 2 倍、4 倍…と自動で広げる（0 で広げない）
MAX_SPAN_ROWS = 30

# --- 平均のとり方 -----------------------------------------------------------------
# loss_avg_50 は 50 step ごとの値なので、1 epoch 分だけだと揺れる。
# 各行は「その epoch の前後 SMOOTH_EPOCHS epoch」の平均（0 ならその epoch 内だけ）。
SMOOTH_EPOCHS = 1

# --- 表示する指標 -----------------------------------------------------------------
# (TensorBoard のタグ, 表示名)。行を消せば列が減り、# を外せば増える。
METRICS = [
    ("loss_avg_50/g/mel", "mel"),
    ("loss_avg_50/g/kl", "kl"),
    ("loss_avg_50/g/fm", "fm"),
    ("loss_avg_50/g/adv", "g_adv"),
    ("loss_avg_50/d/adv", "d_adv"),
    # ("loss_avg_50/g/total", "g_total"),
    # ("grad_avg_50/norm_g", "gnorm_g"),
    # ("grad_avg_50/norm_d", "gnorm_d"),
]
# 2 run 以上のとき、1 つ目の run との差（その run − 1 つ目）を出す指標（表示名で指定）
DIFF_METRICS = ["mel", "kl"]
# 小数点以下の桁数
DECIMALS = 2

# --- 表のレイアウト ---------------------------------------------------------------
# "wide":       1 つの表に全指標を横に並べる（run が少ないとき向き）
# "per_metric": 指標ごとに表を分け、run を列に並べる（run が多いとき向き）
# "auto":       run が WIDE_MAX_RUNS 個以下なら wide、それより多ければ per_metric
TABLE_LAYOUT = "auto"
WIDE_MAX_RUNS = 2

# --- 更新回数（step）をそろえた比較 -----------------------------------------------
# バッチサイズが違う run 同士は epoch では公平に比べにくいので、
# 全 run の最新 step のうち一番小さい step の付近で比べる。
SHOW_MATCHED_STEPS = True
# その比較で平均する幅（step 数）
MATCHED_STEP_WINDOW = 480

# --- 細かい推移（--fine）----------------------------------------------------------
# 「先週より良くなった / 悪くなった」を 1 点ずつ比べると、50 step 平均のばらつき
# （mel で ±0.15 程度）を変化と読み違える。--fine はその幅と傾きを一緒に出す。
FINE_EPOCHS = 150   # 直近何 epoch を見るか（--fine N で上書きできる）
FINE_STEP = 10      # 何 epoch 刻みで行を出すか
SLOPE_WINDOWS = (300, 150, 60)  # 傾きを計算する期間（epoch）

# --- 速度と残り時間 ---------------------------------------------------------------
# 直近何 epoch 分のログの時刻から 秒/epoch を計算するか
SPEED_EPOCHS = 3
# 出力済みモデル（*_Ne_Ms.pth）の一覧は最後の何個まで表示するか
SHOW_LAST_EXPORTS = 6

# ==================================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO_ROOT, "logs")


def resolve_run(name):
    """フォルダ名そのもの、または一意に決まる一部分から run を特定する。"""
    if os.path.isdir(os.path.join(LOGS, name)):
        return name
    candidates = sorted(
        d for d in os.listdir(LOGS) if os.path.isdir(os.path.join(LOGS, d)) and name in d
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        sys.exit(f"logs/ に '{name}' を含むフォルダがありません。")
    sys.exit(f"'{name}' に一致するフォルダが複数あります: {', '.join(candidates)}")


def running_trainings():
    """実行中の train.py: run 名 -> (総 epoch, バッチサイズ)。"""
    try:
        output = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -match 'train.py' } | "
                "ForEach-Object { $_.CommandLine }",
            ],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return {}
    found = {}
    for line in output.splitlines():
        match = re.search(r"train\.py\s+(\S+)\s+(\d+)\s+(\d+)\s", line)
        if not match:
            continue
        name, total = match.group(1), int(match.group(3))
        # 引数の並び: 保存間隔 総epoch pretrainG pretrainD gpu バッチ サンプリングレート ...
        numbers = re.findall(r"(?<=\s)\d+(?=\s|$)", line[match.end(1):])
        batch = int(numbers[3]) if len(numbers) > 3 else None
        found[name] = (total, batch)
    return found


def load_run(run):
    folder = os.path.join(LOGS, run)
    exports = sorted(
        (int(m.group(1)), int(m.group(2)))
        for path in glob.glob(os.path.join(folder, f"{glob.escape(run)}_*e_*s.pth"))
        if (m := re.search(r"_(\d+)e_(\d+)s\.pth$", path))
    )
    scalars = {}
    for event_file in sorted(
        glob.glob(os.path.join(folder, "eval", "events.out.tfevents.*")), key=os.path.getmtime
    ):
        accumulator = EventAccumulator(
            event_file,
            size_guidance={"scalars": 0, "images": 1, "audio": 1, "histograms": 1, "tensors": 1},
        )
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            scalars.setdefault(tag, []).extend(
                (e.step, e.value, e.wall_time) for e in accumulator.Scalars(tag)
            )
    if "loss_avg_50/g/mel" not in scalars:
        sys.exit(f"{run}: TensorBoard のログ（eval/）にまだ値がありません。")
    for values in scalars.values():
        values.sort()

    notes = []
    if exports:
        steps_per_epoch = exports[-1][1] // exports[-1][0]
        if len({s // e for e, s in exports}) > 1:
            notes.append(
                "途中でバッチサイズが変わっています。epoch は最新のバッチ"
                f"（{steps_per_epoch} step/epoch）で換算した近似で、TensorBoard の step も"
                "学習の再開ごとに前後します。"
            )
    elif "learning_rate" in scalars:
        steps_per_epoch = scalars["learning_rate"][0][0]  # 1 epoch 目の終わりに記録される
    else:
        sys.exit(f"{run}: 1 epoch あたりの step 数が分かりません（出力済みモデルも lr の記録もない）。")

    last_step = scalars["loss_avg_50/g/mel"][-1][0]
    if exports and last_step // steps_per_epoch > exports[-1][0] * 1.5 + 50:
        notes.append(
            f"TensorBoard のログ（epoch {last_step // steps_per_epoch} 相当）が最後の出力済みモデル"
            f"（{exports[-1][0]}e）より大幅に先まであります。フォルダをコピーした場合など、"
            "別の学習のログが eval/ に混ざっている可能性があります。"
        )
    info_path = os.path.join(folder, "model_info.json")
    embedder = None
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            embedder = json.load(f).get("embedder_model")
    return dict(
        run=run, spe=steps_per_epoch, scalars=scalars, last_step=last_step,
        last_epoch=max(1, last_step // steps_per_epoch), exports=[e for e, _ in exports],
        embedder=embedder, notes=notes,
    )


def mean_between(run, tag, low_step, high_step):
    values = [
        v for s, v, _ in run["scalars"].get(tag, []) if low_step < s <= high_step and np.isfinite(v)
    ]
    return float(np.mean(values)) if values else float("nan")


def stats_between(run, tag, low_step, high_step):
    values = [
        v for s, v, _ in run["scalars"].get(tag, []) if low_step < s <= high_step and np.isfinite(v)
    ]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def slope_per_100(run, tag, epochs):
    """直近 epochs epoch の傾き（100 epoch あたり、最小二乗）。"""
    spe = run["spe"]
    points = [
        (s / spe, v) for s, v, _ in run["scalars"].get(tag, [])
        if s > run["last_step"] - epochs * spe and np.isfinite(v)
    ]
    if len(points) < 3:
        return float("nan")
    return float(np.polyfit([p[0] for p in points], [p[1] for p in points], 1)[0] * 100)


def print_fine(runs, window):
    """直近の細かい推移と傾き。1 点ずつの上下をノイズと見分けるための表。"""
    for i, r in enumerate(runs, 1):
        print(
            f"\n== #{i} {r['run']}: last {window} epochs, every {FINE_STEP} epochs"
            f" (+-mel = spread of the 50-step averages = noise level)"
        )
        print("epoch |" + "".join(f"{n:>9s}" for _, n in METRICS) + f"{'+-mel':>9s}")
        rows = list(range(max(1, r["last_epoch"] - window), r["last_epoch"] + 1, FINE_STEP))
        if rows[-1] != r["last_epoch"]:
            rows.append(r["last_epoch"])
        for e in rows:
            low, high = (e - FINE_STEP / 2) * r["spe"], (e + FINE_STEP / 2) * r["spe"]
            values = [stats_between(r, tag, low, high)[0] for tag, _ in METRICS]
            _, spread = stats_between(r, "loss_avg_50/g/mel", low, high)
            print(f"{e:5d} |" + "".join(fmt(v, 9) for v in values) + fmt(spread, 9))
        print("   slopes per 100 epochs (negative = improving for mel):")
        for tag, name in METRICS:
            cells = "  ".join(
                f"last {w}: {slope_per_100(r, tag, w):+.3f}" for w in SLOPE_WINDOWS
            )
            print(f"   {name:8s} {cells}")


def epoch_value(run, tag, epoch):
    if epoch > run["last_epoch"]:
        return float("nan")
    spe = run["spe"]
    low = (epoch - 1 - SMOOTH_EPOCHS) * spe
    high = min(epoch + SMOOTH_EPOCHS, run["last_epoch"]) * spe
    return mean_between(run, tag, low, high)


def epoch_rows(latest_epochs):
    top = max(latest_epochs)
    rows = {e for e in EARLY_EPOCHS if e <= top}
    if EPOCH_SPAN > 0:
        start = max(EARLY_EPOCHS) if EARLY_EPOCHS else 0
        span = EPOCH_SPAN
        while MAX_SPAN_ROWS and (top - start) // span > MAX_SPAN_ROWS:
            span *= 2
        rows |= set(range(span * (start // span + 1), top + 1, span))
    rows |= {e for e in EXTRA_EPOCHS if e <= top}
    if SHOW_LATEST:
        rows |= set(latest_epochs)
    return sorted(e for e in rows if e >= 1)


def fmt(value, width, signed=False):
    if not np.isfinite(value):
        return f"{'-':>{width}s}"
    return f"{value:>+{width}.{DECIMALS}f}" if signed else f"{value:>{width}.{DECIMALS}f}"


def print_status(runs, active):
    now = time.time()
    print(f"== status {time.strftime('%m-%d %H:%M')}")
    for i, r in enumerate(runs, 1):
        mel = r["scalars"]["loss_avg_50/g/mel"]
        recent = [x for x in mel if x[0] > r["last_step"] - SPEED_EPOCHS * r["spe"]]
        if len(recent) > 1 and recent[-1][0] > recent[0][0]:
            sec_per_epoch = (recent[-1][2] - recent[0][2]) / (recent[-1][0] - recent[0][0]) * r["spe"]
        else:
            sec_per_epoch = float("nan")
        lr = r["scalars"]["learning_rate"][-1][1] if "learning_rate" in r["scalars"] else float("nan")
        total, batch = active.get(r["run"], (None, None))
        if r["run"] in active:
            state = "running"
            if total and np.isfinite(sec_per_epoch):
                state += f", {total} epochs total, about {(total - r['last_epoch']) * sec_per_epoch / 3600:.1f} h left"
        else:
            state = f"stopped (last log {(now - mel[-1][2]) / 60:.0f} min ago)"
        print(f"#{i} {r['run']}  [{r['embedder'] or '?'}]")
        batch_text = f"batch {batch}, " if batch else ""  # 実行中の学習のみ分かる
        print(
            f"   epoch {r['last_epoch']}, step {r['last_step']}, {batch_text}"
            f"{r['spe']} steps/epoch, {sec_per_epoch:.0f} s/epoch, lr {lr:.3e}, {state}"
        )
        exports = r["exports"]
        shown = exports[-SHOW_LAST_EXPORTS:] if SHOW_LAST_EXPORTS else exports
        more = f" ({len(exports)} files)" if len(shown) < len(exports) else ""
        print(f"   exports: {shown}{more}")
        for note in r["notes"]:
            print(f"   note: {note}")


def print_wide(runs, rows):
    labels = [f"#{i}" for i in range(1, len(runs) + 1)]
    columns = []  # (header, function(epoch) -> text)
    for tag, name in METRICS:
        for label, r in zip(labels, runs):
            columns.append((f"{name}{label}", lambda e, r=r, tag=tag: epoch_value(r, tag, e), False))
        if name in DIFF_METRICS:
            for label, r in zip(labels[1:], runs[1:]):
                columns.append(
                    (f"d{name}{label}",
                     lambda e, r=r, tag=tag: epoch_value(r, tag, e) - epoch_value(runs[0], tag, e),
                     True)
                )
    widths = [max(8, len(h) + 1) for h, _, _ in columns]
    print("epoch |" + "".join(f"{h:>{w}s}" for (h, _, _), w in zip(columns, widths)))
    for e in rows:
        print(f"{e:5d} |" + "".join(fmt(f(e), w, signed) for (_, f, signed), w in zip(columns, widths)))


def print_per_metric(runs, rows):
    labels = [f"#{i}" for i in range(1, len(runs) + 1)]
    for tag, name in METRICS:
        diffs = name in DIFF_METRICS and len(runs) > 1
        header = "".join(f"{l:>9s}" for l in labels)
        if diffs:
            header += " |" + "".join(f"{l + '-#1':>9s}" for l in labels[1:])
        print(f"\n-- {name}")
        print("epoch |" + header)
        for e in rows:
            values = [epoch_value(r, tag, e) for r in runs]
            line = "".join(fmt(v, 9) for v in values)
            if diffs:
                line += " |" + "".join(fmt(v - values[0], 9, signed=True) for v in values[1:])
            print(f"{e:5d} |" + line)


def print_matched_steps(runs):
    step = min(r["last_step"] for r in runs)
    print(f"\n== matched optimizer steps: around step {step} (mean of the last {MATCHED_STEP_WINDOW} steps)")
    print("   epoch at that step: " + ", ".join(f"#{i} {step // r['spe']}" for i, r in enumerate(runs, 1)))
    for tag, name in METRICS:
        values = [mean_between(r, tag, step - MATCHED_STEP_WINDOW, step) for r in runs]
        parts = [f"#1 {fmt(values[0], 0).strip()}"]
        for i, v in enumerate(values[1:], 2):
            parts.append(f"#{i} {fmt(v, 0).strip()} ({fmt(v - values[0], 0, signed=True).strip()})")
        print(f"   {name:8s} " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="logs/ 以下のフォルダ名（一部でも可）。1 つ以上。")
    parser.add_argument(
        "--fine", nargs="?", type=int, const=FINE_EPOCHS, default=None,
        help=f"直近 N epoch を {FINE_STEP} epoch 刻みで、ばらつきと傾き付きで出す（N 省略時 {FINE_EPOCHS}）",
    )
    args = parser.parse_args()
    names = args.runs or DEFAULT_RUNS
    if not names:
        parser.print_help()
        return

    runs = [load_run(resolve_run(name)) for name in names]
    print_status(runs, running_trainings())

    rows = epoch_rows([r["last_epoch"] for r in runs])
    layout = TABLE_LAYOUT
    if layout == "auto":
        layout = "wide" if len(runs) <= WIDE_MAX_RUNS else "per_metric"
    smooth = f"epochs e-{SMOOTH_EPOCHS}..e+{SMOOTH_EPOCHS}" if SMOOTH_EPOCHS else "that epoch only"
    print(f"\n== per epoch (loss_avg_50, mean over {smooth}; d = run - #1)")
    if layout == "wide":
        print_wide(runs, rows)
    else:
        print_per_metric(runs, rows)

    if SHOW_MATCHED_STEPS and len(runs) > 1:
        print_matched_steps(runs)

    if args.fine:
        print_fine(runs, args.fine)


if __name__ == "__main__":
    main()
