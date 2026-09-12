"""Render measured JSON samples; never substitute estimates for missing cases."""

from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

from benchmark_realtime_compile import CASES, ROOT

OUT = ROOT / "benchmarks/results/torchcompile_20260912"
DEST = ROOT / "torchcompile_benchmark_20260912.md"
LABELS = {
    "none": "最適化なし", "crepe": "CREPEのみ", "embedder": "Embedderのみ",
    "rvc": "RVCのみ", "crepe_embedder": "CREPE＋Embedder", "crepe_rvc": "CREPE＋RVC",
    "embedder_rvc": "Embedder＋RVC", "all_reduce": "全てON / reduce-overhead",
    "all_default": "全てON / default", "all_max": "全てON / max-autotune",
}


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for name, *_ in CASES:
        records = []
        for repeat in range(3):
            path = OUT / f"{name}_r{repeat}.json"
            if not path.exists():
                raise RuntimeError(f"Missing result: {path}")
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["success"] and data["flags"] == data["verified_compiled"]
            records.append(data)
        results[name] = records

    def samples(name, field):
        return np.concatenate([r[field] for r in results[name]])

    def mean(name):
        return float(np.mean(samples(name, "total_ms")))

    baseline = mean("none")
    best = min(results, key=mean)
    best_reduce = min((c[0] for c in CASES[:8]), key=mean)
    reductions = {k: (1-mean(k)/baseline)*100 for k in results}
    first = results["none"][0]
    template = manifest["template"]
    perf = template["performanceTab"]
    parameters = template["modelTab"]["parameterValues"]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        cpu = manifest["cpu"]

    lines = [
        "# リアルタイム推論 TorchCompile ベンチマーク",
        "",
        f"作成日時: {datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"テンプレート: `{manifest['template_name']}`  ",
        f"対象コード: `{manifest['git_head']}`。アプリのコード・ユーザー設定を変更せず、専用プロセスで計測。",
        "",
        "## 結果",
        "",
        f"最適化なしは **{baseline:.2f} ms/チャンク**。今回の平均値が最小だったのは **{LABELS[best]}：{mean(best):.2f} ms** で、"
        f"**{reductions[best]:.1f}%（{baseline-mean(best):.2f} ms）短縮**した。",
        f"`reduce-overhead`の8条件の中では **{LABELS[best_reduce]}：{mean(best_reduce):.2f} ms、{reductions[best_reduce]:.1f}%短縮**。",
        "",
        "以下はGPU完了を待った、音声チャンク1個の処理時間。1条件につき40チャンク×3プロセス＝120測定。"
        "p95は全120測定の95パーセンタイル。『巡回平均範囲』は3回それぞれの平均の最小～最大。",
        "",
        "| 条件 | モード | 平均 ms | 中央値 ms | p95 ms | 短縮率 | 巡回平均範囲 ms | RTF |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, _, _, mode in CASES:
        arr = samples(name, "total_ms")
        round_means = [np.mean(r["total_ms"]) for r in results[name]]
        lines.append(
            f"| {LABELS[name]} | {'—' if name == 'none' else mode} | {np.mean(arr):.2f} | {np.median(arr):.2f} | "
            f"{np.percentile(arr,95):.2f} | {reductions[name]:+.1f}% | {min(round_means):.2f}–{max(round_means):.2f} | "
            f"{np.mean(arr)/perf['chunk_size']:.3f} |"
        )
    lines += [
        "",
        "RTF＝処理時間÷入力チャンク時間960ms。RTF < 1はこの負荷で処理がチャンク長に収まることを示す。"
        "これはマイクからスピーカーまでの遅延ではなく、960msの入力待ち・音声ドライバ・出力キューなどは含まない。",
        "",
        "## 処理別の内訳",
        "",
        "各段階の前後でCUDA同期を入れた**別の計測パス**（20チャンク×3回＝60測定）の平均。"
        "CPU処理とGPU完了待ちを含む。同期・計測による影響と対象チャンク数の違いがあるため、上の平均時間とは一致しない。",
        "",
        "| 条件 | CREPE/F0 ms | Embedder ms | RVC ms | インデックス ms | その他 ms | 内訳計測の合計 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stage_means = {}
    for name, *_ in CASES:
        stages = {k: float(np.mean(np.concatenate([r["stages_ms"][k] for r in results[name]])))
                  for k in ("f0", "embedder", "rvc", "index")}
        stage_means[name] = stages
        lines.append(f"| {LABELS[name]} | {stages['f0']:.2f} | {stages['embedder']:.2f} | {stages['rvc']:.2f} | "
                     f"{stages['index']:.2f} | {np.mean(samples(name,'other_ms')):.2f} | "
                     f"{np.mean(samples(name,'profiled_total_ms')):.2f} |")
    lines += [
        "",
        "- **CREPE/F0**：`get_f0`全体。Mangioの正規化、CREPE full_speechのニューラルネット、デコード、周期性処理、F0バッファ更新を含む。ネットワーク単体の時間ではない。",
        "- **Embedder**：`embedder_forward`。入力正規化、Kushinada HuBERT Large、出力層選択・特徴量スケールを含む。ON時はコンパイル出力を保持するためのコピーも含む。",
        "- **RVC**：`net_g.infer`による音声生成。ON時はコンパイル出力のコピーも含む。",
        "- **インデックス**：FAISS検索、CPU/GPU転送、検索した特徴量との混合。TorchCompileの対象外。",
        "- **その他**：リサンプリング、VAD、特徴量補間・保護処理、出力クリップ、SOLA・クロスフェード、CPUへの出力転送、コールバック処理、内訳計測の付加コスト。合計から4項目を引いた残差。",
        "",
        "### 数値から読み取れること",
        "",
    ]
    for name, key in (("crepe", "f0"), ("embedder", "embedder"), ("rvc", "rvc")):
        before, after = stage_means["none"][key], stage_means[name][key]
        lines.append(f"- {LABELS[name]}ONでは、対象段階が {before:.2f} → {after:.2f} ms（{(1-after/before)*100:+.1f}%短縮）。"
                     f"全体の短縮率は {reductions[name]:+.1f}%。")
    for name in ("all_default", "all_max"):
        lines.append(f"- {LABELS[name]}は全てON / reduce-overheadに対し、平均処理時間が "
                     f"{mean(name)-mean('all_reduce'):+.2f} ms（{(mean(name)/mean('all_reduce')-1)*100:+.1f}%）。")
    lines += [
        "- コンパイル対象以外のF0前後処理、FAISS検索、SOLAなどは残るため、段階単体の高速化率がそのまま全体の高速化率になるわけではない。",
        "- 数ms以下の差については上表の巡回平均範囲も参照。今回の順位が別の音声・チャンク長・GPU負荷でも維持されるとは限らない。",
        "",
        "## 準備時間・メモリ・コンパイル確認",
        "",
        "既存のディスクキャッシュを利用しているため、下表は**キャッシュを全消去した初回コンパイル時間ではない**。"
        "『生成』はモデル・インデックスのロードとアプリ既存の開始前ウォームアップを含む。"
        "Embedder/RVCのどちらかがONの場合は、この中で3チャンクのウォームアップが走る。"
        "『追加準備』は全条件共通の実音声12チャンクで、CREPEの初回ロードや遅延コンパイルも含む。"
        "これらと計測前のバッファ充填は、定常推論の測定値から除外した。",
        "",
        "| 条件 | 1巡目の生成 秒 | 1巡目の追加準備 秒 | 2–3巡目の準備合計 秒 | 最大割当VRAM MiB | 最大予約VRAM MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, *_ in CASES:
        rows = results[name]
        later = [r["construction_seconds"] + r["extra_warmup_seconds"] for r in rows[1:]]
        lines.append(f"| {LABELS[name]} | {rows[0]['construction_seconds']:.2f} | {rows[0]['extra_warmup_seconds']:.2f} | "
                     f"{min(later):.2f}–{max(later):.2f} | {max(r['peak_allocated_mib'] for r in rows):.0f} | "
                     f"{max(r['peak_reserved_mib'] for r in rows):.0f} |")
    recompiles = []
    for name, rows in results.items():
        for row in rows:
            before = row["graphs_after_warmup"].get("unique_graphs", 0)
            after = row["graphs_after_total"].get("unique_graphs", 0)
            if after != before:
                recompiles.append(f"{name}/r{row['round']}: {before}→{after}")
    lines += [
        "",
        "VRAMはPyTorchアロケータの定常計測パス中のピーク。CUDAコンテキストやWindowsの表示用途などを含むnvidia-smiの総使用量とは異なる。",
        "",
        "全30プロセスで、ONにした経路が実際にコンパイル済みであることを確認。"
        "Embedder/RVCの通常推論へのフォールバックは発生していない。CREPEは`OptimizedModule`を確認した。",
        ("定常時間計測中の追加グラフ: " + "; ".join(recompiles)) if recompiles else
        "ウォームアップ後から全体時間計測終了まで、Dynamoの`unique_graphs`増加は全30プロセスで0だった。",
        "",
        "## 測定条件と限界",
        "",
        f"- GPU: {manifest['gpu_before'].split(',')[0]}。ドライバ {manifest['gpu_before'].split(',')[1].strip()}。",
        f"- CPU: {cpu}。PyTorch CPUスレッド {first['torch_threads']}、FAISS/OpenMPスレッド {first['faiss_threads']}。既存環境の値を維持。",
        f"- OS: `{manifest['platform']}`。Python `{first['python'].split()[0]}`、PyTorch `{first['torch']}`、CUDA `{first['cuda']}`、Triton `{first['triton']}`。",
        "- 計測前にユーザーがRVCを停止。準備時のGPU状態: `" + manifest["gpu_before"] + "`。バックグラウンド描画を完全停止した専用機ではない。",
        f"- モデル: `{template['modelTab']['voice']['model_path']}`。FAISSインデックスも指定ファイルを使用（{first['faiss_index_vectors']:,}ベクトル）。",
        "- F0: `mangio-crepe-full-speech`、Embedder: `kushinada-hubert-large`、Embedder精度: `fp32`。",
        f"- 入力/出力48kHz、チャンク{perf['chunk_size']}ms、クロスフェード{perf['crossfade_overlap_size']}秒、追加コンテキスト{perf['extra_convert_size']}秒。"
        f"推論窓は16kHzで{first['input_window_samples_16k']}サンプル（{first['input_window_samples_16k']/16000:.3f}秒）。",
        f"- ピッチ{parameters['pitch']}、index rate {parameters['index_rate']}、protect {parameters['protect']}、volume envelope {parameters['volume_envelope']}、VAD ON、autotune OFF、proposed pitch OFF、speaker ID 0。",
        f"- 入力は対象モデルの`sliced_audios_16k`のファイル名順先頭32 WAVを連結し、48kHzに変換した{manifest['input_seconds']:.3f}秒の音声。"
        "全条件に同じ先頭40チャンクを入力。内訳計測は同じ先頭20チャンク。対象モデルの学習用音声であり、ユーザーのマイク入力そのものではない。",
        "- 各条件3回、毎回独立したPythonプロセス。1巡目は依頼の順、2・3巡目は固定シードで順序を変更。条件間でコンパイル済みモデルやCUDA Graphを共有しない。ディスクキャッシュは共有。",
        "- 計測用の独立した設定JSONをプロセス内だけで参照。実際の`assets/config.json`とテンプレートを変更していない。Triton無効化はOFF。",
        "- アプリと同じ`AudioCallbacks.change_voice`を呼び、入力変換・VAD・インデックス・音声生成・SOLA・CPU出力を通して計測。音声デバイスやGradioは起動せず、UI・ドライバ・入出力待ち時間は含めない。",
        "- チャンクは実時間の960ms間隔ではなく連続投入。GPUのクロック・温度は固定せず、各プロセス前後の状態をJSONに記録。実時間で待機を挟む場合と省電力状態が異なる可能性がある。",
        "- CREPEは既存の`dynamic=True`コンパイル、Embedder/RVCは既存の`dynamic=False`＋出力コピーをそのまま使用。"
        "`reduce-overhead`を選んでも全演算のCUDA Graph化や高速化が保証されるわけではなく、この実装での実測値を掲載。",
        "- 出力のサンプル数と有限値を確認した。乱数シードは固定したが、コンパイル前後で乱数列は一致するとは限らない。声色・音質の同等性の聴取評価はこのベンチマークには含めていない。",
        "",
        "## 再現用ファイル",
        "",
        "- [計測スクリプト](benchmarks/benchmark_realtime_compile.py)",
        "- [レポート生成スクリプト](benchmarks/render_realtime_compile_report.py)",
        "- [環境・入力音声一覧・ハッシュ](benchmarks/results/torchcompile_20260912/manifest.json)",
        "- [実行順](benchmarks/results/torchcompile_20260912/order.json)",
        "- `benchmarks/results/torchcompile_20260912/<条件>_r<0–2>.json`: 全サンプルの時間、内訳、コンパイル確認、GPU状態。対応する`.log`にコンパイルログ。",
        "",
        "```powershell",
        r".\env\python.exe -X utf8 benchmarks\benchmark_realtime_compile.py --all --out benchmarks/results/torchcompile_rerun",
        "```",
        "",
        "同じ出力先の成功済み条件はスキップするため、再測定時は新しい`--out`を指定する。",
        "",
    ]
    DEST.write_text("\n".join(lines), encoding="utf-8")
    print(DEST)
    print(f"best={best} baseline={baseline:.3f} best_ms={mean(best):.3f} reduction={reductions[best]:.2f}%")


if __name__ == "__main__":
    main()
