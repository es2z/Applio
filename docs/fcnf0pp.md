# FCNF0++ / FCNF0++-RVC

PENN（[interactiveaudiolab/penn](https://github.com/interactiveaudiolab/penn)、MIT）の
FCNF0++ を、PENN 本来の pitch / periodicity / decoder / framing を変えずに
学習抽出・通常推論・batch 推論・TTS・realtime・F0 utility で使えるようにしたものです。

## まず使うには

1. Pitch extraction algorithm で **FCNF0++-RVC** を選びます。
2. **F0 profile JSON は空欄のまま**にします（同梱の既定値が使われます）。
3. 通常どおり変換・抽出・realtime 開始を実行します。CPU でも動きます（遅くなります）。

| method | 中身 | 用途 |
| --- | --- | --- |
| **fcnf0++** | PENN の pitch をそのまま渡す。全フレーム有声、補間なし | FCNF0++ モデル自体の評価 |
| **fcnf0++-rvc** | 同じ pitch のうち、PENN の periodicity が `periodicity_threshold` 以下のフレームだけを無声（0 Hz）にする | RVC での変換・学習 |

両者の違いは V/UV の判定**だけ**です。聴き比べれば「聴感の問題が voicing 判定から来ているのか、モデルから来ているのか」を 1 回で切り分けられます。median filter、hysteresis、補間、無音ゲートなどは入れていません。

## 同梱の既定値

~~~json
{
  "method": "fcnf0++-rvc",
  "version": 1,
  "decoder": "viterbi",
  "periodicity_threshold": 0.035,
  "center": "zero",
  "coarse_min": 50.0,
  "coarse_max": 1680.0,
  "calibrated": false
}
~~~

`fcnf0++` の既定値は `"periodicity_threshold": null` で、それ以外は同じです。
「Load recommended F0 settings」ボタンで同じ JSON が入ります。優先順位は
**明示 JSON → method が一致する checkpoint の設定 → 同梱の既定値** です（FCN-993 と同じ）。
CLI では `--fcn_profile` に JSON を渡します（FCN-993 と FCNF0++ で共用です。JSON の `method` で振り分けます）。

| 項目 | 既定値 | 意味 |
| --- | --- | --- |
| decoder | `viterbi` | PENN の既定値。`argmax` も選べます（どちらも local expected value ±9 bin を使います）。periodicity は decoder に依存しません |
| periodicity_threshold | 0.035 | `periodicity > threshold` を有声とします（`penn.voicing.threshold` と同じ向き）。`fcnf0++` では `null` |
| center | `zero` | frame i の窓中心を t = i × 10 ms に置きます。`half-hop`（+5 ms）は過去の実装を再現する比較用です |
| coarse_min / coarse_max | 50 / 1680 | PENN が復号してよい範囲であり、同時に coarse F0 の量子化範囲でもあります。50–1100 も指定できます |

設定を変えると学習抽出はやり直しになります（`model_info.json` の `f0_extraction` に記録されます）。
FCNF0++ で学習したモデルは checkpoint に設定が引き継がれ、推論と realtime でも同じ設定が使われます。

## PENN との対応

| 項目 | PENN | この統合 |
| --- | --- | --- |
| 入力 | 任意の sr を 8 kHz へ resample（`torchaudio.transforms.Resample`、既定パラメータ） | 16 kHz 音声を同じクラス・同じパラメータで 8 kHz へ変換します。penn は呼び出しごとに Resample を作り直すので、1 つを使い回しています |
| hop | `HOPSIZE_SECONDS = 0.01` | 0.01 秒。RVC の 160 samples @16 kHz と一致します |
| framing | `penn.preprocess`、`center` は `half-window`（API 既定）/ `half-hop` / `zero` | `penn.preprocess(center='zero')`。frame 数は `len//160 + 1` で、先頭から `p_len` 個を使います。全体 resize や `np.interp` はしていません |
| model | `penn.Model()`、`torch.load(checkpoint)['model']` | 同じ `penn.Model()` を `weights_only=True` で読み込み、一度だけ構築します |
| 推論 | `penn.core.inference_context`（CUDA では autocast） | 同じ関数を使います |
| decoder | `penn.decode.Viterbi()`（torbi）/ `Argmax()`、local expected value 19 bins | 同じクラスです。transition 行列だけ最初から device に置きます（値は同一） |
| periodicity | `penn.periodicity.entropy`（fmin/fmax でマスクした logits から） | 同じ関数です |
| `interp_unvoiced_at` | 既定値 None | **常に None**。補間すると 0 Hz が消え、breath・語尾・子音に音高が残るためです（promonet も viterbi 使用時は補間していません） |

`penn.from_audio(..., center='zero', interp_unvoiced_at=None)` と、pitch・periodicity とも
**ビット一致**します（CPU / CUDA、viterbi / argmax。`tests/test_fcnf0pp.py`）。

## 過去の実装（15936e63〜e1f272ea）との違い

| # | 過去 | 今回 | 根拠 |
| --- | --- | --- | --- |
| 1 | `center="half-hop"` | `center="zero"` | 窓の位置は `half-hop` だと 5 ms 早くなります（純音のグライドで実測 +5.1 ms） |
| 2 | 単一しきい値 0.065。F0 範囲が offline 50–1100、学習・realtime 50–1680 で別々 | 3 経路すべて 50–1680、しきい値は 0.035 | entropy periodicity の無声の床は `1 − log(K)/log(1440)`（K は許可 bin 数）で、1100 Hz なら 0.0407、1680 Hz なら 0.0230 です。同じ 0.065 が経路ごとに違う意味を持っていました |
| 3 | realtime の coarse 量子化が Hz と mel の混在式 | `quantize_f0`（学習・offline と同じ式） | 旧式では 900 Hz が約 30 bin 低くなります。**他方式の realtime は互換性のため旧式のまま**です |

## 測定結果（RTX 4090、2026-09-23）

### しきい値：PENN と同じ方法で選びました

PENN はしきい値を定数として持たず、`penn/evaluate` で有声 F1 を最大化するように決めます。
FCN-993-RVC Balanced v1 を選んだのと同じ 4 例（FCN-f0 の DTB 手動修正済み例 4/5/11/28、
1283 frames）で、同じ基準で評価しました（`tools/evaluate_fcnf0pp_voicing.py`、
結果は [fcnf0pp-voicing-evaluation.json](fcnf0pp-voicing-evaluation.json)）。

| 設定 | 無声→有声の誤り | 有声→無声の誤り | F1 | V/UV 遷移 |
| --- | ---: | ---: | ---: | ---: |
| **FCNF0++ 0.035（PENN の F1 基準で最適）** | **34 / 479（7.1%）** | **30 / 804（3.7%）** | **0.960** | **60** |
| FCNF0++ 0.065（README の例） | 28 / 479（5.8%） | 65 / 804（8.1%） | 0.941 | 76 |
| FCNF0++ 0.1625（promonet の VOICING_THRESHOLD） | 14 / 479（2.9%） | 129 / 804（16.0%） | 0.904 | 86 |
| FCN-993-RVC Balanced v1 | 66 / 479（13.8%） | 20 / 804（2.5%） | | |
| Mangio-CREPE full-speech | 83 / 479（17.3%） | 18 / 804（2.2%） | | |

0.025〜0.04 の範囲で F1 は 0.955 以上を保ち、急変はありません。しきい値を上げるほど弱声が切れ（0.1625 では有声の 16% を落とします）、遷移も増えます。
開発用の 4 例であり、話者を分けた独立評価ではありません。そのため `calibrated: false` としています。

### 時間軸

frame i の音高が実際に鳴っていた時刻から i × 10 ms を引いた値です。上昇と下降のグライドを平均し、音高の系統誤差を打ち消しています。

| 信号 | `center='zero'` | `center='half-hop'` |
| --- | ---: | ---: |
| 純音 300–800 Hz | −0.1 ms | +5 ms 早い |
| 倍音あり 70–180 Hz / 150–400 Hz | −10.9 / −12.3 ms | |
| 倍音あり 300–800 Hz / 500–1200 Hz | −0.4 / −0.1 ms | |
| 実音声（RMVPE・CREPE との最良ラグ） | +11 ms 遅れ | +5〜6 ms 遅れ |
| 参考：RMVPE / CREPE / FCPE / FCN-993（グライド） | +0.5 / +0.6 / +0.1 / −1.3 ms | |

**窓の位置（統合側で決めるもの）は `zero` で正しく揃っています。** 一方、**倍音を含む話し声の帯域でだけ、FCNF0++ の出す音高は約 11 ms（約 1 フレーム）遅れます**。純音や高い声では遅れません。
これは信号の中身によって変わるモデル自身の性質で、固定量をずらす補正では正しくならないため、**補正していません**（今回の範囲外）。語頭・語尾・breath への移行部の聴感に影響する可能性が最も高い、既知の性質です。

定常の倍音音では、音高も平均 −4 cents ずれていました（−10.6〜+2.7 cents）。

### 実音声での比較（logs/reference/reference.wav、34.9 秒、基準 RMVPE）

| method | 有声率 | V/UV 遷移 | 1 フレームの穴 | 音高差の中央値 | ラグ |
| --- | ---: | ---: | ---: | ---: | ---: |
| fcnf0++ | 100% | 0 | 0 | 35.7 c | +11 ms |
| fcnf0++-rvc | 62.5% | 154 | 7 | 33.1 c | +11 ms |
| fcn-993-rvc | 66.0% | 262 | 27 | 10.6 c | +3 ms |
| mangio-crepe-full-speech | 67.6% | 256 | 0 | 12.9 c | −3 ms |
| rmvpe | 67.4% | 126 | 2 | – | – |

### realtime

stateless で、毎ブロック convert 窓全体を再計算します（RMVPE・CREPE と同じ方式）。holdback はありません。
160 ms ブロック、122 frame の窓、kushinada-hubert-large、MRF HiFi-GAN 48k で計測しました。

| method | p50 | p95 |
| --- | ---: | ---: |
| rmvpe | 40.1 ms | 52.0 ms |
| fcnf0++-rvc viterbi | 46.4 ms | 63.1 ms |
| fcnf0++-rvc argmax | 26.1 ms | 30.8 ms |

viterbi は窓全体を大域復号するので、同じ時刻の F0 がブロックごとに再復号されます。
隣り合う窓の重なる部分を比べると、50 cents 超の変化は viterbi で 76 / 12697、
argmax（frame 独立）でも 71 / 12697 でした。V/UV の反転はどちらも 223 です。
揺れの大部分は decoder ではなく、窓の端の reflect padding と resample から来ています（RMVPE・CREPE にも同じことが起きます）。
このため realtime の既定も viterbi です。

## 聴き比べ

~~~powershell
.\env\python.exe tools\compare_f0_methods.py input.wav --output compare.json `
  --pth logs\MODEL\MODEL.pth --embedder_model kushinada-hubert-large --render_dir logs\f0_compare
~~~

`fcnf0++` / `fcnf0++-rvc` / `fcn-993-rvc` / `mangio-crepe-full-speech` / `rmvpe` の F0 指標を JSON に出し、
同じモデル・ピッチシフト 0・index なしで変換した wav を書き出します。
一度に変えるのは 1 項目だけにしてください（method、decoder、しきい値を同時に動かさない）。

症状が出た音源の periodicity 分布を見るには：

~~~powershell
.\env\python.exe tools\measure_fcnf0pp_periodicity.py input.wav --csv periodicity.csv
~~~

## 依存関係と weights

- `penn==1.0.0`。`penn` は import 時に `torbi` を読み込むので、argmax だけを使う場合でも torbi が必要です。
  `torchutil` は import 時に `apprise` と `psutil` を読み込みます。
- **Windows の torbi は `vendor/torbi` の wheel を使います。** PyPI の torbi 1.4.0 は torch 2.11 / cu130 までのバイナリしか同梱しておらず、torch 2.13 / cu132 では import できません。
  vendor の wheel は公式ソース（commit `ca7732cd`）を無改変でビルドしたもので、SHA-256 は `vendor/torbi/README.md` に記載しています。
  この wheel の torbi は、ピークを持つ posteriorgram で `librosa.sequence.viterbi`（torbi 自身の参照実装）と完全一致しました（T=500）。
  py313 プロファイル（torch 2.8 / cu128）は PyPI の wheel を使います。
- weights：HuggingFace `maxrmorrison/fcnf0-plus-plus` の `fcnf0++.pt`（revision `74911e26`、107 MB、sha256 `28d89add…`）は Adam の状態を含む学習 checkpoint です。
  `tools/strip_fcnf0pp.py`（初回の prerequisites download でも自動実行）で `model` だけの 35.7 MB にし、
  `rvc/models/predictors/fcnf0++.manifest.json` に sha256 を記録します。読み込みは `weights_only=True` です。実行時に HuggingFace へアクセスすることはありません。
- License：PENN は MIT です（[licenses/PENN-LICENSE](licenses/PENN-LICENSE)）。
  引用：Max Morrison, Caedon Hsieh, Nathan Pruyne and Bryan Pardo, *Cross-domain Neural Pitch and Periodicity Estimation* (arXiv:2301.12258)。

## 今回やっていないこと

octave-jump の補正、posterior を使った裏返り補正、hysteresis による V/UV、median などの時間平滑化、語尾・breath 専用の処理、約 11 ms の遅れの補正、独自の streaming decoder。
FCNF0++ を正しく統合した状態で聴感を評価してから、必要なら別途検討します。
