# HPA-RMVPE

[PhamHuynhAnh16/HPA-RMVPE](https://github.com/PhamHuynhAnh16/HPA-RMVPE)（重みは
[AnhP/HPA-RMVPE](https://huggingface.co/AnhP/HPA-RMVPE)、MIT）を、
学習抽出・通常推論・batch 推論・TTS・realtime・F0 カーブツール・CLI で使えるようにしたものです。

RMVPE の DeepUnet を、YOLO13 系 encoder + HyperACE（hypergraph attention）+ FullPAD decoder
に置き換えたモデルです。メル（16 kHz / hop 160 / win 1024 / 128 bin / 30–8000 Hz / htk）、
32 フレームへの reflect パディング、32000 フレームのチャンク、local average cents の
デコード、しきい値 0.03 は RMVPE と**完全に同じ**です。違うのはネットワーク本体だけです。
そのため実装は `RMVPE0Predictor` を継承し、差し替えているのは `__init__` だけです
（`rvc/lib/predictors/hpa_rmvpe/`）。

## まず使うには

Pitch extraction algorithm で **HPA-RMVPE (76000, aligned)** か
**HPA-RMVPE (112000, aligned)** を選びます。初めて使うときに重みをダウンロードします（下記）。

| method | 中身 | 用途 |
| --- | --- | --- |
| **hpa-rmvpe-76000** | 上流の 76000 step の重み。出力は上流とビット一致 | モデル自体の評価 |
| **hpa-rmvpe-76000-aligned** | 同じ重みで、窓を 2 フレーム（20 ms）後ろに置いて遅れを打ち消したもの | RVC での変換・学習 |
| **hpa-rmvpe-112000** | 上流の 112000 step の重み（HF の `exp/`） | モデル自体の評価 |
| **hpa-rmvpe-112000-aligned** | 112000 の補正版 | RVC での変換・学習 |

プロファイルはありません。量子化（coarse F0）は rmvpe と同じ mel の式です。

## 重みの取得（初回使用時）

`rvc/models/predictors/hpa-rmvpe-<variant>.pt` と `.manifest.json` が無いときに、
そのメソッドを最初に使った時点で次の処理をします（`weights.py` の `ensure_weight`）。

1. HF のリビジョン `fa65f12635ab1877c11a6087940bcc21a7a309c0` に固定した URL から、
   203 MB の学習チェックポイントをダウンロードします。
2. サイズと sha256（LFS object id）を照合します。途中で切れたダウンロードは、
   サイズの段階で「incomplete」と表示して拒否します。実際に一度 157 MB で切れたことがあります。
3. `checkpoint["model"]` だけを取り出して保存します（68 MB。Adam の状態と scheduler は捨てます）。
   manifest には weight の sha256 も書きます。

| variant | HF のパス | 元ファイルの sha256 |
| --- | --- | --- |
| 76000 | `model_76000.pt` | `82bb44b31774e53b976002bf1a9facd9f7dee354e1738e370b7cb6e7109d0120` |
| 112000 | `exp/model_112000.pt` | `096812f4053b534086c5e4eec45b30772ebf17bc7f941b0d407607602e6f6e39` |

- 元のチェックポイントは `best_rpa` を numpy の float64 で持っているため、素の
  `weights_only=True` では読めません。numpy の scalar / dtype / Float64DType の 3 つだけを
  `torch.serialization.safe_globals` で許可して読みます。`weights_only=False` は使いません。
- 保存する weight は決定的です。`torch.save` にパスを渡すとファイル名がアーカイブ名になるため、
  いったんバッファに書いてからファイルにしています。そのため取り直しても sha256 は変わらず、
  学習抽出の再利用判定も崩れません。
- 取得するタイミング:
  - 学習抽出: `extraction_spec` の中で、ワーカーを起動する前に親プロセスで 1 回だけ取得します。
  - realtime: 開始時に predictor を先に作るので、音声を流し始める前に取得します。
  - 推論・TTS・F0 ツール: 最初の変換の中で取得します。
- 手動で入れる場合: `env\python.exe tools\strip_hpa_rmvpe.py 76000 [model_76000.pt]`
  （元ファイルを省略するとダウンロードします）。

## 上流との一致

上流のコミット `0cdb7db22b381f0ee053bd540e8cdc90180443b7` を clone し、その
`HPARMVPE.infer_from_audio` と本実装を `logs/reference/reference.wav`（34.9 s）で比べました。
76000 と 112000 の両方で、**CPU・CUDA ともビット一致**しました。

`E2E0` の `num_heads=4` は重みに現れません（`query_proj` の出力を分割するだけ）。
値を間違えてもエラーなしで読み込めて、音高だけが変わります。そのため引数にはせず固定しています。

## 遅れ（-aligned の理由）

**話し声では、上流の出力が RMVPE / CREPE より約 20 ms 遅れます。**
RMVPE と CREPE は互いに ±3 ms で一致しているので、遅れているのは HPA-RMVPE の方です。

| 音声 | 76000 | 112000 | RMVPE（CREPE 比） |
| --- | --- | --- | --- |
| reference.wav（34.9 s） | +23 / +24 ms | +19 / +21 ms | -2 ms |
| aaaaa.wav | +21 / +20 ms | +23 / +22 ms | +3 ms |
| bbbb.wav | +21 / +21 ms | +22 / +18 ms | +2 ms |
| cccc.wav | +16 / +17 ms | +17 / +16 ms | +2 ms |

（各セルは CREPE 比 / RMVPE 比。0.1 フレーム刻みで最も一致する遅れを探した値です）

合成の倍音（1.5 Hz でうねる glide、80–600 Hz）では、遅れは 76000 で 2.5–3.5 ms、
112000 で 3–6.5 ms しかありません。FCNF0++ の 11 ms と同じく、信号によって変わる遅れです。
上流の学習コード（`src/dataset.py`）のラベル位置は RMVPE と同じ規約なので、原因は特定できていません。

**-aligned** は、音声の末尾に 320 サンプル（2 hop）の reflect パディングを足して推定し、
先頭 2 フレームを捨てます（`HPA_RMVPE_LAG_FRAMES`）。フレーム数は RMVPE と同じ
`len // 160 + 1` のままで、ネットワークには手を入れていません。補正後の遅れは 4 音声とも
±4 ms に収まり、RMVPE と CREPE の差（±3 ms）と同程度です。代わりに、遅れの小さい合成音では
その分（15 ms 前後）早くなります。

`tools/compare_f0_methods.py` で reference.wav を RMVPE と比べた結果です（有声どうしのフレーム）。

| method | 有声率 | V/UV 遷移 | 中央値 | p95 | >600 c | 遅れ | ここだけ有声 | RMVPE だけ有声 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rmvpe | 67.4% | 126 | – | – | – | – | – | – |
| crepe-full | 80.1% | 240 | 13.1 c | 144.2 c | 49 | +3 ms | 537 | 95 |
| hpa-rmvpe-76000 | 63.7% | 120 | 67.9 c | 243.3 c | 4 | +22 ms | 87 | 218 |
| **hpa-rmvpe-76000-aligned** | 63.7% | 120 | **12.0 c** | 77.8 c | 1 | +3 ms | 18 | 149 |
| hpa-rmvpe-112000 | 62.6% | 120 | 61.9 c | 223.7 c | 3 | +20 ms | 62 | 230 |
| **hpa-rmvpe-112000-aligned** | 62.7% | 120 | **10.2 c** | 59.9 c | 1 | +0 ms | 10 | 177 |

- 補正すると、音高も有声/無声の境界も RMVPE とよく揃います（「ここだけ有声」が 87 → 18）。
- HPA-RMVPE は RMVPE より有声とするフレームが 4 ポイントほど少なくなります。作者のベンチで
  voicing の precision が高いこととも合っています。しきい値は上流・RMVPE と同じ 0.03 のままです。
- 実際の変換（kushinada-hubert-large / RefineGAN のモデル、aaaaa.wav）の出力から RMVPE で
  測り直すと、rmvpe で変換したものとの差は中央値で 76000-aligned が 6.5 c、112000（補正なし）が
  22.1 c でした。

## TorchCompile

TorchCompile 設定の「**Enable TorchCompile for F0 models (CREPE / HPA-RMVPE)**」
（旧「Enable TorchCompile (CREPE)」。設定キー `torch_compile_enabled` はそのまま）に従います。
この設定は、変換・batch・TTS・realtime・F0 ツール・**学習抽出**のすべてに効きます。
学習抽出用の「Enable TorchCompile for Extraction (Training)」は、Embedder と RMVPE / FCPE だけを
対象にしており、HPA-RMVPE には関係しません。モードは共通の「TorchCompile Mode」を使います。

- offline（realtime 以外）: 呼び出しごとに長さが変わるので、`dynamic=True`、CUDA graphs なしで
  コンパイルします。predictor は (variant, device) ごとにプロセス内でキャッシュし、設定が
  変わったときだけ作り直します。
- realtime: 窓の長さが固定なので静的形状でコンパイルし、CUDA graphs はモードに従います。
  compile path を `CompileSession` に渡しているので、ウォームアップが 3 ブロックになり、
  ステータスに `HPA-RMVPE: Using compiled inference` と表示されます。
- 失敗したときは `CompiledPath` がそのまま eager に戻り、理由を表示します。

測定環境は RTX 4090、torch 2.13.0+cu132、76000 です。

| 経路 | eager | compiled | 初回（キャッシュなし） |
| --- | --- | --- | --- |
| realtime 1.5 s 窓、`reduce-overhead`（p50 / p95） | 19.6 / 24.0 ms | **9.7 / 10.8 ms（x2.0）** | 47 s |
| realtime 1.5 s 窓、`default`（p50 / p95） | 19.6 / 23.8 ms | 14.2 / 15.7 ms（x1.4） | 18 s |
| offline 2–11 s、`default` | 20–53 ms | 16–49 ms（x1.08–1.28） | 91 s |
| offline 34.9 s、`default` | 141 ms | 139 ms（x1.02） | – |
| realtime 全体（256 ms ブロック、kushinada、p50） | 83.8 ms | **72.4 ms** | 48 s |

- 同じ条件で rmvpe を使うと、realtime 全体は 88.9 ms でした。
- offline で速くならないのは、時間の大半がネットワーク以外（メル、numpy のデコード）に
  かかっているためです。1 回だけの短い変換では、コンパイルの時間の方が大きくなります。
  学習抽出も同じで、初回は 101 s かかり、キャッシュがあれば 6–30 s です。
- **数値:** realtime 窓 200 個（19,703 有声フレーム）で compiled と eager を比べると、
  中央値 0.012 c、p99 0.10 c、10 c を超えるのは 6 フレームでした（salience がほぼ平らな
  フレームで、local average の窓がずれるため）。eager どうしの再実行は完全に一致します。

Windows では inductor のテンプレートを読むのに UTF-8 モードが必要です。`run-applio.bat` は
`PYTHONUTF8=1` を設定しています。CLI から使うときは `-X utf8` を付けるか、同じ環境変数を
設定してください。付けないと cp932 のエラーで eager に戻ります（動作はします）。

## 学習抽出の記録

`model_info.json` の `pitch_extraction_run.specification` は
`{"method": ..., "weight_sha256": ...}` です。次の場合は再抽出します。

- メソッドを変えたとき（76000 ↔ 112000、plain ↔ -aligned を含む）。
- weight が変わったとき。
- 記録のない古いフォルダ。HPA-RMVPE の f0 ではありえないので再利用しません。

ワーカーは起動時に weight の sha256 を照合し、抽出中に weight が差し替えられていれば止まります。

## テスト

`tests/test_hpa_rmvpe.py`（weight が無い環境では、重みを使うテストを skip します）。

- 4 つのメソッドが CLI の 4 か所と UI の 6 つの Radio に入っていること。
- 公開ファイル以外の strip を拒否すること、途中で切れたダウンロードを拒否して後始末すること、
  manifest と一致しない weight を拒否すること。
- strict ロードとパラメータ数（16,939,452）、2 つの重みが別物であること。
- フレーム数が RMVPE と同じであること（-aligned を含む）。
- 話し声での遅れが 15–28 ms で、-aligned で ±6 ms に収まること。RMVPE との差の中央値が 20 c 未満であること
  （構造や重みを取り違えると、ここで大きく外れます）。
- 抽出の spec と再利用の判定、offline キャッシュ、`CompileSession` の f0 path。
- CUDA + Triton がある環境では、offline（2 つの長さ）と realtime（固定窓）で compiled と eager が
  一致すること（p99 < 1 c）。
