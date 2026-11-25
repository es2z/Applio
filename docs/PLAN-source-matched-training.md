# 実装計画: ソース寄せ特徴量による学習 (srcaug)

男性（ユーザー）→ 女性（naru）変換の精度を上げるための実装計画。
手元の Claude Code に実装を依頼する前提で書いている。ほかの候補案と、それらをこの順位にした理由は
`docs/ideas/m2f-conversion-ideas.md` にある。

- 状態: **未実装**（2026-11-25 作成）
- 推論・リアルタイムへの変更: **なし**（学習データの作り方だけを変える）
- 既存モデル・既存フォルダへの影響: なし（新しい設定の既定値 0 では学習ループが今とビット単位で同じ）

---

## 1. 何をするのか

### 今の問題

naru モデルの学習で、入力と正解の組はこうなっている。

| | 入力する特徴量 | 正解 |
|---|---|---|
| 学習時 | **naru の声** から取った kushinada L24 特徴量 | naru の音声 |
| 使用時 | **ユーザーの声** から取った kushinada L24 特徴量 | （naru らしい声が出てほしい） |

モデルは「男性の声から取った特徴量」を学習中に一度も見ていない。見たことのない入力をそのまま処理しているので、
ユーザーの声質が出力に残り、発音が崩れる。今は index（とSPIN の key index）が、入力の特徴量を naru の特徴量に
部分的に置き換えることで、この差を埋めている。

### やること

学習データに次の組を追加する。

| | 入力する特徴量 | F0 | 正解 |
|---|---|---|---|
| 追加する組 | **naru の台詞を、同じタイミングでユーザーの声で言った音声** の特徴量 | naru の元の F0 | naru の元の音声 |

2 つの音声は内容もタイミングも同じで、声だけが違う。だからモデルは「ユーザーの声の特徴量 → naru の声」の
差を直接学習できる。特徴量に残っている「声の低さ・男性らしさ」の手がかりは F0 入力と食い違うので、
それを無視することも同時に学習する。

「同じ台詞を同じタイミングでユーザーが言った音声」は、**ユーザーの声の RVC モデルで naru の学習スライスを
変換して作る**（ピッチ −12.5 半音前後）。RVC の出力は入力と同じ長さなので、元の特徴量とフレーム単位で揃う。

学習時は、サンプルごとに確率 `srcaug_prob`（0.3〜0.5 を想定）で、元の特徴量の代わりにこの特徴量を使う。
F0・正解音声・index は今と同じ。

### やらないこと

- **embedder を学習し直さない。** SPIN（Praat の性別変換 + EQ で摂動）は特徴量からスペクトルの細部を落とし、
  本学習すると声がこもった（CLAUDE.md の SPIN 節、`logs/_spin/RESULTS.md` 16）。ピッチやフォルマントを変えた音声を
  embedder に追加学習させるのは同じ方向なので、ここではやらない。
- **index に変換後の特徴量を入れない。** index の値は naru 本人の特徴量のまま。
- **推論時の F0 は変わらない。** 今後もピッチ +12.5 前後を指定する。「ピッチ 0 で女性の声が出る」ようにはならない。
  F0 は特徴量とは別の入力で、出力の音高を直接決めるため。

### 期待する効果（未測定）

- 発音がはっきりする（PER が下がる）
- ユーザーの声質が残りにくくなる（naru 方向へのシフトが大きくなる）
- 同じシフト量を、より低い index_rate で得られる → index による雑音・不連続が減る

---

## 2. 全体の流れと分担

| 手順 | 内容 | 誰が |
|---|---|---|
| 0 | ユーザーの声のモデルを作る（`my_voice/train`、kushinada、48k。`my_voice/heldout` は学習に使わない） | ユーザー（GUI） |
| 1 | `tools/srcaug/build_source_features.py` を実装する | Claude Code |
| 2 | `srcaug_prob` を学習側に実装する（data_utils、GPU キャッシュ、config、CLI、Training タブ、テスト） | Claude Code |
| 3 | `tools/srcaug/clone_folder.py` を実装する（A/B 用にモデルフォルダを複製） | Claude Code |
| 4 | 手順 1 のスクリプトで、naru フォルダの変換音声と特徴量を作る | ユーザー（コマンド 1 本） |
| 5 | 変換音声を聴き、ユーザーの声と判定されるか測る | ユーザー + Claude Code |
| 6 | 対照群と srcaug 群を同条件で追加学習する | ユーザー（GUI） |
| 7 | 聴き比べと PER・シフト量の比較 | ユーザー + Claude Code |

---

## 3. 実装 1: `tools/srcaug/build_source_features.py`

naru の学習スライスをユーザーモデルで変換し、naru フォルダの embedder 設定で特徴量を取り、
`logs/<target>/extracted_srcaug/` に保存する。変換と抽出を 1 本のスクリプトで行い、中間の WAV は
指定したときだけ残す。

```
env314\Scripts\python.exe -X utf8 tools\srcaug\build_source_features.py ^
    --target <naru のモデルフォルダ名> ^
    --source-pth logs\<ユーザーモデル>\<model>.pth ^
    --source-index logs\<ユーザーモデル>\<model>.index ^
    --pitch -12.5 --index-rate 0.5 --f0-method hpa-rmvpe-112000-aligned ^
    --keep-audio 40
```

### 入力

- `logs/<target>/sliced_audios/<name>.wav`（モデルのサンプルレート、preprocess 済み）。
  `sliced_audios_16k` ではなくこちらを使う（音質が良いため。変換後に 16 kHz へ落とす）。
- `logs/<target>/model_info.json` の embedder 設定（`embedder_model`、`embedder_output_layer`、
  `embedder_feature_scale`、`embedder_input_std_floor`）。**記録がないフォルダは拒否する**
  （何で抽出したか分からない特徴量と混ぜないため）。
- `logs/<target>/extracted/<name>.npy`: フレーム数の基準。

### 変換

- 既存の推論経路（`rvc/infer/pipeline.py` の `Pipeline`、または `VoiceConverter`）を Python から直接呼ぶ。
  バッチ推論タブ / `core.py batch_infer` は使わない（ファイル名の対応付けと後処理のオフを確実にするため）。
- **ピッチは小数で渡す。** `convert_audio` / `run_batch_infer_script` の `pitch` は `int` 型なので、−12.5 を
  渡せる経路を使う。`Pipeline.get_f0` は `f0 *= pow(2, pitch / 12)` なので小数でそのまま動くはずだが、確認すること。
- 後処理（clean_audio、autotune、formant、エフェクト、split_audio）はすべてオフ。`volume_envelope = 1`、
  `protect = 0.5`（= protect なし）。乱数シードは固定し、manifest に記録する。
- F0 手法は引数で指定する（既定は `hpa-rmvpe-112000-aligned`。gap-filled 版は使わない。息継ぎに音高が付くため。
  CLAUDE.md の F0 Methods 節と `docs/f0-benchmarks.md` 5.8 を参照）。
  オプションで `--reuse-target-f0` を検討してよい: `logs/<target>/f0_voiced/<name>.wav.npy` を 2^(pitch/12) 倍して
  そのまま使い、F0 の再抽出をしない（naru の F0 と完全に整合する）。`Pipeline` に外部 F0 を渡す口が必要なら、
  推論経路には影響しない形で追加すること。
- index_rate の既定は 0.5。ユーザーモデルの index が強すぎると、変換音声がユーザーの声の継ぎはぎになり、
  naru の発音のタイミングから外れやすい。

### 特徴量の抽出

- 変換音声（ユーザーモデルのサンプルレート）を 16 kHz にリサンプルする。
  preprocess と同じ `soxr_vhq` を使う（`PreProcess` の 16k 版の作り方に合わせる）。
- 抽出は `extract.py` と同じ経路を通す: `load_embedding(..., output_layer)` → `embedder_forward`
  （入力正規化・std floor・feature scale を含む）。自前で HuBERT を呼ばない。
- **フレーム数を `extracted/<name>.npy` に合わせる。** 差が 2 フレーム以内なら、末尾を切るか最後のフレームを繰り返して
  揃える。それ以上ずれたファイルは保存せずに一覧に出す。差の分布を manifest に記録する。
- NaN を含む結果は保存しない（`process_file_embedding` と同じ）。

### 出力

- `logs/<target>/extracted_srcaug/<name>.npy`: `extracted/` と同じファイル名・同じ形状・float32。
  **`extracted/` の中には置かない。** `extract_index.py` は `extracted/` の全ファイルを index に入れるため。
- `logs/<target>/extracted_srcaug/manifest.json`: ユーザーモデルのパスと sha256、ピッチ、F0 手法、index_rate、
  protect、シード、embedder 設定（target の `model_info.json` から写したもの）、件数（成功・スキップ・失敗）、
  フレーム差の分布、作成日時、スクリプトの git commit。
- `--keep-audio N`: 先頭 N 件（名前順）の変換音声を `logs/<target>/srcaug_audio/` に WAV で残す（聴く用、測る用）。
  0 なら残さない。`--keep-audio all` で全件。
- 既にあるファイルはスキップし（再開できるように）、`--overwrite` で作り直す。manifest の設定と今回の引数が
  違う場合は、`--overwrite` なしでは止める。

### その他

- GPU 1 枚で逐次処理でよい（1,400 ファイル程度を想定）。embedder とユーザーモデルは 1 回だけ読み込む。
- 進捗は tqdm。最後に要約（件数、フレーム差、所要時間、出力先）を出す。

---

## 4. 実装 2: 学習時に混ぜる（`srcaug_prob`）

### 設定

- `logs/<model>/config.json` の `train.srcaug_prob`（float、0〜1、既定 0 = オフ）。
- CLI `--srcaug_prob`、Training タブの詳細設定にスライダー（0〜1、刻み 0.05）。i18n の en_US / ja_JA キーも追加する。
  スライダーは常に表示し、`extracted_srcaug/` がないフォルダで 0 より大きい値を指定したら、学習開始前に分かる
  エラーで止める。
- **`TRAIN_SETTING_KEYS` には入れない。** `c_reg` / `c_sd` / `cqt_lr_scale` と同じ理由
  （`read_train_settings` はすべてのキーが揃っているときしか config の値を返さないので、キーを足すと既存フォルダが
  全部 stock config にフォールバックして間違った learning_rate を表示する）。
- 学習の再開をまたいで値を変えてよい（構造が変わらないので `architecture_identity` には入れない）。
  ただし使った値は TensorBoard とログに残す。

### データの読み込み（`rvc/train/data_utils.py`）

- `TextAudioLoaderMultiNSFsid` が、元の特徴量のパスから対応する srcaug のパスを決める。
  **filelist.txt のパスからではなく、学習フォルダ（`experiment_dir`）とファイル名から決める:**
  `os.path.join(experiment_dir, "extracted_srcaug", os.path.basename(phone_path))`。
  filelist.txt は絶対パスなので、フォルダを複製すると元フォルダを指したままになるため。
- mute 行（`mute.npy`）は対象外。srcaug ファイルがないサンプルは元の特徴量を使い、件数を数えて開始時に出す。
- `get_labels` で、確率 `srcaug_prob` で srcaug を読む。長さはスクリプト側で揃えてあるが、念のため以降の
  長さ合わせ（`len_phone != len_spec` の処理）はそのまま通す。
- 乱数: DataLoader のワーカーごとにシードが違うことを確認する（torch の乱数を使えば DataLoader が
  ワーカーごとにシードを設定する。numpy の乱数を使う場合はワーカーごとにシードし直す必要がある）。
  学習全体のシードから再現できること。

### GPU キャッシュ（`GPUSampleCache`）との関係: **ここが落とし穴**

`cache_data_in_gpu` が有効だと、`GPUSampleCache.fill` は各サンプルを 1 回だけ読み込んで GPU に置き続ける。
読み込み時に srcaug を選ぶと、**最初に引いた方が学習中ずっと固定される**（CLAUDE.md の
"The GPU data cache holds samples, not batches" と同じ種類の問題）。

対策: `srcaug_prob > 0` のときは、キャッシュに両方の特徴量（元と srcaug）を持たせ、`batches()` で
バッチを組むたびにサンプルごとに選ぶ。特徴量の分だけキャッシュの VRAM が増えるので、増分を開始時に出す。
キャッシュなしの経路とキャッシュありの経路で、同じシードなら選ばれ方の分布が同じであることをテストする。

### 開始時のチェックとログ

- `extracted_srcaug/manifest.json` の embedder 設定が、フォルダの `model_info.json` と一致しなければ止める
  （フォルダを別の embedder で抽出し直した後の古い srcaug を混ぜないため。`resolve_feature_reuse` が
  `extracted/` と index を作り直したときに srcaug は消えないので、ここで止める）。
- 開始時に `srcaug: prob=0.40, 1366/1366 files, manifest pitch=-12.5 f0=... source=...` を出す。
- TensorBoard に `train/srcaug_ratio`（エポックごとに実際に srcaug を使った割合）を出す。
- 評価セット（`rvc/train/evaluation.py`）は変えない。評価は常に実際の音声から抽出した特徴量で行う。

### テスト（`tests/test_srcaug.py`）

- `srcaug_prob = 0` で、データセットの出力が実装前とビット単位で同じ
- `srcaug_prob = 1` で、mute 以外はすべて srcaug、mute 行は元のまま
- srcaug ファイルがないサンプルは元にフォールバックし、数えられる
- GPU キャッシュあり / なしで、選択がエポックごとに変わり、割合が prob に近い
- manifest の embedder 設定が一致しないと止まる
- フォルダを複製した後も、複製先の `extracted_srcaug/` が使われる
- `build_source_features.py` のフレーム合わせ（±2 以内は揃える、超えたら保存しない）

---

## 5. 実装 3: `tools/srcaug/clone_folder.py`

A/B（対照群と srcaug 群）のために、今の naru フォルダを複製する。今のモデルの root G/D を上書きしないためでもある。

```
env314\Scripts\python.exe -X utf8 tools\srcaug\clone_folder.py <元フォルダ名> <新フォルダ名>
```

- 複製するもの: `sliced_audios/`、`sliced_audios_16k/`、`f0/`、`f0_voiced/`、`extracted/`、`extracted_srcaug/`（あれば）、
  `mute.npy`、`config.json`、`model_info.json`、root の `G_*.pth` / `D_*.pth`、index 一式（key index と
  `.kv.json` / `.values.npy` を含む）。
- 複製しないもの: `checkpoints/`、`eval/` 系の出力、TensorBoard のログ、エクスポート済みの `.pth`（必要なら引数で）。
- `filelist.txt` のパスを新フォルダに書き換える（`generate_filelist` で作り直してもよい。ただし shuffle の結果は変わる）。
- 容量が大きいので、可能ならハードリンクを使う（同じドライブなら）。書き込みが起きるファイル（G/D、config.json、
  filelist.txt）は必ず実体をコピーする。

---

## 6. 実行手順（ユーザー向け）

### 手順 0: ユーザーの声のモデル

- データ: `my_voice/train`。`my_voice/heldout` は入れない（手順 7 の評価に使うため）。
- embedder は kushinada（naru と同じでなくても動くが、揃えておくのが無難）。48k、vocoder は普段のもの。
- 出来の確認: `my_voice/heldout` を自分のモデルで変換して、自分の声に聞こえること。

### 手順 4: srcaug 特徴量を作る

上のコマンドを naru のフォルダに対して実行する。`--keep-audio 40` で 40 件の変換音声が残る。

### 手順 5: 変換音声の確認（学習の前に必ず）

- `logs/<naru>/srcaug_audio/` を聴き、「naru の台詞をユーザーが言っている」ように聞こえるか確かめる。
  naru の声質が残っていたり、崩れていたりしたら、手順 0 のモデルか `--index-rate` を見直す。
- 数値: `tools/spin/measure_speaker.py` を変換音声のフォルダに当てる（今は `my_voice/heldout` 固定なので、
  `--query-dir` のような引数を Claude Code に足してもらう）。kushinada L24 で、変換音声がユーザー（temp4 話者 59）に
  判定される割合が高いほどよい。参考までに、本物の held-out は L22 で 100%。

### 手順 6: 追加学習（対照群つき）

追加学習そのものでも音は変わるので、**srcaug なしで同じだけ追加学習した対照群**と比べないと効果が分からない。

1. `clone_folder.py` で 2 つ複製する: `<naru>_ctrl` と `<naru>_srcaug`。
2. 両方とも Training タブで「重みを引き継いで学習をリセット」にチェックを入れて開始する（今の重みから epoch 1）。
3. 設定は両方同じにし、`srcaug_prob` だけ変える: ctrl は 0、srcaug は 0.4。
   - 学習率は普段の追加学習と同じか少し低め、エポック数は 200〜400 程度から。
   - GPU キャッシュを使う場合、srcaug 側だけ VRAM が増える。
4. TensorBoard では `kl` と `train/srcaug_ratio` を見る。srcaug 側の `kl` が ctrl より高いのは想定内
   （入力が難しくなっているため）。mel が大きく崩れ続けるなら prob を下げる。

### 手順 7: 評価

- `my_voice/heldout` を ctrl と srcaug のモデルで、**普段と同じ設定**（ピッチ +12.5、同じ index、
  `hpa-rmvpe-112000-aligned-filled-150ms`、同じシード）で変換し、ブラインドで聴き比べる
  （`logs/_layercmp/listen/` と同じやり方）。
- 数値は `logs/_spin/scripts/rt_rate_sweep.py` で、index_rate 0.4〜1.0 を振って PER と naru 方向へのシフトを
  両群で出す。期待どおりなら、srcaug 側は同じ index_rate でシフトが大きく、同じシフトを低い index_rate で出せる。
  通常 index と key index の両方で測る。
- リアルタイム（1080 ms ブロック）でも確認する。モデル構造は同じなので、処理時間は変わらないはず。

### 結果の記録

結果は `logs/_spin/RESULTS.md` に新しい節として追記し、CLAUDE.md の SPIN 節の "Not tried yet" を更新する。

---

## 7. うまくいかないときに疑う順

1. **変換音声の質。** ユーザーモデルの音が崩れていれば、naru モデルはその崩れを入力として学習してしまう。
   手順 5 をやり直す。prob を 0.2〜0.3 に下げる。
2. **index_rate（ユーザーモデル側）。** 高すぎると変換音声がユーザーの声の継ぎはぎになり、タイミングと発音がずれる。
3. **F0。** gap-filled の F0 で変換していないか。息継ぎに音高が付いた変換音声は、naru の息継ぎと噛み合わない。
4. **キャッシュ。** `train/srcaug_ratio` が prob から大きく外れていないか（選択が固定されていないか）。
5. **追加学習の長さ。** 短すぎると差が出ず、長すぎると対照群も変わる。両群を同じエポックで比べる。

## 8. この計画の発展（実装しない。効果が出たら検討）

- 摂動の種類を増やす: Praat の性別変換（`tools/spin/perturb.py`）で naru 音声を男性側に下げたものも混ぜる。
  ユーザーモデルを使わずに作れるので安いが、特定のソース（ユーザー）には寄らない。
- 変換元を増やす: ユーザーの声だけでなく、temp4 の他の男性話者モデルでも作る（ユーザー以外が使う場合の頑健性）。
- 特徴量の写像（`docs/ideas/m2f-conversion-ideas.md` の案 2）: この計画で作る (srcaug, 元) の特徴量の組は
  フレーム単位で揃ったパラレルデータなので、線形写像の学習にそのまま使える。
