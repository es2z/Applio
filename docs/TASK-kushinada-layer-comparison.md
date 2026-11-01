# 夜間タスク: kushinada の出力層 16 / 18 / 20 / 22 / 24 の聴き比べ用モデルを作る

Claude Code への指示書。2026-10-26 に作成。ユーザーは寝ているので、途中で質問はできない。
判断に迷ったら「この文書の意図（下の目的）に一番近い、安全な方」を選び、何をどう判断したかを
最後の `RESULTS.md` に書く。応答・報告は日本語。

## 目的

M5（kushinada の出力層を最上層 24 より下にする）の効果を、**耳で聴き比べられる状態にする**。
話者性を消す追加学習（M4 / SPIN）は**やらない**。ユーザーの声で何かを学習することもしない。

各層の特徴量で naru の APEX-GAN+ モデルを同じ条件で短く学習し直し、同じ入力を変換して並べる。
どの層が良いかを決めるのはユーザーの耳で、このタスクは判断材料（音声と数字）を揃えるだけ。

## 守ること

- **元のフォルダーは読むだけ**: `logs/naru_20261015_48k_APEXPlus_1100_rmvpe112ali_kushinada`
  （以下「元フォルダー」）には何も書かない・消さない・移さない。
- **コードは変更しない**（`rvc/`, `tabs/`, `core.py`, `tools/` など）。必要な一時スクリプトは
  `logs/_layercmp/scripts/` に置く。
- 作業はすべて `logs/_layercmp/` と、下で作る 5 つのモデルフォルダーの中で行う。
- git の操作（commit / checkout / branch 切替など）はしない。今のブランチ（`apex-gan-plus-plus`）のまま。
- GPU を使うのは 1 度に 1 つの処理だけ（学習を並列にしない）。
- Python は `C:\dev\myapplio2\env\python.exe`（Python 3.11 の環境）を使い、すべてのコマンドの前に
  PowerShell で `$env:PYTHONUTF8='1'` を設定する。作業ディレクトリは `C:\dev\myapplio2`。
- あるコマンドが失敗したら、原因が運用上のもの（パス、引数、空き容量など）なら直して再実行してよい。
  コードの不具合が原因と思われる場合は直さずに、その層を飛ばして次へ進み、`RESULTS.md` に
  エラー全文と状況を書く。
- 自動モードの判定などでコマンドが拒否されたら、回避しようとせず、その手順を飛ばして記録する。

## 前提（2026-10-26 に確認済みの事実）

- 元フォルダー: 48 kHz、APEX-GAN+、embedder `kushinada-hubert-large`（出力層は既定＝24）、
  F0 は `hpa-rmvpe-112000-aligned`、F0 coarse 1100 Hz、1 話者、1335 クリップ（36 分 35 秒）、
  filelist は 1338 行（うち mute が 3 行）、82 step/epoch（batch 16）、約 33 秒/epoch。
- 学習率は `8e-5 × 0.9992^(epoch-1)`。c_mel は 750e から 36、1550e から 32、1800e から 26。
- 重みだけの G/D が `元フォルダー\checkpoints\every\G_1000e.pth` と `D_1000e.pth` にある
  （1000e、その時点の学習率 3.6e-5、c_mel 36）。ここから始める。
- 空き容量は約 42 GB。1 フォルダーあたり約 3.2 GB 使う見込み（合計 16 GB 程度）。

## 共通の設定（5 つの学習すべてで同じにする）

| 項目 | 値 |
|---|---|
| ボコーダー（レシピ） | `APEX-GAN+` |
| 始点 G / D | 元フォルダーの `checkpoints\every\G_1000e.pth` / `D_1000e.pth`（層 24 だけは下の手順 3 で作るもの） |
| batch size | 16 |
| total epoch | 100（約 55 分） |
| save every epoch | 25（書き出しモデルが 25 / 50 / 75 / 100e にできる） |
| learning rate / lr decay | 8e-5 / 0.9992 |
| c_mel | 36 |
| save_only_latest | True（G/D の保存は最新 1 つだけ。容量のため） |
| save_every_weights | True（書き出しモデルは全部残す） |
| cache_data_in_gpu | True |
| その他 | 既定値のまま（LR ブースト無し、overtraining detector 無し、checkpointing 無し） |

学習率を元の 1000e の値（3.6e-5）ではなく 8e-5 にするのは、層を変えると `enc_p.emb_phone`
（embedder の特徴量を読む唯一の重み）が初期化されるため。100 epoch で学び直せるように、
naru の元の run の開始時と同じ値にしている。

## 層とフォルダー名（この順に実行する）

| 順 | 層 | `--embedder_output_layer` | モデル名（`logs/` 以下のフォルダー名） |
|---|---|---|---|
| 1 | 24（対照） | 0 | `layercmp_L24_naru_APEXPlus` |
| 2 | 18 | 18 | `layercmp_L18_naru_APEXPlus` |
| 3 | 20 | 20 | `layercmp_L20_naru_APEXPlus` |
| 4 | 22 | 22 | `layercmp_L22_naru_APEXPlus` |
| 5 | 16 | 16 | `layercmp_L16_naru_APEXPlus` |

24 は今の設定と同じ特徴量（`0` は「最後の層」の意味）。朝までに全部終わらない場合でも、
対照と中間の層が先に揃うように、この順にする。

## 手順

### 0. 準備

1. Applio や他の GPU を使うアプリが動いていないことを確認する（`nvidia-smi` で他のプロセスが
   大きなメモリを使っていないこと）。動いていたら止めずに、`RESULTS.md` に書いて終了する。
2. `logs/_layercmp/`, `logs/_layercmp/scripts/`, `logs/_layercmp/inputs/`,
   `logs/_layercmp/listen/` を作る（既にあれば中身はそのまま使う）。
3. 変換の入力を決める:
   - `logs/_layercmp/inputs/` に wav があれば（ユーザーが寝る前に入れた自分の声の録音）、それを使う。
     ピッチは **+12.7**（ユーザーの realtime 設定）。
   - **30 秒より長いファイルは、そのまま使わずに抜粋を作る**（数分の録音 1 本だけ置かれることがある）。
     元のファイルは変更せず、抜粋は `logs/_layercmp/inputs_cut/` に置いて、それを入力にする:
     1. 読み込みは `librosa.load(path, sr=None, mono=True)`（サンプリングレートはそのまま）。
     2. `librosa.effects.split(y, top_db=40)` で無音でない区間を出し、隣り合う区間を順につないで
        10〜20 秒のまとまりを作る。区切りは必ず無音の位置（区間の切れ目）にする。区間の前後は
        0.1 秒ずつ余白を残す。
     3. ファイル全体に散らばるよう、先頭・1/3・2/3・終わり付近から 1 つずつ、合計 4 本まで選ぶ
        （ファイルが短ければ取れるだけ）。
     4. 名前は `<元のファイル名>_cut1.wav` 〜 `_cut4.wav`。`soundfile.write` で元と同じサンプリング
        レートの 16-bit PCM で書く。どこを切り出したか（開始・終了の秒数）を `RESULTS.md` に書く。
   - 30 秒以下のファイルはそのまま使う。
   - 加えて、必ず `logs/reference/reference.wav` も使う。誰の声か分からないので、ピッチ **0** と
     **+12.7** の 2 通りで変換する。

### 1. 各層のフォルダーを作る（層ごとに 1〜5 を繰り返す）

元フォルダーから、学習の出力を除いたデータだけをコピーする。

```powershell
$src = 'C:\dev\myapplio2\logs\naru_20261015_48k_APEXPlus_1100_rmvpe112ali_kushinada'
$dst = 'C:\dev\myapplio2\logs\layercmp_L18_naru_APEXPlus'   # 層ごとに変える
foreach ($d in 'sliced_audios','sliced_audios_16k','f0','f0_voiced') {
    robocopy "$src\$d" "$dst\$d" /E /NFL /NDL /NJH /NJS /NP | Out-Null
}
Copy-Item "$src\config.json","$src\model_info.json" $dst
```

コピーしないもの: `extracted`, `*.pth`, `*.index`, `checkpoints`, `eval`, `events.out.tfevents.*`,
`mute.npy`, `filelist.txt`（抽出で作り直される）。

コピーした `config.json` から `process_pids` と `checkpoint_policy` のキーを消す（中身は Python で
JSON を読み書きして消す。他のキーは変えない）。

### 2. 特徴量を抽出する

```powershell
$env:PYTHONUTF8='1'
.\env\python.exe core.py extract --model_name layercmp_L18_naru_APEXPlus `
  --f0_method hpa-rmvpe-112000-aligned --f0_coarse_max 1100 --sample_rate 48000 `
  --embedder_model kushinada-hubert-large --embedder_output_layer 18 `
  --include_mutes 3 --gpu 0
```

（層 24 は `--embedder_output_layer 0`。）

確認すること:
- `logs\layercmp_..\model_info.json` の `embedder_output_layer` が、層 24 では `null`、
  他の層ではその番号になっていること。
- `filelist.txt` が 1338 行で、パスが**新しいフォルダー**を指していること。
- F0 はコピーしたものが再利用されるはず（再抽出されても問題はない。ログに書いておく）。

### 3. 層 24（対照）だけ: emb_phone を初期化した G を作る

他の層は embedder の記録（出力層）が違うので、学習開始時に `enc_p.emb_phone.weight` だけが
自動で初期化される。層 24 は記録が同じなので、そのままだと引き継がれてしまい、比較が不公平になる。
同じ条件にするため、層 24 用に emb_phone.weight だけを初期化した G を作る（bias はそのまま。
自動の初期化と同じ扱い）。

```python
# logs/_layercmp/scripts/reset_emb_phone.py
import torch
src = r"C:\dev\myapplio2\logs\naru_20261015_48k_APEXPlus_1100_rmvpe112ali_kushinada\checkpoints\every\G_1000e.pth"
dst = r"C:\dev\myapplio2\logs\_layercmp\G_1000e_embphone_reset.pth"
c = torch.load(src, map_location="cpu", weights_only=True)
key = "enc_p.emb_phone.weight"
w = c["model"][key]
torch.manual_seed(0)
c["model"][key] = torch.nn.Linear(w.shape[1], w.shape[0]).weight.detach().to(w.dtype).clone()
torch.save(c, dst)
print("reset", key, tuple(w.shape))
```

層 24 の学習では `--g_pretrained_path` にこのファイルを使う（D は他と同じ `D_1000e.pth`）。

### 4. 学習する

```powershell
$env:PYTHONUTF8='1'
$every = 'C:\dev\myapplio2\logs\naru_20261015_48k_APEXPlus_1100_rmvpe112ali_kushinada\checkpoints\every'
.\env\python.exe core.py train --model_name layercmp_L18_naru_APEXPlus `
  --vocoder APEX-GAN+ --sample_rate 48000 --batch_size 16 --total_epoch 100 --save_every_epoch 25 `
  --pretrained True --custom_pretrained True `
  --g_pretrained_path "$every\G_1000e.pth" --d_pretrained_path "$every\D_1000e.pth" `
  --learning_rate 8e-5 --lr_decay 0.9992 --c_mel 36 `
  --save_only_latest True --save_every_weights True --cache_data_in_gpu True --gpu 0
```

（層 24 は `--g_pretrained_path "C:\dev\myapplio2\logs\_layercmp\G_1000e_embphone_reset.pth"`。）

ログの出力はファイルに残す（例: `logs/_layercmp/train_L18.log`）。確認すること:
- `Warm start (G)` の要約で、層 24 以外は `enc_p.emb_phone.weight` が「初期化（from scratch）」、
  残りは全部引き継ぎになっていること。層 24 は全部引き継ぎ（初期化済みの値を引き継ぐ）になること。
- `Warm start (D)` が 100% であること（同じ APEX-GAN+ の判別器）。
- 書き出しモデル `layercmp_L18_naru_APEXPlus_{25,50,75,100}e_*.pth` ができていること。
- 学習中に `nan` が出続けていないこと（最初の数 step の fp16 スキップは APEX-GAN では正常）。

### 5. index を作る

```powershell
.\env\python.exe core.py index --model_name layercmp_L18_naru_APEXPlus
```

### 6. 変換する（5 つの学習が終わってから、まとめて）

各層の **100e** の書き出しモデルで、手順 0 で決めた入力を変換する。比較の参照として、元フォルダーの
`…_1000e_82000s.pth` と `…_1100e_90200s.pth`（元の index）も同じ入力で変換する。

共通の変換設定はユーザーの realtime テンプレートと同じ。**`core.py infer` の `--pitch` は整数しか
受け付けない（12.7 が渡せない）**ので、CLI ではなく Python から `core.run_infer_script` を呼ぶ
（内部の計算は `pow(2, pitch / 12)` なので小数がそのまま効く）。毎回 `torch.manual_seed(0)` を
してから呼び、全モデルで同じ乱数にする。

```python
# logs/_layercmp/scripts/convert.py（作業ディレクトリ C:\dev\myapplio2、PYTHONUTF8=1 で実行）
import contextlib, io, os, sys, torch
sys.path.insert(0, os.getcwd())
import core

def convert(input_path, output_path, pth_path, index_path, pitch, index_rate):
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        core.run_infer_script(
            pitch=pitch, index_rate=index_rate, volume_envelope=1.0, protect=0.0,
            f0_method="hpa-rmvpe-112000-aligned-filled",
            input_path=os.path.abspath(input_path), output_path=os.path.abspath(output_path),
            pth_path=os.path.abspath(pth_path), index_path=os.path.abspath(index_path),
            split_audio=False, f0_autotune=False, f0_autotune_strength=1.0,
            proposed_pitch=False, proposed_pitch_threshold=155.0,
            clean_audio=False, clean_strength=0.5, export_format="WAV",
            embedder_model="kushinada-hubert-large",
        )
    assert os.path.isfile(output_path), output_path
```

- `index_rate` は **0 と 0.6 の 2 通り**作る。0 は特徴量そのものの差が出やすく、0.6 は普段の設定。
- 出力層は書き出しモデルに記録されていて、変換時に自動で使われる。念のため、各書き出しモデルの
  `embedder_output_layer`（`torch.load(..., weights_only=True)` で読める）が層の番号
  （24 は `None`）になっていることを確認して `RESULTS.md` に書く。
- 出力の置き場所と名前:
  - `logs/_layercmp/listen/named/` に、中身が分かる名前で置く
    （例: `reference_p12.7_idx0.6__L18.wav`、参照は `__orig1000e` / `__orig1100e`）。
  - `logs/_layercmp/listen/blind/` に同じファイルを、入力・ピッチ・index ごとにまとめた上で、
    層をランダムな記号（A, B, C …、入力ごとに並びを変える）に置き換えた名前でコピーし、
    対応表を `logs/_layercmp/listen/blind_key.csv` に書く。

### 7. 数字をまとめる

```powershell
.\env\python.exe -X utf8 tools\training_status.py layercmp_L24 layercmp_L18 layercmp_L20 layercmp_L22 layercmp_L16
```

の出力をそのまま `RESULTS.md` に貼る。見方のメモも書く:
- **kl**（低いほど、その層の特徴量から音の潜在表現をよく予測できている）が、層の比較で一番意味のある
  数字。mel は特徴量を通らない経路（事後分布）で計算されるので、層による差は小さいはず。
- fm / adv は判別器が同じなので比べてよいが、音の良さとは直接対応しない。

## 最後に書くもの: `logs/_layercmp/RESULTS.md`

- 何をどの順にやったか、各ステップの所要時間、開始・終了時刻。
- 各層の `Warm start (G)` / `(D)` の要約（emb_phone が意図どおり初期化されたか）。
- `training_status.py` の表と、層ごとの 25 / 50 / 75 / 100e の kl・mel の短い表。
- 変換したファイルの一覧（`listen/named/` と `listen/blind/` の場所）。`blind_key.csv` は開かずに
  聴いてほしい、と一言。
- エラー、飛ばした手順、判断したこと（あれば）。
- 空き容量の残り。

## やらないこと

- SPIN（M4）や、ユーザーの声を使った学習。
- 元フォルダーへの書き込み、コードの変更、git 操作。
- 結果から「どの層が良い」と結論を書くこと（数字の傾向を書くのはよいが、判断はユーザーの耳で行う）。
