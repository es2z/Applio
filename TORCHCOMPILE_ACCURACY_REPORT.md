# torch.compile は音質を落とすのか — 実測レポート

作成: 2026-09-13 / 対象ブランチ: `feature_expand_tourch_compillation`

## 背景

リアルタイム推論で TorchCompile のチェック（CREPE / Embedder / RVC）を増やすほど
「音質が落ちた」「声がかなり変わる」と感じられた。本当に劣化しているのか、
それとも別の理由で音が変わっているのかを実測した。

**結論を先に**: 音は確かに変わる。ただし **精度は落ちていない。むしろコンパイル側の方が
わずかに高精度**。変化の正体はモデル自身が持つ乱数と混沌性であり、
その変化量は「アプリを再起動して乱数が変わったとき」より小さい。

## 測定環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| torch / CUDA / triton | 2.13.0+cu132 / 13.2 / 3.7.1 |
| Python | 3.11.16 |
| モデル | `naru_20260906_kushinada_hubert_large_test1_2150e_894400s.pth` |
| Embedder | `kushinada-hubert-large` (1024-dim, fp32) |
| F0 | `mangio-crepe-full-speech` |
| テンプレート | `00000_少し低めlittleAlterなし_..._2150.yaml` (chunk 960ms / crossfade 0.138 / extra 0.22) |
| 入力 | 実音声 44 秒 (`benchmarks/results/torchcompile_20260912/input_48k.npy`) |

`torch.backends.cudnn.allow_tf32 = True`（eager の畳み込みは TF32）、
`torch.backends.cuda.matmul.allow_tf32 = False`、`float32_matmul_precision = "highest"`。

---

## 1. 精度: コンパイル側の方が正確

Embedder の出力を **fp64 CPU** で計算した値を正解とし、そこからの相対誤差を測った。

| 実行方法 | fp64 基準の相対誤差 |
|---|---|
| eager fp32 CUDA | 2.216e-04 |
| compiled `default` | 2.183e-04 |
| compiled `reduce-overhead` | 2.183e-04 |
| compiled `max-autotune` | **2.176e-04** |

eager と compiled の差は 6.5e-05（コサイン類似度 1.0000）で、
**どちらも正解から離れている量 (2.2e-04) より小さい**。
つまり精度が落ちたのではなく、丸め方が変わって別の等価解に落ちているだけ。
順位をつけるなら compiled の方が正しい。

---

## 2. 出力音声の差 — 同一入力・同一 seed・44 秒

まず対照実験として、**同じ設定・同じ seed で 2 回レンダリングした結果は bit-identical**
（SNR 218.9 dB）だった。よってこのパイプラインは seed を固定すれば完全に決定的であり、
以下の差はすべて本物である。

| 条件 | mel 差 (中央値) | MFCC 相対差 | 備考 |
|---|---|---|---|
| 同設定・同 seed | 0.000 dB | 0.0000 | 対照（bit-identical） |
| CREPE のみコンパイル | 0.000 dB | 0.0012 | ほぼ差なし |
| CREPE + Embedder | 0.870 dB | 0.0291 | |
| CREPE + Embedder + RVC | 1.135 dB | 0.0323 | |
| **コンパイルなし・seed のみ変更** | **1.188 dB** | **0.0351** | ← 最大 |

チェックを増やすほど差が増えるという体感は正しい。
しかし **3 つ全部入れた差より、乱数の seed を変えただけの差の方が大きい**。

（波形 SNR は SOLA のアライメントが数サンプル動くため負の値になり指標にならない。
上表はフレーム単位のスペクトル距離で、ピークから 40 dB 以内のフレームのみを対象にしている。）

---

## 3. なぜ 1e-5 の差が声の違いになるのか

1. `Synthesizer.infer` は毎チャンク `z_p = m_p + exp(logs_p)·randn·0.66666` を引く
   (`rvc/lib/algorithm/synthesizers.py:228`)。
   HiFi-GAN の source module も位相 `torch.rand` とノイズ `randn_like` を引く
   (`rvc/lib/algorithm/generators/hifigan.py:189,223`)。
   → **このモデルは元から実行ごとに違う音を出す。**
2. encoder → flow → decoder が混沌的で、1e-5 の入力差が出力では相関を失うまで増幅される。
3. RVC をコンパイルすると乱数の実装が inductor 側に替わるため、実質「別 seed」になる。

FAISS index 検索の増幅を疑って `index_rate = 0` でも測ったが差は変わらなかった
（-2.48 dB → -3.16 dB）。**index ではなくモデル本体**が増幅源。

なお CUDA グラフによる乱数の固着（毎チャンク同じノイズが出る）も確認したが、
3 モードすべてで `FROZEN_RNG = False`、1 回ごとのばらつき幅も eager と同一 (0.0025) だった。

---

## 4. CREPE の F0

単一チャンク（133 フレーム）での eager 対 compiled の差:

| mode | max\|Δf0\| | RMS | voicing 反転 |
|---|---|---|---|
| default | 7.75 Hz | 9.02 cents | 0 / 133 |
| reduce-overhead | 5.82 Hz | 7.95 cents | 0 / 133 |
| max-autotune | 6.51 Hz | 8.38 cents | 0 / 133 |

viterbi デコードのビン境界（20 cents 刻み）が稀に反転するため。
ただし 44 秒通しでの出力ピッチへの影響は **0.05 cents** で、聴感上は無関係。

---

## 5. 速度 — チェックごとの実効果

`benchmarks/benchmark_realtime_compile.py --all` の 1 チャンクあたり中央値（3 ラウンド）:

| ケース | 中央値 | none 比 |
|---|---|---|
| none | 108.2 ms | — |
| **crepe** | 107.4 ms | **-0.7 ms** |
| embedder | 99.8 ms | -8.4 ms |
| rvc | 95.9 ms | -12.2 ms |
| crepe + embedder | 100.5 ms | -7.7 ms |
| crepe + rvc | 94.7 ms | -13.4 ms |
| embedder + rvc | 86.3 ms | -21.9 ms |
| all (`reduce-overhead`) | 85.2 ms | -22.9 ms |
| all (`default`) | 92.8 ms | -15.3 ms |
| all (`max-autotune`) | 85.1 ms | -23.0 ms |

- **CREPE のチェックは音も速度もほとんど変えない**（-0.7 ms）。
  `setup_torch_compile_cache()` がプロセス全体で `TORCHINDUCTOR_CUDAGRAPH_TREES=0` /
  `triton.cudagraphs=False` を立てており、かつ torchcrepe 側は `dynamic=True` で
  コンパイルされるため、CUDA グラフの恩恵を受けられない。
- 速度を稼いでいるのは **Embedder と RVC**。
- `default` モードは CUDA グラフを使わないぶん他の 2 モードより 7〜8 ms 遅い。

---

## 6. 実務上の結論

- 「コンパイルで音質が劣化する」は **誤り**。精度指標ではコンパイル側が上。
- 「チェックを増やすと音が変わる」は **正しい**。ただし変化量は再起動 1 回分と同程度。
- 音を動かしたくない → Embedder / RVC のチェックを外す（-23 ms を失う）。
- CREPE のチェックは損得がほぼ無いので任意。

## 7. 付随して判明した不具合（修正済み）

`reset_torchcrepe_compiled_model()` が `torchcrepe.core.infer.model` だけを `None` にし、
リロード判定に使われる `infer.capacity` を残していたため、
**TorchCompile の設定を一度でも変更すると次回のリアルタイム開始が必ず失敗**していた
(`AttributeError: 'NoneType' object has no attribute 'to'`)。
`infer.capacity` の無効化に変更し、不要かつ有害だった `torch._dynamo.reset()` を削除。
リグレッションテストを `tests/test_realtime_compile_session.py` に追加済み。

## 再現方法

```bat
env\python.exe -X utf8 benchmarks\benchmark_realtime_compile.py --all --out benchmarks/results/<name>
env\python.exe -m unittest discover -s tests -v
```

---

# 追補: 学習側に torch.compile を入れるとどうなるか

「コンパイルで精度が上がるなら、学習時にも使えば整合が取れて音が良くなるのでは」という
仮説と、「速くなるなら入れたい」という要望を受けて測定した。測定環境は本編と同じ。

## 結論

- **音質目的では意味がない。** eager と compiled の差 (6.5e-05) は、どちらも fp64 の
  正解から離れている量 (2.2e-04) より小さい。学習時に compiled を使っても特徴量が
  「正解」に近づくわけではなく、モデルは与えられた特徴量からの写像を学ぶだけなので、
  推論時 1e-4 の摂動は学習時の注入ノイズより 4 桁小さい。
- **速度目的では、学習ステップは効かない。抽出は条件付きで効く。**

## 学習ステップ (net_g / net_d) — 見送り

batch 8 / segment 17280 / phone_len 112-144 / mode=default / MSVC 環境あり:

| 条件 | ms/step | 倍率 | peak VRAM |
|---|---|---|---|
| eager | 242.3 | 1.00 | 6.38 GiB |
| compile net_d のみ | 243.8 | 0.99 | 5.70 GiB |
| compile net_g のみ | 233.8 | 1.03 | 5.57 GiB |
| compile 両方 | 234.2 | **1.03** | 4.84 GiB |
| eager + TF32 matmul | 244.1 | 0.99 | 6.38 GiB |
| compile 両方 + TF32 | 233.6 | 1.04 | 4.84 GiB |
| eager 再測定（対照） | 242.4 | **1.00** | 6.38 GiB |

対照の eager 再測定が 1.00 に戻っているので測定自体は健全。**利得は 3%**。

実装しない理由:
1. 3% に対し、コンパイルに 60〜240 秒かかる。
2. inductor の C++ ラッパに **MSVC (`cl.exe`) が PATH 上に必要**。この環境では
   Visual Studio 2022 は入っているが PATH には無く、素の状態では
   `InductorError: InvalidCxxCompiler: Compiler: cl is not found` で失敗する。
3. `torch.compile(net_g)` は `OptimizedModule` を返すため、`state_dict()` のキーに
   `_orig_mod.` が付き、保存されるチェックポイントの互換性が壊れる。DDP や
   gradient checkpointing との組み合わせも要検証になる。

なお TF32 matmul が効かないのは、このモデルが畳み込み律速で、かつ cuDNN が既に
畳み込みを TF32 で回している (`torch.backends.cudnn.conv.fp32_precision = "tf32"`) ため。

**副産物**: コンパイルすると peak VRAM が 6.38 → 4.84 GiB (**-24%**) になる。
速度ではなくバッチサイズを上げたい場合には意味がある数字。

## 抽出 (extract) — 実装した。ただし既定はオフ

1 ファイル当たり（60 クリップ / 223 秒、eager をコンパイル前後の両方で測り、
各 3 回の最小値。ウォームアップ偏りを速度向上と誤認しないための A-B-A 測定）:

| 対象 | eager | compiled (default) | 倍率 |
|---|---|---|---|
| Embedder (kushinada-hubert-large) | 10.4 ms | 7.8 ms | **x1.32** |
| RMVPE | 24.3 ms | 18.1 ms | **x1.33** |
| FCPE | 2.6 ms | 1.7 ms | **x1.52** |
| CREPE / mangio-crepe | 既存の「Enable TorchCompile (CREPE)」で対応済み | | |

> 初回に 2.1x / 1.69x / 2.38x という値を得たが、これは eager を先に測っていたための
> ウォームアップ偏りだった。また FCPE は `torch.compile(module)` が返す
> `OptimizedModule` の `.infer()` が元のメソッドに委譲されるため、実際には
> コンパイルされていなかった。どちらも上表で修正済み。

**ところが end-to-end では遅くなる。** 200 クリップ (709 秒) で実測:

| | 1 回目 | 2 回目 |
|---|---|---|
| compile オフ | 27 s | 27 s |
| compile オン | 61 s | 62 s |

トレースに 1 ワーカープロセス・1 ステージあたり約 17 秒かかり、inductor のキャッシュでも
消えない（2 回目も 62 秒）。ファイル当たりの節約は約 8.8 ms なので、
**損益分岐は約 4000 クリップ（およそ 4 時間分のデータセット）**。
参考までに本レポートのモデルのデータセットは 1364 クリップ (41 分) なので、
この規模では**オフのままが速い**。

そのため実装はしたが既定はオフで、UI にも分岐点を明記してある。

## 出力の一致

抽出を compile オン/オフで実行し、生成された `.npy` を比較（16 ファイル）:

| | 最大相対誤差 | 最大絶対差 |
|---|---|---|
| `extracted/` (embedder 特徴量) | 2.10e-04 | 0.00225 |
| `f0/` (RMVPE) | **0.0** | **0.0** |

F0 は cents ビンに量子化されるためビット一致。embedder 特徴量の差は fp32 の丸め相当で、
本編で示したとおり eager と compiled のどちらも fp64 からは 2.2e-04 離れている。

## 追加した設定

`assets/config.json` の `training_compile_extraction`（既定 `false`）と、
TorchCompile Settings 内のチェックボックス
「Enable TorchCompile for Extraction (Training)」。モードは既存の
`torch_compile_mode` を共有する。コンパイルに失敗しても eager にフォールバックするだけで、
抽出は止まらない。

---

# 追補 2: 「意図した音が出なくなる / 余韻が変わる / 音量が上がる」の調査

体感の説明が「音質低下」ではなく「余韻がそっけなくなる」「音量が大きくなりやすい」
「声がかなり変わる」だったため、無音区間の処理と実行間の再現性を測り直した。

## 結論

コンパイルは原因ではない。原因は独立した 3 つの実装上の問題だった。

| # | 問題 | 状態 |
|---|---|---|
| 1 | VAD と Silence Threshold がどこからも参照されず、入力ゲートが存在しなかった | 修正済み |
| 2 | `webrtcvad` の使い方が壊れており、室内騒音を「音声」と判定していた | 修正済み |
| 3 | `mangio-crepe` の F0 が**非決定的**で、同一音声でも最大 25.9 cents ぶれる | 未対応（要判断） |

## 1. 入力ゲートが存在しなかった

`rvc/realtime/core.py` で `is_input_silent` を計算していたが、**この変数はどこからも
読まれていなかった**。つまり `vad_enabled` も `silent_threshold` も完全に無効で、
室内騒音も常に変換され、`audio_model * sqrt(入力RMS)` で増幅されていた。

修正では `is_input_silent` を実際に使う。ただし最初の無音ブロックで切ると余韻を
途中で断ち切るので、**変換ウィンドウ全体が無音になるまで待つ**
(`silence_blocks_to_stop`、このテンプレートでは 2 ブロック = 1.92 秒)。
また 0 dBFS 以上のしきい値は「ゲートなし」として扱う（`10**(0/20) = 1.0` は
あらゆる入力を無音と判定してしまうため）。

## 2. webrtcvad の使い方

実測（0.96 秒ブロック、-50 dBFS の室内騒音）:

| 検出器の状態 | 室内騒音の音声フレーム数 | 実音声 |
|---|---|---|
| セッションを通して使い回し | **27〜31 / 32** | 25〜32 / 32 |
| ブロックごとに新規生成 | 3 / 32 | 14〜28 / 32 |
| 新規生成 + 先頭3フレーム破棄 | **0 / 29** | 2〜29 / 29 |

webrtcvad は聞いた音に適応するため、大きな音声と室内騒音が交互に来るストリームでは
室内騒音を音声と判定するようになる。さらにデジタル無音に対しても、実音声の直後は
「音声」を返す（フレッシュなら False、実音声通過後は True, True, False）。

修正では検出器をブロックごとに作り直し、**先頭 3 フレーム（90 ms）は判定に使わない**。
これで室内騒音 0 / 29、実音声 2 / 29 以上と完全に分離するため、
「1 フレームでも音声なら音声」という元の判定を保ったまま（＝発話の立ち上がりを
取りこぼさないまま）室内騒音をゲートできる。

### 検証（実音声3秒 + -50 dBFS 室内騒音3秒 × 8、VAD 有効、しきい値 -90 のまま）

- ミュートされたブロック: 11 / 19 の室内騒音ブロック
- **音声ブロックのミュート: 0**
- 音声区間の RMS: -48.96 → -48.95 dBFS（不変）
- ミュートされなかった室内騒音ブロックは発話直後のもの＝余韻がまだウィンドウ内にある区間

Silence Threshold のスライダ上限も -60 → -20 dB に広げた（既定値 -90 は変更なし）。
現実の室内騒音は -50 dBFS 前後なので、従来の範囲では届かなかった。

## 3. mangio-crepe の F0 が非決定的（未対応）

同一プロセス・同一入力で 5 回連続実行した結果:

| F0 手法 | 再現性 | 最大ずれ |
|---|---|---|
| rmvpe | **ビット一致** | 0 |
| fcpe | **ビット一致** | 0 |
| crepe-full | **ビット一致** | 0 |
| mangio-crepe-full-speech | ✗ | 3.60 Hz / **25.92 cents** |
| mangio-crepe-full | ✗ | 4.34 Hz / **26.70 cents** |

原因はデコーダ:

| decoder | 再現性 |
|---|---|
| `viterbi`（mangio-crepe の既定） | ✗ |
| `argmax` | ✗ |
| `weighted_argmax`（`CREPE` クラスが明示指定） | **ビット一致** |

CUDA の `argmax` は同値のタイブレークが非決定的で、CREPE の確率分布に同値が出ると
選ばれるビンが変わる。ビン間隔は 20 cents なので、1 ビンずれるとそのまま
20〜26 cents のピッチ変動になる。

**これが「意図した音が出なくなる」の最有力候補**。ピッチが実行ごとに 1/4 半音ぶれれば、
声の印象も余韻の聞こえ方も変わる。パイプライン全体としても、
このぶれが実行間の差の主因だった（シードを固定しても残差 0.75 dB が残る）。

対処の選択肢:
1. `MANGIO_CREPE` のデコーダを `weighted_argmax` にする（`CREPE` クラスは既にそうしている）。
   決定的になるが F0 の値自体が変わるので音も変わる。
2. F0 手法を `crepe-full` / `rmvpe` / `fcpe` に変える。いずれもビット一致。
3. 現状維持。

## 4. リアルタイムの固定 RNG シード（実装済み）

`Synthesizer.infer` と HiFi-GAN の source module が毎チャンク乱数を引くため、
同じ音声でもセッションごとに声が変わる。`realtime_seed`（`assets/config.json`、
Realtime タブの "RNG Seed"、-1 で従来どおりランダム）で固定できる。

シードはウォームアップの後に適用する。ウォームアップはコンパイル有無に関わらず
常に 1 回以上走るようにした（遅延生成される F0 モデルの重み初期化が乱数を消費するため、
それより後にシードを当てないとセッション 2 でずれる）。

効果の実測（セッション間の mel 中央値差）:

| | 事前状態が同じ | 事前状態が違う |
|---|---|---|
| シード -1（ランダム） | 0.824 dB | 1.154 dB |
| シード 4242 | 0.754 dB | **0.790 dB** |

シードは事前状態への依存を消すが、**残差 0.75 dB は消えない**。その残差の正体が
上の mangio-crepe の非決定性である。

---

# 追補 3: Mangio-CREPE デコーダを選択可能にした

追補 2 §3 で見つけた非決定性について、既定を変えずに選べるようにした。

## 設定

`assets/config.json` の `mangio_crepe_decoder`（既定 `viterbi` = 従来どおり）。
実装は `rvc/lib/predictors/crepe_decoder.py`、UI は `tabs/components.py` の共通部品で、
**mangio-crepe 系が選ばれている間だけ「Pitch extraction algorithm」の直下に表示される**。
配置先は Realtime / Inference / Batch Inference / TTS / Training(Extract) /
Extra の F0 Curve の 6 か所。設定はグローバルなので、どこで変えても全タブに効く
（表示中の値は f0 手法を切り替えたタイミングで読み直す）。

## 実測（RTX 4090、同一入力を同一プロセスで 5 回連続）

| decoder | 再現性 | 最大ずれ | リアルタイム処理時間 |
|---|---|---|---|
| `viterbi`（既定） | ✗ | 4.02 Hz / 23.53 cents | 107〜109 ms/block |
| `weighted_argmax` | **ビット一致** | 0 | **84〜86 ms/block** |
| `argmax` | ✗ | 4.27 Hz / 30.86 cents | 76〜77 ms/block |

処理時間は順序を入れ替えても同じ傾向だったので、ウォームアップ偏りではない。
`viterbi` は librosa の Viterbi デコードを CPU で回すため、**`weighted_argmax` に
すると再現性が得られるうえに 1 ブロックあたり約 24 ms 軽くなる**。

## 音がどれだけ変わるか（同一入力・同一シード、`viterbi` 基準）

| decoder | mel 中央値差 | MFCC 相対差 |
|---|---|---|
| `weighted_argmax` | 1.043 dB | 0.0334 |
| `argmax` | 0.869 dB | 0.0319 |

参考: 乱数シードを変えただけの差が 1.188 dB / 0.0351。つまり
**デコーダを変えたときの音の変化は、シードを変えた程度**で、劇的ではない。

試聴用: `scratchpad/decoder/{viterbi,weighted_argmax,argmax}.wav`

## 既定を変えていない理由

`weighted_argmax` は再現性でも速度でも有利だが、F0 の推定方法そのものが変わるので
音は変わる。既存のテンプレートで作り込んだ音を黙って変えないよう、既定は `viterbi` の
まま、選択肢として出すに留めた。

---

# 追補 4: 学習時に mangio / viterbi を使ってよいか

## 前提: 抽出は1回きりで凍結される

F0 の非決定性は「実行するたびに変わる」性質だが、学習では `extract` が一度走って
`logs/<model>/f0*` に `.npy` として保存され、以降のエポックはその同じファイルを読む。
**エポック間でラベルが揺れることはないので、収束が不安定になる形の害は起きない。**
問題になり得るのは (a) ラベル自体の質、(b) 学習時と推論時の系統差、の 2 つ。

## (a) ラベルの質 — viterbi が最も滑らか

学習データ 40 クリップ（125 秒）で測定。jitter は有声区間内の隣接フレーム間
|cents| の中央値で、小さいほど滑らかなピッチ軌跡。

| 抽出手法 | 有声率 | jitter (cents) |
|---|---|---|
| **mangio viterbi** | 80.6% | **12.23 / 12.01**（2回実行） |
| rmvpe | 76.7% | 15.38 |
| crepe-full-speech | 98.4% | 18.82 |
| mangio weighted_argmax | 97.9% | 18.97 |

Viterbi は時間的連続性を前提に経路を選ぶデコーダなので、当然ながら最も滑らかな軌跡を出す。
**ピッチラベルとしての質では viterbi が一番良い。**

有声率も見るべき点で、`weighted_argmax` と `crepe-full-speech` は **ほぼ全フレームを
有声と判定している**（98%）。NSF 音源は無声フレームでは倍音ではなく雑音を出すので、
98% 有声は無声子音にも倍音を当てることを意味する。
「推論では mangio の方が良い」という体感は、ここが効いている可能性が高い。

## (b) 非決定性のコストは品質ではなく再現性

同じデータを viterbi で 2 回抽出した差:

| | 値 |
|---|---|
| 連続 F0 の差（中央値 / p90） | 5.64 / 14.11 cents |
| 有声判定の不一致 | **0.00%** |
| coarse ビン（`pitch`）が変わるフレーム | 25.98% |

有声判定はぶれない。coarse ビンが 26% 変わるのは、294.8 Hz 付近で 1 ビン = 26.4 cents
しかなく、5.6 cents の摂動でも境界付近のフレームは移るため。
ただしこれは**手法を変えたときの差より小さい**（mangio vs rmvpe は 49% が別ビン）。
つまり「再抽出すると別の等価なラベルになる」だけで、劣化ではない。

## (c) 本当に効くのは学習と推論の一致

手法間の系統差（有声共通フレームの中央値 / p90、有声判定の不一致）:

| 比較 | 中央値 | p90 | 有声判定の不一致 |
|---|---|---|---|
| mangio viterbi vs 同 2 回目 | 5.64 | 14.11 | **0.00%** |
| mangio viterbi vs mangio weighted_argmax | 5.49 | 16.78 | **17.34%** |
| mangio viterbi vs crepe-full-speech | 9.35 | 36.50 | 17.87% |
| mangio viterbi vs rmvpe | 11.11 | 45.73 | 14.15% |

**デコーダを変えると有声/無声マスクが 17% 変わる。** これは cents のずれよりずっと大きな
条件変化で、RVC では `pitchf == 0` が NSF 音源の励振を倍音から雑音に切り替えるスイッチに
なっている。

### 既存モデルが何で抽出されたかの特定

`logs/naru_20260906_kushinada_hubert_large_test1/f0_voiced/*.npy`（25 ファイル）を
その場で再抽出した各手法と照合:

| 候補 | 完全一致フレーム | 中央値 cents | 有声判定一致 |
|---|---|---|---|
| **mangio viterbi** | **18.29%** | **5.51** | **100.00%** |
| fcpe | 13.81% | 10.47 | 89.88% |
| rmvpe | 13.51% | 10.05 | 88.73% |
| mangio weighted_argmax | 3.01% | 6.08 | 84.72% |
| crepe-full-speech | 2.24% | 8.89 | 83.95% |

有声判定が **100% 一致**するのは mangio viterbi だけ。中央値 5.51 cents というずれは、
上で測った viterbi の実行間ばらつき（5.64 cents）そのもの。
**このモデルは mangio-crepe-full-speech + viterbi で抽出されており、現在の推論設定と
一致している。**

## 結論

- **学習で mangio / viterbi を避ける理由はない。** 収束は不安定にならず（ラベルは凍結される）、
  ラベルの滑らかさはむしろ全手法中で最良。
- **推論も学習と同じ viterbi のままが安全。** 決定性目当てに推論だけ `weighted_argmax` に
  変えると、有声マスクが 17% 学習時とずれる。
- **決定性が欲しいなら、抽出と推論の両方を `weighted_argmax` に揃えて学習し直す**のが筋。
  ただし有声率が 98% になる副作用込みで評価すること。
- 非決定性が実際に痛いのは「同じデータセットを再抽出しても同じラベルにならない」点だけで、
  実験の再現性の問題であって品質の問題ではない。
