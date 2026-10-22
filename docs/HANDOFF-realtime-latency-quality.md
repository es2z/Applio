# 引き継ぎ: リアルタイム変換の遅延・音質改善（ブランチ `realtime-latency-quality`）

次の Claude（ローカルの Claude Code）向けの引き継ぎです。2026-10-22 時点。
これと `CLAUDE.md`、`docs/realtime-latency-quality.md`（各オプションの説明と A/B の手順）を
読めば、ここまでの経緯を知らなくても次の判断ができるように書いています。

---

## 0. 最初に守ること

1. **応答は日本語。** コード、コードのコメント、コミットメッセージ、識別子は英語（`CLAUDE.md`）。
2. **既定の動作を変えない。** この作業の変更は、すべて「初期値 = 従来の動作」のオプトインです。
   全オプションが初期値のとき、出力は main（`5ad003c`）と**ビット単位で一致**しています。
   リアルタイム経路を触ったら、`tools/realtime_e2e_compare.py` で必ず確認すること（5 章）。
3. **音色は耳で決める。** ユーザーは音色の悪化を最も警戒しています。数値（seam、mel 距離）は
   「どこを聴くか」の手がかりで、良し悪しの判定には使わない。推測で「良くなる」と言わない。
4. **ユーザーの設定・好み**（会話の userPreferences より）:
   - 回答は日本語。調べ物は英語の情報源を優先（日本固有の話題でない限り日本語サイトは見ない）。
   - 過度に同調せず中立に。
   - 完成形のコンポーネントを求められたら、社会通念上必要な追加機能（削除時の確認など）を
     コメントアウト＋注釈付きで含める。
   - より良い代替手段があれば、本回答の後に二次提案として添える。ただしユーザーのスキル
     （JavaScript、C#、VBA、React、WPF、Python）から大きく外れる言語は勧めない。
     ミドルウェアやライブラリ（ffmpeg など）の提案は歓迎。
   - React の質問には JavaScript で答える。
   - ユーザーが書いたコメントをむやみに削除・変更しない。
   - ファイルの変更を頼まれたら、指示が無い限り完全なファイルを渡す。
   - ユーザーはプログラマ経験 5〜10 年。前提知識の説明は短くてよい。

---

## 1. ユーザーの環境と前提

- Windows。`env\python.exe -X utf8` で動かす（utf8 が無いと torch.compile が cp932 で落ちる）。
  Python 3.13、torch 2.13 + cu132（`requirements.txt`）。GPU は RTX 4090（CLAUDE.md の実測環境）。
- 使っているモデル: **APEX-GAN+（48 kHz）**、embedder **kushinada-hubert-large（1024 次元）**、
  F0 は学習時 **HPA-RMVPE 112000 aligned**、推論時 **…-aligned-filled（穴埋めあり）**。
  torch.compile 有効。リアルタイム時の GPU 負荷は約 10%。
- リアルタイムのテンプレート: **Chunk 960 ms / Crossfade 0.138 s / Extra 0.22 s**、移調 **+12.3 半音**。
  従来の総遅延は 2 秒超（内訳は 3 章）。
- 入力は WASAPI。出力は **WDM-KS** にしていた。理由: 出力を WASAPI にすると必ず遅延が約 1 バッファ
  増えたため。ring 転送（段階 2）で入出力とも WASAPI にする前提で進めている。
- 入力のノイズ除去は**外部で実施済み**。Applio の後ろに **VST のディエッサとリミッタ**を入れている。
- 新しい録音を用意するのは難しい。事前学習（pretrain）は何度も回している。
- 判断は「ローカルの Claude Code と一緒に、聴きながら確認する」と本人が言っている。

---

## 2. ユーザーの指示（時系列、要点）

1. Applio フォーク（es2z/Applio、main。master は無い）を調べ、音質（最優先）・遅延・アプリ動作の
   改善案をレポートにまとめる。学習とリアルタイムのみ。外部ソフトによるデータ加工の提案は可。
   → claude.ai の Docs「Applio改造版 改善提案レポート」（https://claude.ai/artifact/3KEZtBbyUR7gGbdvXrR7qR）。
   ローカルからは読めない前提で、項目 ID の意味は 7 章にまとめた。
2. 「以前破棄したブランチで WASAPI→WASAPI を WASAPI→WDM-KS 並みにできた。盛り込めるか」
   → `new_main_testversion`（`29eae3d`）の転送層だけを移植すると回答。
   **破棄した理由はユーザーから未回答**（必要なら聞く）。
3. 「段階 1 と WASAPI の遅延減少を行い、各段階でコミット。可能ならリングバッファ化、ドリフト補正、
   乱数・位相・F0 を時刻に固定も。**IO1 は不要**（ノイズ除去は行っている、音色が悪化しそうで怖い）。
   **P4 は今までのものも使えるように**して比べられるように。**基本的に音色が悪化する可能性を
   危惧している**」→ 全部オプトインにした理由。
4. 「ブランチ切って実行して。結果はローカルの Claude Code を交えながら聴きながら確認する」
5. 最新: 残りのロードマップ図表と質問への回答（Docs「リアルタイム変換 残りのロードマップと
   追加の回答」https://claude.ai/artifact/TiXtbJ32xmk8aASq81WY29）、
   **IO1〜IO5 で音質に悪影響の可能性があるものは選択式（初期値オフ）に**、この引き継ぎ文書。

---

## 3. 遅延の内訳（基準テンプレート、計測と見積もり）

| | 従来 | trim | trim + ring |
|---|---|---|---|
| ブロックを集める | 960 | 960 | 960 |
| ウィンドウ遅延（実測、恒等モデル） | 370 | 150 | 150 |
| 出力側の待ち | 約 960 | 約 960 | 処理時間 p95 + 10 ms 程度 |
| 合計（目安） | 約 2.3 s | 約 2.1 s | 約 1.2 s |

- 従来の窓は「Extra + Crossfade + SOLA + Block」で、**最も古い部分を再生**していたので Extra が
  先読みとして遅延に乗っていた（370 = 220 + 138 + 10、10 ms 単位に丸め）。
- 出力側の約 1 チャンクは、従来の転送（`rvc/realtime/audio.py`）が blocksize = Chunk で
  コールバック内で変換するため。WDM-KS 出力にしていたのはこの一部を隠すため。
- 実機の遅延はまだ誰も測っていない（ring のステータス `est. total` は推定値）。

---

## 4. 実装済み（コミット順）

すべてブランチ `realtime-latency-quality`。UI は Realtime タブ > Performance Settings >
「遅延・音質の実験的オプション（初期値 = 従来の動作）」。Start 時に適用、テンプレートの
`experimentTab` に保存。オプションの実体は `rvc/realtime/options.py` の `RealtimeOptions`。

| コミット | 段階 | 内容 |
|---|---|---|
| `5b8cf9b` | 1 | 出力位置 trim・先読み（L1・L4）、入力の連続リサンプル（S5 = IO3）、Protect の有声判定（P3）、出力ゲインの時刻合わせ（P6）、APEX-GAN ノイズ包絡 window/fixed（P4）、オフライン A/B ツール（S0）、有声率ツール |
| `474bc1c` | 2 | ring 転送とドリフト補正（L2・L5・A1、A4 の一部） |
| `d076ba7` | 3 | ノイズ・正弦波の位相・F0 を時刻に固定（S2・S3・S4） |
| `ac3ec1b` | 4 | 入力ハイパス（IO1）、クリップ位置（IO4）、ゲートのフェードと保持（IO5）、潜在ノイズの温度（P1）、レベル表示 |
| （この文書のコミット） | — | 引き継ぎ文書、`tools/realtime_e2e_compare.py` |

### オプション一覧（`FIELD_ORDER` の順。UI の `option_controls` と同じ順でなければならない）

| フィールド | 初期値 | 意味 |
|---|---|---|
| `output_position` | legacy | trim: 最新の「Block + Crossfade + SOLA + 先読み」だけ生成・再生（flow の余白 24 フレーム） |
| `lookahead_s` | 0.0 | trim 時の先読み。遅延に 1:1 で加算 |
| `input_resample` | stateless | stateful: 48→16 kHz を連続ストリームで（遅延 24 サンプル）。FCN 系 F0 では無効 |
| `protect_voicing` | f0 | unfilled: Protect に穴埋め前の有声判定を使う（Protect < 0.5 のときだけ意味がある） |
| `gain_mode` | legacy | aligned: sqrt(RMS) ゲインを掛ける時刻の周辺で測る |
| `apex_noise_envelope` | window | fixed: 学習バッチ相当の固定値でノイズ包絡を正規化（APEX-GAN のみ） |
| `apex_noise_voiced_ratio` | 0.65 | fixed の基準になる学習データの有声率（`tools/apex_noise_voiced_ratio.py`） |
| `transport` | legacy | ring: 小さいデバイスバッファ＋リングバッファ＋変換スレッド |
| `ring_stream_mode` | auto | separate: 常に 2 ストリーム＋ドリフト補正（入出力が別機器なら推奨） |
| `time_locked_noise` | False | flow と APEX-GAN のノイズを時刻に固定 |
| `phase_continuity` | False | APEX-GAN の正弦波の位相をブロック間で連続に |
| `f0_cache` | False | 「1 ブロック + マージン」より古い F0 を保持 |
| `f0_cache_margin_ms` | 100 | 上のマージン |
| `latent_noise_scale` | 0.66666 | P1。スライダーの 0.67 は初期値扱い |
| `input_highpass_hz` | 0 | IO1。48 = 学習時の Noise filter と同じ（5 次バターワース） |
| `clip_stage` | generator | IO4。output: 生成器の出力は切らず、最終出力だけ ±1 |
| `gate_fade_ms` | 0 | IO5。無音ゲートが閉じるとき、続くはずの音をフェードアウト |
| `gate_hold` | False | IO5。無音中にゲートが周期的に開く既存の挙動（6 章）を止める |

時刻固定の 3 つは **Chunk が 40 ms の倍数**のときだけ有効（960 は対象）。

### 主なファイル

- `rvc/realtime/options.py` オプション。**フィールドは末尾に追加するだけ**（UI の並び、テンプレートの互換）。
- `rvc/realtime/core.py` 窓の配置、stateful 入力、aligned ゲイン、クリップ位置、ゲートのフェード・保持、
  `algorithmic_delay_ms`、`level_status()`。
- `rvc/realtime/pipeline.py` `configure_stream`、時刻固定の状態、Protect の有声判定、生成器の
  ピーク計測（`last_peak`）。
- `rvc/realtime/synthesis.py` trim・ノイズ指定・位相指定・包絡指定・潜在ノイズ倍率付きの生成器呼び出し
  （オプションが全部初期値なら従来どおり `Synthesizer.infer`）。
- `rvc/lib/algorithm/generators/apex_gan.py` 包絡の固定参照値、与えられたノイズと位相での生成。
  学習とオフライン変換の経路は変更なし（APEX-GAN のテストは通過）。
- `rvc/realtime/ring_transport.py`, `ring_buffer.py` ring 転送。
- `rvc/realtime/resample.py` stateful リサンプラ。`rvc/realtime/io_processing.py` ハイパス、フェード、レベル表示。
- `rvc/realtime/warmup.py` ウォームアップ（ring 用に 5 ブロックの時間計測も、RNG を消費しない形で）。
- `rvc/realtime/stream_eval.py`, `tools/realtime_stream_eval.py` オフライン A/B（デバイス不要）。
- `tabs/realtime/realtime.py`, `tabs/realtime/template.py`、i18n は `assets/i18n/languages/{ja_JA,en_US}.json`。
- テスト: `tests/test_realtime_options.py`, `test_realtime_synthesis.py`, `test_realtime_ring_transport.py`,
  `test_realtime_io.py`（既存の `test_realtime_silence_and_seed.py`, `test_realtime_compile_session.py` も通過）。

---

## 5. 確認のしかた

### テスト（CPU で動く、GPU・重み不要）

```
env\python.exe -m pytest -q tests\test_realtime_options.py tests\test_realtime_synthesis.py tests\test_realtime_ring_transport.py tests\test_realtime_io.py tests\test_realtime_silence_and_seed.py tests\test_realtime_compile_session.py
env\python.exe -m pytest -q tests\test_apex_gan.py
```

クラウド側では 65 件 + 10 subtests と APEX-GAN の 52 件が通過（Linux、CPU）。`test_apex_gan.py` は
メモリを食うので他と同時に回さない。

### 初期値の動作がビット単位で同じかの確認

```
git worktree add ..\applio-base 5ad003c
env\python.exe tools\realtime_e2e_compare.py run --repo ..\applio-base --out base.npy
env\python.exe tools\realtime_e2e_compare.py run --repo . --out new.npy
env\python.exe tools\realtime_e2e_compare.py compare base.npy new.npy
git worktree remove ..\applio-base
```

ランダム重みの APEX-GAN、固定の偽 embedder、偽 F0 で VoiceChanger を CPU で回す。
`--options '{"output_position":"trim"}'` でオプション付きの動作確認（こちらは一致しなくてよい）。

### 実際の音で比べる（オフライン）

```
env\python.exe -X utf8 tools\realtime_stream_eval.py --template <テンプレート名> --input voice.wav --preset legacy --preset stage1 --preset locked --offline
```

`logs/stream_eval/<日時>/` に WAV と `results.json`（ウィンドウ遅延、処理時間、seam、mel_vs_offline）。
任意の組み合わせは `--preset "name={...JSON...}"`。手順の詳細は `docs/realtime-latency-quality.md`。

---

## 6. 判断の記録（なぜそうしたか）

- **全部オプトイン、初期値 = 従来の動作。** ユーザーが音色の悪化を最も恐れているため。
  比べられること自体が要件（P4 は明示的に「今までのものも使えるように」）。
- **trim（L1）**: RVC WebUI と同じ配置（`flow_head = max(head - 24, 0)`）。trim のときだけ、Index の
  ブレンド範囲と RMS ミックスの参照区間を生成区間に合わせた。legacy では従来の（Index が再生区間の
  頭 220 ms に混ざらない）挙動を残している。trim は再生部分が見られる未来の音声が Crossfade 分
  だけになるので、劣化があれば先読みで補う。
- **P4（APEX-GAN のノイズ包絡）**: 学習はバッチ全体の平均で割るが、リアルタイムは窓 1 つで、
  gaps-filled F0 では窓全体が有声 → 有声部のノイズが学習時の 3〜6 倍。fixed は
  `per_unit_mag × (r × 0.003 + (1 − r) × 0.1/3)` を基準にする。今まで聴いてきたのは window の音で、
  fixed が聴感上良いかは未確認。
- **P3**: gaps-filled では Protect が何もしない。これは穴埋めの意図どおりの動作（バグではない）なので、
  既定は変えずに選択肢にした。
- **ring 転送**: `new_main_testversion` の転送層を移植。ただし出力予備の制御は書き直した。
  元の実装は、アンダーフローで空いた時間がそのまま遅延に加わっても、それを管理していなかった。
  今は「出力リングの量 + 処理時間」を p95 + 予備（10 ms〜）に合わせ、ずれの補正は
  **-60 dBFS 未満の静かな出力でだけ**行う（声には触れない）。リングの量は再生中のデバイス周期の
  残りまで足して測る（周期単位の量子化があるとドリフト補正が振動したため、シミュレーションで確認）。
  ドリフト補正は separate モードだけで、可変レートの soxr（HQ、21 kHz まで平坦）を PI 制御
  （±1000 ppm）。チャンネル数は従来の転送と同じ開き方。ring 中は Python のスレッド切替間隔を 1 ms に
  する（コールバックが GIL を待たないように、停止時に戻す）。
- **時刻固定（S2〜S4）**: 正弦波の位相は float64 で累積して 2^-24 周期に丸める。float32 だと
  1e-6 の差でデコーダ出力が約 5% 変わったため（APEX-GAN は正弦波の STFT の位相だけを使い、
  エネルギーの小さいビンで丸め誤差が効く）。時刻固定の乱数は RNG シードの後に引き直すので、
  シード固定で再現する。F0 キャッシュは、保持した F0 が「マージン分の未来しか見ていない」点に
  注意（legacy の配置では不利になるので trim と組み合わせる。gaps-filled の穴埋めもその時点の値で固定）。
- **段階 4**:
  - Applio 内のリミッタ・ディエッサは**入れない**。pedalboard の Limiter はしきい値未満でもゲインを
    上げて圧縮する（実測: しきい値 -1 dB で -20 dBFS の正弦波が +4.8 dB）。ユーザーは VST を使っている。
  - 代わりに、VST では直せない「生成器の ±1 クリップが √ゲインの前に起きる」点を `clip_stage` で
    選べるようにし、生成器と出力のピークを表示する。
  - **IO2（ノイズ抑制）は実装しない。** 外部で除去済みで、重ねると子音が削れやすい。DeepFilterNet3 の
    `deepfilterlib` は Python 3.13 用のビルド済みが無い（pip ではソース配布のみ、Rust ビルドが必要）。
  - **見つけた既存の挙動**: `Realtime.flush_buffers()` が `consecutive_input_silence` も 0 に戻すので、
    部屋の雑音が続くとゲートが周期的に開き、変換した雑音が短く出ては切れる（240 ms・-40 dB の
    シミュレーションで「閉じる 3・開く 3」の繰り返し）。既定は変えず `gate_hold` で止められるようにした。
  - 出力レベル: 出力 = 生成器の出力（±1 で切る）× √(入力 RMS) なので、Output Gain 100% なら
    出力のピークは √(入力 RMS) を超えない（入力 RMS -20 dBFS なら -10 dBFS 以下）。
    ユーザーの「リミッタが効くほど出ていない」という認識はおそらく正しい。

---

## 7. レポートの項目 ID（何を指すか、状態）

| ID | 内容 | 状態 |
|---|---|---|
| L1 / L4 | 出力位置の修正 / 先読みの独立 | 済（段階 1） |
| L2 / L5 | リングバッファ化 / 別ストリームのドリフト補正 | 済（段階 2） |
| L3 | ブロックを 960 → 480 → 240 ms に短縮 | 未（設定だけで試せる。前提は済） |
| L6 | デバイスと OS の設定（同じホスト API、NVIDIA の電源管理を最大パフォーマンス） | 未（設定） |
| S0 | ストリーミングによる劣化を測る仕組み | 済（`tools/realtime_stream_eval.py`） |
| S1 | 過去の文脈を約 2.5 s に（trim なら Extra を増やしても遅延は増えない） | 未（設定のみ） |
| S2 / S3 / S4 | ノイズ / 位相 / F0 を時刻に固定 | 済（段階 3） |
| S5 | 48→16 kHz の状態付き変換（= IO3） | 済（段階 1） |
| S6 | embedder の入力正規化（窓ごとの平均 0・分散 1）を安定させる | 未（S1 でかなり揃う見込み） |
| S7 | 位相ボコーダ式のクロスフェード（本家 `c1a3782`） | 未（S2〜S4 があるので優先度低） |
| S8 / T7 | リアルタイムと同じ条件で作った特徴量で微調整 | 未（中〜大） |
| P1 | z_p の温度 | 済（段階 4、`latent_noise_scale`） |
| P2 | Index 検索の GPU 化と重み付け（今の `(1/score)²` はほぼ最近傍 1 本） | 未 |
| P3 / P4 / P6 | Protect / APEX ノイズ包絡 / ゲイン | 済（段階 1） |
| P5 | F0 の後処理（移調後の範囲確認、短いオクターブ跳びの補正） | 未 |
| M1 | APEX-GAN+ に CQT 判別器（RingFormer+ の `cqt_discriminator.py` を流用） | 未（学習 +16%・VRAM +2.2 GiB 見込み、推論・.pth 変化なし） |
| M2 | Snake をエイリアシング対策付きに（BigVGAN 式） | 未（試作の CPU 計測でデコーダ 1.6〜1.8 倍、重みは 100% 引き継げる） |
| M3 | Wavehax を新ボコーダとして | 未 |
| M4 | kushinada を SPIN で話者不変に微調整 | 未（SPIN 自体 1〜3 時間見込み＋再抽出＋事前学習やり直し） |
| M5 | embedder の出力層を中間層に（`embedder_output_layer` は既にある） | 未（再抽出＋事前学習やり直し） |
| M6 | F0 は差し替えより一貫性と後処理（hybrid） | 未 |
| IO1 / IO3 / IO4 / IO5 | ハイパス / 連続リサンプル / クリップ位置 / ゲート | 済（選択式、初期値オフ） |
| IO2 | ノイズ抑制 | 実装しない（6 章） |
| T1 | 学習データのクリーニング（残響、リップノイズ、吹かれ） | 未 |
| T2 | 変形した音声から特徴量を作る拡張（話者情報の漏れを減らす） | 未（事前学習から） |
| T3 | F0 の範囲が足りているか（+12.3 半音後の分布） | 未 |
| T4 | ラウドネス基準の正規化 | 未（優先度低） |
| T5 | ベスト選びの指標を耳に合うものへ（`docs/checkpoint-noise-vs-listening.md`） | 未 |
| T6 | 評価にリアルタイム経路を加える | 未 |
| A1 | コールバック内で待たない | 済（ring 転送のとき） |
| A2 | GPU 同期の削減（1 ブロックに 6 か所以上。本家 `d973943` はゲート時に推論を飛ばす） | 未 |
| A3 | torch.compile と周辺設定（MMCSS で推論スレッドの優先度を上げる案あり） | 一部（設定） |
| A4 | 状態欄の情報（遅延の推定、p50/p95、xrun） | 済（ring 転送のとき） |

M1・M2・M4・M5 の学習・推論の重さとモデルサイズの詳しい根拠は、2 つ目の Docs の 2 章。
書き出したモデル（.pth）の大きさはどれも変わらない（APEX-GAN 48k・1024 次元で 2,842 万パラメータ、fp16 で約 56.8 MB）。

---

## 8. 検証済みと未検証

**検証済み（クラウド、Linux、CPU、GPU なし）**: 上のテスト、初期値でのビット一致、各オプションの
組み合わせが落ちずに動くこと、ring 転送のデバイスなしシミュレーション（遅延 = チャンク +
処理時間 + 10 ms、p95 以内のぶれでアンダーフローなし、+200 ppm のクロック差を補正）、
UI の組み立てとテンプレートの往復。

**未検証（ローカルで最初に確かめること）**:
1. 実機・実モデル・GPU での動作全般。**音色の評価はまだ誰もしていない。**
2. ring 転送を実際の WASAPI で（xrun、GIL、排他モード、auto と separate の違い、実際の遅延）。
3. 新しいオプションと **torch.compile の組み合わせ**（trim や時刻固定は `realtime_infer` を通り、
   ノイズ・位相のテンソルを引数で渡す。コンパイルに失敗すると `CompiledPath` が通常推論に
   フォールバックしてログを出すはず）。float64 の位相計算が inductor でどう扱われるかも未確認。
4. FCN 系の F0 手法と各オプションの組み合わせ（stateful は FCN では自動で無効）。
5. ユーザーの環境の soxr が可変レート（`vr=True`）に対応しているか（無ければ警告してドリフト補正なし）。
6. レベル表示の数字（generator peak が 0 dBFS を超えるか）。

---

## 9. 次にやること（推奨順）

1. ブランチを取り込み、テストと `realtime_e2e_compare.py` を Windows で通す。
2. `tools/realtime_stream_eval.py` で legacy / stage1 / locked を同じ録音で作り、ユーザーと聴く。
   次に `apex_noise_envelope` の window と fixed（有声率は `tools/apex_noise_voiced_ratio.py`）。
3. 実機で `transport = ring`（auto → 必要なら separate）。ステータスの xruns と `est. total`、
   レベル表示を見る。話し終わりのノイズが気になれば `gate_fade_ms = 30` と `gate_hold`。
4. 良ければ L3（Chunk を 480 → 240 ms）と S1（Extra 2.5 s）を設定だけで試す。
   処理時間 p95 がチャンクより十分短いこと、時刻固定のために 40 ms の倍数にすることに注意。
5. 学習側は、今の事前学習の重みから続けられる M1・M2 が試しやすい。M4・M5 は特徴量空間が
   変わるので、同じ回の事前学習でまとめて比べる。

---

## 10. 落とし穴

- `RealtimeOptions` のフィールドは**末尾にだけ追加**。`tabs/realtime/realtime.py` の
  `option_controls` も同じ順に足す（`assert len(option_controls) == len(OPTION_FIELDS)` がある）。
  テンプレートの読み込み更新は `template.py` の `extract_option_updates` が順番どおりに作る。
- 生成器の呼び出しは、全オプションが初期値なら `Synthesizer.infer` そのもの。`realtime_infer` を
  触ったら `test_without_options_it_is_synthesizer_infer` を必ず通す。
- `Realtime_Pipeline.configure_stream` はセッションごとに 1 回、コンパイル前に呼ばれる前提。
  `SynthesisPlan` は frozen（torch.compile が値で特殊化するため）。
- `assets/config.json` はコミットしない（ユーザー設定）。
- 本家 Applio の参考コミット: `811dbf4`（生成器の出力を新しい側だけ切り出す）、`c1a3782`（位相ボコーダ式
  クロスフェード）、`f5f0564`（ゲート時のフェード）、`d973943`（ゲート時に推論を飛ばす）。
