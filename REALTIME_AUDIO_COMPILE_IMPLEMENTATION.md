# Realtime 音声経路・Torch Compile 改修記録

## TL;DR（ユーザー向け）

- `Chunk Size` は可変値のままであり、`960 ms` は実機試験例にだけ使用した。コードに
  固定値として埋め込んでいない。
- `Fixed Chunk` と `Overlap` の両方を追加した。Overlap でもモデルが見る Context は
  Chunk Size のまま、開始時に決めた短い Hop だけを一定周期で進める。Start 後に
  Chunk/Hop/Compile shape は変化しない。
- 推論遅れへの対処は Chunk を勝手に拡大する方式ではない。追加出力バッファだけを
  `10 ms` ずつ増減する。2 回の deadline miss で `+10 ms`、実 underflow は即
  `+10 ms`、安定した無音中だけ `-10 ms` とする。2 倍、4 倍には増えない。
- PortAudio callback から推論・resample・待機ロックを外した。本当に同じ PA device が
  input/output の両方向を持つ場合だけ duplex とし、PA ID が異なるdeviceはHost APIが
  同じでも separate streamにする。別デバイスのクロック差は Chunk/Hop を変えず最大
  ±1000 ppm の可変resampleで吸収する。
- デバイスの表示番号を内部 ID として再利用しない。Dropdown は現在の PortAudio ID
  を `PA 37: ...` のように表示する一方、保存値には安定 fingerprint を使う。古い
  `15: Name (Host API)` と template は起動時に移行する。
- MOTU WASAPI Loopback は callback stream の exclusive open を `Invalid device -9996` で
  拒否する実機挙動を確認した。Exclusive の既定を OFF にし、ON の設定が残っていても
  同じ PA ID を shared mode で自動再試行する。
- Torch Compile は `CREPE / Mangio-CREPE` と `RVC generator (experimental)` を
  分離した。設定変更は次回 Start だけに反映され、実行中モデルをリセットしない。
  Compile 失敗時の cache 再構築は audio stream を開く前の warmup 中に一度だけ行い、
  実行開始後の失敗はその component だけ eager に戻す。
- Windows の通常要件に `triton-windows==3.7.1.post27`、Python 3.13 / Torch 2.8
  要件に Triton 3.4 系を追加した。torchcrepe fork は検証した commit に固定した。
- RTX 4090 実測（あくまで今回の代表設定）では Mangio-CREPE-full、Chunk 960 ms、
  Hop 320 ms、`reduce-overhead` で p95 は約 109.8 ms。CREPE と RVC の両方を
  Compile した試験は約 101.4 ms で、両 component が Inductor で動作した。

## 1. 依頼の背景と非交渉条件

用途は Windows 11 23H2 上の real-time voice changer で、音質を最優先する。
調査時の代表的なデバイス構成は次のとおりだった。

- Input: `10: Loopback (MOTU M Series) (Windows WASAPI)`
- Output: `4: Speakers (VB-Audio Point) (Windows WDM-KS)`
- WDM-KS input の候補: `15: Loopback (Loopback) (Windows WDM-KS)`
- GPU: NVIDIA GeForce RTX 4090
- Python 3.11.14 / PyTorch 2.13.0+cu132 / CUDA runtime 13.2

`10`、`4`、`15` は旧 GUI が並べ直した表示上の番号であり、必ずしも PortAudio が
stream open に要求する index ではなかった。調査時の実 PortAudio index は順に
28、42、37 だったが、これらも reboot、driver、接続順で変化し得る。

ユーザーが求めた挙動は以下である。

1. `960 ms` は現在 GUI で選んでいる一例にすぎない。どの Chunk Size でも同じ設計で
   動作しなければならず、この値を定数、分岐条件、既定の設計単位にしてはならない。
2. 長い Chunk を音質のために維持できること。解決策として Chunk Size を小さくする
   ことを要求しない。
3. Fixed と Overlap の両方式を選べること。Overlap は固定 Context + 固定 Hop であり、
   実行中に model tensor shape を動的変更しないこと。
4. 一時的な処理遅れには最小限の追加 buffer を足し、回復後は自動的に除去すること。
   増加量は 10 ms 単位であり、指数的に増やさないこと。
5. Compile 設定を触った瞬間に実行中推論を破壊しないこと。壊れた cache から安全に
   復旧できること。
6. 主な負荷は最重量設定をさらに重くした Mangio-CREPE である。Hubert を無理に
   Compile するより、まず CREPE を確実に速くすること。RVC は簡単かつ安全なら対応する。

## 2. 「960 ms + 223 ms になるのか」への回答

Fixed Chunk の初回出力は概ね次の要素を含む。

```text
Chunk の収録時間 + 推論時間 + 出力 reserve + audio driver I/O
```

したがって Chunk 960 ms、測定推論 223 ms なら、初回は概ね 960 + 223 ms に
reserve と driver latency が加わる。これは「二重実行すれば常に 960 + 223 ms で
安定する」という保証ではない。継続可能性は 1 回の推論時間ではなく、平均処理時間が
処理 cadence 以下か、backlog が増加し続けないかで決まる。

Overlap は初回に Context 分の入力を必要とするが、立ち上がった後の cadence は固定 Hop
となる。当初は次の概算を表示していた。

```text
steady estimate ≈ Hop + inference p95 + reserve + driver I/O
```

今回の実測では 960 ms Context / 320 ms Hop に対して p95 約 101 ms だったため、
推論時間だけを見れば cadence の約 31% であり大きな余裕がある。ただしゲーム同時実行時、
driver 自身の buffering、実ケーブルを含む end-to-end latency は別途測定が必要である。

後の実測で、この表示は初回Context待ちと実ring残量を表さず、正常に見えても古い音を検出できない
と分かった。現在は`Chunk + inference p95 + reserve + reported I/O`を`lower-bound est.`として
表示し、実ring残量を`queue in/out`として別表示する。どちらも物理end-to-end実測値ではない。

Auto Hop は warmup の `ceil10(p95 / 0.70)` で一度だけ決める。たとえば p95 が 109.8 ms
なら 160 ms となる。これは Chunk を変える処理ではなく、固定 Context を何 ms ごとに
更新するかを決める処理である。Manual Hop は 10 ms 刻みで指定できる。

## 3. 音声 transport の設計

### 3.1 固定 shape

[`rvc/realtime/runtime.py`](rvc/realtime/runtime.py) の `RuntimeAudioShape` が 1 回の
Start/Stop session の shape を保持する。dataclass は immutable である。

- `context_frames`: GUI Chunk Size を内部 48 kHz / 128 frame 単位へ丸めた値
- `hop_frames`: Fixed なら Context と同じ、Overlap なら Auto または Manual の固定値
- `effective_chunk_ms`, `effective_hop_ms`: 実際に使われる値

GUI は要求値とは別に Effective Chunk Size を表示する。2.7 ms のような小さい値も
128 frame（48 kHz で約 2.667 ms）へ丸めるだけで、960 ms 特例は存在しない。

### 3.2 callback と worker の分離

[`rvc/realtime/audio.py`](rvc/realtime/audio.py) は以下の責務に分けた。

```text
PortAudio input callback  -> preallocated ring -> inference worker
inference worker          -> output ring       -> PortAudio output callback
```

callback は preallocated scratch/ring への copy と xrun signal の記録だけを行う。
推論、SOXR、buffer 伸縮は dedicated worker に置いた。ring lock が競合した場合 callback は
待たずに xrun として記録する。host callback `blocksize=0` を使うため、GUI Chunk Size を
WDM-KS/WASAPI の host buffer size として渡さない。

同じPA device IDがInput/Output両方を持つ場合だけsingle duplex streamを試す。
PA IDが異なれば、同じWASAPI同士でもseparate InputStream/OutputStreamを使う。異なる API の
WASAPI -> WDM-KSもseparate streamになる。Start は worker、input callback、output
callback の heartbeat をすべて確認するまで Ready を返さないため、「表示は動作中だが
実 stream は固まった」という状態を成功扱いしない。

### 3.3 sample rate と device clock

Input/Output は各 device の native sample rate で開き、モデル内部だけ 48 kHz とする。
調査時の WASAPI input 48 kHz -> WDM-KS output 44.1 kHz も stateful SOXR で変換する。

separate stream は nominal sample rate が同じでも物理 clock が独立しており、長時間では
ring が徐々に増減する。output ring の reserve 誤差から SOXR の variable-rate ratio を
低域通過制御し、`MAX_CLOCK_CORRECTION_PPM = 1000` の範囲だけ output rate を補正する。
Chunk/Hop、推論 shape、音声の時間単位を変更する処理ではない。status の `drift` に現在値を
表示する。duplex は共有 clock のため補正を 0 ppm とする。

### 3.4 追加 buffer

適応対象は Chunk ではなく output reserve だけである。主要定数は同じ
[`runtime.py`](rvc/realtime/runtime.py) に集約した。

| 定数 | 値 | 意味 |
|---|---:|---|
| `BUFFER_STEP_MS` | 10 ms | 1 回に増減する量 |
| `MISSES_BEFORE_BUFFER_STEP` | 2 | 2 回の miss で 1 step。倍率ではない |
| `MISS_WINDOW_MS` | 5000 ms | miss を数える窓 |
| `BUFFER_ADJUST_COOLDOWN_MS` | 2000 ms | 増減の過剰反応防止 |
| `SILENCE_BEFORE_SHRINK_MS` | 1000 ms | 縮小に必要な連続無音 |
| `STABLE_BEFORE_SHRINK_MS` | 5000 ms | 最終 miss 後の安定時間 |
| `DEFAULT_MAX_EXTRA_BUFFER_MS` | 200 ms | GUIで変更できる追加reserve上限の初期値 |
| `OVERLOAD_DETECTION_MS` | 30000 ms | 維持不能判定の観測時間 |
| `MAX_CLOCK_CORRECTION_PPM` | 1000 ppm | separate clock 補正上限 |

base reserve は `max(10 ms, ceil10(warmup p95 - p50))`、extra は 0 ms から始まる。
deadline missを2回観測するか実underflowが起きるとreserve目標を10 ms増やす。初期実装は
低energy区間を複製挿入していたが、実音でclickが報告されたため廃止した。現在はseparate stream用
SOXR rate補正で目標reserveへ徐々に追従する。削除は安定した無音blockに限定する。

平均推論が cadence 以上で backlog が 30 秒にわたり増える場合は `OVERLOADED` を表示する。
この場合も Chunk/Hop を勝手に変更しない。これは buffer 追加で一時的 jitter は吸収できても、
永続的に生産速度が消費速度を下回る状態は有限 buffer では解決できないためである。

## 4. デバイス選択の修正

[`rvc/realtime/devices.py`](rvc/realtime/devices.py) は Refresh ごとに metadata snapshot を
1 回だけ取得する。全 device を open して探索する処理や PortAudio の terminate/reinitialize
は行わない。

Dropdown の label は実 PortAudio index を含む。

```text
PA 28: Loopback (MOTU M Series) (Windows WASAPI)
PA 37: Loopback (Loopback) (Windows WDM-KS)
PA 42: Speakers (VB-Audio Point) (Windows WDM-KS)
```

value は host API、正規化 name、方向、channel 数、同名 device の ordinal から作る
fingerprint である。変更可能な default sample rate と runtime PortAudio index は fingerprint
に含めない。Start は同じ snapshot 内で fingerprint を実 index に解決する。古い表示 string、
手入力した新しい `PA N:` label、古い template の device string も name + host API で移行する。
選択 device が本当に消えた場合は別 device に黙って fallback せず Refresh を求める。

PortAudio ID は Input と Output を合わせた全 device table の global index である。
Input dropdown だけを見ると `0, 1, 2, 3, 4, 11, ... 28, ...` のように飛ぶのが正常で、
独自採番ではない。今回の `PA 28` と `PA 42` は `sounddevice.query_devices()` の実 index と
一致することを再確認した。

追加の実機再現で、PA 28 を callback なしの blocking WASAPI exclusive stream としては
open できた一方、Realtime と同じ callback stream + exclusive では PortAudio が
`Invalid device [PaErrorCode -9996]` を返した。shared callback stream は正常だった。
そのため番号を置換するのではなく、WASAPI exclusive の constructor が拒否された場合だけ
同じ device/index の shared mode へ再試行する。Input、Output、Monitor、duplex の各経路に
同じ原則を適用し、status に fallback 理由を残す。UI の Exclusive Mode 既定値も OFF とした。

## 5. Torch Compile の設計

### 5.1 推奨設定

この利用形態では次を推奨する。

1. まず `Compile CREPE / Mangio-CREPE = ON`、`mode = reduce-overhead`。
   CREPE が支配的で、固定 shape の反復、小 batch、RTX 4090 という条件に合う。
2. ゲームと同時使用して CUDA Graph/VRAM の相性が悪い場合は `default` を安定基準にする。
   kernel tuning は欲しいが CUDA Graph を避けたい場合は
   `max-autotune-no-cudagraphs` を比較する。
3. `max-autotune` は長い初回 compile と追加の探索コストを許容できる検証用候補であり、
   名前だけで常に最速とは判断しない。
4. `Compile RVC generator` は実機で動いたが experimental のまま既定 OFF とする。
   まず CREPE のみを有効にし、同梱 benchmark で差を確認してから足す。
5. Hubert は今回 Compile しない。主 bottleneck ではなく、変更範囲と失敗面を増やすため。

PyTorch の公式説明でも `reduce-overhead` は CUDA Graphs で Python overhead を減らす mode
で、workspace cache により memory 使用量が増え得る。GPU 負荷を低くする mode ではない。
4 mode (`default`, `reduce-overhead`, `max-autotune`,
`max-autotune-no-cudagraphs`) はすべて UI に残した。

参考:

- https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html

### 5.2 session snapshot と復旧

[`rvc/realtime/compile_session.py`](rvc/realtime/compile_session.py) が CREPE と RVC を別 session
として管理する。GUI 変更は config に保存するだけで、Start 時に immutable snapshot を取る。
実行中の checkbox/mode 変更は次の Start まで反映されず、`torch.compiler.reset()` や model
reload を発生させない。

cache namespace は Python、Torch、CUDA、Triton、GPU、component、model shape signature、
mode の hash で分離し、`.torch_compile_cache/v2/<component>/<hash>` に置く。
`TORCHINDUCTOR_FX_GRAPH_CACHE=1` も application 起動時、Torch import より前に設定する。

warmup 中に compiled call が失敗した場合だけ、その失敗 namespace を検証して削除し、
compiler reset + 1 回の再 compile を行う。再失敗時はその component だけ eager fallback。
warmup 終了後は cache 削除や global compiler reset を禁止し、compiled call の実行時失敗は
保存してある eager callable へ即 fallback する。status は backend、cache rebuilt、warning を
表示する。手動の `Clear inactive compile caches` も Realtime 停止中だけ許可する。

### 5.3 Triton Windows

旧 `Disable Triton` checkbox が設定していた変数は現在の Inductor を切り替える有効な契約では
なく、UI 上は誤解を生むため削除した。CUDA の `torch.compile` では Inductor が Triton kernel
や template、CUDA Graph を組み合わせるため、Windows でも互換版 Triton を正しく入れる方を
採用した。

- `requirements.txt`: Torch 2.13.0+cu132 と検証した
  `triton-windows==3.7.1.post27`
- `requirementspy313.txt`: Torch 2.8+cu128 と公式表に対応する `>=3.4,<3.5`
- torchcrepe fork: commit
  `fb524c97c2f4bb74f1da4ae7b53b097072b872bf` に固定

Triton Windows の公式 compatibility 表は Torch 2.8 -> Triton 3.4、2.12 -> 3.7 を示す。
2026-08-18 時点で表にまだ 2.13 行はないが、upstream の release policy は奇数 PyTorch
minor が同じ Triton branch の patch release を使うとしており、この環境では 2.13 + 3.7.1
を実 compile で確認した。この組を requirements で固定し、未検証の 3.8 へ自動追従しない。

参考:

- https://github.com/triton-lang/triton-windows
- https://github.com/triton-lang/triton/blob/main/RELEASE.md

## 6. GUI と status

Realtime / Performance Settings に以下を追加した。

- Effective Chunk Size
- Processing Mode: Fixed Chunk / Overlap (experimental)
- Overlap Hop: Auto / Manual
- Manual Hop Size: 10 ms step
- Maximum Extra Buffer: 0～2000 ms、10 ms step、初期値200 ms

TorchCompile Settings には以下を追加・整理した。

- Compile CREPE / Mangio-CREPE
- Compile RVC generator (experimental)
- 4 compile modes
- 検出した Triton backend/version
- Clear inactive compile caches

実行中 status は transport (`duplex/...` または `separate/...`)、Chunk、Hop、推論 p50/p95、
reserve、PortAudio I/O latency、lower-bound estimate、実queue in/out、clock correction ppm、
input/output xrun、catch-up/dropped量、各compile backend、overload/warningを表示する。

## 7. 検証結果

### 7.1 自動テスト

```powershell
.\env\python.exe -X utf8 -m unittest discover -s tests -v
```

23 tests passed。対象は以下を含む。

- 複数の可変 Chunk が 128 frame alignment され、Fixed の Hop と一致する
- 960/320 はあくまで 1 test case として Context/Hop が分離される
- Auto Hop が p95/0.70 を 10 ms grid へ切り上げる
- ring wraparound の順序
- 2 misses ごとに +10 ms で、指数増加しない
- 安定無音時だけ -10 ms ずつ縮小する
- 30 秒の持続的な平均超過と backlog が overload を立てる
- 10 ms 挿入後の長さと finite sample
- legacy/new device label の解決と消失 device error
- 同じHost APIでも異なるPA deviceは別streamを使う
- 半速device clockとmulti-second I/O latencyを拒否する
- `running=False`でも残存streamをstop/closeする
- live compile failure が global cache reset をせず eager fallback する

`py_compile`、`import app`、`pip check` も成功した。

### 7.2 実 audio device smoke test

モデルを使わない transport 単体試験を行った。

```text
PA37 WDM-KS input -> PA42 WDM-KS output
duplex/Windows WDM-KS
I/O 20.0 ms, drift 0 ppm, xruns 0/0, input/output heartbeat True

PA28 WASAPI 48 kHz input -> PA42 WDM-KS 44.1 kHz output
separate/Windows WASAPI->Windows WDM-KS
I/O 32.0 ms, drift -129 ppm（3秒時点）, xruns 0/0,
input/output heartbeat True

同じ pair、Exclusive Mode ON（自動 shared fallback）
I/O 32.0 ms, drift +54 ppm（1秒時点）, xruns 0/0,
input/output heartbeat True
status: input PA 28 rejected WASAPI exclusive; using shared
```

これにより、以前音が出ず固まったように見えた WDM-KS -> WDM-KS pair は少なくとも
PortAudio duplex transport と callback のレベルでは開けることを確認した。Loopback に
実際の信号が routing されているかは MOTU mixer 側の状態にも依存するため、この試験だけで
物理 routing を保証はしない。

### 7.3 RTX 4090 model benchmark

同梱の [`rvc/realtime/benchmark.py`](rvc/realtime/benchmark.py) を使い、実モデル、index、
Mangio-CREPE-full で voiced 220 Hz 入力を処理した。960/320 はユーザーの現設定に近い
代表値としてのみ使用した。

| compile | Chunk | Hop | p50 | p95 | realtime ratio | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| CREPE only, reduce-overhead | 960 ms | 320 ms | 109.2 ms | 109.8 ms | 0.341 | 895.2 MiB |
| CREPE + RVC, reduce-overhead | 960 ms | 320 ms | 98.1 ms | 101.4 ms | 0.308 | 894.2 MiB |

両者で CREPE は `inductor/reduce-overhead`、後者では RVC も同 backend が active だった。
数回の短い benchmark なので 1～2 ms の優劣を一般化しないこと。ゲーム負荷下では次の
PowerShell 例を使って各 mode を再測定する。

```powershell
$env:PYTHONUTF8='1'
.\env\python.exe -m rvc.realtime.benchmark `
  --model logs\naru_A2\naru_A2_105e_78540s.pth `
  --index logs\naru_A2\naru_A2.index `
  --f0 mangio-crepe-full --chunk 960 --processing overlap `
  --hop-mode manual --hop 320 `
  --modes default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs
```

出力は JSON と Markdown で、p50/p95/p99、平均、cadence ratio、peak VRAM、component status
を保存する。別 Chunk/Hop を評価する場合は引数だけを変更する。

## 8. 変更ファイルの案内

- `rvc/realtime/runtime.py`: immutable shape、ring、10 ms elastic controller と定数
- `rvc/realtime/devices.py`: metadata snapshot、fingerprint、legacy/template migration
- `rvc/realtime/audio.py`: callback/worker transport、duplex fallback、SOXR、clock drift、status
- `rvc/realtime/core.py`: Context/Hop 分離、time-based silence、voiced warmup
- `rvc/realtime/callbacks.py`: session shape と compile status の接続
- `rvc/realtime/pipeline.py`: CREPE/RVC session wiring。Hubert は eager のまま
- `rvc/lib/predictors/f0.py`: CREPE session 利用、Mangio zero-input 安全化
- `rvc/realtime/compile_session.py`: component 別 compile/rebuild/eager fallback
- `tabs/realtime/realtime.py`: device/processing UI、Start snapshot、Auto Hop
- `tabs/realtime/template.py`: legacy device migration と Exclusive の安全な既定値
- `tabs/settings/sections/torch_compile.py`: setting migration、version check、cache namespace
- `app.py`: Torch import 前 bootstrap と Compile UI
- `requirements.txt`, `requirementspy313.txt`: Triton Windows と torchcrepe pin
- `rvc/realtime/benchmark.py`: offline benchmark
- `tests/`: shape/buffer/device/live fallback tests

## 9. 次のインスタンスが守るべき再実装 invariant

1. `context_frames` と `hop_frames` を混同しない。Context は品質設定、Hop は cadence である。
2. Start 後に Context/Hop/Compile mode を変更しない。UI change は config に保存するだけ。
3. PortAudio callback で推論、resample、blocking lock、cache reset、device refresh をしない。
4. miss count は倍率ではない。`MISSES_BEFORE_BUFFER_STEP=2` は「2 miss で +10 ms」を意味する。
5. 可聴 sample を latency 回収のため捨てない。縮小は確認済みの無音だけで行う。
6. 追加 buffer と永続 overload を区別する。30 秒平均が cadence 以上で backlog が増えるなら
   警告し、Chunk/Hop を勝手に変更しない。
7. Dropdown の label の数字と保存 token を混同しない。open は snapshot の実 PA index だけ。
8. Compile cache を消す範囲は `.torch_compile_cache/v2` 内の該当 namespace に限定する。
9. live session 中に `torch.compiler.reset()`、torchcrepe reload、全 cache delete を行わない。
10. Triton/PyTorch の組を requirements で固定し、更新時は全 4 mode、CREPE-full、RVC、
    representative GPU contention を再 benchmark する。

## 10. 未検証・今後の測定

- ASIO device は実機が列挙されなかったため、stable fingerprint と channel selector path は
  実装したが実 stream smoke test は未実施。
- 3 秒の separate clock test は機能確認であり、数時間の game/voice-chat soak test ではない。
- driver報告I/Oとは別に、MOTU loopback区間41.3 ms、VB-Cable区間76.7 ms、RVCを外した
  アプリtransport全体約0.98～1.17秒（Chunk 960/Hop 800の試験値）を物理loopで測定した。
  実RVC + mangio-crepeの最終聴感遅延はユーザー試験が必要である。
- WDM-KS Loopback の callback は確認したが、MOTU mixer の Loopback routing 自体はこの
  application から設定できない。無音なら device open failure と routing silence を status/xrun
  と mixer meter で分けて確認する。
- Overlap は実モデルで finite/seamless processing を確認したが、最終的な音質判断はユーザーの
  聴感が基準。Fixed を削除していないので即時比較できる。

## 11. 2026-08-18 WASAPI先行修正

### 11.1 TL;DR

MOTUのWASAPI入力とVB-AudioのWASAPI出力は、異なるdeviceなのに同じHost APIという理由だけで
1個のduplex streamへまとめられていた。これを別streamへ変更した。PA28→PA22の実機試験では
I/O 44 msで5回連続Start/Stopに成功したため、この構成ではWDM-KSを使う必要はない。

新しい200 ms reserve UIやbuffer方式の変更は意図的に保留した。まず実モデルのWASAPI同士を
ユーザーが聴き、問題が残った場合だけ第二段階へ進む。

### 11.2 発見した問題と測定結果

ユーザー環境で次のstatusが報告された。

```text
duplex/Windows WASAPI | Chunk 960.0 ms | Hop 800.0 ms
| infer p50/p95 94.5/116.9 ms | reserve 60 ms
| I/O 4740.7 ms | steady est. 5717.6 ms
```

4.7秒はモデル推論ではなく`stream.latency`が返したdriver I/Oだった。ただしアプリを完全終了した
fresh processでは、PA28と全WASAPI出力のduplexは54～55 ms、PA28→PA22の別streamは44 msだった。
したがってWASAPI固有の不可避な遅延ではなく、誤ったduplex判定または残留streamを含む異常状態で
ある。

PA37 WDM-KS入力はさらに次の異常を示した。

- 44.1 kHz: 3秒間にcallback 0回
- 48 kHz指定: 3秒間に72,192 frames、実効約24,060 frames/s
- PortAudioのreported rateとstatus flagは48 kHz、xrun 0

この半速sample供給が水中音の直接原因であり、通常の数百ppmのclock drift補正では直せない。
PA37/WDM-KSの修正は今回の対象外とし、WASAPI経路を推奨する。

### 11.3 実装判断

`input.host_api_index == output.host_api_index`は共有clockの証明にならない。今後1個のduplex streamを
使えるのは、PortAudio indexが同じで、そのraw deviceが入出力の両方を持つ場合だけである。
異なるPA indexはHost APIが同じでも必ず既存の別stream + SOXR ASRC経路を使う。

ASRCは10分のmono音声を0.60秒で処理し、リアルタイム比0.001だった。推論worker、RVC、CREPEは
1個のままであり、別stream化によってモデル負荷は増えない。

Start時には1秒のcallback probeを行う。要求sample rateとの差が5%を超えるか、reported I/Oが
1000 msを超える場合はstreamをcloseして明示的に失敗させる。probe中の入力はringから消去し、
実セッションへ混ぜない。

UIの`stop_realtime()`は`running`が既にFalseでも`audio_manager`が存在すれば必ずstop/closeする。
新しいStartも残っている旧managerを先に停止する。これにより、Realtime Stop後もPA28が保持され、
アプリ全体を終了するまで`-9996 Invalid device`になる経路を塞いだ。

### 11.4 WASAPI先行修正の時点で変更しなかったもの

- Fixed/Overlap、Chunk、Hop、推論shape
- 既存の10 ms adaptive grow/shrinkと`_insert_sola_style`
- GUI設定可能な追加reserve上限（初期値200 ms）
- ring lockとunderflow concealment
- Torch Compile、Triton、requirements
- WDM-KS host implementation

短い128 ms無音dummyによるPA28→PA22実機試験では、各Start後にoutput underflowが1～3回発生し、
一部の試験で既存reserveが10→20 msへ増えた。WASAPI 4.7秒問題とは別のため今回変更していないが、
実モデルでクリックが残った場合の次の調査点である。

実際のユーザー試験では約10秒の遅延とreserve変化時のclickが残った。そのため、ここで保留した
ring/adaptive処理を次のSection 12で修正した。I/O 44 msだけを根拠に解決済みと判断したのは誤りで、
driver報告値と実end-to-end、実queue残量を分けて検証する必要があった。

ユーザー向けの用語説明、推奨経路、Statusの読み方、実音確認手順は
[`REALTIME_AUDIO_GUIDE_JA.md`](REALTIME_AUDIO_GUIDE_JA.md)に分離した。

## 12. 2026-08-18 実queue追従修正

### 12.1 再調査の契機

Section 11の修正後、ユーザー実音では約10秒の遅延が残り、Fixed/Overlapの両方でadaptive bufferが
増える瞬間にclickが聞こえた。reported I/O 44 msはPortAudioが返すstream設定値であり、実際に
何秒前のsampleがringに残っているかを測っていなかった。

調査順序はユーザー指定どおり、(1) WASAPIだけで低遅延、(2) WDM-KS input正常化、
(3) buffer戦略再設計とした。まずアプリを外した物理loopを相互相関で測った。

| 測定 | 結果 |
|---|---:|
| PA23 MOTU WASAPI output → MOTU loopback → PA28 input | 41.3 ms、correlation約1.0 |
| PA22 VB-Cable WASAPI output → PA26 capture | 76.7 ms、correlation約1.0 |
| 上記両区間 + アプリpass-through、Chunk 960/Hop 800 | 約0.98～1.17秒 |

最後の960/800は再現試験用のGUI値であり、実装定数ではない。pass-throughでもContext収録時間があるため
約1秒は説明できるが、10秒は説明できない。WASAPI経路は優先順位1を継続できると判断した。

### 12.2 根本の設計不良

旧`_create_buffers()`はnative input、internal input、outputをそれぞれ最低8秒確保していた。
`FloatRingBuffer.write()`は満杯になるとincoming tail、つまり新しいsampleを捨て、古いprefixを残した。
workerはnative inputをinternalへ全量移してから古い順に推論した。

これはファイル変換なら正しいがrealtime voice chatでは正しくない。compile/autotune、GPU contention、
OS schedulingなどでworkerが一度停止すると、復帰後も過去の時刻を順に処理し続けられる。output ringも
同じ方針なので、既に変換した古い音をさらに保持できる。`steady est.`にはこれらの実残量が入らない。

### 12.3 `FloatRingBuffer`の追加操作

[`rvc/realtime/runtime.py`](rvc/realtime/runtime.py)へ以下を追加した。

- `overflow_policy="drop_oldest"`: incoming最新sampleを保存するため最古sampleを破棄する。
- `discard(count)`: oldestから指定数を破棄する。
- `trim_to_latest(count)`: 最新count sampleだけを残す。
- `replace(samples)`: queueを新しいblockへatomicに置換する。
- `overwritten_samples`: realtime catch-upで失ったsample数を診断する。

従来動作を必要とする呼び出しの既定値は`drop_newest`のままにし、audio transportが明示的に
`drop_oldest`を選ぶ。

### 12.4 容量はruntime shapeから導出

固定8秒を廃止し、Start時の可変Chunk/Hopとdevice rateから次を計算する。

```text
input capacity
  = native-rate Chunk context + input grace + worker-copy余裕

internal capacity
  = 48 kHz Chunk context + input grace + worker-copy余裕

output capacity
  = native-rate Hop + (base reserve + GUI max extra + output grace) + copy余裕
```

`INPUT_BACKLOG_GRACE_MS`と`OUTPUT_QUEUE_GRACE_MS`は現在100 ms、`WORKER_READ_FRAMES`は8192である。
graceは選択Chunkを置き換えず、callback overshootを誤ってcatastrophic backlogと見なさないために使う。
容量には常にGUIから作った`context_frames`/`hop_frames`を使い、960 ms条件分岐はない。

### 12.5 catch-up state transition

次のいずれかでdiscontinuityと判定する。

- native input ringが満杯になりoldest sampleを上書きした。
- callbackがring lockを取得できずsampleを失った。
- internal queueが`context_frames + 100 ms`を超えた。

完全なContextがあれば`trim_to_latest(context_frames)`で最新Contextを残す。不完全なら、gapの前後を
1 Contextとして接続しないようinternal queueをclearし、新しいContextが溜まるまで待つ。その後:

1. `vc.sola_buffer`をzeroにする。
2. `vc.vc_model.flush_buffers()`でaudio/convert/pitch historyを消す。
3. local worker stateをinitialへ戻し、固定Context shapeで再開する。
4. 次の変換結果が完成してから、残る古いoutput queueを`replace()`する。
5. callbackが最後に実際に出したsampleから新blockへ`RECOVERY_CROSSFADE_MS = 10`で補間する。

outputを先にclearして推論完了まで無音にするのではなく、新block完成時に置換する。古いoutputがまだ
残る場合は推論中のgapを減らせる。既にqueueが空ならgapは不可避だが、数秒古い発話の再生は再開しない。

### 12.6 adaptive reserve

`ElasticBufferController`は`max_extra_buffer_ms`をsessionごとに保持する。GUIへ
`Maximum Extra Buffer (ms)`を追加し、0～2000 ms、10 ms刻み、初期値200 msとした。
`assets/config.json`の`realtime.max_extra_buffer_ms`へ保存する。設定は次回Startから使われる。

miss判定、10 ms step、cooldown、安定無音時のshrinkは維持する。ただしgrow時の
`_insert_sola_style()`呼び出しは削除した。音声内へ複製sampleを突然挿入せず、separate-device経路で
常時使用するSOXR variable-rate feedbackのtargetだけを10 ms動かす。これにより実queueは小さな
rate差で徐々に目標へ近づく。silent blockのshrinkだけは無音sampleを10 ms削除できる。

最大値を200 msとするのは既定値であって固定要件ではない。ユーザーはGUIで0にも2000にも変更できる。
Chunk/Hop/model shapeには影響しない。

### 12.7 Statusとテスト

Status変更:

- `steady est.`を`lower-bound est.`へ変更し、式をChunk基準にした。
- `queue in/out`でnative+internal inputとoutputのsnapshotをms表示する。
- `catch-up`で復旧回数を表示する。
- `dropped in/out`で古い時刻を意図的に破棄した累計msを表示する。

追加unit testは以下を検証する。

- ring wraparound order。
- `drop_oldest`が最新sampleを残すこと。
- `trim_to_latest`と`replace`が古いprefixを残さないこと。
- 代表shapeでbufferが固定8秒ではなくChunk/Hopから導出されること。
- `Chunk + grace`を超えたworker backlogが最新Contextへcatch-upすること。
- 通常のHop出力後に残るsawtooth queueをstale backlogと誤判定しないこと。
- extra reserveがGUI由来上限を超えないこと。
- Chunkが複数の値で可変、Overlap context/hopが固定shapeであること。

### 12.8 残る検証と次の判断

自動試験とpass-through物理loopではtransportを検証できるが、mangio-crepe最重量設定の聴感click、
ゲーム同時負荷、長時間clock driftはユーザーの実運転が必要である。次回はStatus全文、特に
`queue in/out`、`catch-up`、`dropped`、`xruns`を約10秒が再現した時点で保存する。

- WASAPI実モデルで低遅延になればWDM-KS対応は不要。
- queueが小さいまま約10秒なら、RVC/CREPE内部またはVB-Cable後段の消費アプリをcontent timestampで測る。
- PA37 WDM-KSは約24 ksample/sの半速供給があるため、WASAPI失敗時だけhost-specific pin/rateを調べる。

今回のqueue修正ではdependencyを追加・更新していない。Torch/Triton/torchcrepeのrequirements固定は
既存Sectionのままである。
