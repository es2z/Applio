# Realtime 音声経路・デバイス選択・Torch Compile 改修記録

更新日: 2026-08-18

## TL;DR（ユーザー向け）

- クリック、音切れ、大きな遅延を起こした新しいWDM-KS低遅延化設計を撤回した。
- Realtime音声経路は変更前の「固定Chunk SizeをPortAudio blocksizeとして使い、同一APIは
  duplex、WDM-KSまたは異なるAPIは元のQueue付きseparate stream」に戻した。
- Hop、Overlap Processing、ring buffer、推論worker、SOXR clock補正、10 ms単位の可変
  reserveは削除した。Chunk Sizeを実行中に変更する処理もない。
- デバイス選択の修正は維持した。表示は実PortAudio ID（`PA 37`など）、保存値は安定した
  fingerprintであり、Start時に同じdevice snapshotから実IDへ解決する。
- Torch Compileの修正も維持した。CREPEとRVCを分離し、設定変更は次回Startで反映、
  warmup中の失敗時だけ該当cacheを再構築し、それでも失敗すればeagerへ戻す。
- まず既知の動作状態へ戻すことを優先した。WDM-KSの追加低遅延化は今回実装していない。

## 1. 今回の復元理由

最初の改修では、PortAudio callbackから推論を分離するring buffer、固定Context＋Hop、
推論遅れに応じた追加reserve、無音時のreserve削減、別device clockのSOXR補正を導入した。
しかし実機では次の問題が継続した。

- Fixed/Overlapの両方で発話中に「ぷちっ」と切れる。
- WASAPI同士でも期待した一系統の安定したstreamにならず、遅延と音切れを両立できない。
- WDM-KS入力で水中のような音になる場合がある。
- 表示上のI/O latencyや実聴遅延が異常に大きくなる場合がある。
- 調整を重ねても、改修前より正常性が下がった。

ユーザーの現在の優先事項は低遅延化の継続ではなく、「取り敢えず正常に動く元の状態」へ
戻すことである。このため新設計を部分修正せず、音声transport全体を変更前コミット
`6cffa574`の方式へ戻した。

## 2. 現在の音声経路

### 2.1 共通仕様

GUIの`Chunk Size`は可変値であり、次の式で48 kHzのframe数へ変換する。

```text
read_chunk_size = int(chunk_ms * 48000 / 1000 / 128)
block_frame = read_chunk_size * 128
```

`960 ms`はユーザーの代表的な実験値にすぎず、定数、既定値、分岐条件にはしていない。
PortAudioとモデルは同じ固定`block_frame`で処理する。Start後にChunkやtensor shapeを
動的変更しない。

### 2.2 WASAPI同士など、WDM-KSを含まない同一host API

`sounddevice.Stream`によるduplex streamを使う。

```text
PortAudio duplex callback
  input Chunk -> 同じcallback内で推論 -> output Chunk
```

追加のHop、output reserve、ring buffer、別推論workerは存在しない。これは改修前の経路である。

### 2.3 WDM-KS、または異なるhost APIの組み合わせ

変更前と同じく`InputStream`と`OutputStream`を分け、`Queue`で接続する。

```text
InputStream callback -> 推論 -> Queue -> OutputStream callback
```

- WDM-KSがInputまたはOutputに含まれる場合はseparate streamを使う。
- Input/Outputのhost APIが異なる場合もseparate streamを使う。
- OutputStreamを先に開始し、その後InputStreamを開始する。
- Queueは古いChunkを蓄積しないよう小さく保つ。
- 長い無音では元実装どおりQueueをdrain/clearする。
- WASAPIとWDM-KSを混在させる場合、WASAPI exclusiveは互換性のためsharedへ落とす。

このQueue方式にも設計上の限界はあるが、今回の目的は新しい遅延戦略を追加することではなく、
ユーザーが使えていた既知の挙動へ戻すことである。

## 3. 削除した低遅延化実装

以下は現行実装ではない。

- `RuntimeAudioShape`
- Fixed Chunk / Overlap Processing Modeの切り替え
- Auto/Manual Hop
- model Contextと出力Hopの分離
- callback専用ring bufferと推論worker
- `blocksize=0`によるhost側blocksize自動選択
- SOXRによるnative sample rate変換と別device clock補正
- p50/p95から作るbase reserve
- deadline missやunderflowによる10 ms単位のreserve追加
- 無音時の10 ms単位reserve削減
- transport/Hop/reserve/drift/xrunをまとめた新status表示
- `rvc/realtime/runtime.py`とその専用テスト

GUIは元のChunk Size、Crossfade、Extra Conversion Sizeだけに戻した。statusも元の
`Latency: ... ms`表示へ戻した。

## 4. 維持したデバイス選択修正

`rvc/realtime/devices.py`はRefresh時にPortAudio metadataを一度取得し、Input/Outputの
snapshotを作る。各deviceを試しにopenしたり、Refresh時にPortAudioをterminate/reinitialize
したりしない。

Dropdownのlabelは実PortAudio global indexを表示する。

```text
PA 28: Loopback (MOTU M Series) (Windows WASAPI)
PA 37: Loopback (Loopback) (Windows WDM-KS)
PA 42: Speakers (VB-Audio Point) (Windows WDM-KS)
```

Input一覧やOutput一覧だけを見ると番号が飛ぶが、独自の連番ではなく全device tableのglobal
indexなので正常である。Dropdownのvalueには、host API、device名、方向、channel数、同名
deviceのordinalから作るfingerprintを保存する。Start時は選択時と同じsnapshotでfingerprintを
実indexへ解決する。

旧形式の`15: Name (Host API)`、新形式の`PA N: Name (Host API)`、旧template値は可能な範囲で
fingerprintへ移行する。選択deviceが消えた場合は別deviceへ黙ってfallbackせず、Refreshを促す。

音声transportを元へ戻しても、解決済みの`Invalid device [PaErrorCode -9996]`の原因だった
「表示用の番号とPortAudio IDの混同」は再導入しない。`Audio.start()`は解決済み
`AudioDeviceRef`から直接indexを使用する。stream open失敗も握り潰さずGUIへ返す。

Exclusive ModeのGUI既定値はOFFを維持した。ASIOのInput/Output channel selectorはそれぞれ
選択した正しい方向のdeviceへ適用する。

## 5. 維持したTorch Compile修正

Torch Compileは音声transportとは独立して残した。

- CREPE / Mangio-CREPEとRVC generatorを別々にON/OFFできる。
- RVC compileはexperimentalで既定OFF。
- modeは`default`、`reduce-overhead`、`max-autotune`、
  `max-autotune-no-cudagraphs`から選ぶ。
- 設定変更はconfigへ保存するだけで、実行中sessionをresetしない。
- Start時にimmutableな設定snapshotを読み込む。
- Compileが有効な場合だけ、audio streamを開く前に固定Chunk shapeでwarmupする。
- warmup中のcompile失敗時だけ該当namespaceを一度再構築する。
- 再失敗または実行中失敗は、そのcomponentだけeagerへfallbackする。
- cache namespaceはPython/Torch/CUDA/Triton/GPU/component/shape/modeで分離する。
- cache削除範囲はworkspaceの`.torch_compile_cache/v2`配下に限定する。

推奨の開始点は、主要負荷であるCREPE / Mangio-CREPEのみON、modeは
`reduce-overhead`である。不安定なら`default`を比較する。RVCは必要な場合だけ追加する。

Windows要件にはPyTorch pinに対応する`triton-windows`を記載済みである。

## 6. 変更ファイル

- `rvc/realtime/audio.py`: 変更前のduplex/separate Queue transportへ復元。安定device refのみ接続。
- `rvc/realtime/core.py`: Context/Hop分離を撤回。固定Chunk処理へ復元。compile warmupのみ維持。
- `rvc/realtime/callbacks.py`: runtime shape連携を撤回。compile snapshot/status取得のみ維持。
- `tabs/realtime/realtime.py`: Hop/Overlap UIを撤回。stable device registryとcompile開始処理を維持。
- `rvc/realtime/benchmark.py`: Hop/Overlap比較を削除し、固定Chunkのcompile benchmarkへ変更。
- `rvc/realtime/runtime.py`: 削除。
- `tests/test_realtime_runtime.py`: 削除。
- `rvc/realtime/devices.py`: 維持。
- `rvc/realtime/compile_session.py`、`rvc/realtime/pipeline.py`、
  `rvc/lib/predictors/f0.py`: 維持。
- `tabs/settings/sections/torch_compile.py`、`app.py`: 維持。
- `requirements.txt`、`requirementspy313.txt`: Triton/torchcrepe記載を維持。

## 7. 検証

2026-08-18に以下を実行した。

```powershell
.\env\python.exe -m py_compile `
  rvc/realtime/audio.py rvc/realtime/callbacks.py rvc/realtime/core.py `
  rvc/realtime/benchmark.py tabs/realtime/realtime.py rvc/realtime/devices.py

.\env\python.exe -m unittest `
  tests.test_realtime_devices tests.test_realtime_compile_session
```

結果:

- 構文チェック成功。
- device/compileの3テスト成功。
- `git diff --check`成功。

実音声deviceはユーザーのMOTU/VB-Audio routingと聴感が必要なため、この復元作業では開いて
いない。次の確認はアプリ上で従来の代表構成をStartし、まず音切れや異常遅延が改修前の状態へ
戻ったかを確認する。

## 8. 次のインスタンスが守ること

1. ユーザーの明示的な許可なしにHop、可変reserve、ring buffer、clock補正を再導入しない。
2. `960 ms`を固定値や設計前提にしない。GUIで選べるChunk Sizeの一例としてのみ扱う。
3. デバイスlabelの数字を独自採番しない。openにはsnapshotが持つ実PortAudio indexを使う。
4. RefreshでPortAudioをterminate/reinitializeし、取得済みindexを無効化しない。
5. 実行中のcompile設定変更でmodelやcompilerをresetしない。
6. compile cacheを消す場合は失敗した停止中warmup sessionのnamespaceだけに限定する。
7. 音声経路を再設計する場合は、WASAPI同士、WASAPI→WDM-KS、WDM-KS→WDM-KSを別々に
   実音声で検証し、pass-throughだけを成功判定にしない。
8. クリック、音切れ、遅延はユーザーの聴感を最終判定とし、status値だけで改善を主張しない。
