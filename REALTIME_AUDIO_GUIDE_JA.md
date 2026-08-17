# Realtime音声経路と遅延修正の説明

## TL;DR

- このPCではMOTU、VB-AudioのWASAPI区間そのものは低遅延で動作している。
- 実測はMOTU WASAPI loopback区間が約41.3 ms、VB-Cable WASAPI区間が約76.7 msだった。
- RVC/CREPEを外し、アプリの現行transportだけを通した反復試験は、GUI上の試験値
  Chunk 960 ms / Hop 800 msで約0.98～1.17秒だった。960 msは試験値であり固定値ではない。
- したがって約10秒の遅延をWASAPI driverだけで説明することはできない。現行コードには、
  一度処理が遅れた際に最大数秒の古い入力・出力を維持し、満杯になると新しい入力を捨てる問題が
  実際にあった。
- 修正後はChunk/Hopを変更しない。許容範囲を超えた場合だけ古い時刻を捨て、最新の完全な
  Chunk contextからモデルを再同期する。
- separate fallbackの追加bufferは10 ms単位のまま、GUIの`Maximum Extra Buffer (ms)`で上限を
  設定する。初期値は200 ms、0なら追加増加を無効化する。WASAPI duplexでは使用しない。
- bufferを増やすための「音声断片の複製」は廃止した。別device間のSOXR rate補正でreserveへ
  徐々に追従するため、増加の瞬間に10 msの音を挿入しない。
- PA28→PA22は現在、1本のWASAPI duplex callbackを最初に試す。この経路では実行中の
  10 ms reserve増減とclock補正を無効化し、warmupから決めた固定の起動reserveだけを使う。
- 実モデル + mangio-crepeで聴感上の約10秒が消えたかは、ユーザー環境での最終確認が必要である。
  driver値だけを見て「解決済み」とは扱わない。

## 調査の優先順位

ユーザーが求めた順序は次のとおりである。

1. WASAPI input/outputだけで低遅延にする。
2. それが成立しない場合はWDM-KS inputの水中音を直す。
3. それでも必要なら遅延buffer戦略を作り直す。

今回の実測により1の物理経路が成立すると分かった。一方、同じWASAPI経路を使うアプリ内部に
古い時刻を保持する問題が見つかったため、WDM-KSへ移る前にこの問題を修正した。

## 実際に測った値

### RVCを通さないdevice区間

| 区間 | 測定結果 | 相関 |
|---|---:|---:|
| PA23 MOTU WASAPI output → MOTU内部loopback → PA28 input | 41.3 ms | 約1.0 |
| PA22 VB-Cable WASAPI output → PA26 CABLE output側input | 76.7 ms | 約1.0 |

PortAudioがアプリへ報告したI/O合計は44 msだった。外部loopbackを含む実測値とdriver報告値は
同じ意味ではないため、Statusの`I/O`だけをend-to-end遅延として扱ってはいけない。

### アプリtransportを含む試験

次の経路でRVC/CREPEだけをpass-throughへ置換した。

```text
PA23 MOTU output
  -> MOTU loopback
  -> PA28 WASAPI input
  -> このアプリのinput ring / worker / output ring / SOXR
  -> PA22 VB-Cable WASAPI output
  -> PA26 capture
```

Chunk 960 ms / Hop 800 msの反復試験で約0.98～1.17秒だった。Chunk contextを集める時間があるため
約1秒は設計上自然だが、10秒は自然ではない。この結果から次を区別できる。

- WASAPI device経路: 数十ms単位で動作している。
- 新規開始直後のアプリtransport: 選択したChunkに近い遅延で動作している。
- 実モデル運転中の約10秒: 処理停止や一時的な遅れの後に、古いqueueを再生する経路を疑うべき。

1本のWASAPI duplexへ変更後の同じpass-through試験は約1.15秒、相関約1.0、I/O 54 ms、
`xruns 0/0`だった。3回連続Start/Stopもすべてduplex/I/O 54 msで成功した。

## 見つかったアプリ内部の問題

以前の実装は次の容量を確保していた。

- native input ring: 最低8秒
- 48 kHz internal input ring: 最低8秒
- output ring: 最低8秒
- monitor ring: 最低8秒

さらにringが満杯になると、先に到着していた古い音を残し、新しく到着した音を捨てていた。
リアルタイム通話ではこれは逆である。モデルcompile、GPUの一時停止、スケジューラ遅延などが起きた後、
処理は正常速度へ戻っていても古い音を何秒も順番に再生し得る。

以前のStatusにあった`steady est.`は、Hop、推論p95、reserve、driver報告I/Oを加えただけだった。
実ringに何ms残っているかを表示していなかったため、Statusが正常に見えても古いqueueを検出できなかった。

## 新しいqueue設計

### Chunk/Hopは不変

GUIのChunk Sizeはモデルcontext、Hopは推論cadenceであり、StartからStopまで変更しない。
960 msをコード上の前提にはしていない。128、512、960、1200 msなど、現在のGUI値からframe数を
毎回計算する。

### 通常時

- input ringの容量は`Chunk context + grace + worker copy余裕`から計算する。
- internal ringも`Chunk context + grace + worker copy余裕`から計算する。
- output ringは`Hop + 最大reserve + grace + copy余裕`から計算する。
- 固定8秒bufferは使用しない。
- input ringが一杯なら最新のsampleを保存し、最古のsampleを捨てる。

`grace`は現在100 msで、`INPUT_BACKLOG_GRACE_MS`および`OUTPUT_QUEUE_GRACE_MS`という定数である。
これはChunkでも追加reserveでもない。通常のcallback単位の揺れでcatch-upを誤作動させないための幅である。

### 実時間から外れた場合

internal inputが`現在のChunk + 100 ms`を超えた場合、またはinput callbackでsample lossを検出した場合:

1. 古いprefixを捨て、最新の完全なChunk contextだけを残す。
2. RVCの変換historyとSOLA historyをflushする。
3. 最新contextを通常と同じ固定shapeで1回処理する。
4. その出力が完成した時点で、queueに残る古い変換済み音声を置換する。
5. 最後に実際に再生したsampleから新しい出力へ10 msで連続的に接続する。

既に数秒遅れた後は、「一切捨てずに数秒遅れを維持する」と「現在時刻へ戻る」を同時には満たせない。
このアプリではvoice chatの実時間性を優先し、古い音を捨てる。10 ms接続はclickを弱めるための処理であり、
捨てる区間に存在した発話を復元するものではない。

## 追加bufferの変更

以前はunderflowなどを検出すると、出力音声内の静かな10 ms区間を探して複製・挿入していた。
この挿入が固定Chunk、Overlapの両方で聞こえた「ぷちっ」の直接候補だった。

修正後:

- 1本のduplex経路ではreserveを実行中に増減しない。
- duplexではwarmup p95-p50から決めた固定の起動reserveだけを使う。最低値は10 msで、これは
  workerの推論jitterを吸収するための固定遅延であり、発話中に挿入・削除されない。
- duplex開始やtiming probeが失敗してseparate streamへfallbackした場合だけ、reserve目標値を
  10 ms単位で増減する。
- separateでは2回のmiss、または実underflowで10 ms増える既存判定を維持する。
- 音声sampleは複製しない。
- 別deviceのWASAPI経路では、既に使用しているSOXRの小さな出力rate補正により、新しいreserveへ
  徐々に近づける。
- 安定した無音時には10 msずつreserve目標を下げ、無音sampleを削って実queueも追従させる。
- separate時の最大追加量はGUIの`Maximum Extra Buffer (ms)`。初期値200 ms、範囲0～2000 ms。

この値はChunk Sizeではない。例えばChunkがGUI上960 msでも512 msでも、追加buffer上限は独立している。

## Statusの読み方

修正後は概ね次の項目が表示される。

```text
duplex/Windows WASAPI
| route PA28@48000Hz->PA22@48000Hz (clock .../...Hz)
| Chunk ... ms | Hop ... ms | infer p50/p95 .../... ms
| reserve ... ms (fixed) | I/O ... ms | lower-bound est. ... ms
| queue in/out .../... ms | drift ... ppm
| xruns in/out .../... | catch-up ... (dropped in/out .../... ms)
```

- `I/O`: PortAudio/driverの報告値。物理end-to-end実測ではない。
- `lower-bound est.`: `Chunk + infer p95 + reserve + reported I/O`。実測値ではなく下限の目安。
- `queue in/out`: その瞬間にアプリringへ残っている入力合計と出力。数秒へ増え続けてはいけない。
- `catch-up`: 実時間へ戻す処理を行った回数。
- `dropped in/out`: catch-upなどで意図的に捨てた古い音声の累計時間。
- `xruns`: callbackで入力を失った回数と、出力が不足した回数。
- `reserve ... (fixed)`: duplexでStart後に増減しない起動reserve。
- `reserve ... (adaptive)`: separate fallbackで10 ms制御が有効なreserve。

Statusは画面更新時点のsnapshotなので、`queue out`は1 Hop出力直後に大きく、再生につれて小さくなる。
重要なのは数秒単位へ単調増加しないことと、遅延発生時に`catch-up`が作動することである。

## 2つのdeviceを1つのWASAPI streamにする方法

MOTUとVB-Audioは両方48,000 Hz表示でも、別々のhardware/software clockを持つ。
同じ`Windows WASAPI`というHost API名だけでは物理clock共有の証明にならないが、PortAudioは
異なるcapture/render endpointを1本のfull-duplex callbackとして開くことができる。

- 同じPA番号がinput/output両方を持つ場合: PortAudioの1 duplex streamを使用可能。
- PA番号が異なっても両方WASAPIの場合: まず1本のduplex streamを試す。
- duplexのreported I/Oが1000 ms超、callback rate異常、またはopen失敗: 自動的にseparateへfallback。
- WDM-KSなど異なるdeviceの非WASAPI Host API: separate streamを使う。

duplexではinput/outputが同じcallback frame数で進むため、独立clock用SOXR補正と動的10 ms reserveを
止める。fallbackしたseparate経路だけSOXRでclock差を補正する。どちらもRVC/CREPEの実行は1回である。

## WDM-KS inputを今は優先しない理由

PA37 `Loopback (Loopback) (Windows WDM-KS)`は次の異常を再現した。

- 44.1 kHz: 3秒間callbackが0回。
- 48 kHz指定: 実際には約24,060 samples/sしか供給しない。

48 kHzの音として半分程度の速度でsampleが届くため、水中音・約2倍の時間軸になる。これはRVC推論前の
device入力で起きており、通常の数百ppmのclock補正では直せない。WASAPI経路が低遅延であることを
実測できたため、現段階ではPA37を推奨しない。WASAPI実モデル試験後も問題が残る場合だけ優先順位2へ進む。

## ユーザーが確認する内容

推奨する最初の試験:

- Input: `PA 28: Loopback (MOTU M Series) (Windows WASAPI)`
- Output: `PA 22: CABLE Input (VB-Audio Virtual Cable) (Windows WASAPI)`
- Exclusive Mode: OFF
- Maximum Extra Buffer: duplexでは未使用。separate fallback時だけ初期値200 msを使う。

確認点:

1. Start後のStatus全体を保存する。
2. 約10秒ではなく、概ね選択Chunkに近い遅延へ戻ったかを実音で確認する。
3. 長時間、ゲーム負荷、torch compile後に`queue in/out`が数秒へ増えないかを見る。
4. 遅れが起きたとき`catch-up`と`dropped`が増え、現在時刻へ戻るかを見る。
5. Statusが`duplex/Windows WASAPI`かつ`reserve ... (fixed)`になり、発話中にreserveが変化しないことを確認する。
6. Stop→Startを繰り返し、`-9996 Invalid device`が再発しないか確認する。

実モデルの音質・clickは自動試験だけでは判定できない。Statusと聴感結果の両方で次の修正要否を決める。

## 今回変更していないもの

- Torch/Triton packageのversionと`requirements.txt`
- CREPEやmangio-crepeのモデル計算内容
- WDM-KS driver固有処理
- ユーザーが選んだChunk Size

今回は依存packageを更新していないため、`requirements.txt`へ追加すべき新規dependencyはない。
