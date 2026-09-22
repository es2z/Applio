<proposed_plan>

# FCN-993／FCN-993-RVC追加計画

調査対象は `C:\dev\myapplio2` の `main`、HEADは `424f85dde8e4779edeed87ed9163518b7a425102`。現行working treeのコードを基準とする。

**採用案は、FCN-993をPyTorchへ忠実移植し、同一network・同一weightに対して二つのadapterを提供する方式。**

- 表示名 **FCN-993**、method ID `fcn-993`：原実装再現用。
- 表示名 **FCN-993-RVC**、method ID `fcn-993-rvc`：RVC向けV/UV判定・時間処理を追加。
- 両methodとも学習抽出・通常推論・batch・realtime・F0 utilityで使用可能にする。
- FCN-929は実装せず、architecture metadataで追加可能な構造にする。

今回、コード変更、ファイル追加、dependency追加、weight変換、weightの保存・ダウンロード、branch変更は行っていない。原weightのHDF5内部、実際のconfidence分布、速度・聴感は未検証であり、以下で実装時の検証工程を明示する。

## A. 現在のApplio F0 architecture

### 調査した主要コードと役割

| ファイル | 現在の役割・確認事項 |
|---|---|
| [rvc/lib/predictors/f0.py](/C:/dev/myapplio2/rvc/lib/predictors/f0.py) | RMVPE、FCPE、CREPE、Mangio-CREPE、SwiftF0、CREPE_ONNXのwrapper |
| [crepe_models.py](/C:/dev/myapplio2/rvc/lib/predictors/crepe_models.py) | CREPE method→model対応、UI/CLI用method一覧 |
| [crepe_decoder.py](/C:/dev/myapplio2/rvc/lib/predictors/crepe_decoder.py) | Mangio decoder設定。既定は`viterbi` |
| [F0Extractor.py](/C:/dev/myapplio2/rvc/lib/predictors/F0Extractor.py) | 補助F0抽出。主経路と異なる直接呼び出しを含み、戻り値はcents |
| [train/extract/extract.py](/C:/dev/myapplio2/rvc/train/extract/extract.py) | 学習用F0・特徴抽出、device別worker、coarse/full F0保存 |
| [train/extract/compile_extract.py](/C:/dev/myapplio2/rvc/train/extract/compile_extract.py) | RMVPE/FCPE等の任意compile、失敗時eager fallback |
| [infer/pipeline.py](/C:/dev/myapplio2/rvc/infer/pipeline.py) | 通常推論のF0 dispatch、pitch補正、量子化、音声chunk処理 |
| [infer/infer.py](/C:/dev/myapplio2/rvc/infer/infer.py) | 音声読込、変換モデル・Pipelineの再利用、batch処理 |
| [realtime/pipeline.py](/C:/dev/myapplio2/rvc/realtime/pipeline.py) | realtimeの実際のF0推定担当。F0 modelをinstance内で保持 |
| [realtime/core.py](/C:/dev/myapplio2/rvc/realtime/core.py) | 48→16 kHz、音声/F0 buffer、無音処理、SOLA |
| [realtime/callbacks.py](/C:/dev/myapplio2/rvc/realtime/callbacks.py) | session構築、audio開始前warm-up、callback仲介 |
| [realtime/audio.py](/C:/dev/myapplio2/rvc/realtime/audio.py) | sounddevice、入出力queue、再接続、処理時間表示 |
| [realtime/compile_session.py](/C:/dev/myapplio2/rvc/realtime/compile_session.py) | session所有のembedder/RVC compile。現在F0専用枠はない |

### 呼び出し経路

```text
学習
tabs/train/train.py または core.py CLI
  → core.run_extract_script()
  → rvc/train/extract/extract.py
  → deviceごとのprocess_files()
  → FeatureInput → predictor.get_f0()
  → f0/*.npy：coarse
  → f0_voiced/*.npy：Hz、UVでは0

通常・batch
UI / core.py
  → rvc/infer/infer.py
  → Pipeline.pipeline()
  → Pipeline.get_f0()
  → predictor
  → pitch補正 → coarse化
  → 共通F0を音声chunkに合わせて切り出して合成

realtime
tabs/realtime/realtime.py
  → AudioCallbacks / VoiceChanger / Realtime
  → audio.py callback
  → Realtime.inference()
  → Realtime_Pipeline.voice_conversion()
  → Realtime_Pipeline.get_f0()
  → predictor → pitch buffer → 合成 → SOLA
```

### 現行実装で設計上重要な事実

**学習抽出**

- 16 kHz、hop 160、10 ms。
- `FeatureInput`の量子化範囲は50–1680 Hz、256 bins、実際のcoarse値は1–255。
- `f0_voiced`はUVを補間した完全なcontinuous pitchではなく、predictor出力をそのまま保存する。
- multiprocessingは`spawn`。deviceごとに`FeatureInput`を一度作り、複数ファイルで再利用する。
- 両F0ファイルが存在するとmethodに関係なく再利用する。**FCN切替時の取り違え防止が必要。**
- 学習対応methodはRMVPE、FCPE、CREPE/Mangio各種。Swiftは学習dispatchにない。

**通常推論**

- 16 kHz、hop 160、量子化範囲50–1100 Hz。
- 48 Hz high-pass、reflect padding後にF0を計算する。
- 基本的に合成chunk分割前にF0をまとめて抽出する。
- wrapperは`get_f0()`ごとに生成・破棄。torchcrepeのnetwork自体はpackage側のglobal cacheに残る。
- `Autotune.autotune_f0()`は0 Hzにも音程補正を行うため、FCNのUV maskは補正後に再適用する必要がある。

**realtime**

- `Realtime_Pipeline.get_f0()`が入力Tensorを一律CPU NumPyへ移す。
- F0 modelはlazy生成後に再利用し、callback開始前のwarm-upでも構築される。
- 音声のrolling buffer全体を再解析しており、F0推定器に絶対sample位置を渡していない。
- 48→16 kHzはcached `torchaudio.transforms.Resample`だが、各入力blockに独立適用している。**object cacheはstreaming filter stateとは異なる。**
- 現在のcoarse化はmel値からHzの`f0_min/max`を直接引いており、学習・通常推論の式と一致しない。
- UIの“Latency”は主に処理時間であり、capture・lookahead・queueを含む総遅延ではない。
- UI初期値のchunkは512 ms。したがってFCNのlookaheadだけで実際の体感遅延を説明できない。

### CREPE-speechから再利用できるもの

`crepe-full-speech`と`mangio-crepe-full-speech`はいずれも、installed `torchcrepe_plus`の`full_speech`へdispatchされる。weightはpackage内`assets/full_speech.pth`、ライセンスファイルも同梱されている。

- plain CREPE：weighted argmax、F0/pdに3-frame median、`pd < 0.1`でUV。
- Mangio：入力を0.999 quantileで正規化、padding、設定可能decoder、3-frame median、pd threshold、NaNを介した`p_len`への補間。
- 学習Mangioだけはdecoderを`viterbi`に固定する。
- torchcrepeのcacheはcapacity中心のglobal管理。FCNは独立した所有者管理にする。
- decoderファイルの非再現性の説明だけを根拠にしない。installed torchcrepeのbin→centsにはrandom ditherも存在する。

再利用対象はmethod登録の形式、device受渡し、session lifecycle、compile fallbackの仕組み。**360-bin decoder、pd閾値、CREPE network、quantile正規化、長さ比によるalignmentは流用しない。**

`fcnf0++.pt`や古いcacheは存在するが、現行Python dispatchにFCN-F0++はない。これらをFCN-993の既存実装・weightとして扱わない。CREPE_ONNXもwrapperはあるが、今回調べた主経路では選択対象になっていない。CLIのhybrid choicesも実dispatchとは一致していないため、FCN追加と同時にhybrid機能の新規実装は行わない。

## B. FCN-993原実装の解析

### Network

原実装の`core.py`と`model.json`を確認した。JSONはKeras 2.2.4、TensorFlow backend、固定長training graphであり、通常のfully-convolutional推論は`load_model.py`から`inputSize=None`で構築する。[原network](https://github.com/ardaillon/FCN-f0/blob/master/models/FCN_993/core.py)、[model.json](https://github.com/ardaillon/FCN-f0/blob/master/models/FCN_993/model.json)、[loader](https://github.com/ardaillon/FCN-f0/blob/master/models/load_model.py)

| 段 | 出力channel | kernel / stride | 後続処理 | 累積受容野 | 累積stride |
|---|---:|---|---|---:|---:|
| Conv1 | 256 | 32 / 1 | ReLU → MaxPool 2/2 → BN | 33 | 2 |
| Conv2 | 32 | 32 / 1 | ReLU → MaxPool 2/2 → BN | 97 | 4 |
| Conv3 | 32 | 32 / 1 | ReLU → MaxPool 2/2 → BN | 225 | 8 |
| Conv4 | 128 | 32 / 1 | ReLU → BN | 473 | 8 |
| Conv5 | 256 | 32 / 1 | ReLU → BN | 721 | 8 |
| Conv6 | 512 | 32 / 1 | ReLU → BN | 969 | 8 |
| Classifier | 486 | 4 / 1 | sigmoid | **993** | **8** |

全Conv・Poolは`valid`。biasあり。BNは`epsilon=0.001`、`momentum=0.99`、gamma/betaあり。推論時dropoutなし。

計算式は`R_next = R + (kernel−1) × stride_accumulated`。

- sample rate：8 kHz。
- network受容野：993 samples＝124.125 ms。
- frame中心から左右端までは496 samples＝62 ms。
- native output stride：8 samples＝**1 ms**。
- 993 samples入力時の時間長：`962→481→450→225→194→97→66→35→4→1`。

### 前処理・decoder

以下は原`prediction.py`に基づく。[prediction.py](https://github.com/ardaillon/FCN-f0/blob/master/prediction.py)

1. mono化、float32化。
2. sample rateが異なれば`resampy.resample()`。
3. 左右496 samplesをzero-pad。
4. `sliding_norm(frame_sizes=993)`。
5. network。
6. 最大activationをconfidenceとする。
7. argmax binの左右4 bins、最大9 binsで**centsのactivation加重平均**。
8. `10 × 2^(cents/1200)`でHz化。NaN周波数は0。
9. native時刻は`j × 8 / 8000`。

**normalizationの実処理は994 samples。**

奇数窓を1増やし、zero-pad済み配列をさらに497 samplesずつ`wrap` padする。sample `i`の統計範囲は`[i−497, i+496]`。population stdを用い、stdが正確に0の場合だけfloat32 epsilonへ置換する。

**confidenceによるV/UV thresholdは公式defaultにはない。**  
baselineにCREPEの0.1や独自energy gateを入れると、原実装比較の意味が変わる。

### Pitch mapping

486 binsは30–1000 Hzをcents上で等間隔に分割し、bin間隔は約12.51685 cents。

`f0_to_target_convertor.py`には「先頭binがUV」という説明があるが、実コードは30 Hzから始まるmappingであり、その説明どおりのUV専用binはない。**bin 0をUVへ置換しない。** [変換コード](https://github.com/ardaillon/FCN-f0/blob/master/models/f0_to_target_convertor.py)

### Viterbi

公式処理は486状態、uniform prior、距離に応じた三角形のtransitionで、`max(12−|i−j|, 0)`を行正規化する。観測はactivation全体ではなくargmax bin。emissionはself成分0.1とuniform成分から構成し、復元pathの周辺でlocal averageを行う。[Viterbi実装](https://github.com/ardaillon/FCN-f0/blob/master/prediction.py)

- UV状態を持たない。
- 全系列pathの確定は無制限の未来に依存し得る。
- 現行hmmlearnへ旧`MultinomialHMM`呼び出しをそのまま移すべきではない。
- 初期製品defaultには採用しない。

### 正規化を含む受容範囲

内部区間では、networkの`[−496,+496]`にnormalizationの`[−497,+496]`が加わる。

```text
必要な元8 kHz音声：frame中心に対して [−993, +992]
合計：1986 samples
未来context：992 / 8000 = 124 ms
```

これはarchitectureと前処理コードからの導出。READMEの約62 msはnetwork中心窓の半分に対応し、**完全な前処理を再現したstreamingの待ち時間とは分ける必要がある。** [README](https://github.com/ardaillon/FCN-f0#readme)

## C. Baseline FCN-993の設計

### 処理契約

```text
mono float32 16 kHz
 → 原resampy相当16→8 kHz
 → 左右496 zero padding
 → 994-sample sliding normalization、公式端処理
 → FCN-993 FP32 eval
 → 486-bin sigmoid activation
 → 公式local average cents
 → Hz / confidence
 → 10 ms gridから直接選択
```

- target時刻は`k × 160 / 16000`。nativeの`j=10k`を選ぶ。
- `p_len`に合わせて音声全体を伸縮する`np.interp`は使用しない。
- 通常契約は`p_len=floor(N16/160)`。指定長が実音声の有効gridを超える場合は、呼出側の明示paddingを要求する。
- 出力は厳密に`p_len`、float32 Hz。coarseは呼出側で作る。
- 空入力・0 framesは空配列。NaN/Inf入力は明示エラーとする。
- 無音にnetworkが有限pitchを出した場合、それをbaseline独自処理で0に変えない。
- native activation/confidenceを診断APIで取得できるようにする。

### Baselineに含めない処理

median、hysteresis、octave補正、aggregation、quantile正規化、confidence gateは既定で無効。

公式Viterbiは**比較用decoderとして検証ツールに実装**するが、初回の通常method設定には出さない。realtimeで全系列Viterbiと同じ結果を有限遅延で保証できないためである。後から公開する場合は、別profileと学習設定記録を必要とする。

### 共通API

FCN内部は既存predictorと独立した以下の責務に分ける。

- `FCNModel`：正規化済みTensor→activation。
- `FCNPreprocessor`：resampling、normalization、境界処理。
- `FCNDecoder`：activation→cents、Hz、confidence。
- `FCNRVCAdapter`：grid変換、V/UV、temporal処理。
- `FCNStream`：sample clock、context、確定frame、reset/flush。
- 互換wrapper：既存経路向け`get_f0(x, p_len)`。

内部結果は`frame_index / pitch_hz / voiced / confidence`を保持する。GPU経路はTensorのまま扱い、NumPyが必要な既存境界でだけ変換する。

## D. 改善型FCNの設計

network・weight・bin mapping・基本normalizationはbaselineと共有する。

### 1. 1 ms出力のrobust aggregation

10 ms frame `k`に対し、native時刻`[10k−5, 10k+5)` ms、すなわち10点を使う。真の音声端では存在する点だけで評価する。

候補比較は以下とする。

| 方法 | 利点 | 欠点 |
|---|---|---|
| Hz平均 | 単純 | octave outlierに弱い |
| confidence加重平均 | 弱い候補の寄与を下げる | 高confidenceの誤推定に引かれる |
| median | outlierに強い | confidenceを利用できない |
| **cents上のconfidence加重median** | octave外れ値へ比較的強く、confidenceを利用 | 急変がframe内にあると片側へ寄る |
| activation先行平均 | bin単位でconsensusを作れる | 複数pitchを混ぜる可能性 |

初期採用は**voiced候補だけのcents加重median**。累積重みが半分に達する最小pitchを選ぶ決定的規則とし、random tie-breakingを使わない。

### 2. Pitch・confidence・V/UVを分離

- local average pitchはconfidence gate前に保持。
- confidenceは最大activationとして保持し、確率とは呼ばない。
- 最終V/UV maskを独立に決定。
- 最後に`f0[~voiced]=0`。
- UV gapのpitch補間、最後のpitchの持ち越しは既定で行わない。
- 0 Hzと有声pitchのlinear interpolationを禁止する。

### 3. Temporal median

初期候補はnative **5 frames＝5 ms、未来2 ms**。

- confidenceに5点median。
- pitchは現在点と近傍の有効候補を用いてcents median。
- 現在点にconfidenceの支持がない場合、近傍pitchだけで有声化しない。
- 追加の10 ms grid上の3-frame medianは既定OFF。両方を重ねて約30 ms以上平滑化しない。

5 msは初期評価値であり、**要実測**。OFF・3・5・9 msを比較し、同等成績なら短い窓を選ぶ。

### 4. Confidenceとhysteresis

10 ms frame confidenceは、そのframe内の平滑化confidenceのmedianを使用する。

- UV→V：`confidence >= enter_threshold`。
- V→V：`confidence >= exit_threshold`かつ有効pitch候補あり。
- その他：UV。
- `exit_threshold <= enter_threshold`。
- 有効候補はraw confidenceと平滑化confidenceの双方がexit thresholdを満たす点。
- 開始状態はUV。
- hangover、UV中のpitch hold、gap fillingは0 ms／OFF。
- 完全なdigital silenceは改善型だけでUVとする。固定dB thresholdは初回defaultに追加しない。

これにより弱声のexit条件を緩めつつ、confidenceの支持がなくなった無声音へpitchを延長しない。ただしconfidenceの誤較正による偽有声は残り得るため、noise/breathで独立評価する。

**threshold値は現段階で決めない。** 実weightで分布を測っていないため、0.1を含め特定値を正当化できない。

決定手順は固定する。

1. 話者を分けたcalibration/validation/test setを作る。
2. raw・5 ms平滑化後・10 ms集約後のconfidence分布をV/UV別に測る。
3. enter/exitを0–1、0.01刻みで探索。
4. baselineの単一threshold方式も比較する。
5. validationでUV→V誤判定率が比較対象Mangio以下となる候補を優先し、その中で弱声・語尾のV→UV誤判定を最小化。
6. 同等ならhysteresis幅が小さい候補を選ぶ。
7. held-out testと聴感評価で合格した値をversioned profileへ固定。

条件を満たさない場合は「改善済みdefault」として出荷せず、評価失敗として扱う。実装者が任意の閾値を選んで穴埋めしない。

### 5. Octave jump

初回はlocal medianとaggregationによる短時間outlier抑制までとする。

前frameとの比較だけで自動的に÷2／×2する処理は入れない。真のoctave移動、falsetto、急なpitch bendを壊すためである。

持続的octave errorが残った場合はactivation内の第二候補を利用する処理を別実験として評価し、現profileへ無検証で追加しない。

### 6. Normalization

Mangioのquantile正規化は追加しない。FCNが期待するsliding mean/stdを維持する。

改善型では最外周のnormalization境界をzero extensionとしてstreamableにする。baselineの`wrap`との差は真の音声端に限定し、内部block境界ではどちらも実contextを使う。

## E. 学習用F0 extraction統合

`FeatureInput`へ二つのmethodを追加し、worker内でFCN predictorを一度構築する。

- 入力は既存`load_audio_16k()`のmono 16 kHz。
- `p_len=len(audio)//160`を明示。
- `f0_voiced`には最終mask適用後のHzを保存。内部continuous pitchを保存してUVを消さない。
- coarseは既存学習範囲50–1680 Hzを維持する。
- FCN検出範囲30–1000 Hzと、coarse量子化範囲を別の設定として管理する。
- profileは親processで確定し、workerに明示渡しする。workerが別tabのglobal設定を再読込しない。
- ファイルごとにtemporal stateをresetする。
- 長いファイルは固定出力block＋必要contextで処理し、内部境界へzero paddingを入れない。

### 抽出再利用と再現性

`model_info.json`に以下を記録する。

- method ID、profile versionと内容。
- weight SHA-256、architecture ID。
- resampler ID、normalization ID、grid origin/hop。
- decoder、threshold、median/aggregation設定。
- coarse量子化仕様。

F0出力はprofile fingerprintが一致する場合だけ再利用する。FCNへの切替、baseline↔改善型、threshold変更時は再抽出する。

途中失敗で旧/new F0が混在した状態を完了扱いしない。各ファイルの二つの出力が揃ったことを検証し、runの完了metadataは全件成功後に確定する。workerの例外を親へ伝え、`subprocess.run()`も失敗をUI/CLIに返す。

`extract_model.py`でprofileと量子化仕様をRVC checkpointへ引き継ぐ。未知の旧モデルにFCN profileがあったことを推測しない。

## F. 通常推論統合

### Pipelineとcache

`Pipeline`にFCN predictorのinstance cacheを追加する。

- keyはarchitecture、weight hash、device、dtype。
- baseline/improvedは同一networkを共有し、adapter stateは独立。
- 同じ変換モデルでの通常・batch処理ではnetworkを再利用。
- ファイル間でhysteresis等はreset。
- cleanup時に解放する。globalな単一model変数は使わない。

### F0抽出とpadding

FCNは真の16 kHz音声区間でF0を確定し、その後、合成用reflect paddingに対応するF0を構成する。reflectされた冒頭contextが改善型hysteresisの開始状態を変えないようにする。

- target時刻は常に10 ms整数grid。
- F0は合成chunk分割前に確定する。
- 合成用paddingはF0・mask・confidenceへ同じindex対応で適用する。
- 長音声のFCN内部分割は結果を変えない。
- batchの各入力は独立したstreamとして扱う。

既存のhigh-pass、学習preprocessing、realtime capture resamplingには上流差がある。共通性はまず**同じ16 kHz waveformをFCNへ渡した場合**に保証し、上流を含めた差は別試験で測る。既存全methodの前処理を同時に変更しない。

### Pitch補正と量子化

FCN推定はpitch shift前の入力F0を返す。

- 既存autotune／proposed pitch／手動shiftの呼出順は維持。
- FCNのUV maskを補正後に再適用。
- shift後Hzを1000 Hzへclipしない。
- coarseだけ所定範囲へ飽和させ、Hz値は保持する。

FCN経路には共通mel quantizerを使う。

- FCNで新規学習したcheckpoint：保存された50–1680 Hz仕様。
- metadataのないモデル：現ブランチの学習仕様50–1680 HzをFCN経路の既定。
- 外部モデル用に50–1100 Hzの互換指定を提供し、通常/realtimeで同じ値を使用。
- 既存methodを選択したときの量子化挙動は今回変更しない。

この互換指定はFCNの検出上限を変更する設定ではない。

## G. リアルタイム推論統合

### 基本方針

**初回はcentered処理＋有限lookaheadを採用する。**  
右端を毎回zero-padして最新F0を推定する低遅延近似は、初回defaultにも隠れたrealtime専用処理にも入れない。

### Streaming state

session所有の`FCNStream`に以下を保持する。

- 絶対16 kHz sample count。
- resamplerのphaseと左右context。
- 8 kHz raw/normalized ring buffer。
- 次に出力するnative frame index。
- median/aggregationの必要context。
- hysteresis state。
- 確定10 ms F0 ringとframe index。
- model参照。weightの再loadはしない。

新規入力samplesを一度だけ消費し、既に確定したframeをrolling window再解析で上書きしない。内部blockの開始位置は8 kHzのstride 8へ整列する。

### Resampling

16→8 kHzは、公式の`resampy`既定`kaiser_best`と同じ係数・phaseを使う固定比率FIRとして実装する。

- 係数はstartup時に一度取得・構築し、deviceへ配置。
- offline/streaming共通の処理を使う。
- ratio 1/2におけるresampyの加算式と端処理をoracleにして比較。
- sample countを保持し、odd-length chunkでも位相をずらさない。
- chunk端では未到着contextを待つ。真の開始/終了だけzero extension。
- Torch実装とresampyの演算順による差は数値一致試験で管理する。

`kaiser_best`は公式APIの既定で、50 zero crossingsのfilterである。正確な係数長からlookaheadを算出し、hard-codeだけに依存しない。[resampy仕様](https://resampy.readthedocs.io/en/stable/api.html)

48→16 kHzについても、FCN選択時は既存torchaudio filterの係数を維持したcontext付きstreaming wrapperを使用する。毎callbackの独立resampleと整数切捨てによるsample driftを避ける。他methodのcapture経路は維持する。

### Latency budget

| 要素 | baseline | 改善型初期候補 |
|---|---:|---:|
| networkだけの未来context | 62 ms | 62 ms |
| normalization込み | **124 ms** | **124 ms** |
| native median | 0 | 2 ms |
| 10点集約の右側 | 0 | 4 ms |
| 16→8 kHz FIR | 約6.25 ms以内を保守的初期見積り | 同左 |
| 合計の初期見積り | 約130.25 ms | 約136.25 ms |

capture resamplerの有限supportも加算し、10 ms単位へ切り上げる。**初期設計のholdbackは両者140 ms**とし、係数由来の厳密計算で不足しないことをテストする。

この140 msは総end-to-end latencyではない。実際にはcapture block待ち、計算、SOLA、入出力device、queueが加わる。

- block内の到着時刻により待ち時間は変わる。
- `extra_convert_size`は過去contextであり、未来lookaheadの代用にはならない。
- 現在の512 ms chunk defaultをFCN追加だけで高速realtimeと評価しない。
- 20/40/80/160 ms等のchunkでも計測し、実機がdeadlineを満たす範囲を提示する。

### 音声とF0の同期

`core.py`にFCN用の確定時刻を導入する。

1. 最新入力をanalysis ringへ書く。
2. 必要な未来contextが揃ったF0 frameだけ確定する。
3. その確定時刻を終端とする音声windowをembedder/RVCへ渡す。
4. pitch/pitchfも同じ絶対時刻範囲から取得する。
5. 既存SOLAへ渡す。

**F0だけ140 ms遅らせて最新音声へ当てない。**  
volume envelope・無音判定も対応する遅延音声に合わせ、pending tailが残っている間にflushしない。

`circular_write()`への「長さだけでの追記」はFCN経路では使わず、frame index対応で格納する。warm-up後、停止、再接続、sample discontinuityではresampler/FCN/temporal stateをまとめてresetする。

### Baselineの境界例外

公式はzero paddingの外側へ`wrap`を使うため、冒頭の最外側normalization sampleがファイル末尾に依存し得る。無限streamでは未知の末尾を参照できない。

- offline baselineは公式端処理を完全再現する。
- realtime baselineの真のsession開始では未知外側を0とする。
- この例外は明示し、開始端を除いた一致と開始端の差を別々に検証する。
- 通常のcallback境界では例外を認めず、実contextで一致させる。
- 改善型はofflineも同じstreamable端処理を使う。

### Cache・compile・CPU

- networkとresamplerはaudio deviceを開く前に構築・warm-up。
- `callbacks.py`のwarm-upを必要contextが埋まるまで実行する。
- CPU時は現行の無条件`torch.cuda.synchronize()`をguardする。
- FCN入力を既存の一律`.cpu().numpy()`より前にdispatchする。
- activationとdecoderはdevice上に保持。CPU V/UV state処理が必要なら10 msへ縮約した小配列だけ転送する。
- compileは初期OFF。CPUはeager。
- 任意compileはnetworkのみ。Python state machine、resampler state、hysteresisは対象外。
- CUDA compile失敗はsession所有のeager networkへfallbackし、stream stateは維持する。
- CPUでdeadline未達の場合は測定結果を表示し、別methodへ黙って変更しない。

## H. Model／weight移植方式

| 方式 | 現行Applioとの適合性 | realtime上の評価 | 採否 |
|---|---|---|---|
| TensorFlow/Keras runtime | 別runtime・GPU管理が増える | sessionとbufferの統合が複雑 | 不採用 |
| **PyTorch移植＋一度だけ変換** | 既存device・Tensor・compileと整合 | 転送を抑え、同一processで管理可能 | **採用** |
| ONNX | 既存ONNX codeはある | provider/I/O binding・別cacheが必要 | 初回は不採用 |

速度優位は実測前には断定しない。採用理由は現行構成との整合性と、単一Tensor経路にできる点。

### 完全対応させる項目

- Keras Conv2D kernel `[K,1,Cin,Cout]` → PyTorch Conv1d `[Cout,Cin,K]`。
- biasはそのまま。
- `Conv → ReLU → Pool → BN`の順序を維持する。CREPEの順序へ合わせない。
- Poolはkernel 2、stride 2、padding 0、`ceil_mode=False`。
- BN：gamma→weight、beta→bias、moving mean/variance→running buffers。
- `eps=0.001`、`eval()`固定、勾配不要。
- momentumの意味の違いは推論結果に影響させず、学習用BN更新を行わない。
- classifierはkernel 4のConv1d＋sigmoid。Linearやsoftmaxへ変更しない。
- output layoutは内部`[B,T,486]`へ統一。
- 初期はFP32、AMP/FP16/BF16・Conv/BN fusionを使わない。

### 変換ツール

実装段階で、手元のHDF5を入力する専用scriptを作る。

- network downloadを変換scriptに埋め込まない。
- datasetはlayer名・parameter名で対応し、列挙順に依存しない。
- 実HDF5の各shape・dtype・finite値を検証。
- unknown/missing/duplicate parameterはエラー。
- `num_batches_tracked`等のPyTorch側だけのbufferを明示初期化。
- `state_dict`は`strict=True`でload検証。
- source commit、HDF5 SHA-256、変換version、出力hashをmanifestへ保存。
- baselineと改善型で同じweight hashを要求する。

**HDF5内部の実dataset名・shape・値は今回未確認。** Web経由のbinary内容取得はできず、ローカルの`fcnf0++.pt`も代替ではない。実装開始時に原HDF5を検査することを最初のgateにする。

## I. Model asset／dependency／license

### Asset

配置予定：

```text
rvc/models/predictors/fcn-993.pt
```

architecture・profile定義、配布manifest、licenseは追跡対象のcode/docs側に置く。現行`.gitignore`は`rvc/models`とweight拡張子を除外するため、重要なmanifestをasset directoryだけに置かない。

配布方針：

1. 最初は検証済みconverterによるlocal provisioningを正式にサポート。
2. loaderはasset不在なら必要パスを明示して開始前に失敗。
3. 検証済みconverted weightを公開できた段階で、immutable URLとSHA-256をmanifestへ登録。
4. 既存prerequisite downloaderにFCN assetを追加するが、未公開URLを推測して登録しない。
5. downloadはsetup段階だけ。callbackやpredictor forwardでは実行しない。

### Dependencies

- runtimeは既存PyTorch、NumPy、torchaudio、resampyを中心とする。
- resampyは現環境に存在するが、requirementsには直接記載されていない。採用版を検証後、直接dependencyとして固定する。
- TensorFlow/Kerasは数値一致検証用の隔離環境だけ。
- h5pyは変換用dependencyだけ。
- hmmlearnはruntime不要。公式Viterbi検証にはversionを固定したoracleを使う。
- resampling係数を同梱する場合は生成条件・checksum・元licenseも記録する。

### License・attribution

原repositoryのLICENSEはMITで、Luc Ardaillon（2019）とJong Wook Kim（2018）のcopyrightが記載される。READMEは論文引用を求めている。[LICENSE](https://github.com/ardaillon/FCN-f0/blob/master/LICENSE)、[引用情報](https://github.com/ardaillon/FCN-f0#references)

確認したREADME・LICENSEには、pretrained weightだけを別条件にする記載は見つからなかった。ただし今回HDF5内部metadataは未確認なので、「weight専用の明示許諾を確認済み」とは記載しない。

配布前に固定commitの関連文書・weight metadataを再確認し、原MIT全文、両copyright、論文、source URL、改変・変換内容を同梱する。追加条件が見つかった場合はconverted weight公開を停止し、local conversion経路を維持する。

## J. UI／CLI変更

### Method choices

追加対象：

- training extraction。
- ordinary inference。
- batch inference。
- realtime inference。
- TTSのRVC変換。
- F0 extractor utility。
- `core.py`のinfer、batch、TTS、extract用CLI。

新規の軽量`f0_methods.py`で、FCN method IDと用途別choicesを共通化する。既存CREPE mappingと既存dispatch classを大規模factoryへ置換しない。

### Profile設定

FCN選択時だけ共通の詳細設定を表示する。

- baseline／改善型の説明。
- 改善型enter/exit threshold。
- versioned profile選択・読込。
- coarse量子化互換指定。
- 必要lookaheadと現在の処理時間。
- 設定変更は次の抽出run／推論開始／realtime session開始時に固定。

CLIには`--fcn_profile`でprofile JSONを渡せるようにする。未指定は同梱profile。UIとCLIが同じresolverを通る。

推論の優先順位は、明示profile→選択methodに一致するcheckpoint内profile→同梱default。選択methodとcheckpointのmethodが異なる場合は自動的にmethodを切り替えず、差を表示する。

### F0 utility

現状はcentsを返すのにHzと表示し、時刻計算も10 ms gridと整合しない。

- 従来`extract_f0()`のcents互換を保つ。
- Hz・timestamp・confidence・maskを返す`extract_track()`を追加。
- FCNのutilityは`extract_track()`を使い、CSVはseconds/Hzを明記。
- plot/CSVの検証を行い、FCNが比較に使える正しい時刻軸を提供する。

## K. テスト計画

### 1. 原実装との数値一致

二段階で切り分ける。

**Network単体**

同じ正規化済み8 kHz TensorをKeras/PyTorchへ入力し、各Conv、Pool、BN、classifier、sigmoid出力を比較する。

- 993、1000、1001 samples、複数native frames、複数秒。
- 正弦、倍音音声、noise、impulse、DC、silence、非常に小さい振幅。
- CPU FP32を最初の基準にし、続いてCUDA。
- TF32・AMPを無効化。
- 初期許容値はactivation/confidenceで`atol=1e-5, rtol=1e-4`。
- 超過時はlayer単位で原因確認し、理由なく許容値を緩めない。

**前処理込み**

- resample waveform。
- sliding mean/stdとnormalized samples。
- activation。
- confidence。
- decoded cents/Hz。
- native時刻と10 ms frame index。

非退化の有声frameではdecoded差0.1 cent以内を初期合格基準とする。argmax tie・ほぼゼロactivationは分離報告し、tie規則とfinite処理を検証する。

公式端処理、994窓、population std、epsilon置換条件を独立に検証する。Keras runtimeの互換修正が必要なら、optimizer等の推論無関係部分の変更を記録する。

### 2. Streaming一致

同一16 kHz音声を、全長・固定chunk・不規則chunkで入力する。

- odd sample数、10 ms未満、strideを跨ぐchunk。
- 全長とstreamingで同じ確定frame数・時刻・mask。
- 改善型hysteresisがchunkサイズで変わらない。
- 1時間相当でframe driftなし。
- callback境界付近に局所的なpitch jumpが増えない。
- reset、warm-up、無音、再接続、終端flush。
- baselineの公式wrap開始端の差は別枠で説明し、通常境界の不一致をその例外へ混ぜない。

### 3. F0比較

最低限、同一音声に対して以下を比較する。

1. `mangio-crepe-full-speech`、既定Viterbi。
2. 同methodのweighted argmaxを再現性確認用に追加。
3. `fcn-993`。
4. `fcn-993-rvc`。
5. 改善型の各処理をOFFにしたablation。

対象区間は、男性通常声、高めの男性声、falsetto、vibrato、pitch bend、立ち上がり、語尾、breath移行、弱声、無声音、子音、急激なpitch changeをすべて含める。

測定項目：

- referenceに対するcents error、RPA、octave error。
- V→UV／UV→V誤判定。
- onset/offset時間誤差。
- voiced island、gap数、連続frameの変化量。
- 持続音のjitterとvibrato深さ・周期の保存。
- 語尾の欠落時間、breathへ誤延長した時間。

### 4. 聴感

同一RVC checkpoint、seed、index、protect、volume、pitch shiftで比較する。最初はautotune/proposed pitchをOFFにする。

- 男性→女性を想定し、+12 semitoneを主要条件、+7/+19を補助条件。
- 余韻・弱声・語尾の自然さ、息への移行、子音のbuzz、急変の追従を評価。
- blind A/BまたはABXと、区間ごとの評価を記録。
- jitterが減っただけで採用しない。vibratoや表現が失われた場合は不合格。
- F0 extractor差の評価には同一checkpointを使う。
- 学習方式の評価は同一dataset・seed・recipeで別途行い、両者を混同しない。

### 5. 性能・realtime

CPU/CUDA、compile OFF/ON、cold/warm、複数chunk長で測定する。

- F0単体・全pipelineのRTF。
- p50/p95/p99処理時間。
- model load時間、compile時間、再compile回数。
- GPU使用率、VRAM、CPU負荷、RSS、転送量。
- 30分以上の連続運転、underflow/drop、queue増大、memory増大。
- loopbackで総遅延を測り、UI処理時間と分ける。

realtime合格は、対象設定で処理がblock deadlineを継続的に満たし、steady-stateで再load・再compile・queue増大が起きないこと。compileは数値・mask一致を満たし、startupを除いても実効改善がある場合だけ推奨する。

### 6. Regression

既存CREPE全capacity・alias・speech、Mangio全種、RMVPE、FCPE、Swiftの対応経路を検証する。

- 現在Swiftが未対応の学習経路を、FCN追加で対応済みと誤認させない。
- 未接続CREPE_ONNXは最低限import/API smokeを維持する。
- 既存decoder設定、compile設定、CPU fallback、CUDA、batch model reuse。
- 既存`test_crepe_models.py`、`test_crepe_decoder.py`、`test_extract_compile.py`、realtime compile・silence/seedテスト。
- 非FCN methodではF0値・mask・量子化結果を意図せず変えない。

## L. 変更予定ファイル一覧

以下は実装時の予定であり、今回作成・変更していない。

| ファイル | 区分 | 変更内容 |
|---|---|---|
| `rvc/lib/predictors/fcn/__init__.py` | 新規 | 公開API |
| `rvc/lib/predictors/fcn/model.py` | 新規 | FCN-993 network、architecture metadata |
| `rvc/lib/predictors/fcn/preprocess.py` | 新規 | resampler、公式normalization、端処理 |
| `rvc/lib/predictors/fcn/decoder.py` | 新規 | 486-bin local average、confidence |
| `rvc/lib/predictors/fcn/adapter.py` | 新規 | baseline/improved、grid、V/UV、aggregation |
| `rvc/lib/predictors/fcn/streaming.py` | 新規 | ring buffer、sample clock、確定frame |
| `rvc/lib/predictors/fcn/profiles.py` | 新規 | profile validation・解決・fingerprint |
| `rvc/lib/predictors/fcn/assets.json` | 新規 | architecture・source/hash・配布manifest |
| `rvc/lib/predictors/f0_methods.py` | 新規 | 用途別method choices |
| `rvc/lib/predictors/f0_quantization.py` | 新規 | FCN経路共通mel quantizer・互換指定 |
| `rvc/lib/predictors/f0.py` | 修正 | FCN互換wrapper公開 |
| `rvc/lib/predictors/F0Extractor.py` | 修正 | FCNとtrack API |
| `rvc/train/extract/extract.py` | 修正 | dispatch、profile、再利用判定、失敗伝播 |
| `rvc/train/extract/compile_extract.py` | 修正 | FCN任意compile |
| `rvc/train/process/extract_model.py` | 修正 | checkpointへprofile・量子化仕様保存 |
| `rvc/infer/pipeline.py` | 修正 | cache、FCN padding/dispatch、UV保持、量子化 |
| `rvc/infer/infer.py` | 修正 | checkpoint profile受渡し、cache lifecycle |
| `rvc/realtime/pipeline.py` | 修正 | Tensor dispatch、timestamp対応、共通量子化 |
| `rvc/realtime/core.py` | 修正 | holdback、同期window、streaming capture、flush |
| `rvc/realtime/callbacks.py` | 修正 | preloading、warm-up、CPU guard |
| `rvc/realtime/audio.py` | 修正 | discontinuity/reset通知、遅延表示用情報 |
| `rvc/realtime/compile_session.py` | 修正 | FCN任意compileのsession所有・status |
| `core.py` | 修正 | choices、profile CLI引数、subprocess失敗伝播 |
| `tabs/components.py` | 修正 | FCN共通設定UI |
| `tabs/train/train.py` | 修正 | extraction method/profile |
| `tabs/inference/inference.py` | 修正 | single/batch method/profile |
| `tabs/realtime/realtime.py` | 修正 | method/profile、lookahead表示 |
| `tabs/realtime/template.py` | 修正 | profile・互換設定の保存 |
| `tabs/tts/tts.py` | 修正 | method/profile |
| `tabs/extra/sections/f0_extractor.py` | 修正 | choices、正しい時刻/単位 |
| `tabs/settings/sections/torch_compile.py` | 修正 | FCN専用opt-in設定 |
| `rvc/lib/tools/prerequisites_download.py` | 修正 | 公開済みFCN assetの取得 |
| `requirements.txt` / `requirementspy313.txt` | 修正 | resampyの直接依存宣言・検証版固定 |
| `tools/convert_fcn993.py` | 新規 | HDF5→PyTorch変換 |
| `tools/validate_fcn993.py` | 新規 | Keras parity、公式Viterbi比較 |
| `tools/benchmark_fcn_f0.py` | 新規 | F0・性能・streaming評価 |
| `tools/requirements-fcn-validation.txt` | 新規 | 隔離検証/変換dependency |
| `tests/test_fcn_model.py` | 新規 | shape、mapping、変換検証 |
| `tests/test_fcn_adapter.py` | 新規 | V/UV、grid、aggregation |
| `tests/test_fcn_streaming.py` | 新規 | chunk一致、latency、reset |
| `tests/test_fcn_integration.py` | 新規 | train/infer/realtime/CLI/cache |
| 既存extract/realtimeテスト | 修正 | 共通境界とCPU regression |
| `docs/fcn-993.md` | 新規 | profile、制約、検証結果、利用方法 |
| `docs/licenses/FCN-f0-LICENSE` | 新規 | 原MIT全文 |
| `.gitignore` | 修正 | 検証requirementsのみ必要な除外解除 |
| `rvc/models/predictors/fcn-993.pt` | 新規asset | 検証後に配置。通常Git追跡対象外 |

`crepe_models.py`、`crepe_decoder.py`、既存torchcrepe package自体は原則変更しない。

## M. 実装順序

1. **参照固定**  
   upstream commit、実HDF5、license、dependency版を固定。HDF5 dataset一覧とchecksumを取得する。

2. **Network移植とweight変換**  
   architecture・BN・layoutを再現し、layer単位parityを通す。

3. **Baseline前処理とdecoder**  
   resampling、994 normalization、padding、local average、1 ms時刻を再現する。

4. **Baselineの10 ms統合**  
   学習・通常・utilityへ追加。profile記録、再抽出判定、量子化仕様を揃える。

5. **Baseline realtime**  
   resampler state、sample clock、holdback、audio/F0同期を実装。chunk一致を通す。

6. **改善adapter**  
   baselineに変更を入れず、median、aggregation、V/UV、hysteresisを追加する。

7. **Confidence校正**  
   固定手順でthreshold/windowを決め、profileをversion固定する。

8. **全UI／CLI／batch統合**  
   同じprofileが全経路へ届くことを確認する。

9. **性能最適化**  
   eagerを基準にcompile、block長、転送を評価する。数値一致を維持できない最適化は採用しない。

10. **聴感・長時間realtime・regression**  
    合格後にasset配布manifestとdefaultを確定する。

baseline parityとbaseline realtimeが通る前に、改善処理で誤差を覆い隠さない。

## N. リスク

| リスク | 対応 |
|---|---|
| 8 kHz化で高域情報が失われる | 学習条件として維持。alias抑制filterを固定し、16 kHzをnetworkへ直接入れない |
| 1000 Hz超の入力 | 正しく範囲外検出できる保証はない。1000 Hz付近・subharmonic・低confidence等になり得ることを試験する |
| 男性→高女性声 | shift前入力が1000 Hz未満ならFCN上限と変換後上限は別問題。shift後Hzは1000 Hzでclipしない |
| BaselineにUV判定がない | 原実装どおりとして明記。改善型のUV性能と分けて評価する |
| 弱声を残すとbreathも有声化する | exit threshold校正、持越しなし、breath/noiseの独立評価 |
| normalizationがノイズを増幅 | 全体quantileで隠さず、実confidence分布とnoise条件で検証する |
| realtime lookaheadを62 msと誤認 | normalization込み124 ms＋filter・aggregationを計上する |
| chunk boundary／phase drift | 絶対sample clock、有限context、確定frame方式 |
| 公式wrap境界の非因果性 | baseline開始端の例外を明示。内部境界とは区別する |
| Keras weight変換 | 実dataset名、shape、BN順序、epsilonを検証し、strict loadとlayer parity |
| compileでF0が閾値を跨ぐ | activationだけでなくmask一致を確認。eager fallback |
| profile変更後に古い学習F0を再利用 | fingerprintと完了metadataで検出 |
| coarse仕様の不一致 | FCN経路の共通quantizerとcheckpoint metadata |
| 上流preprocessing差 | predictor単体一致とend-to-end差を別測定する |
| CPU速度不足 | 対応chunk条件を実測。別methodへ無通知fallbackしない |

## O. 推奨default

| 項目 | FCN-993 | FCN-993-RVC |
|---|---|---|
| Network / weight | 原FCN-993 | 同一 |
| 精度 | FP32 eval | FP32 eval |
| Decoder | 公式local weighted average | 同左 |
| Native Viterbi | OFF、検証ツールのみ | OFF、検証ツールのみ |
| Confidence threshold | **無効** | enter/exit分離、**値は要実測** |
| Temporal filtering | OFF | native 5 ms medianを初期候補、**要実測** |
| 10 ms化 | nativeの10点おき直接選択 | 10点区間のvoiced候補cents加重median |
| Pitch / UV補間 | なし | なし |
| Hysteresis | なし | ON、hangover 0 ms |
| Octave強制補正 | OFF | OFF |
| Quantile正規化 | なし | なし |
| 16→8 kHz | 原resampy相当FIR | 同一 |
| Realtime | centered、確定frameのみ | 同一＋共通temporal state |
| Holdback | 初期設計140 ms、係数から検証 | 同左 |
| torch.compile | OFF | OFF |
| FCN新規学習のcoarse範囲 | 50–1680 Hz | 同左 |
| 改善型profile確定条件 | 該当なし | 数値・V/UV・語尾聴感・realtime試験合格 |

**完了条件は、二つのmethodを選択できることではなく、原実装parity、学習設定の再現性、通常・streamingの時刻整合、変換後の語尾・弱声の評価まで通ることとする。**

</proposed_plan>
