# チェックポイントのノイズを「耳に合う形」で測る（知見）

2026-10-05、48k RingFormer+ の事前学習
`logs/pretrain_20261004_48k_codenamePlus_1100_rmvpe112ali_kushinada` の 165〜200e を比べたときの記録。
**耳の判断と自動指標が食い違ったときに、何をどう測れば耳に合ったか**と、
その結果わかった**学習中のベスト保存の問題（2026-10-05 時点で未修正）**をまとめる。

## 結論（先に）

- 耳: 「165e より 185e の方がノイズのような音が明らかに少ない」。
- 当時の自動指標（`rvc/train/evaluation.py` の `voiced_hf_flatness`、`A_/B_/C_noise`）は
  **165e が最良**と出し、耳と逆だった。
- **ノイズの「量」（エネルギー）を元音声と比べ、倍音の間・無音・無声に分けて測ると耳と一致した。**
  185e は聴いた1クリップの 8 シードすべてで 165e より少なく、70 クリップでも 58〜61 クリップで少なかった。
- 食い違いの原因は、平坦さ（spectral flatness）が**スペクトルの形**しか見ていないこと。
  ノイズが減っても、倍音側も一緒に変われば平坦さはほとんど動かない。

## 耳に合う判定にたどり着いた依頼の仕方

最初は「元音声とのノイズが大きいか小さいか」を平坦さで測っていて、外れていた。
効いた指示・手順は次のとおり。別のセッションで同じことをするなら、最初からこう頼む（こうする）。

1. **「耳で聴いたファイルそのものを測って」**
   TensorBoard の `gen/audio_*` や自分で書き出した wav など、実際に聴いた音を測る。
   別のクリップの平均値を見せても、耳との比較にならない。
2. **「同じチェックポイントを複数シードで生成して、毎回起きるのか偶然なのかを見せて」**
   推論は乱数を使う（`z_p` のノイズと SineGen）。同じ G でもシードで値が揺れる
   （このクリップの平坦さで 0.90〜1.16）。**エポック間の差は、このシード間の揺れと比べる。**
   TensorBoard の `gen/audio_*` は保存のたびに違う乱数なので、エポックの差とシードの差が混ざっている。
3. **「ノイズの量（エネルギー）を、元音声と同じ区間・同じ帯域で比べて」**
   形（平坦さ）ではなく量。dB で、元音声に対して何 dB 多い・少ないか。
4. **「無音・無声・有声の倍音の間に分けて、帯域ごとに出して」**
   有声部分だけ、または中央値だけで見ると、息継ぎや語尾の無音に乗るノイズが消える。
   今回、165e は無音部分で元音声より +1.4 dB ノイズを足していて、185e は −0.5 dB だった。
5. **「1クリップで分かったことを、多数クリップで確かめて」**
   聴いた1クリップは話者 6 の 3 秒だけ。70 クリップ（`pick_auto_trained(filelist, 70)`、7 話者均等）で、
   クリップごとの対応比較（165e との差と、そのブートストラップ 95% 区間）で確かめた。

逆に、**やってはいけなかったこと**:

- 1つの代理指標だけで「165e がはっきり最良」と結論した。耳と食い違った時点で、指標の方を疑うべきだった。
- 4 kHz 以上をナイキストまで全部見ていた。音声には 12 kHz 以上がほとんど無く、平坦さがノイズの床を測ってしまう。
  元音声を 16 ビットにしただけで値が 1.43 倍になった（4〜12 kHz なら 1.005 倍）。
  TensorBoard の音声は 16 ビットで保存されるので、float の元音声と比べると不公平になる。
  → 帯域を 4〜12 kHz に直した（`HF_BAND`、修正済み）。それでも耳とは合わなかった。
- 測定スクリプトを片付けで消し、前回と同じ測り方を再現できなくなった。測り方はこの文書の末尾に残す。

## 測った値

### 聴いたクリップ（`6_6069_1`、話者 6、3 秒）、各 8 シード

元音声に対するエネルギーの差（dB、マイナス = 元より少ない）。± はシード間の標準偏差。

| 区間・帯域 | 165e | 180e | 185e | 185e − 165e |
|---|---|---|---|---|
| 無音 4〜8 kHz | +1.4 ±0.3 | +1.6 ±0.4 | −0.5 ±0.4 | −1.9（8/8 シードで少ない） |
| 無音 8〜12 kHz | +1.2 ±0.4 | +0.7 ±0.3 | −0.0 ±0.4 | −1.2（8/8） |
| 無声 4〜8 kHz | +0.5 ±0.4 | +0.8 ±0.4 | −1.2 ±0.2 | −1.7（8/8） |
| 有声・倍音の間 4〜8 kHz | −1.3 ±0.1 | −2.0 ±0.1 | −2.7 ±0.1 | −1.4（8/8） |
| 有声・倍音の間 8〜12 kHz | −0.1 ±0.4 | −0.9 ±0.3 | −1.2 ±0.4 | −1.1（8/8） |
| 16〜24 kHz（全区間） | ほぼ ±0.5 以内で差なし | | | |

同じクリップ・同じシードでの平坦さ（4〜12 kHz、元 = 1）は 165e 1.03、180e 1.01、185e 1.05 で、
**差がシードの揺れ（±0.1）に埋もれていた。** エネルギーで見ると、差は 1〜2 dB でシードの揺れ（0.1〜0.4 dB）よりずっと大きい。

### 70 クリップ（学習データ、7 話者均等、シード 2 と 3 の平均）

165e との差（dB）と、165e より少なかったクリップ数。

| エポック | 有声・倍音の間 4〜8 kHz | 有声・倍音の間 8〜12 kHz | 有声以外 4〜12 kHz | 平坦さ 4〜12 kHz（当時の指標） |
|---|---|---|---|---|
| 60 | −0.25 | −0.36 | −0.56 | ×1.05 |
| 95 | −0.17 | −0.29 | −0.44 | ×0.99 |
| 165 | 0 | 0 | 0 | 1 |
| 170 | +0.23 | −0.21 | −0.28 | ×1.02 |
| 175 | −0.31 | −0.60 | −0.64 | ×1.04 |
| 180 | −0.28 | −0.17 | −0.01 | ×1.07 |
| **185** | **−0.74**（61/70） | **−0.86**（61/70） | −0.89（58/69） | ×1.05 |
| 190 | −0.72 | −0.47 | −0.68 | ×1.13 |
| 195 | −0.66 | −0.21 | −0.73 | ×1.12 |
| 200 | −0.62 | −0.72 | **−0.94** | ×1.10 |

- エネルギーでは 185e がほぼ最少、165e は 165〜200e の中でむしろ多い方。耳と一致。
- 平坦さでは 165e が最良に見え、順位がほぼ逆。
- mel L1 は 165〜200e で ±0.5% 以内の横ばいで、どのエポックも区別できない。
- 生成音は、ほとんどの区間で元音声よりノイズが少ない（マイナス）。
  **「元より少ないほど良い」とは限らない**（下げすぎると子音や息が弱まり、こもる）。
  今回は耳が 185e を選んでいるので問題にならなかったが、指標にするときは歯止めが要る（後述）。

## 学習中のベスト保存への影響（2026-10-05 時点で未修正）

保存の設定の「評価が良い上位を保存する」は、`rvc/train/evaluation.py` の指標で順位を付けている。
今のままだと、**耳に合わないチェックポイントを残す可能性が高い。**

- **`noise` 基準:** `checkpoint_policy.score_of` は `B_noise`・`C_noise`（無ければ `A_noise`）の平均を使う。
  中身は平坦さなので、今回のランだと 165e のような「平坦さは低いがノイズ量は多い」ものを選ぶ。
- **`combined` 基準（既定）:** mel ÷ 最初の評価の mel + noise ÷ 最初の評価の noise。
  mel が横ばいの区間では、実質的に平坦さで決まるので、`noise` と同じ問題が出る。
- **`mel` 基準:** 平坦さの影響は受けないが、165〜200e のように mel が ±0.5% で横ばいの区間では、区別できない。
- つまり**今は3つの基準のどれも、耳が選ぶチェックポイントを選べない。**
  直すまでの代わり: 「G/D も残す（間隔 × n）」で一定間隔に残し、あとからこの文書の方法で測る。

### 直すときに考えること

1. **指標の定義:** 有声部分の倍音の間のエネルギー（4〜12 kHz）を、元音声との比（線形、1 = 元と同じ）で測る案。
   `score_of` の `combined` は割り算で正規化するので、dB ではなく線形の比にする。
   無音部分のノイズ（今回 165e だけが元より +1.4 dB）も見るなら、別の項として足す。
   無声部分の 4〜12 kHz は子音（s, sh）そのものなので、ノイズとして数えない方がよい。
2. **下げすぎの歯止め:** 「少ないほど良い」だと、こもった出力が勝つ。
   `combined` なら mel の誤差が増えるので歯止めになる。`noise` 単独の基準では効かないので、
   元より少ない側は頭打ちにする（例: `max(ratio, 下限)`）などを検討する。
3. **倍音の間の分解能:** STFT 2048 点（48 kHz で 23.4 Hz 刻み、Hann の主ローブ半幅 約 47 Hz）では、
   F0 が 約 95 Hz 未満だと倍音の間の bin が主ローブに入る。低い男声で値が濁る。未検証。
   また、窓の中でピッチが動くと高次倍音がにじむ（8 kHz で 40 次なら、1% のピッチ変化で 80 Hz）。
   元音声も同じ条件なので比を取れば相殺される方向だが、未検証。
4. **記録の互換性:** 指標の定義を変えると、`checkpoints/state.json` の `evaluations`・`baseline`・`best` が
   古い定義のまま残る。`state.json` に指標の版（例: `metric_version`）を持たせ、
   変わっていたら baseline と順位を作り直す処理が要る。
   2026-10-05 時点では、どの学習フォルダにも `state.json` は無い（まだ誰も使っていない）。
5. **触る場所:**
   - `rvc/train/evaluation.py`: `voiced_hf_flatness`、`Evaluator.evaluate` の `*_noise`。
   - `rvc/train/checkpoint_policy.py`: `score_of`、`baseline_of`、`CheckpointKeeper._load_state`（版の確認）。
   - テスト: `tests/test_evaluation.py` の `MetricTest`、`tests/test_checkpoint_policy.py` の `ScoreTest`。
   - 説明: `CLAUDE.md` の Checkpoint policy の節、`docs/TRAINING_STATUS.md`、
     Training タブの「ベストの基準」の説明文（en_US / ja_JA）、`docs/pretrain-manual-guide.md` の推奨基準。
6. **シード:** 評価器はシード (0, 1) の 2 回平均。エネルギーのシード間の揺れは 1 クリップで 0.1〜0.4 dB なので、
   クリップ数が数本でも足りる見込み。ただし 1 クリップ 1 シードの値で判断しない。

## 再現のしかた

G の読み込み（`G_2333333.pth` も書き出しモデルの `weight` も同じ。書き出しは fp16 だが、測った値は G と一致した）:

```python
net_g = Synthesizer(config.data.filter_length // 2 + 1,
                    config.train.segment_size // config.data.hop_length,
                    **config.model, use_f0=True, sr=config.data.sample_rate,
                    vocoder="CodenameRingFormer").cuda().eval()
ck = torch.load(path, map_location="cpu", weights_only=True)["model"]
w = {k.replace(".weight_v", ".parametrizations.weight.original1")
      .replace(".weight_g", ".parametrizations.weight.original0"): v for k, v in ck.items()}
net_g.load_state_dict({k: w[k] for k in net_g.state_dict()})
```

クリップは `Evaluator(..., pick_auto_trained(filelist, N), ...)` の `items`（phone・coarse・f0・audio が揃う）を使い、
`torch.random.fork_rng` の中で `torch.manual_seed(seed)` してから fp16 autocast で `net_g.infer(...)` する。

ノイズ量の分解（元音声と生成音は同じ長さに切りそろえる。生成音は元より 2 フレーム短いことがある）:

```python
from scipy import signal
import numpy as np

def noise_energy_db(generated, original, f0, sr=48000, n_fft=2048):
    """Energy of generated vs original (dB) per frame type and band. f0 at 100 frames/s."""
    hop = sr // 100
    n = min(len(generated), len(original))
    def spec(a):
        f, t, Z = signal.stft(a[:n], sr, nperseg=n_fft, noverlap=n_fft - hop)
        return f, t, np.abs(Z) ** 2 + 1e-20
    f, t, PO = spec(original)
    _, _, PG = spec(generated)
    fr = np.asarray(f0)[np.clip(np.round(t * 100).astype(int), 0, len(f0) - 1)]
    voiced = fr > 0
    level = 10 * np.log10(PO.mean(0))
    silent = ~voiced & (level < np.percentile(level, 30))   # pauses
    unvoiced = ~voiced & ~silent                              # consonants
    out = {}
    for lo, hi in ((4000, 8000), (8000, 12000)):
        band = (f >= lo) & (f < hi)
        for name, frames in (("silence", silent), ("unvoiced", unvoiced)):
            if frames.any():
                out[f"{name} {lo//1000}-{hi//1000}k"] = 10 * np.log10(
                    PG[band][:, frames].sum() / PO[band][:, frames].sum())
        if voiced.any():
            d = (f[:, None] / fr[voiced][None, :]) % 1.0
            between = band[:, None] & (d > 0.3) & (d < 0.7)     # bins between harmonics
            out[f"voiced-between {lo//1000}-{hi//1000}k"] = 10 * np.log10(
                (PG[:, voiced] * between).sum() / (PO[:, voiced] * between).sum())
    return out
```

比べ方: エポックごとに 8 シード以上で生成し、エポック間の差がシード間の揺れより大きいかを見る。
多数クリップでは、基準のエポックとのクリップごとの差を取り、平均と 95% 区間（ブートストラップ）と、
基準より少なかったクリップ数を出す。
