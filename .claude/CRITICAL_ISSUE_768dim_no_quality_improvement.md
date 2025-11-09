# 【重大問題】768次元モデル - Loss低下するも品質改善せず

**日時:** 2025-11-06 23:00+
**状態:** 🔴 CRITICAL - 根本的な問題の可能性
**モデル:** `naru_x_C768_3` (50 epochs, 9400 steps)

---

## 問題の概要

### 異常な症状

**学習の進行:**
```
8エポック:  lowest_value ~35    → ほぼノイズ、声として認識できない
43エポック: lowest_value ~30.7  → 依然としてほぼノイズ、改善なし
50エポック: lowest_value ~30程度（推定）→ 状態不明
```

**期待される挙動との乖離:**
- ✅ lowest_valueは順調に低下（35 → 30.7）
- ❌ しかし品質は全く改善していない
- ❌ 「声として認識できる音がほとんどない」状態が継続

### 正常な学習との比較

**通常の192次元モデル（pretrain有り）:**
```
5エポック:  lowest_value ~50-60  → 実用的な品質、声として明確に認識できる
50エポック: lowest_value ~30-35  → 良好な品質
```

**現在の768次元モデル（pretrain無し）:**
```
50エポック: lowest_value ~30     → ほぼノイズ ← 異常！
```

---

## 診断結果

### ✅ 正常な項目

1. **モデルの次元:** すべて正しい
   ```
   inter_channels: 768
   hidden_channels: 768
   filter_channels: 2048
   gin_channels: 512
   text_enc_hidden_dim: 768
   ```

2. **重みの統計:**
   ```
   enc_p.emb_phone.weight: Mean=0.000000, Std=0.020721
   dec.conv_pre.weight:    Mean=-0.000004, Std=0.009766
   Has NaN: False
   Has Inf: False
   ```

3. **FAISSインデックス:**
   ```
   Dimension: 768 (正しい)
   Total vectors: 126,777
   ```

4. **メタデータ:**
   ```
   Embedder: japanese-hubert-base
   Sample rate: 48000
   Epoch: 50, Step: 9400
   ```

### ❌ 異常な項目

1. **品質とlossの不一致:**
   - lowest_valueは正常に低下
   - しかし品質は改善しない
   - → **学習しているものが間違っている可能性**

2. **重みの標準偏差が小さい:**
   ```
   enc_p.emb_phone.weight: Std=0.020721
   dec.conv_pre.weight:    Std=0.009766
   ```
   - 192次元の学習済みモデルと比較が必要
   - もしかしたら重みがほとんど更新されていない？

---

## 可能性のある原因

### 仮説1: c_mel支配による偏った学習 ⚠️ 最有力

**問題:**
```python
loss_gen_all = loss_gen + 2*loss_fm + 45*loss_mel + loss_kl
```

`c_mel=45`が大きすぎて、メルスペクトログラム再構成だけを最適化し、実際の音声品質（loss_gen）が無視されている可能性

**確認方法:**
- TensorBoardで各損失を個別確認
- `loss_gen`が全く下がっていないのでは？
- `loss_mel`だけが下がっている？

**対策:**
```json
"c_mel": 35  // 45 → 35 に下げる
```

---

### 仮説2: 学習率が不適切

**問題:**
```
learning_rate = 8e-5 (0.00008)
```

768次元・pretrain無しには低すぎる？

**確認方法:**
- 勾配のノルムを確認（TensorBoard）
- 勾配が極端に小さい？

**対策:**
```json
"learning_rate": 1e-4  // 8e-5 → 1e-4 に上げる
```

---

### 仮説3: データセットの問題

**可能性:**
- 音声データが破損している
- 前処理が失敗している
- embedder抽出が失敗している

**確認方法:**
- 192次元で同じデータセットを学習
- 正常に動作するならデータセットは問題なし
- 動作しないならデータセット自体に問題

**対策:**
```bash
# 192次元で比較実験
env\python.exe core.py train \
  --model_name test_192_comparison \
  --sample_rate 48000 \
  --total_epoch 50 \
  --batch_size 12 \
  --hidden_channels 192 \
  --pretrained True
```

---

### 仮説4: Discriminatorが強すぎる

**問題:**
Discriminatorが強すぎて、Generatorが学習できない（mode collapse）

**確認方法:**
- TensorBoardで`loss_disc`を確認
- Discriminatorが完璧に識別している（loss_disc ≈ 0）？

**対策:**
- Discriminatorの学習率を下げる
- またはGeneratorの学習率を上げる

---

### 仮説5: FAISSインデックスの使用方法

**可能性:**
推論時に`index_rate`が不適切で、学習データの埋め込みが正しく適用されていない

**確認方法:**
```bash
# index_rate=0で推論（インデックスなし）
```

---

### 仮説6: 768次元の根本的な問題（最悪ケース）

**可能性:**
768次元アーキテクチャの実装に気づいていないバグがある

**確認方法:**
- 192次元で学習が正常に動作することを確認
- 768次元だけで問題が再現するなら実装のバグ

---

## 次回確認すべきこと（優先順位順）

### 🔴 Priority 1: 個別損失の確認

```bash
run-tensorboard.bat
# ブラウザで http://localhost:6006
```

**確認項目:**
1. `loss_gen`（GAN loss）の推移
   - 5-15の範囲で安定しているか？
   - 全く下がっていないのでは？
2. `loss_mel`（重み付け前）の推移
   - 0.5以下に収束しているか？
3. `loss_fm`と`loss_kl`の推移
4. `loss_disc`（Discriminator loss）
   - Discriminatorが強すぎないか？

### 🔴 Priority 2: 192次元比較実験

同じデータセットで192次元モデルを50エポック学習：

```bash
env\python.exe core.py preprocess \
  --model_name test_192_comparison \
  --dataset_path logs/naru_x_C768_3/sliced_audios \
  --sample_rate 48000

env\python.exe core.py extract \
  --model_name test_192_comparison \
  --sample_rate 48000 \
  --f0_method mangio-crepe \
  --embedder_model japanese-hubert-base

env\python.exe core.py train \
  --model_name test_192_comparison \
  --sample_rate 48000 \
  --total_epoch 50 \
  --batch_size 12 \
  --hidden_channels 192 \
  --pretrained True
```

**期待結果:**
- 5-10エポックで実用的な品質
- 50エポックで良好な品質

**もし192次元でも失敗:**
→ データセットに問題

**もし192次元は成功:**
→ 768次元実装に問題

### 🟡 Priority 3: c_mel調整実験

現在の学習を止めて、`c_mel`を下げて再学習：

```bash
# config.jsonを編集
nano logs/naru_x_C768_3/config.json
# "c_mel": 45 → "c_mel": 35

# 新しいモデルで最初から
env\python.exe core.py train \
  --model_name naru_x_C768_c35 \
  --sample_rate 48000 \
  --total_epoch 100 \
  --batch_size 8 \
  --hidden_channels 768 \
  --pretrained False
```

### 🟡 Priority 4: 学習率調整実験

`learning_rate`を上げて実験：

```json
{
  "learning_rate": 1e-4,  // 8e-5 → 1e-4
  "c_mel": 35,
  "lr_decay": 0.9997
}
```

---

## 緊急対策案

### Option A: c_mel下げ + 学習率上げ（推奨）

```json
{
  "learning_rate": 1e-4,   // 8e-5 → 1e-4（高め）
  "c_mel": 35,             // 45 → 35（GAN重視）
  "c_kl": 1.0,
  "lr_decay": 0.9997
}
```

**理由:**
- c_melを下げることでGAN lossの寄与を増やす
- 学習率を上げて学習を加速

### Option B: 192次元に一旦戻る

768次元は一旦保留にして、192次元で確実に動作するモデルを作る：

```bash
env\python.exe core.py train \
  --model_name naru_192_reliable \
  --sample_rate 48000 \
  --total_epoch 500 \
  --batch_size 12 \
  --hidden_channels 192 \
  --pretrained True
```

**理由:**
- 確実に動作する
- データセットが問題ないことを確認
- 768次元の問題を切り分け

### Option C: 完全リセット（最終手段）

すべてをクリーンな状態から再実行：

```bash
# 1. データセット再処理
env\python.exe core.py preprocess \
  --model_name naru_fresh_768 \
  --dataset_path <original_audio_dir> \
  --sample_rate 48000

# 2. 特徴抽出
env\python.exe core.py extract \
  --model_name naru_fresh_768 \
  --sample_rate 48000 \
  --f0_method rmvpe \
  --embedder_model japanese-hubert-base \
  --hidden_channels 768

# 3. 設定を調整して学習
# config: learning_rate=1e-4, c_mel=35
env\python.exe core.py train \
  --model_name naru_fresh_768 \
  --sample_rate 48000 \
  --total_epoch 200 \
  --batch_size 8 \
  --hidden_channels 768 \
  --pretrained False
```

---

## ユーザーの懸念（重要）

> 学習すべき対象を本当に学習できているか根本的に疑問

**この懸念は正当です。**

現在の状況は明らかに異常であり、以下のいずれかの可能性があります：

1. ❌ **学習が間違った方向に進んでいる**（c_melの過剰重視）
2. ❌ **学習がほとんど進んでいない**（学習率が低すぎる）
3. ❌ **データに問題がある**（前処理の失敗）
4. ❌ **768次元実装にバグがある**（気づいていない問題）

**次回の診断で必ず原因を特定します。**

---

## 技術的メモ

### lowest_valueの計算式（再確認）

```python
loss_gen_all = loss_gen + 2*loss_fm + 45*loss_mel + 1*loss_kl
```

**典型的な値の比例:**
```
loss_gen:  ~10    (10%)
loss_fm:   ~20    (20%)  [2.0 × 10]
loss_mel:  ~900   (90%)  [45 × 20]
loss_kl:   ~2     (2%)
------------------------
Total:     ~932   (100%)
```

**問題:**
`loss_mel`が支配的すぎる（90%）
→ `loss_gen`（音声のリアルさ）が無視される可能性

### 正常な学習の兆候

**以下が確認できれば正常:**
1. `loss_gen`が 10-15 → 5-8 に低下
2. `loss_mel`（重み付け前）が 1.0 → 0.3-0.5 に低下
3. `loss_disc`が 安定して 1.0-3.0 の範囲
4. 勾配のノルム（grad_norm_g, grad_norm_d）が適切な範囲

### 異常な学習の兆候

**以下が見られたら問題:**
1. ❌ `loss_gen`が全く下がらない（10-15で停滞）
2. ❌ `loss_mel`だけが下がる
3. ❌ `loss_disc`が 0に近い（Discriminatorが強すぎ）
4. ❌ 勾配が極端に小さい（学習率不足）

---

## 現在のファイル状況

```
Model: logs/naru_x_C768_3/naru_x_C768_3_50e_9400s.pth (489 MB)
Config: logs/naru_x_C768_3/config.json
Index: logs/naru_x_C768_3/naru_x_C768_3.index (382 MB, 126,777 vectors)
Dataset: logs/naru_x_C768_3/sliced_audios/ (41分43秒)
Embeddings: logs/naru_x_C768_3/extracted/*.npy (1597 files)
```

---

## まとめ

### 現状

🔴 **CRITICAL:** 50エポック学習したが、品質はほぼノイズのまま
- lowest_value: 35 → 30.7（低下している）
- 品質: ノイズ → ノイズ（改善なし）

### 最有力の原因

**c_mel=45が高すぎて、メル再構成だけを最適化している可能性**

### 次回やること

1. 🔴 **TensorBoardで個別損失を確認**（最優先）
2. 🔴 **192次元比較実験**（データセット検証）
3. 🟡 **c_mel=35に下げて再学習**
4. 🟡 **学習率を1e-4に上げる**

### 次回までに準備すること

- TensorBoardのスクリーンショットまたは値
- 学習時のコンソール出力（可能なら）
- 192次元比較実験の結果

---

**最終更新:** 2025-11-06 23:00+
**次回確認予定:** ユーザーが起床後
**状態:** 原因調査中 - 複数の仮説あり
