# PyTorch 2.8.0 アップグレードガイド

## 更新内容

このアップデートでは、Applioの依存関係を最新バージョンにアップグレードしました。

### 主な変更点

**PyTorch関連**
- PyTorch: `2.7.1` → `2.8.0` (存在しないバージョンから正しいバージョンへ修正)
- TorchAudio: `2.7.1` → `2.8.0`
- TorchVision: `0.22.1` → `0.23.0`
- CUDA: 12.8サポート継続

**その他のライブラリ**
- faiss-cpu: `1.7.3` → `1.9.0` (Python 3.12対応)
- scipy: `1.11.1` → `1.14.1` (Windows/macOS: Python 3.12対応)

### パフォーマンス向上

PyTorch 2.8.0の主な改善点：
- NVIDIA Blackwell GPUアーキテクチャ対応（RTX 50xxシリーズ）
- コンパイラとランタイムの最適化
- メモリ効率の改善
- より高速な推論速度

### インストール方法

#### 方法1: Python 3.11を使用（推奨・安定版）

```batch
run-install.bat
```

- 最も安定した動作
- 全依存関係の完全サポート確認済み
- 既存環境との互換性が高い

#### 方法2: Python 3.12を使用（パフォーマンス重視）

```batch
run-install-py312.bat
```

- Python 3.12の性能向上（約5-10%高速）
- より新しいPython機能の利用
- 推奨環境: 新規インストール時
- **注意**: faiss-cpu 1.9.0が必要（requirements.txt更新済み）

### 再インストール手順

既存のインストールをアップグレードする場合：

1. **環境のバックアップ（推奨）**
   ```batch
   rename env env_backup
   ```

2. **新規インストール**
   ```batch
   run-install.bat
   ```
   または
   ```batch
   run-install-py312.bat
   ```

3. **動作確認**
   ```batch
   run-applio.bat
   ```

4. **問題がなければバックアップ削除**
   ```batch
   rmdir /s /q env_backup
   ```

### アップグレード後の確認

インストール後、Pythonコンソールで確認：

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

期待される出力：
```
PyTorch version: 2.8.0+cu128
CUDA available: True
CUDA version: 12.8
```

### システム要件

**最小要件**
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA Compute Capability 5.0以上
  - ⚠️ Maxwell (5.x)、Pascal (6.x)、Volta (7.0) GPUはCUDA 12.8非サポート
  - 古いGPUの場合: CUDA 12.6ビルドを使用
- 8GB以上のRAM
- 20GB以上の空きディスク容量

**推奨要件**
- Windows 11 (64-bit)
- NVIDIA RTX 20/30/40/50シリーズ
- 16GB以上のRAM
- SSD推奨

### 古いGPUをお使いの場合

Maxwell、Pascal、Volta世代のGPUをお使いの場合：

requirements.txtを編集して、CUDA 12.6を使用：
```
torch==2.8.0+cu126; sys_platform == 'linux' or sys_platform == 'win32'
```

インストールコマンドも変更：
```batch
--extra-index-url https://download.pytorch.org/whl/cu126
```

### トラブルシューティング

#### インストールエラー

**症状**: `Could not find a version that satisfies the requirement`

**解決策**:
1. キャッシュをクリア
   ```batch
   env\python.exe -m pip cache purge
   ```

2. 再インストール
   ```batch
   run-install.bat
   ```

#### CUDA未検出

**症状**: `torch.cuda.is_available()` が `False`

**解決策**:
1. NVIDIAドライバーが最新か確認
   ```batch
   nvidia-smi
   ```

2. 必要に応じてドライバー更新
   - [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)

#### パフォーマンスが改善しない

**確認項目**:
1. GPU使用率を確認
   ```batch
   nvidia-smi -l 1
   ```

2. CUDAベンチマークを実行
   ```python
   import torch
   torch.backends.cudnn.benchmark = True
   ```

### サポート

問題が解決しない場合：
1. [Applio GitHub Issues](https://github.com/IAHispano/Applio/issues)
2. [Applio Discord](https://discord.gg/urxFjYmYYh)

### ロールバック

問題が発生した場合、元のバージョンに戻すには：

1. `env`フォルダを削除
2. `env_backup`を`env`にリネーム
3. または、元のrequirements.txtに戻して再インストール

---

**更新日**: 2025年1月
**対象バージョン**: Applio 3.5.0+
