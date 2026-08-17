# PyTorch / CUDA アップグレードガイド

## 現在の構成

- Python: `3.11`
- PyTorch: `2.13.0+cu132`
- TorchVision: `0.28.0+cu132`
- TorchAudio: `2.11.0`（最終リリース）
- Triton Windows: `3.7.1.post27`
- CUDA runtime: `13.2`（PyTorch wheelに同梱）

Windows / Linux向けの旧`cu128`指定は`requirements.txt`内にコメントとして残し、
現在のバージョンを別行で固定しています。

TorchAudio 2.11はTorch 2.11と将来のTorchに対応する最終リリースです。
CUDA別のネイティブwheelではなくなったため、通常のPyPIからインストールされます。
`torchaudio.__version__`が`2.11.0+cpu`と表示されても、`transforms`と`functional`
のPyTorch演算はCUDA Tensorを受け取りGPU上で実行できます。

## インストール

Windowsでは次を実行します。

```batch
run-install.bat
```

`requirements.txt` は Torch 2.13 系に合わせて
`triton-windows==3.7.1.post27` を固定しているため、別環境でも通常の
インストール手順だけで Realtime の `torch.compile` backend が揃います。
Python 3.13 / Torch 2.8 用の `requirementspy313.txt` は Triton 3.4 系を指定します。

既存の`env`を更新する場合は次を実行します。

```batch
env\python.exe -m pip install --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

## 確認

```batch
env\python.exe -c "import importlib.metadata as m, torch, torchvision, torchaudio; print(torch.__version__); print(torchvision.__version__); print(torchaudio.__version__); print(m.version('triton-windows')); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
env\python.exe -m pip check
```

この環境で確認した値は次のとおりです。

```text
torch: 2.13.0+cu132
torchvision: 0.28.0+cu132
torchaudio: 2.11.0+cpu
triton-windows: 3.7.1.post27
CUDA runtime: 13.2
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

## 注意点

- PyTorch wheelにはCUDA runtimeが同梱されるため、通常の実行にローカルCUDA Toolkitは不要です。
- CUDA拡張をローカルでコンパイルする場合は、`nvcc`側のToolkitバージョンも別途合わせる必要があります。
- CUDA 13系にはNVIDIAドライバー580以降が必要です。この環境の591.44では実GPU演算を確認済みです。
- `torchaudio.load()`によるファイル読み込みにはTorchCodecと共有ライブラリ版FFmpegが別途必要です。Applio本体の音声読み込み経路はLibrosa/SoundFileを使うため、通常動作には不要です。
- 更新前からApplioを起動していた場合、旧DLLがプロセスに読み込まれたままなのでアプリを再起動してください。
- Triton の minor は PyTorch の minor と組で扱います。PyTorch だけ、または
  Triton だけを先に更新せず、`requirements.txt` の組を同時に更新してください。

## 公式情報

- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- TorchAudio 2.11 release: https://github.com/pytorch/audio/releases/tag/v2.11.0
- CUDA compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- Triton Windows / PyTorch compatibility: https://github.com/triton-lang/triton-windows
