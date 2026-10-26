# PyTorch / CUDA アップグレードガイド

## 現在の構成

- Python: `3.11`
- PyTorch: `2.13.0+cu132`
- TorchVision: `0.28.0+cu132`
- TorchAudio: `2.11.0`（最終リリース）
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

既存の`env`を更新する場合は次を実行します。

```batch
env\python.exe -m pip install --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

## 確認

```batch
env\python.exe -c "import torch, torchvision, torchaudio; print(torch.__version__); print(torchvision.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
env\python.exe -m pip check
```

この環境で確認した値は次のとおりです。

```text
torch: 2.13.0+cu132
torchvision: 0.28.0+cu132
torchaudio: 2.11.0+cpu
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

## Python 3.14 + PyTorch 2.15 環境（env314）

上の`env`とは別に、uvだけで作る2つ目の環境です。両方を残したまま起動バッチで切り替えて比べられます。

| | `env`（従来） | `env314`（新） |
| --- | --- | --- |
| Python | 3.11（conda） | 3.14（uv管理のCPython） |
| PyTorch | 2.13.0+cu132 | 2.15.0+cu134（RC、testチャンネル） |
| triton-windows | 3.7 | 3.8（torch 2.15対応の3.9は未公開） |
| インストール | `run-install.bat` | `run-install3.14.bat` |
| 起動 | `run-applio.bat` | `run-applio3.14.bat` |
| 依存一覧 | `requirements.txt` | `requirements3.14.txt` |

```batch
run-install3.14.bat
run-applio3.14.bat
```

`run-install3.14.bat`はuvが無ければ公式インストーラで入れ、`uv python install 3.14`、
`uv venv env314`、`uv pip install -r requirements3.14.txt`の順に実行します。
既に`env314`があればそれを更新します。作り直すときは`env314`フォルダーを消してから実行します。

PyTorch 2.15.0の正式版が出たら、`run-install3.14.bat`の`TORCH_INDEX`を
`https://download.pytorch.org/whl/cu134`に変えて再実行します（バージョン文字列は同じです）。

確認:

```batch
env314\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
uv pip check --python env314\Scripts\python.exe
```

注意点:

- Python 3.15は見送りました。faiss-cpu、torchaudio、onnxruntime、triton-windows、loudness、
  webrtcvad-wheelsにcp315のWindows wheelがありません（2026-10-26時点）。
- torch 2.15に対応するTritonは3.9ですが、triton-windowsは3.8までです。3.8で`torch.compile`
  （default / reduce-overhead / max-autotune / backward）が動くことは確認済みです。
  3.9が出たら`run-install3.14.bat`を再実行すると入ります（上限を`<3.10`にしています）。
- FCNF0++（penn / torbi）はまだ入れていません。torch 2.15用のtorbiバイナリが公開されていないためです。
  `tools\build_torbi_wheel.bat`でビルドし、`requirements3.14.txt`の2行のコメントを外すと使えます。
- torchvisionは入れていません。どのコードからもimportされていません。

2026-10-26の計測（RTX 4090、詳細はCLAUDE.md）:

- realtime（APEX-GAN+ / kushinada / hpa-rmvpe、1080 msブロック、RVCをcompile）: 1ブロックの処理時間p50が
  env 75.0 / 72.9 ms、env314 56.1 / 55.9 ms（約24%短縮）。
- 学習1ステップ（APEX-GAN+、fp16、batch 24、compile無し）: env 513 / 509 ms、env314 521 / 519 ms（約2%遅い）。
- embedder 5種の特徴量とRMVPEのF0は両環境で一致（相対誤差2.2e-5以下）。テストはFCNF0++（penn未導入）以外すべて通過
  （HPA-RMVPEのcompileテスト1件は窓ごとのp99判定によるもので、全体の一致度はenv314のほうが良い）。

## 公式情報

- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- TorchAudio 2.11 release: https://github.com/pytorch/audio/releases/tag/v2.11.0
- CUDA compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
