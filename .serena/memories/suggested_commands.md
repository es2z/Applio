# Suggested Commands

## Installation & Setup

### Windows
```batch
# First-time installation (Python 3.11 + PyTorch 2.7.1)
run-install.bat

# Python 3.13 + PyTorch 2.8 installation (fork-specific)
run-install-py313.bat

# Launch Applio web interface
run-applio.bat

# Launch TensorBoard
run-tensorboard.bat
```

### Linux/macOS
```bash
# Installation
./run-install.sh

# Launch Applio
./run-applio.sh

# Launch TensorBoard
./run-tensorboard.sh

# Makefile commands (Linux/macOS only)
make run-install  # Install dependencies
make run-applio   # Run with --share flag
make run-tensorboard
```

## Running Applio

### Web Interface
```bash
# Basic launch (opens browser automatically on Windows)
env\python.exe app.py --open

# Custom port and server
env\python.exe app.py --open --port 6969 --server-name 127.0.0.1

# Share publicly via Gradio
env\python.exe app.py --share
```

## CLI Operations (via core.py)

### Voice Inference
```bash
# Single file inference
env\python.exe core.py infer ^
  --input_path input.wav ^
  --output_path output.wav ^
  --pth_path logs\model\model.pth ^
  --index_path logs\model\model.index ^
  --f0_method rmvpe ^
  --pitch 0 ^
  --index_rate 0.75

# Batch inference
env\python.exe core.py batch_infer ^
  --input_folder input_dir ^
  --output_folder output_dir ^
  --pth_path logs\model\model.pth ^
  --index_path logs\model\model.index
```

### TTS with Voice Conversion
```bash
env\python.exe core.py tts ^
  --tts_text "Hello world" ^
  --tts_voice en-US-ChristopherNeural ^
  --output_rvc_path output.wav ^
  --pth_path logs\model\model.pth ^
  --index_path logs\model\model.index
```

### Training Pipeline
```bash
# 1. Preprocess audio dataset
env\python.exe core.py preprocess ^
  --model_name my_model ^
  --dataset_path dataset_folder ^
  --sample_rate 40000

# 2. Extract features (F0 + embeddings)
env\python.exe core.py extract ^
  --model_name my_model ^
  --sample_rate 40000 ^
  --f0_method rmvpe ^
  --embedder_model contentvec

# 3. Train model
env\python.exe core.py train ^
  --model_name my_model ^
  --sample_rate 40000 ^
  --total_epoch 500 ^
  --save_every_epoch 50 ^
  --batch_size 8

# 4. Generate index file
env\python.exe core.py index --model_name my_model
```

### Utilities
```bash
# Model information
env\python.exe core.py model_information --pth_path model.pth

# Blend two models
env\python.exe core.py model_blender ^
  --model_name blended ^
  --pth_path_1 model1.pth ^
  --pth_path_2 model2.pth ^
  --ratio 0.5

# Audio analysis
env\python.exe core.py audio_analyzer --input_path audio.wav

# Launch TensorBoard
env\python.exe core.py tensorboard

# Download prerequisites
env\python.exe core.py prerequisites
```

## Git Commands (Windows)

```bash
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "message"

# Push
git push

# Pull latest changes
git pull

# View commit history
git log --oneline -10

# View diff
git diff
```

## File Operations (Windows PowerShell/CMD)

```bash
# List files
dir
ls  # PowerShell

# Change directory
cd path\to\directory

# Create directory
mkdir directory_name

# Copy files
copy source.txt destination.txt
cp source.txt destination.txt  # PowerShell

# Move/Rename files
move old.txt new.txt
mv old.txt new.txt  # PowerShell

# Delete files
del file.txt
rm file.txt  # PowerShell

# View file contents
type file.txt
cat file.txt  # PowerShell

# Find files
dir /s /b *.py
Get-ChildItem -Recurse -Filter *.py  # PowerShell
```

## Python Environment

```bash
# Activate environment (if using conda directly)
conda activate env

# Check Python version
env\python.exe --version

# Install package
env\python.exe -m pip install package_name

# List installed packages
env\python.exe -m pip list
```

## Important Notes

- **No formal test suite**: Testing is done through Gradio UI or CLI commands
- **No linting/formatting configuration**: No pyproject.toml, .flake8, .pylintrc, or black config
- **Windows batch files**: Primary installation and execution method on Windows
- **env folder**: Contains Python virtual environment (created by installer)
