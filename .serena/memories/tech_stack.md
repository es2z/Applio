# Technology Stack

## Core Technologies
- **Python**: 3.11 (default), 3.13 (fork-specific)
- **PyTorch**: 2.7.1 (default), 2.8.0 (fork) with CUDA 12.8 support
- **Gradio**: 5.23.1 - Web UI framework

## Deep Learning & Audio Processing
- **Audio Processing**:
  - librosa==0.11.0 - Audio analysis and processing
  - soundfile==0.12.1 - Audio file I/O
  - ffmpeg-python - Audio format conversion
  - pedalboard - Audio effects
  - noisereduce - Noise reduction
  - stftpitchshift - Pitch shifting
  - soxr - High-quality audio resampling
  
- **Machine Learning**:
  - torch, torchaudio, torchvision (2.7.1/2.8.0)
  - transformers==4.44.2 - Hubert/ContentVec models
  - torchcrepe==0.0.23 - CREPE pitch extraction
  - torchfcpe - FCPE pitch extraction
  - swift_f0 - SWIFT pitch extraction
  - einops - Tensor operations
  - numba==0.61.0 - JIT compilation for performance
  
- **Vector Search & Indexing**:
  - faiss-cpu==1.7.3 - Speaker embedding retrieval

## Visualization & Monitoring
- matplotlib==3.7.2
- tensorboard, tensorboardX - Training visualization

## UI & Utilities
- gradio==5.23.1 - Web interface
- edge-tts==7.2.0 - Text-to-speech
- pypresence - Discord presence
- beautifulsoup4 - Web scraping
- sounddevice - Real-time audio I/O
- webrtcvad - Voice activity detection
- loudness - Audio loudness measurement

## Scientific Computing
- numpy==1.26.4
- scipy==1.11.1

## Additional Tools
- tqdm - Progress bars
- requests - HTTP requests
- wget - File downloads

## GPU Support
- CUDA 12.8 (NVIDIA GPUs)
- ZLUDA (AMD GPU support via emulation layer)

## Environment Management
- Miniconda/Conda for Python environment
- uv - Fast package installer (used in install script)
