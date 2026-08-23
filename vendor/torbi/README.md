# torbi compatibility wheel

This directory contains an unmodified wheel built from the official
[`maxrmorrison/torbi`](https://github.com/maxrmorrison/torbi) source.

- Version: `1.4.0`
- Source commit: `ca7732cd2344ba0d7449204f8805fb594a912701`
- Target: Windows x86-64, CPython 3.9+ stable ABI
- Build environment: PyTorch `2.13.0+cu132`, CUDA Toolkit 13.0, MSVC,
  `TORCH_CUDA_ARCH_LIST=8.9`
- SHA-256: `e901fae7ecb4b1633557f4197b1afb575c070158bdec40324dc8310a275212de`

The wheel is needed because the upstream 1.4.0 Windows wheel does not contain
a binary matching this repository's PyTorch 2.13 / CUDA 13.2 pin. The bundled
binary was tested on an NVIDIA RTX 4090. Other dependency profiles use the
official PyPI wheel.

The upstream MIT license is included both here and inside the wheel.
