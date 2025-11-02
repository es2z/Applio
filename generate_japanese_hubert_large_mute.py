"""
Script to generate 1024-dimensional mute files for japanese-hubert-large embedder.
This creates the necessary mute files in logs/mute_japanese_hubert_large/ directory.
"""
import os
import shutil
import torch
import numpy as np
from rvc.lib.utils import load_embedding, load_audio_16k

def create_directory_structure(base_path):
    """Create the directory structure for mute files."""
    dirs = [
        os.path.join(base_path, "extracted"),
        os.path.join(base_path, "f0"),
        os.path.join(base_path, "f0_voiced"),
        os.path.join(base_path, "sliced_audios"),
        os.path.join(base_path, "sliced_audios_16k"),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory structure at {base_path}")

def copy_audio_and_f0_files(source_base, target_base):
    """Copy audio and F0 files from source mute directory (embedder-agnostic)."""
    # Copy audio files
    audio_files = [
        ("sliced_audios/mute32000.wav", "sliced_audios/mute32000.wav"),
        ("sliced_audios/mute40000.wav", "sliced_audios/mute40000.wav"),
        ("sliced_audios/mute44100.wav", "sliced_audios/mute44100.wav"),
        ("sliced_audios/mute48000.wav", "sliced_audios/mute48000.wav"),
        ("sliced_audios/mute48000.spec.pt", "sliced_audios/mute48000.spec.pt"),
        ("sliced_audios_16k/mute.wav", "sliced_audios_16k/mute.wav"),
    ]

    for src_rel, dst_rel in audio_files:
        src = os.path.join(source_base, src_rel)
        dst = os.path.join(target_base, dst_rel)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src_rel}")

    # Copy F0 files (dimension-agnostic)
    f0_files = [
        ("f0/mute.wav.npy", "f0/mute.wav.npy"),
        ("f0_voiced/mute.wav.npy", "f0_voiced/mute.wav.npy"),
    ]

    for src_rel, dst_rel in f0_files:
        src = os.path.join(source_base, src_rel)
        dst = os.path.join(target_base, dst_rel)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src_rel}")

def generate_1024dim_embeddings(target_base, device):
    """Generate 1024-dimensional embeddings using japanese-hubert-large."""
    print("\nGenerating 1024-dimensional embeddings...")

    # Load the japanese-hubert-large model
    model = load_embedding("japanese-hubert-large", None).to(device).float()
    model.eval()

    # Load the 16kHz mute audio
    wav_file = os.path.join(target_base, "sliced_audios_16k", "mute.wav")
    feats = torch.from_numpy(load_audio_16k(wav_file)).to(device).float()
    feats = feats.view(1, -1)

    # Extract embeddings
    with torch.no_grad():
        result = model(feats)["last_hidden_state"]

    feats_out = result.squeeze(0).float().cpu().numpy()

    # Verify dimensions
    print(f"Generated embeddings shape: {feats_out.shape}")
    if feats_out.shape[1] != 1024:
        raise ValueError(f"Expected 1024 dimensions, got {feats_out.shape[1]}")

    # Save embeddings
    output_path = os.path.join(target_base, "extracted", "mute.npy")
    np.save(output_path, feats_out, allow_pickle=False)
    print(f"Saved 1024-dim embeddings to {output_path}")
    print(f"Final shape: {feats_out.shape}")

def main():
    # Paths
    source_mute = os.path.join("logs", "mute")
    target_mute = os.path.join("logs", "mute_japanese_hubert_large")

    # Check if source exists
    if not os.path.exists(source_mute):
        print(f"Error: Source mute directory not found at {source_mute}")
        return

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory structure
    create_directory_structure(target_mute)

    # Copy audio and F0 files
    print("\nCopying audio and F0 files...")
    copy_audio_and_f0_files(source_mute, target_mute)

    # Generate 1024-dim embeddings
    generate_1024dim_embeddings(target_mute, device)

    print("\n✓ Successfully generated japanese-hubert-large mute files!")
    print(f"  Location: {target_mute}")

if __name__ == "__main__":
    main()
