"""
Test script to compare embedding statistics between japanese-hubert-base and japanese-hubert-large
"""
import torch
import numpy as np
from rvc.lib.utils import load_embedding, load_audio_16k
import os

def analyze_embedder(embedder_name, test_audio_path):
    """Analyze embedder output statistics"""
    print(f"\n{'='*60}")
    print(f"Analyzing {embedder_name}")
    print(f"{'='*60}")

    # Load model
    model = load_embedding(embedder_name).cuda().float()
    model.eval()

    # Load test audio
    if not os.path.exists(test_audio_path):
        print(f"Warning: {test_audio_path} not found, creating silent audio")
        # Create 3 seconds of silence
        audio = np.zeros(16000 * 3, dtype=np.float32)
    else:
        audio = load_audio_16k(test_audio_path)

    # Process
    feats = torch.from_numpy(audio).cuda().float()
    feats = feats.view(1, -1)

    with torch.no_grad():
        result = model(feats)["last_hidden_state"]

    feats_out = result.squeeze(0).float().cpu().numpy()

    # Calculate statistics
    print(f"Shape: {feats_out.shape}")
    print(f"Dimension: {feats_out.shape[1]}")
    print(f"\nStatistics across all features:")
    print(f"  Mean: {feats_out.mean():.6f}")
    print(f"  Std:  {feats_out.std():.6f}")
    print(f"  Min:  {feats_out.min():.6f}")
    print(f"  Max:  {feats_out.max():.6f}")

    # Per-dimension statistics
    dim_means = feats_out.mean(axis=0)
    dim_stds = feats_out.std(axis=0)

    print(f"\nPer-dimension statistics:")
    print(f"  Mean of means: {dim_means.mean():.6f}")
    print(f"  Std of means:  {dim_means.std():.6f}")
    print(f"  Mean of stds:  {dim_stds.mean():.6f}")
    print(f"  Std of stds:   {dim_stds.std():.6f}")

    # Check for outlier dimensions
    outlier_dims = np.where(np.abs(dim_means) > 2.0)[0]
    if len(outlier_dims) > 0:
        print(f"\nWarning: {len(outlier_dims)} dimensions have |mean| > 2.0")
        print(f"  Outlier dimension indices (first 10): {outlier_dims[:10].tolist()}")

    return {
        'shape': feats_out.shape,
        'mean': feats_out.mean(),
        'std': feats_out.std(),
        'min': feats_out.min(),
        'max': feats_out.max(),
        'dim_means': dim_means,
        'dim_stds': dim_stds,
    }

if __name__ == "__main__":
    # Test audio path - use a mute file or any existing audio
    test_audio = r"logs\mute\sliced_audios_16k\mute.wav"

    print("Testing embedder output statistics...")
    print("This helps identify if japanese-hubert-large has different normalization")

    # Analyze both embedders
    stats_base = analyze_embedder("japanese-hubert-base", test_audio)
    stats_large = analyze_embedder("japanese-hubert-large", test_audio)

    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"\nDimension ratio (large/base): {stats_large['shape'][1] / stats_base['shape'][1]:.3f}")
    print(f"Mean ratio (large/base):      {stats_large['mean'] / stats_base['mean'] if stats_base['mean'] != 0 else float('inf'):.3f}")
    print(f"Std ratio (large/base):       {stats_large['std'] / stats_base['std']:.3f}")

    print("\nRecommendations:")
    if abs(stats_large['mean'] - stats_base['mean']) > 0.5:
        print("  ⚠ WARNING: Large difference in mean values detected!")
        print("  → Consider adding normalization after embedding extraction")
    if abs(stats_large['std'] / stats_base['std'] - 1.0) > 0.3:
        print("  ⚠ WARNING: Large difference in std deviation detected!")
        print("  → Consider scaling embeddings to match base model statistics")
    if abs(stats_large['mean']) > 1.0 or abs(stats_base['mean']) > 1.0:
        print("  ⚠ WARNING: Embeddings are not centered around zero!")
        print("  → Consider mean normalization")
