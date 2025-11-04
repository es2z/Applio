"""
Analyze trained model weights to identify potential issues with japanese-hubert-large
"""
import torch
import numpy as np
import os
import glob

def analyze_model(model_path):
    """Analyze model weights"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(model_path)}")
    print(f"{'='*80}")

    try:
        # Load checkpoint
        cpt = torch.load(model_path, map_location='cpu', weights_only=True)

        # Get model info
        embedder_model = cpt.get('embedder_model', 'Unknown')
        text_enc_hidden_dim = cpt.get('text_enc_hidden_dim', 'Not saved')
        version = cpt.get('version', 'Unknown')
        epoch = cpt.get('epoch', 'Unknown')
        sr = cpt.get('sr', 'Unknown')

        print(f"Embedder: {embedder_model}")
        print(f"Text Enc Hidden Dim: {text_enc_hidden_dim}")
        print(f"Version: {version}")
        print(f"Epoch: {epoch}")
        print(f"Sample Rate: {sr}")

        # Analyze enc_p (text encoder) weights
        weight = cpt.get('weight', {})

        # Check emb_phone weights (enc_p.emb_phone.weight)
        if 'enc_p.emb_phone.weight' in weight:
            emb_phone_weight = weight['enc_p.emb_phone.weight']
            print(f"\n--- enc_p.emb_phone (phone embedding layer) ---")
            print(f"Shape: {emb_phone_weight.shape}")  # Should be [192, text_enc_hidden_dim]
            print(f"Mean: {emb_phone_weight.mean():.6f}")
            print(f"Std:  {emb_phone_weight.std():.6f}")
            print(f"Min:  {emb_phone_weight.min():.6f}")
            print(f"Max:  {emb_phone_weight.max():.6f}")

            # Check if weights seem properly initialized/trained
            if abs(emb_phone_weight.mean()) > 0.5:
                print("  ⚠ WARNING: Mean is far from zero!")
            if emb_phone_weight.std() < 0.01 or emb_phone_weight.std() > 2.0:
                print(f"  ⚠ WARNING: Unusual std deviation: {emb_phone_weight.std():.6f}")

        # Check emb_pitch weights (enc_p.emb_pitch.weight)
        if 'enc_p.emb_pitch.weight' in weight:
            emb_pitch_weight = weight['enc_p.emb_pitch.weight']
            print(f"\n--- enc_p.emb_pitch (pitch embedding layer) ---")
            print(f"Shape: {emb_pitch_weight.shape}")  # Should be [256, 192]
            print(f"Mean: {emb_pitch_weight.mean():.6f}")
            print(f"Std:  {emb_pitch_weight.std():.6f}")
            print(f"Min:  {emb_pitch_weight.min():.6f}")
            print(f"Max:  {emb_pitch_weight.max():.6f}")

            if abs(emb_pitch_weight.mean()) > 0.5:
                print("  ⚠ WARNING: Mean is far from zero!")
            if emb_pitch_weight.std() < 0.01 or emb_pitch_weight.std() > 2.0:
                print(f"  ⚠ WARNING: Unusual std deviation: {emb_pitch_weight.std():.6f}")

        # Compare relative magnitudes
        if 'enc_p.emb_phone.weight' in weight and 'enc_p.emb_pitch.weight' in weight:
            phone_magnitude = weight['enc_p.emb_phone.weight'].abs().mean()
            pitch_magnitude = weight['enc_p.emb_pitch.weight'].abs().mean()
            ratio = pitch_magnitude / phone_magnitude
            print(f"\n--- Relative Magnitudes ---")
            print(f"Phone embedding avg magnitude: {phone_magnitude:.6f}")
            print(f"Pitch embedding avg magnitude: {pitch_magnitude:.6f}")
            print(f"Pitch/Phone ratio: {ratio:.3f}")

            if ratio > 2.0:
                print(f"  ⚠ WARNING: Pitch embedding is {ratio:.1f}x larger than phone!")
                print("  → This could cause pitch to dominate, resulting in incorrect pitch learning")
            elif ratio < 0.5:
                print(f"  ⚠ WARNING: Phone embedding is {1/ratio:.1f}x larger than pitch!")
                print("  → This could cause underfitting of pitch information")

        # Check generator output layer
        if 'dec.conv_post.weight' in weight:
            conv_post = weight['dec.conv_post.weight']
            print(f"\n--- dec.conv_post (final generator layer) ---")
            print(f"Shape: {conv_post.shape}")
            print(f"Mean: {conv_post.mean():.6f}")
            print(f"Std:  {conv_post.std():.6f}")

        # Check for any NaN or Inf values
        has_nan = False
        has_inf = False
        for key, val in weight.items():
            if torch.isnan(val).any():
                print(f"\n⚠⚠⚠ NaN detected in {key}!")
                has_nan = True
            if torch.isinf(val).any():
                print(f"\n⚠⚠⚠ Inf detected in {key}!")
                has_inf = True

        if not has_nan and not has_inf:
            print(f"\n✓ No NaN or Inf values detected")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Searching for models trained with japanese-hubert-large...")

    # Search for models
    model_dirs = glob.glob("logs/*/")

    japanese_large_models = []
    japanese_base_models = []
    other_models = []

    for model_dir in model_dirs:
        model_info_path = os.path.join(model_dir, "model_info.json")
        if os.path.exists(model_info_path):
            import json
            try:
                with open(model_info_path, 'r') as f:
                    info = json.load(f)
                    embedder = info.get('embedder_model', '')

                    # Find latest checkpoint
                    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
                    if pth_files:
                        latest_pth = max(pth_files, key=os.path.getmtime)

                        if 'japanese-hubert-large' in embedder or 'japanese_hubert_large' in embedder:
                            japanese_large_models.append(latest_pth)
                        elif 'japanese-hubert-base' in embedder or 'japanese_hubert_base' in embedder:
                            japanese_base_models.append(latest_pth)
                        else:
                            other_models.append((latest_pth, embedder))
            except:
                pass

    print(f"\nFound:")
    print(f"  {len(japanese_large_models)} models with japanese-hubert-large")
    print(f"  {len(japanese_base_models)} models with japanese-hubert-base")
    print(f"  {len(other_models)} models with other embedders")

    # Analyze japanese-hubert-large models
    if japanese_large_models:
        print(f"\n{'#'*80}")
        print("JAPANESE-HUBERT-LARGE MODELS")
        print(f"{'#'*80}")
        for model_path in japanese_large_models[:3]:  # Analyze up to 3 models
            analyze_model(model_path)

    # Compare with japanese-hubert-base if available
    if japanese_base_models:
        print(f"\n{'#'*80}")
        print("JAPANESE-HUBERT-BASE MODELS (for comparison)")
        print(f"{'#'*80}")
        for model_path in japanese_base_models[:1]:  # Analyze 1 model for comparison
            analyze_model(model_path)

    if not japanese_large_models and not japanese_base_models:
        print("\nNo trained models found. Please train a model first.")
        print("Or specify a model path manually:")
        print('  python analyze_model_weights.py "logs/your_model/your_model_e100.pth"')
