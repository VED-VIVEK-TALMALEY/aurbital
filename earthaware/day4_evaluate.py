"""
Evaluate trained multispectral VLM.

Day 4 - Model Evaluation
"""

import json
from pathlib import Path

import numpy as np
import torch

from day4_multimodal_model import MultispectralVLM

def load_model(checkpoint_path='checkpoints/isro_eo_enhanced_best.pt'):
    
    print("Loading trained model...")

    model = MultispectralVLM(use_lora=True, lora_rank=8)

    selected_path = checkpoint_path
    if not Path(selected_path).exists():
        fallback = 'checkpoints/best_model.pt'
        if Path(fallback).exists():
            selected_path = fallback
        else:
            raise FileNotFoundError("No checkpoint found in checkpoints/")

    checkpoint = torch.load(selected_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint: {selected_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'n/a')}")
    if 'loss' in checkpoint:
        print(f"Training loss: {checkpoint['loss']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    return model

def load_multispectral_image(bands_dict):
    
    band_ids = [
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
        'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
    ]

    bands = []
    for band_id in band_ids:
        if band_id in bands_dict:
            band_path = bands_dict[band_id]
            band_data = np.load(band_path)
            band_data = band_data.astype(np.float32) / 10000.0
            bands.append(band_data)

    image = np.stack(bands, axis=0)
    return torch.from_numpy(image).float()

def generate_caption(model, image, device='cuda', max_length=100):
    
    model.eval()

    with torch.no_grad():
        visual_embeddings = model.encode_image(image.unsqueeze(0).to(device))

        prompt = "This satellite image shows"
        prompt_tokens = model.tokenizer(prompt, return_tensors="pt").to(device)

        generated_ids = prompt_tokens.input_ids.clone()

        for _ in range(max_length):
            current_text_emb = model.language_model.get_input_embeddings()(generated_ids)
            current_combined = torch.cat([visual_embeddings, current_text_emb], dim=1)

            current_mask = torch.ones(
                current_combined.shape[:2],
                device=device,
                dtype=torch.long,
            )

            outputs = model.language_model(
                inputs_embeds=current_combined,
                attention_mask=current_mask,
            )

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == model.tokenizer.eos_token_id:
                break

        generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

def evaluate_samples(model, data_path='data/training/training_data.json', num_samples=10):
    
    print("=" * 60)
    print("EVALUATING TRAINED MODEL")
    print("=" * 60)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for i, sample in enumerate(data[:num_samples]):
        print(f"\n{'=' * 60}")
        print(f"SAMPLE {i + 1}/{num_samples}")
        print(f"{'=' * 60}")

        image = load_multispectral_image(sample['bands'])

        land_cover = sample['land_cover']
        ndvi = sample['spectral_indices']['NDVI']
        ground_truth = sample['captions'][0]

        print(f"\nLand Cover: {land_cover}")
        print(f"NDVI: {ndvi:.3f}")
        print("\nGround Truth:")
        print(f"  {ground_truth}")

        print("\nGenerating caption...")
        generated = generate_caption(model, image, device, max_length=50)

        print("\nGenerated:")
        print(f"  {generated}")

        keywords = ['NDVI', 'NIR', 'SWIR', 'infrared', 'vegetation', 'reflectance']
        found_keywords = [kw for kw in keywords if kw.lower() in generated.lower()]

        if found_keywords:
            print(f"\nSpectral keywords found: {', '.join(found_keywords)}")
        else:
            print("\nNo spectral keywords in generation")

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")

def main():
    model = load_model()
    evaluate_samples(model, num_samples=5)

if __name__ == "__main__":
    main()
