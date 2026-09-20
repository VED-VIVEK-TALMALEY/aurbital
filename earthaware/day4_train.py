

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

from day4_multimodal_model import MultispectralVLM

class MultispectralDataset(Dataset):

    def __init__(self, data_path='data/training/training_data.json',
                 multispectral_dir='data/raw/sentinel2_multispectral'):

        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.multispectral_dir = Path(multispectral_dir)

        self.samples = []
        for item in self.data:
            for caption in item['captions']:
                self.samples.append({
                    : item['sample_id'],
                    : caption,
                    : item['bands']
                })
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def load_multispectral_image(self, bands_dict):
        
        band_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                    , 'B8A', 'B09', 'B10', 'B11', 'B12']
        
        bands = []
        for band_id in band_ids:
            if band_id in bands_dict:
                band_path = bands_dict[band_id]
                band_data = np.load(band_path)
                
                band_data = band_data.astype(np.float32) / 10000.0
                bands.append(band_data)

        image = np.stack(bands, axis=0)
        return torch.from_numpy(image).float()
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = self.load_multispectral_image(sample['bands'])

        caption = sample['caption']
        
        return {
            : image,
            : caption
        }

def collate_fn(batch, tokenizer, max_length=128):
    
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]

    encodings = tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        : images,
        : encodings.input_ids,
        : encodings.attention_mask,
        : encodings.input_ids.clone()  
    }

def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    
    model.train()
    
    total_loss = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return avg_loss

def main():

    print("="*60)
    print("MULTISPECTRAL VLM TRAINING")
    print("="*60)

    batch_size = 1  
    num_epochs = 10
    learning_rate = 5e-5
    gradient_accumulation_steps = 4  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print("\nInitializing model...")
    model = MultispectralVLM(use_lora=True, lora_rank=8)
    model = model.to(device)

    print("\nLoading dataset...")
    dataset = MultispectralDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer),
        num_workers=0  
    )
    
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    scaler = GradScaler()

    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, scaler, device, epoch)
        
        print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                : epoch,
                : model.state_dict(),
                : optimizer.state_dict(),
                : avg_loss,
            }, checkpoint_path)
            
            print(f"  ✓ Saved best model (loss: {avg_loss:.4f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: checkpoints/best_model.pt")

    print("\nTesting generation...")
    model.eval()

    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)
    
    print(f"Ground truth: {sample['caption']}")
    
    with torch.no_grad():
        generated = model.generate(
            image,
            prompt_text="Describe this satellite image:",
            max_new_tokens=30
        )
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main()
