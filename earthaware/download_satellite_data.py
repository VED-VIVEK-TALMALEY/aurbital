

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, destination):
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def setup_directories():
    
    base_dir = Path('.')
    
    dirs = [
        ,
        ,
        ,
        ,
        ,
        
    ]
    
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")
    return base_dir

def download_eurosat_samples():
    
    print("\n" + "="*60)
    print("DOWNLOADING EUROSAT SAMPLE IMAGES")
    print("="*60)

    from datasets import load_dataset
    
    try:
        print("\nDownloading EuroSAT dataset (this may take a few minutes)...")
        dataset = load_dataset("tanganke/eurosat", split="train[:100]")

        data_dir = Path('data/raw/eurosat')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(dataset)} images to {data_dir}...")
        
        for idx, item in enumerate(dataset):
            
            image = item['image']
            label = item['label']

            label_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 
                          , 'Industrial', 'Pasture', 'PermanentCrop',
                          , 'River', 'SeaLake']
            
            label_name = label_names[label]

            image_path = data_dir / f"{idx:04d}_{label_name}.jpg"
            image.save(image_path)
            
            if (idx + 1) % 10 == 0:
                print(f"  Saved {idx + 1}/100 images...")
        
        print(f"\n✓ Downloaded 100 EuroSAT images to {data_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading EuroSAT: {e}")
        print("Will try alternative source...")
        return False

def download_sample_images_manual():
    
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE SATELLITE IMAGES")
    print("="*60)

    samples = [
        {
            : 'https://raw.githubusercontent.com/EuroSAT/EuroSAT/master/samples/Forest_1.jpg',
            : 'sample_forest.jpg',
            : 'forest'
        },
        
    ]
    
    data_dir = Path('data/raw/samples')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading sample images to {data_dir}...")
    
    for sample in samples:
        try:
            destination = data_dir / sample['filename']
            print(f"\nDownloading {sample['filename']}...")
            download_file(sample['url'], destination)
            print(f"✓ Saved to {destination}")
        except Exception as e:
            print(f"✗ Failed to download {sample['filename']}: {e}")
    
    return True

def create_sample_images():
    
    print("\n" + "="*60)
    print("CREATING SYNTHETIC TEST IMAGES")
    print("="*60)
    
    from PIL import Image
    import numpy as np
    
    data_dir = Path('data/raw/synthetic')
    data_dir.mkdir(parents=True, exist_ok=True)

    image_types = {
        : (34, 139, 34),      
        : (128, 128, 128),     
        : (0, 119, 190),       
        : (255, 215, 0), 
        : (237, 201, 175),    
    }
    
    print(f"\nCreating synthetic images in {data_dir}...")
    
    for name, base_color in image_types.items():
        
        img_array = np.ones((224, 224, 3), dtype=np.uint8)
        
        for i in range(3):
            img_array[:, :, i] = base_color[i]

        noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)
        img.save(data_dir / f'synthetic_{name}.jpg')
        print(f"  ✓ Created {name} image")
    
    print(f"\n✓ Created 5 synthetic test images")
    return True

def main():
    
    print("="*60)
    print("SATELLITE IMAGE DOWNLOAD SCRIPT")
    print("="*60)

    setup_directories()

    print("\nAttempting to download real satellite imagery...")
    
    try:
        success = download_eurosat_samples()
        if success:
            print("\n✓ Successfully downloaded satellite imagery!")
            print("\nYou can find images in:")
            print("  data/raw/eurosat/")
            return
    except:
        print("EuroSAT download failed, trying alternatives...")

    print("\nCreating synthetic test images as backup...")
    create_sample_images()
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check data/raw/ for downloaded images")
    print("2. Run baseline model testing")
    print("3. Evaluate results")

if __name__ == "__main__":
    
    try:
        import datasets
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'datasets'])
    
    main()
