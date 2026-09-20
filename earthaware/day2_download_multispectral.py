

import numpy as np
from pathlib import Path
import requests
from PIL import Image
import json
from tqdm import tqdm

def download_sentinel2_sample():
    
    print("="*60)
    print("DOWNLOADING SENTINEL-2 MULTISPECTRAL DATA")
    print("="*60)

    try:
        from datasets import load_dataset
        
        print("\nAttempting to download BigEarthNet subset...")
        print("(This dataset has all 13 Sentinel-2 bands)")

        dataset = load_dataset(
            ,
            split="train[:50]"  
        )
        
        print(f"\n✓ Downloaded {len(dataset)} multispectral images")

        output_dir = Path('data/raw/sentinel2_multispectral')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing multispectral data")):
            
            sample_info = {
                : idx,
                : {}
            }

            for band_name, band_data in item.items():
                if isinstance(band_data, Image.Image):
                    band_path = output_dir / f"sample_{idx:04d}_{band_name}.tif"
                    band_data.save(band_path)
                    sample_info['bands'][band_name] = str(band_path)
            
            metadata.append(sample_info)

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved {len(metadata)} multispectral samples to {output_dir}")
        return True
        
    except Exception as e:
        print(f"\nBigEarthNet download failed: {e}")
        print("Falling back to synthetic multispectral generation...")
        return False

def create_synthetic_multispectral():
    
    print("\n" + "="*60)
    print("CREATING SYNTHETIC MULTISPECTRAL DATA")
    print("="*60)
    
    output_dir = Path('data/raw/sentinel2_multispectral')
    output_dir.mkdir(parents=True, exist_ok=True)

    bands_info = {
        : {'name': 'Coastal Aerosol', 'wavelength': 443, 'resolution': 60},
        : {'name': 'Blue', 'wavelength': 490, 'resolution': 10},
        : {'name': 'Green', 'wavelength': 560, 'resolution': 10},
        : {'name': 'Red', 'wavelength': 665, 'resolution': 10},
        : {'name': 'Red Edge 1', 'wavelength': 705, 'resolution': 20},
        : {'name': 'Red Edge 2', 'wavelength': 740, 'resolution': 20},
        : {'name': 'Red Edge 3', 'wavelength': 783, 'resolution': 20},
        : {'name': 'NIR', 'wavelength': 842, 'resolution': 10},
        : {'name': 'NIR Narrow', 'wavelength': 865, 'resolution': 20},
        : {'name': 'Water Vapor', 'wavelength': 945, 'resolution': 60},
        : {'name': 'SWIR Cirrus', 'wavelength': 1375, 'resolution': 60},
        : {'name': 'SWIR 1', 'wavelength': 1610, 'resolution': 20},
        : {'name': 'SWIR 2', 'wavelength': 2190, 'resolution': 20},
    }

    spectral_signatures = {
        : {
            : 0.03, 'B03': 0.04, 'B04': 0.03,  
            : 0.10, 'B06': 0.15, 'B07': 0.20,  
            : 0.45, 'B8A': 0.45,               
            : 0.25, 'B12': 0.15                
        },
        : {
            : 0.15, 'B03': 0.12, 'B04': 0.08,  
            : 0.04, 'B06': 0.02, 'B07': 0.01,  
            : 0.01, 'B8A': 0.01,               
            : 0.00, 'B12': 0.00                
        },
        : {
            : 0.12, 'B03': 0.15, 'B04': 0.18,  
            : 0.20, 'B06': 0.22, 'B07': 0.24,  
            : 0.28, 'B8A': 0.28,               
            : 0.32, 'B12': 0.35                
        },
        : {
            : 0.05, 'B03': 0.06, 'B04': 0.05,  
            : 0.12, 'B06': 0.18, 'B07': 0.25,  
            : 0.40, 'B8A': 0.40,               
            : 0.22, 'B12': 0.12                
        },
        : {
            : 0.10, 'B03': 0.13, 'B04': 0.16,  
            : 0.18, 'B06': 0.20, 'B07': 0.22,  
            : 0.24, 'B8A': 0.24,               
            : 0.28, 'B12': 0.30                
        }
    }
    
    print("\nCreating synthetic multispectral images...")
    print("Land cover types: forest, water, urban, agriculture, bare_soil")
    
    metadata = []
    image_size = 64  
    
    for land_cover, signature in spectral_signatures.items():
        for sample_num in range(10):  
            
            sample_id = len(metadata)
            sample_info = {
                : sample_id,
                : land_cover,
                : {},
                : {}
            }

            for band_id, band_spec in bands_info.items():

                if band_id in signature:
                    base_value = signature[band_id]
                else:
                    
                    base_value = 0.15

                band_array = np.random.normal(
                    base_value, 
                    0.05,  
                    size=(image_size, image_size)
                )

                band_array = np.clip(band_array, 0, 1)

                band_uint16 = (band_array * 10000).astype(np.uint16)

                band_path = output_dir / f"sample_{sample_id:04d}_{band_id}.npy"
                np.save(band_path, band_uint16)
                
                sample_info['bands'][band_id] = str(band_path)

            red = np.load(sample_info['bands']['B04']) / 10000.0
            green = np.load(sample_info['bands']['B03']) / 10000.0
            nir = np.load(sample_info['bands']['B08']) / 10000.0
            swir1 = np.load(sample_info['bands']['B11']) / 10000.0

            ndvi = (nir - red) / (nir + red + 1e-8)
            sample_info['spectral_indices']['NDVI'] = float(np.mean(ndvi))

            ndwi = (green - nir) / (green + nir + 1e-8)
            sample_info['spectral_indices']['NDWI'] = float(np.mean(ndwi))

            ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
            sample_info['spectral_indices']['NDBI'] = float(np.mean(ndbi))
            
            metadata.append(sample_info)

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created {len(metadata)} synthetic multispectral samples")
    print(f"  Saved to: {output_dir}")

    print("\n" + "="*60)
    print("EXAMPLE SPECTRAL INDICES")
    print("="*60)
    
    for land_cover in spectral_signatures.keys():
        samples = [m for m in metadata if m['land_cover'] == land_cover]
        avg_ndvi = np.mean([s['spectral_indices']['NDVI'] for s in samples])
        avg_ndwi = np.mean([s['spectral_indices']['NDWI'] for s in samples])
        avg_ndbi = np.mean([s['spectral_indices']['NDBI'] for s in samples])
        
        print(f"\n{land_cover.upper()}:")
        print(f"  NDVI (vegetation): {avg_ndvi:+.3f}")
        print(f"  NDWI (water):      {avg_ndwi:+.3f}")
        print(f"  NDBI (built-up):   {avg_ndbi:+.3f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("NDVI: >0.3 = vegetation, <0.2 = bare soil/urban")
    print("NDWI: >0.0 = water, <0.0 = land")
    print("NDBI: >0.0 = urban/built-up, <0.0 = vegetation")
    
    return True

def main():

    print("="*60)
    print("DAY 2: MULTISPECTRAL DATA ACQUISITION")
    print("="*60)

    success = download_sentinel2_sample()

    if not success:
        create_synthetic_multispectral()
    
    print("\n" + "="*60)
    print("DATA ACQUISITION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Create RGB and false-color composites")
    print("2. Test baseline model on different band combinations")
    print("3. Compare results to demonstrate information loss")

if __name__ == "__main__":
    main()
