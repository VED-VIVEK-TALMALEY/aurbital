"""Expand EO training dataset with synthetic paraphrases and QA augmentation."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy

def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def jitter(v: float, scale: float = 0.03) -> float:
    return clamp(v + random.uniform(-scale, scale))

def build_augmented_item(item: dict, sample_id: int) -> dict:
    out = deepcopy(item)
    out["sample_id"] = sample_id

    ndvi = float(item["spectral_indices"]["NDVI"])
    ndwi = float(item["spectral_indices"]["NDWI"])
    ndbi = float(item["spectral_indices"]["NDBI"])

    ndvi_j = jitter(ndvi)
    ndwi_j = jitter(ndwi)
    ndbi_j = jitter(ndbi)

    out["spectral_indices"] = {
        : ndvi_j,
        : ndwi_j,
        : ndbi_j,
    }

    land = str(item.get("land_cover", "mixed"))

    out["captions"] = [
        f"EO scene with dominant {land} patterns. NDVI {ndvi_j:.2f}, NDWI {ndwi_j:.2f}, NDBI {ndbi_j:.2f} indicate spectral behavior consistent with this class.",
        f"Remote-sensing interpretation: {land} land cover with vegetation-water-builtup profile (NDVI={ndvi_j:.2f}, NDWI={ndwi_j:.2f}, NDBI={ndbi_j:.2f}).",
        f"Spectral-layer summary suggests {land} surface distribution; near-infrared, green, and SWIR relationships support this classification.",
    ]

    out["qa_pairs"] = [
        {
            : "Classify the dominant land cover in this image.",
            : land.capitalize(),
        },
        {
            : "What are the key spectral indices (NDVI/NDWI/NDBI)?",
            : f"NDVI {ndvi_j:.2f}, NDWI {ndwi_j:.2f}, NDBI {ndbi_j:.2f}.",
        },
        {
            : "Which layers are most informative for this classification?",
            : "NIR/Red for NDVI, Green/NIR for NDWI, and SWIR/NIR for NDBI are the main discriminative layers.",
        },
        {
            : "Is this scene likely vegetated, water-dominant, or built-up?",
            : f"It is primarily {land} based on spectral index patterns.",
        },
    ]

    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/training_data.json")
    parser.add_argument("--output", default="data/training/training_data_expanded.json")
    parser.add_argument("--factor", type=int, default=3, help="Augmented copies per original sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input, "r", encoding="utf-8") as f:
        base = json.load(f)

    expanded = []
    next_id = 0
    for item in base:
        original = deepcopy(item)
        original["sample_id"] = next_id
        expanded.append(original)
        next_id += 1

        for _ in range(args.factor):
            expanded.append(build_augmented_item(item, next_id))
            next_id += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2)

    print(f"Base samples: {len(base)}")
    print(f"Expanded samples: {len(expanded)}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
