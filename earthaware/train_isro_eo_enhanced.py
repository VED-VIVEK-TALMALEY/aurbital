"""
Enhanced ISRO EO training script.

Uses the existing multispectral Sentinel-style dataset as ISRO EO domain data,
expands caption + QA supervision, and trains with better optimization defaults.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from day4_multimodal_model import MultispectralVLM

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

class ISROEODataset(Dataset):

    # Sentinel-2 Multispectral Band IDs
    BAND_IDS = [
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
    ]

    # Keywords to infer geographic classes from metadata/text
    GEO_CLASS_KEYWORDS = {
        'mountain': ["mountain", "hill", "ridge", "alpine", "highland"],
        'urban': ["building", "urban", "city", "settlement", "infrastructure", "industrial"],
        'river': ["river", "stream", "canal", "drainage"],
        'sea': ["sea", "ocean", "coast", "shore", "marine"],
        'forest': ["forest", "woodland", "tree", "canopy"],
        'agriculture': ["crop", "agri", "farmland", "field", "plantation"],
        'wetland': ["wetland", "marsh", "swamp", "waterlogged", "floodplain"],
        'barren': ["barren", "sparse", "desert", "rocky", "bare soil"],
        'snow': ["snow", "ice", "glacier"],
    }

    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.samples: List[Dict] = []
        for item in raw_items:
            sample_id = item.get("sample_id")
            land_cover = str(item.get("land_cover", "unknown")).lower()
            indices = item.get("spectral_indices", {})
            bands = item["bands"]
            geo_class = self._infer_geo_class(item)

            for cap in item.get("captions", []):
                prompt = self._caption_prompt(sample_id, land_cover, indices)
                self.samples.append({"bands": bands, "prompt": prompt, "answer": cap, "geo_class": geo_class})

            for qa in item.get("qa_pairs", []):
                question = qa.get("question", "Describe the EO scene")
                answer = qa.get("answer", "")
                prompt = self._qa_prompt(sample_id, question, indices)
                self.samples.append({"bands": bands, "prompt": prompt, "answer": answer, "geo_class": geo_class})

        if not self.samples:
            raise ValueError("No training samples were built from training_data.json")

        print(f"Built {len(self.samples)} instruction samples")

    @classmethod
    def _infer_geo_class(cls, item: Dict) -> str:
        land_cover = str(item.get("land_cover", "")).lower()
        text = " ".join(
            [land_cover]
            + [str(x).lower() for x in item.get("captions", [])]
            + [str(qa.get("question", "")).lower() for qa in item.get("qa_pairs", [])]
            + [str(qa.get("answer", "")).lower() for qa in item.get("qa_pairs", [])]
        )

        for cls_name, kws in cls.GEO_CLASS_KEYWORDS.items():
            if any(k in text for k in kws):
                return cls_name
        if "water" in text:
            return "river"
        if "built" in text:
            return "building"
        if "vegetation" in text:
            return "forest"
        return "other"

    @staticmethod
    def _caption_prompt(sample_id: int, land_cover: str, idx: Dict) -> str:
        ndvi = idx.get("NDVI", 0.0)
        ndwi = idx.get("NDWI", 0.0)
        ndbi = idx.get("NDBI", 0.0)
        return (
            "Describe this Earth Observation scene including land-cover and spectral indicators. "
            f"Sample={sample_id}; expected land-cover={land_cover}; "
            f"NDVI={ndvi:.3f}, NDWI={ndwi:.3f}, NDBI={ndbi:.3f}. "
        )

    @staticmethod
    def _qa_prompt(sample_id: int, question: str, idx: Dict) -> str:
        return (
            "Analyze the multispectral image and answer the question provided. "
            f"Sample={sample_id}; NDVI={idx.get('NDVI', 0.0):.3f}; "
            f"NDWI={idx.get('NDWI', 0.0):.3f}; NDBI={idx.get('NDBI', 0.0):.3f}. "
            f"Question: {question}"
        )

    def __len__(self):
        return len(self.samples)

    def _load_multispectral_image(self, bands_dict: Dict[str, str]) -> torch.Tensor:
        bands = []
        for band_id in self.BAND_IDS:
            path = bands_dict.get(band_id)
            if not path:
                raise ValueError(f"Missing band {band_id}")
            arr = np.load(path).astype(np.float32) / 10000.0
            bands.append(arr)
        image = np.stack(bands, axis=0)
        return torch.from_numpy(image).float()

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = self._load_multispectral_image(s["bands"])
        return {"image": image, "prompt": s["prompt"], "answer": s["answer"], "geo_class": s["geo_class"]}

def parse_required_classes(raw: str) -> List[str]:
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return out

def compute_dataset_class_coverage(raw_items: Sequence[Dict]) -> Dict:
    scene_counts: Dict[str, int] = {}
    sample_counts: Dict[str, int] = {}

    for item in raw_items:
        geo_class = ISROEODataset._infer_geo_class(item)
        scene_counts[geo_class] = scene_counts.get(geo_class, 0) + 1

        captions = item.get("captions", []) or []
        qas = item.get("qa_pairs", []) or []
        n_samples = len(captions) + len(qas)
        if n_samples == 0:
            n_samples = 1
        sample_counts[geo_class] = sample_counts.get(geo_class, 0) + n_samples

    # Summarize the coverage of different geographic classes
    return {
        "scene_counts": dict(sorted(scene_counts.items(), key=lambda kv: kv[0])),
        "sample_counts": dict(sorted(sample_counts.items(), key=lambda kv: kv[0])),
        "total_scenes": int(sum(scene_counts.values())),
        "total_samples": int(sum(sample_counts.values())),
    }

def validate_dataset_coverage(
    data_path: str,
    required_classes: Sequence[str],
    report_path: str,
    block_on_missing: bool = True,
) -> Tuple[Dict, List[str]]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    coverage = compute_dataset_class_coverage(raw_items)
    seen = set(coverage["scene_counts"].keys())
    required = [c.lower() for c in required_classes]
    missing = [c for c in required if c not in seen]

    # Build the final coverage report dictionary
    report = {
        "data_path": str(data_path),
        "required_classes": required,
        "missing_classes": missing,
        "coverage": coverage,
        "status": "failed" if (missing and block_on_missing) else ("warning" if missing else "ok"),
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Dataset coverage report written: {report_file}")
    print(f"Scene coverage: {coverage['scene_counts']}")
    if missing:
        print(f"Missing required classes: {missing}")

    if missing and block_on_missing:
        raise ValueError(
            
            f"{missing}. Add data for these classes or run with --allow-missing-classes."
        )

    return report, missing

def build_weighted_train_sampler(dataset: ISROEODataset, train_subset) -> WeightedRandomSampler:
    subset_classes = [dataset.samples[i].get("geo_class", "other") for i in train_subset.indices]
    class_counts: Dict[str, int] = {}
    for c in subset_classes:
        class_counts[c] = class_counts.get(c, 0) + 1

    weights = [1.0 / max(class_counts[c], 1) for c in subset_classes]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

def make_collate_fn(tokenizer, max_length: int):
    def collate(batch):
        images = torch.stack([x["image"] for x in batch])
        texts = [f"{x['prompt']}\nAnswer: {x['answer']}" for x in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100

        # Dictionary containing tensors for model input
        return {
            "images": images,
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
        }

    return collate

def evaluate(model, loader, device, amp_enabled: bool) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=amp_enabled):
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            losses.append(outputs.loss.item())

    return float(sum(losses) / max(len(losses), 1))

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    print(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    coverage_report_path = args.coverage_report_path or str(ckpt_dir / "dataset_class_coverage_report.json")
    required_classes = parse_required_classes(args.required_classes)
    validate_dataset_coverage(
        data_path=args.data_path,
        required_classes=required_classes,
        report_path=coverage_report_path,
        block_on_missing=not args.allow_missing_classes,
    )

    if args.validate_only:
        print("Validation-only mode enabled. Stopping before training.")
        return

    dataset = ISROEODataset(args.data_path)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    model = MultispectralVLM(use_lora=True, lora_rank=args.lora_rank)
    model = model.to(device)

    collate_fn = make_collate_fn(model.tokenizer, args.max_length)
    train_sampler = build_weighted_train_sampler(dataset, train_set) if args.class_balance else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        rem = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * rem))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=amp_enabled)

    best_path = ckpt_dir / "isro_eo_enhanced_best.pt"
    metrics_path = ckpt_dir / "isro_eo_enhanced_metrics.json"
    history_csv_path = metrics_dir / "training_history.csv"
    history_json_path = metrics_dir / "training_history.json"
    loss_png_path = metrics_dir / "training_loss_curve.png"

    best_val = float("inf")
    global_step = 0
    history = []

    print("Starting enhanced ISRO EO fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step_idx, batch in enumerate(pbar, start=1):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=amp_enabled):
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum

            if step_idx % args.grad_accum == 0 or step_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            # Update the progress bar with training status
            pbar.set_postfix(
                {
                    "loss": f"{running_loss / step_idx:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device, amp_enabled)
        # Record metrics for the current epoch
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            # Save detailed checkpoint for later recovery or inference
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "config": vars(args),
                },
                best_path,
            )
            print(f"Saved new best checkpoint: {best_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        # Create final summary dictionary
        json.dump(
            {
                "best_val_loss": best_val,
                "epochs": args.epochs,
                "total_steps": global_step,
                "history": history,
            },
            f,
            indent=2,
        )

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, indent=2)

    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "train_loss": row.get("train_loss"),
                    "val_loss": row.get("val_loss"),
                    "lr": row.get("lr"),
                }
            )

    if plt is not None and history:
        epochs = [x["epoch"] for x in history]
        train_losses = [x["train_loss"] for x in history]
        val_losses = [x["val_loss"] for x in history]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, marker="o", label="train_loss")
        plt.plot(epochs, val_losses, marker="o", label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ISRO EO Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_png_path, dpi=180)
        plt.close()
    else:
        with open(metrics_dir / "training_loss_curve.txt", "w", encoding="utf-8") as f:
            f.write("matplotlib not available. Install matplotlib to generate PNG loss curves.\n")

    print("Training complete")
    print(f"Best checkpoint: {best_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Metrics visuals folder: {metrics_dir}")
    print(f"History CSV: {history_csv_path}")
    print(f"Loss curve: {loss_png_path if plt is not None else metrics_dir / 'training_loss_curve.txt'}")

def parse_args():
    p = argparse.ArgumentParser(description="Enhanced ISRO EO training")
    p.add_argument("--data-path", default="data/training/training_data.json")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--metrics-dir", default="metrics")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-balance", action="store_true", help="Use weighted sampling to balance geo classes")
    p.add_argument(
        "--required-classes",
        default="mountain,building,river,sea,forest,agriculture,wetland,barren",
        help="Comma-separated required geo classes. Training stops if any class is missing.",
    )
    p.add_argument(
        "--coverage-report-path",
        default="",
        help="Optional JSON path for dataset class coverage report (default: <checkpoint-dir>/dataset_class_coverage_report.json).",
    )
    p.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help="Allow training even when required classes are missing (not recommended).",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Run dataset class validation and exit without training.",
    )
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
