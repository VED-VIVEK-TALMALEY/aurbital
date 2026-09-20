# 🔬 Research Assets — Metrics Summary

Consolidated tables for the research paper. Charts live in
[`../visualizations/`](../visualizations/) — regenerate with
`python generate_visualizations.py` after every training/evaluation run.

---

## 1. Model Configuration

| Component | Specification |
|-----------|--------------|
| Vision encoder | SpectralViT (ViT + spectral adapters, ≤13 bands, 512×512) |
| Language decoder | GPT-2 |
| Fine-tuning | LoRA (PEFT), rank 8, target q/k/v/o projections |
| Trainable params | < 1% of total |
| Input | Sentinel-2 style multispectral stacks (13 bands) |
| Max sequence length | 192 tokens |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01, cosine schedule, warmup 5% |

## 2. Dataset Coverage

| Class | Scenes | Samples |
|-------|--------|---------|
| Agriculture | 10 | 80 |
| Barren | 10 | 80 |
| Forest | 10 | 80 |
| River | 10 | 80 |
| Urban | 10 | 80 |
| **Total** | **50** | **400** |

> Coverage report: `checkpoints/dataset_class_coverage_report.json`
> (missing classes flagged: mountain, building, sea, wetland — train with
> `--allow-missing-classes` or extend the dataset)

## 3. Training Metrics

Source of truth: `../metrics/training_history.json` + `../metrics/training_history.csv`

| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|----------|----|
| _populated from metrics/ after each run_ | | | |

Historic reference run (7 epochs, 315 steps):

| Epoch | Train Loss | Val Loss | Improvement |
|-------|-----------|----------|-------------|
| 1 | 5.632 | 4.271 | — |
| 2 | 4.112 | 4.209 | ↓ 0.062 |
| 3 | 3.809 | 4.091 | ↓ 0.118 |
| 4 | 3.615 | 3.987 | ↓ 0.104 |
| 5 | 3.433 | 3.780 | ↓ 0.207 |
| 6 | 3.339 | **3.514** | ↓ 0.266 |
| 7 | 3.304 | 3.570 | ↑ 0.056 |

Best checkpoint: **epoch 6** (val loss 3.514, train loss ↓ 41.3%).

## 4. Comprehensive Evaluation (day5)

Source: `../results/day5_evaluation.json`

| Metric | Trained | Baseline (BLIP) |
|--------|---------|-----------------|
| NDVI usage | 85.0% | 0.0% |
| NIR usage | 15.0% | 0.0% |
| SWIR usage | 0.0% | 0.0% |
| Reflectance usage | 65.0% | 0.0% |
| NDVI coverage | 85.0% | 0.0% |
| NDVI avg error | 0.0589 | — |

## 5. Figures

| Figure | File |
|--------|------|
| Training/validation loss curve | `../visualizations/training_loss_curve.png` |
| Metrics dashboard (4-panel) | `../visualizations/metrics_dashboard.png` |
| System & model architecture | `../visualizations/architecture_diagram.png` |
| Architecture (mermaid source) | [`architecture_diagram.md`](architecture_diagram.md) |

## Refresh checklist

```bash
cd earthaware
python train_isro_eo_enhanced.py --epochs 7 --batch-size 1 --grad-accum 45 --lr 2e-5 --class-balance --allow-missing-classes
python day5_evaluate_comprehensive.py
python generate_visualizations.py
# then update Section 3 above from metrics/training_history.csv
```
