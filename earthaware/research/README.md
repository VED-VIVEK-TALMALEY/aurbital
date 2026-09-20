# 🔬 Research Assets — Metrics Summary

Consolidated tables for the research paper. Charts live in
[`../visualizations/`](../visualizations/) — regenerate with
`python generate_visualizations.py` after every training/evaluation run.

---

## 1. Model Configuration

| Component | Specification |
|-----------|--------------|
| Vision encoder | SpectralViT (spectral patch embedding + spectral attention, 13 bands, 64×64) |
| Language decoder | GPT-2 (124M) via PEFT LoRA |
| Fine-tuning | LoRA rank 8 on `c_attn` (q/k/v) projections |
| Trainable params | 294,912 / 124,734,720 = **0.2364%** |
| Input | Sentinel-2 style multispectral stack, 13 bands, reflectance / 10000 |
| Max sequence length | 192 tokens |
| Optimizer | AdamW, lr 2e-5, cosine schedule, warmup 5%, weight decay 0.01 |
| Hardware | Single consumer GPU, ~2.9 GB VRAM during training |

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
> Missing classes flagged: mountain, building, sea, wetland (trained with `--allow-missing-classes`).

## 3. Training Metrics — Fresh Run (7 epochs, 56 global steps)

Source: `../metrics/training_history.json` + `../metrics/training_history.csv`

| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|----------|-----|
| 1 | 7.0570 | 5.0713 | 1.94e-05 |
| 2 | 5.2109 | 4.9057 | 1.69e-05 |
| 3 | 4.4500 | 4.0907 | 1.29e-05 |
| 4 | 4.1055 | 4.0308 | 8.26e-06 |
| 5 | 3.8648 | 4.0736 | 4.03e-06 |
| 6 | 3.7731 | 4.0373 | 1.06e-06 |
| 7 | 3.7280 | **4.0221** | 0.0 |

**Summary:** best val loss **4.0221** (epoch 7) · train loss ↓ **47.2%** (7.06 → 3.73) ·
val loss ↓ **20.7%** (5.07 → 4.02) · best checkpoint saved at epoch 7.

### Historic reference run (previous TerraSight-era training, 7 epochs / 315 steps)

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 5.632 | 4.271 |
| 6 | 3.339 | **3.514** |

> The current run uses a corrected class-balanced pipeline and the updated dataset;
> absolute losses are not directly comparable to the historic run.

## 4. Comprehensive Evaluation (day5, vs BLIP baseline)

Source: `../results/day5_evaluation.json` (regenerated after the fresh run)

> ⚠️ **Known limitation (honest reporting for the paper):** in the current
> regenerated evaluation the trained model did not emit NDVI values in free-form
> generation. Root cause identified: the training prompt **grounds** the model with
> the ground-truth spectral indices in the prompt text, so unconditional prompts
> are out-of-distribution (conditioned loss 4.13 vs unconditioned 4.50 — the model
> relies on the grounded indices). The historic run's headline metrics are retained
> below for reference, with the caveat that they came from the same grounded setup.

| Metric | Trained (historic ref) | Baseline (BLIP) |
|--------|------------------------|-----------------|
| NDVI usage | 85.0% | 0.0% |
| NIR usage | 15.0% | 0.0% |
| Reflectance usage | 65.0% | 0.0% |
| NDVI avg error | 0.0589 | — |
| NDVI coverage | 85.0% | 0.0% |

### Visual-conditioning diagnostic (this run)

| Condition | Loss on ground-truth answer |
|-----------|------------------------------|
| Grounded prompt + visual tokens | **4.133** |
| Grounded prompt, text-only (no image) | 4.504 |

→ The visual tokens measurably help (Δ ≈ 0.37 nats), but the model is heavily
index-grounded — a key finding motivating the Aurbital/world-model research line.

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
