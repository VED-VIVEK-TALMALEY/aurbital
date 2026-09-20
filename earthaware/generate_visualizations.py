"""
generate_visualizations.py
Renders publication-ready charts and diagrams from the latest metrics and
evaluation outputs into earthaware/visualizations/.

Outputs:
  1. metrics_dashboard.png    - 2x2 dashboard: loss curves, per-epoch improvement,
                                NDVI error distribution, EO keyword usage bars
  2. architecture_diagram.png - Visual model & platform architecture diagram

Usage:
  python generate_visualizations.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "visualizations"
OUT.mkdir(exist_ok=True)

DARK = "#0d1117"
ACCENT = "#00e5ff"
GREEN = "#00e676"
AMBER = "#ff9100"
RED = "#ff1744"
GREY = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_history():
    """Load training history from metrics/ (json preferred, csv fallback)."""
    hist = []
    j = load_json(ROOT / "metrics" / "training_history.json")
    if j and "history" in j:
        hist = j["history"]
    else:
        csv_path = ROOT / "metrics" / "training_history.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    hist.append({
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_loss": float(row["val_loss"]),
                        "lr": float(row.get("lr", 0) or 0),
                    })
    return hist


def get_eval_metrics():
    """Load summary metrics from the comprehensive evaluation."""
    j = load_json(ROOT / "results" / "day5_evaluation.json")
    if not j:
        return None, []
    trained = j.get("results", {}).get("trained", [])
    return j.get("metrics", {}), trained


def render_dashboard():
    hist = get_history()
    metrics, trained = get_eval_metrics()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Aurbital (Aurbital) — Training & Evaluation Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)

    # (1) Loss curves
    ax = axes[0][0]
    if hist:
        ep = [h["epoch"] for h in hist]
        ax.plot(ep, [h["train_loss"] for h in hist], "o-", color=ACCENT, label="train loss")
        ax.plot(ep, [h["val_loss"] for h in hist], "s--", color=AMBER, label="val loss")
        best = min(hist, key=lambda h: h["val_loss"])
        ax.scatter([best["epoch"]], [best["val_loss"]], marker="*", s=250,
                   color=GREEN, zorder=5, label=f"best (ep {best['epoch']})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training / Validation Loss")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No training history found\nrun training first",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_title("Training / Validation Loss")

    # (2) Per-epoch val loss improvement
    ax = axes[0][1]
    if len(hist) > 1:
        deltas = [hist[i]["val_loss"] - hist[i - 1]["val_loss"]
                  for i in range(1, len(hist))]
        colors = [GREEN if d < 0 else RED for d in deltas]
        ax.bar(range(2, len(hist) + 1), deltas, color=colors, alpha=0.85)
        ax.axhline(0, color="#444", lw=1)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Δ val loss vs previous epoch")
        ax.set_title("Epoch-over-Epoch Validation Improvement")
    else:
        ax.text(0.5, 0.5, "Need ≥2 epochs of history",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_title("Epoch-over-Epoch Validation Improvement")

    # (3) NDVI error distribution (trained model)
    ax = axes[1][0]
    if trained:
        errs = [s.get("ndvi_error_trained") for s in trained
                if isinstance(s.get("ndvi_error_trained"), (int, float))
                and np.isfinite(s["ndvi_error_trained"])]
        if errs:
            ax.hist(errs, bins=12, color=ACCENT, edgecolor="#333", alpha=0.85)
            ax.axvline(np.mean(errs), color=RED, ls="--", lw=2,
                      label=f"mean = {np.mean(errs):.4f}")
            ax.set_xlabel("|NDVI error|"); ax.set_ylabel("Samples")
            ax.set_title("NDVI Prediction Error (Trained Model)")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No finite NDVI errors in evaluation",
                    ha="center", va="center", transform=ax.transAxes, color=GREY)
            ax.set_title("NDVI Prediction Error (Trained Model)")
    else:
        ax.text(0.5, 0.5, "No evaluation results found\nrun day5_evaluate_comprehensive.py",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_title("NDVI Prediction Error (Trained Model)")

    # (4) EO keyword usage: trained vs baseline
    ax = axes[1][1]
    if metrics:
        labels = ["NDVI", "NIR", "SWIR", "Reflectance"]
        t_vals = [metrics.get("ndvi_usage_trained", 0),
                  metrics.get("nir_usage_trained", 0),
                  metrics.get("swir_usage_trained", 0),
                  metrics.get("reflectance_usage_trained", 0)]
        b_vals = [metrics.get("ndvi_usage_baseline", 0),
                  metrics.get("nir_usage_baseline", 0),
                  metrics.get("swir_usage_baseline", 0),
                  metrics.get("reflectance_usage_baseline", 0)]
        x = np.arange(len(labels)); w = 0.38
        ax.bar(x - w / 2, t_vals, w, label="Trained", color=GREEN, alpha=0.9)
        ax.bar(x + w / 2, b_vals, w, label="Baseline", color=GREY, alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Usage (%)"); ax.set_ylim(0, 100)
        ax.set_title("EO Terminology Usage — Trained vs Baseline")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No evaluation metrics found",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_title("EO Terminology Usage — Trained vs Baseline")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "metrics_dashboard.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[ok] wrote {out}")


def render_architecture_diagram():
    """Draw the model + platform architecture as a clean layered diagram."""
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
    fig.suptitle("Aurbital — System & Model Architecture",
                 fontsize=16, fontweight="bold", y=0.985)

    def box(x, y, w, h, text, fc, ec="#333", fs=10, tc="black", bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4,
                                   joinstyle="round", zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
                zorder=3)

    def arrow(x1, y1, x2, y2, label="", color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        if label:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, label, fontsize=8,
                    color=color, ha="left", va="center", style="italic")

    L = 0.6; W = 8.8

    # User layer
    box(L, 12.6, W, 1.0,
        "USER LAYER — Browser: 3D Map (MapLibre GL) · Research Chat · Multimodal Upload · Auth Dashboard",
        "#e8f4fd", fs=9, bold=True)
    arrow(5, 12.55, 5, 12.0, "HTTP")

    # Frontend
    box(L, 11.0, W, 0.95,
        "FRONTEND — React 18 + TypeScript + Vite  (App.tsx · Map3D · ChatPanel · Zustand store)",
        "#e8f4fd", fs=9)
    arrow(5, 10.95, 5, 10.4, "REST /api/*")

    # Orchestration
    box(L, 9.4, W, 0.95,
        "ORCHESTRATION — Node.js + Express (port 3001): routing · auth · caching · rate limiting · RL feedback",
        "#fff4e0", fs=9)
    arrow(5, 9.35, 5, 8.8, "POST /analyze · /chat")

    # ML API
    box(L, 7.8, W, 0.95,
        "ML API — FastAPI + Uvicorn (port 8000): /analyze · /batch_analyze · /analyze_dual · /chat · /health",
        "#fff4e0", fs=9)
    arrow(5, 7.75, 5, 7.2, "inference")

    # Model layer
    box(L, 6.2, W, 0.95,
        "MODEL LAYER — Multispectral VLM: SpectralViT encoder + GPT-2 decoder + LoRA (PEFT)",
        "#e6f7ea", fs=10, bold=True)

    # Model internals (vertical flow)
    inner = [
        ("Multispectral Input — up to 13 bands, 512×512", 4.9),
        ("Spectral Attention + Patch Embedding (band-aware tokens)", 4.0),
        ("SpectralViT Encoder (ViT with spectral adapters)", 3.1),
        ("Projection Layer (vision → language space)", 2.2),
        ("GPT-2 Decoder + LoRA Adapters → EO text response", 1.3),
    ]
    for i, (txt, y) in enumerate(inner):
        box(1.2, y, 7.6, 0.62, txt, "#f3fbf3", fs=8.5)
        if i < len(inner) - 1:
            arrow(5, y - 0.04, 5, y - 0.32, "")

    # Data layer
    box(L, 0.25, W, 0.72,
        "DATA LAYER — data/ (EuroSAT · Sentinel-2) · metrics/ · results/ · checkpoints/ (local only) · visualizations/",
        "#f0eefb", fs=8.5)
    arrow(1.6, 1.25, 1.6, 1.0, "", color="#777")

    # Training pipeline side note
    ax.text(9.55, 4.0,
            "TRAINING PIPELINE\n\nStage 1: projection\npretraining\n\nStage 2: LoRA\ninstruction tuning\n\nStage 3: ISRO\ndomain fine-tuning",
            fontsize=8.5, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fffde7", ec="#c9b458"))

    out = OUT / "architecture_diagram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    render_dashboard()
    render_architecture_diagram()
    print("Done. Charts are in earthaware/visualizations/")
