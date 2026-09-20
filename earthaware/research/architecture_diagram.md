# 🏗️ Architecture Diagram (Mermaid)

Rendered PNG version: [`../visualizations/architecture_diagram.png`](../visualizations/architecture_diagram.png)
Dashboard charts: [`../visualizations/metrics_dashboard.png`](../visualizations/metrics_dashboard.png)

## System & Model Architecture

```mermaid
flowchart TB
    subgraph USER["👤 User Layer"]
        B["Browser — 3D Map (MapLibre GL)<br/>Research Chat · Multimodal Upload · Auth Dashboard"]
    end

    subgraph FE["🎨 Frontend — React 18 + TypeScript + Vite"]
        A["App.tsx · Map3D.tsx · ChatPanel.tsx<br/>useGeoStore.ts (Zustand) · api.ts"]
    end

    subgraph ORCH["🔀 Orchestration — Node.js + Express :3001"]
        E["server.ts · research.ts · auth.ts<br/>openaiService.ts · cache.ts · Zod validation"]
    end

    subgraph ML["🤖 ML API — FastAPI + Uvicorn :8000"]
        API["/analyze · /batch_analyze · /analyze_dual<br/>/chat · /chat/reset · /health"]
    end

    subgraph MODEL["🧠 Model Layer — Multispectral VLM"]
        direction TB
        IN["Multispectral Input (≤13 bands, 512×512)"] --> SA["Spectral Attention +<br/>Patch Embedding"]
        SA --> VIT["SpectralViT Encoder<br/>(spectral adapters)"]
        VIT --> PROJ["Projection Layer<br/>(vision → language space)"]
        PROJ --> GPT["GPT-2 Decoder<br/>+ LoRA Adapters (PEFT)"]
    end

    subgraph DATA["🗄️ Data Layer"]
        D1["data/ — EuroSAT · Sentinel-2 multispectral"]
        D2["metrics/ · results/ · visualizations/"]
        D3["checkpoints/ (local only, gitignored)"]
    end

    B -->|HTTP| A
    A -->|REST /api/*| E
    E -->|POST /analyze · /chat| API
    API -->|inference| IN
    GPT -->|EO text response| API
    D1 -.training data.-> MODEL
    MODEL -.artifacts.-> D2
    MODEL -.weights.-> D3
```

## Training Pipeline

```mermaid
flowchart LR
    S1["Stage 1<br/>Projection Pretraining<br/>(frozen encoder + LM)"] --> S2["Stage 2<br/>LoRA Instruction Tuning<br/>(PEFT adapters)"]
    S2 --> S3["Stage 3<br/>ISRO Domain Fine-Tuning"]
    S3 --> EVAL["Comprehensive Evaluation<br/>day5_evaluate_comprehensive.py"]
    EVAL --> VIZ["Charts & Tables<br/>generate_visualizations.py"]
```
