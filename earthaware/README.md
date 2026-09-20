# EarthAware EO-VLM (ISRO-focused)

End-to-end Earth Observation Vision-Language project with:
- Multispectral model training
- ISRO-style instruction fine-tuning
- Evaluation pipeline
- FastAPI backend (`/analyze`, `/batch_analyze`, `/chat`)
- Frontends: Streamlit app + HTML research chat UI (your design language)

## Project Status

Completed and runnable in current workspace:
- Enhanced training script: `train_isro_eo_enhanced.py`
- Enhanced checkpoint: `checkpoints/isro_eo_enhanced_best.pt`
- Enhanced metrics: `checkpoints/isro_eo_enhanced_metrics.json`
- Backend auto-loads enhanced checkpoint first, then falls back to `checkpoints/best_model.pt`

## Directory (core)

- `train_isro_eo_enhanced.py`: improved ISRO EO fine-tuning
- `expand_training_dataset.py`: expands EO dataset with synthetic paraphrase/QA augmentation
- `day4_evaluate.py`: sample-level evaluation
- `day5_evaluate_comprehensive.py`: benchmark-style evaluation
- `api_server.py`: FastAPI inference + chat backend
- `streamlit_app.py`: Streamlit frontend
- `research_chat_ui.html`: static chat frontend with your visual design language
- `run_project.py`: single CLI runner for the whole project

## Environment

Use the project virtual env (`earthaware/venv`) if available.

Install dependencies (if needed):

```powershell
cd earthaware
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## One-Command Runner

From `earthaware/`:

```powershell
venv\Scripts\python.exe run_project.py check
venv\Scripts\python.exe run_project.py train --epochs 2 --batch-size 1 --grad-accum 4 --lr 2e-5
venv\Scripts\python.exe run_project.py eval
venv\Scripts\python.exe run_project.py api
venv\Scripts\python.exe run_project.py streamlit
```

## Manual Run Flow

1. Train enhanced model

```powershell
cd earthaware
$env:PYTHONIOENCODING='utf-8'
venv\Scripts\python.exe train_isro_eo_enhanced.py --epochs 2 --batch-size 1 --grad-accum 4 --lr 2e-5
```

2. Evaluate

```powershell
venv\Scripts\python.exe day4_evaluate.py
# optional comprehensive
venv\Scripts\python.exe day5_evaluate_comprehensive.py
```

3. Start backend API

```powershell
venv\Scripts\python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

4. Use frontend

- Streamlit: run `venv\Scripts\python.exe -m streamlit run streamlit_app.py`
- HTML UI: open `research_chat_ui.html` in browser and use `/chat`

## API Endpoints

- `POST /analyze`: single image + question
- `POST /batch_analyze`: batch inference
- `POST /analyze_dual`: two-image comparative analysis + change report
- `POST /chat`: research assistant chat with memory
- `POST /chat/reset`: reset chat session
- `GET /health`: service health

## Notes

- If `meta-llama/Llama-2-7b-hf` is gated/unavailable, model code falls back to GPT-2.
- With 4GB VRAM constraints, keep batch size at `1` and use gradient accumulation.
- To use a stronger base model, update `day4_multimodal_model.py` to an accessible instruction-tuned checkpoint.

## Current Trained Result Snapshot

From `checkpoints/isro_eo_enhanced_metrics.json`:
- Epoch 1: train loss `5.0410`, val loss `4.1517`
- Epoch 2: train loss `3.8561`, val loss `3.9485`

This indicates successful convergence compared to initial state.

## Expanded Dataset Run

Generate extra EO data:

```powershell
venv\Scripts\python.exe expand_training_dataset.py --factor 3 --output data/training/training_data_expanded.json
```

Train on expanded dataset:

```powershell
venv\Scripts\python.exe run_project.py train --data-path data/training/training_data_expanded.json --epochs 4 --batch-size 1 --grad-accum 8 --lr 1e-5
```

