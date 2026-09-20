# EXPLANATION

## Project Overview
`EarthAware` is a geospatial AI research system that combines:
- a local vision-language model (EO-focused) for image/question analysis,
- a geospatial web app for map-based area selection,
- a research chat assistant for structured findings and follow-up questions,
- researcher-in-the-loop feedback and manual retraining controls.

The implementation is split into two major parts:
1. `earthaware/` (core Python model/training/API stack)
2. `earthaware/geo-research-assistant/` (React + Node geospatial interface layer)

---

## High-Level Architecture

### A) Core AI Stack (`earthaware/`)
- `api_server.py` (FastAPI)
  - Loads EO multimodal model checkpoint on startup.
  - Exposes inference and chat APIs:
    - `POST /analyze`
    - `POST /batch_analyze`
    - `POST /analyze_dual`
    - `POST /chat`
    - `POST /chat/reset`
    - `GET /health`
  - Performs synthetic multispectral conversion from RGB input.
  - Computes EO metadata and indices (`NDVI`, `NDWI`, `NDBI`) and rule-based scene class.
- Training and evaluation scripts
  - `day4_train.py`, `train_isro_eo_enhanced.py`, `day4_evaluate.py`, `day5_evaluate_comprehensive.py`, etc.
- Data and checkpoints
  - `data/training/...`
  - `checkpoints/...`
  - `results/...`

### B) Geospatial Research Interface (`earthaware/geo-research-assistant/`)
- Frontend (React + TypeScript + Vite + Tailwind)
  - 3D-like satellite map interaction via `MapLibre GL` + free satellite tiles.
  - Box selection (Shift + Drag) and point selection.
  - Captures selected box image pixels from map canvas and sends as `imageDataUrl`.
  - Chat panel for analysis display, follow-up questions, RL feedback, and manual training controls.
- Backend (Node + Express + TypeScript)
  - Routes geospatial area payloads, chat, feedback, and manual training.
  - Calls local Python model API (`http://localhost:8000`) for both text and image-grounded analysis.
  - Caching, validation (Zod), rate limiting, and training job tracking.

---

## Frontend Technical Breakdown

### Core files
- `src/components/Map3D.tsx`
  - Renders map with free imagery source (no Mapbox billing/token dependency).
  - Supports:
    - click point selection,
    - Shift+Drag polygon box selection,
    - selection overlay rendering.
  - Uses `preserveDrawingBuffer: true` and crops selected rectangle from canvas.
  - Sends payload:
    - `coordinates`
    - `center`
    - `areaType`
    - `imageDataUrl` (for boxed image inference).
- `src/components/ChatPanel.tsx`
  - Displays structured analysis sections.
  - Supports follow-up Q&A (`/api/research/chat`).
  - Includes:
    - `Submit RL Feedback` (reward + quality metrics)
    - `Start Manual Training` (researcher-defined hyperparameters)
  - Implements loop guards for analysis:
    - one analysis per selected geometry
    - separate loading states for area analysis and follow-up chat.
- `src/lib/api.ts`
  - Typed API client for:
    - area analysis,
    - follow-up chat,
    - feedback submission,
    - manual training start/status polling.
- `src/store/useGeoStore.ts`
  - Zustand state for selected area, messages, session id, and cached analyses.

### UI behavior
- Select region -> auto-analysis triggered once.
- Receive structured findings + prediction.
- Ask follow-up questions in same session context.
- Submit human feedback and run manual retraining from UI.

---

## Backend (Node) Technical Breakdown

### Core files
- `backend/src/server.ts`
  - Express app bootstrap.
  - CORS setup.
  - Rate limiting on `/api/research`.
  - JSON payload limit increased for image data transfer.
- `backend/src/routes/research.ts`
  - `POST /area`:
    - validates payload,
    - caches analysis by area key,
    - returns structured response + assistant message.
  - `POST /chat`:
    - session-aware follow-up responses.
  - `POST /feedback`:
    - stores reinforcement feedback records as JSONL,
    - computes aggregated reward score.
  - `POST /manual-train`:
    - starts Python training process in background,
    - logs to runtime artifacts,
    - stores job metadata.
  - `GET /manual-train/:jobId`:
    - returns status for researcher polling.
- `backend/src/services/openaiService.ts`
  - Unified inference orchestration for local/openai providers.
  - Local-first behavior configured through `.env`.
  - NLP layer for follow-up:
    - question normalization,
    - intent detection,
    - keyword extraction,
    - prompt routing,
    - low-quality retry handling.
  - Image-grounded analysis path:
    - if `imageDataUrl` exists, calls local Python `/analyze` endpoint.
  - Structured fallback and prediction synthesis if model output is weak.

### Validation and types
- `validation/geoValidation.ts` validates coordinates, optional image data URL, feedback schema, and training schema.
- `types.ts` defines contracts shared by router/services.

---

## Data Flow End-to-End

1. User draws a box on map.
2. Frontend computes polygon + captures boxed image from map canvas.
3. Frontend sends area payload to Node backend (`/api/research/area`).
4. Node backend calls local model API:
   - image-aware path via `POST /analyze` when `imageDataUrl` is present.
5. Python model returns analysis (+ metadata).
6. Node transforms/normalizes into research sections and prediction.
7. Frontend renders structured research output.
8. Follow-up questions go through NLP-enhanced backend chat path.
9. Researcher can log feedback and launch manual retraining jobs.

---

## Model and Training Design

### Base inference
- EO VLM handles image+question behavior.
- Synthetic multispectral representation is used from RGB map image where true multispectral bands are unavailable.

### Metrics and derived EO features
- NDVI/NDWI/NDBI used for quick EO cues.
- Rule-based scene class supports explainability in responses.

### Reinforcement-style loop
- Feedback endpoint collects:
  - scalar reward (`researcherScore`),
  - quality axes (`relevance`, `factuality`, `usefulness`),
  - notes/labels.
- Logged to JSONL for later reward-aware retraining.

### Manual retraining
- Manual control API launches Python training with user hyperparameters.
- Job status is tracked and polled from UI.

---

## Security and Stability
- API keys and provider settings from environment variables.
- Input validation on all critical payloads.
- Rate limiting on backend routes.
- Caching to reduce repeated heavy calls.
- Build/typing checks added across backend/frontend.
- Error handling improved for map load, model output quality, and training jobs.

---

## Key Implemented Improvements During Iteration
- Map provider switched to no-payment/free setup.
- Fixed multiple config and encoding issues (BOM, ESM/CJS mismatch).
- Stabilized area-analysis loop and follow-up chat interactivity.
- Added image-grounded boxed selection analysis.
- Added research-specific output with prediction section.
- Added RL feedback and manual training controls from UI.

---

## How to Run (3 terminals)

1. Local model API:
```powershell
cd "C:\Users\talma\Desktop\earthaware-project - Copy\earthaware"
venv\Scripts\python.exe run_project.py api
```

2. Geo backend:
```powershell
cd "C:\Users\talma\Desktop\earthaware-project - Copy\earthaware\geo-research-assistant\backend"
npm run dev
```

3. Geo frontend:
```powershell
cd "C:\Users\talma\Desktop\earthaware-project - Copy\earthaware\geo-research-assistant\frontend"
npm run dev
```

UI: `http://localhost:5173`

---

## Current Scope and Limitations
- Boxed map imagery is RGB/satellite tile-based, not true multispectral Sentinel pixel stacks.
- Predictions are research-guidance quality; production-grade forecasting needs multi-temporal model calibration.
- For publication-grade claims, always validate with external authoritative datasets (census, OSM QA, ground truth, temporal stacks).

---

## Suggested Next Technical Upgrades
- Add true geospatial tile fetch pipeline (STAC/COG) for real multispectral region crops.
- Add temporal pair selection (T1/T2) directly from map UI for change prediction.
- Add quantitative confidence intervals for each section.
- Add experiment tracking linkage (W&B or MLflow) from manual training jobs.
- Add exportable research report generation (JSON/PDF/LaTeX table output).

