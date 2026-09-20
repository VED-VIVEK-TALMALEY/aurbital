"""
FastAPI server for Terra Sight Multispectral VLM.

Implements guide-style endpoints:
- POST /analyze
- POST /batch_analyze
- GET /health
"""

from __future__ import annotations

import io
import uuid
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from day4_multimodal_model import MultispectralVLM

app = FastAPI(
    title="ISRO Multimodal GPT API",
    description="Visual Question Answering for Earth Observation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: MultispectralVLM | None = None
chat_sessions: Dict[str, List[dict]] = {}
MAX_CHAT_TURNS = 12

RESEARCH_SYSTEM_PROMPT = (

)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    max_tokens: int = Field(default=256, ge=16, le=512)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

class ChatResetRequest(BaseModel):
    session_id: str

def _rgb_to_multispectral(pil_image: Image.Image) -> torch.Tensor:
    
    resized = pil_image.convert("RGB").resize((64, 64))
    arr = np.asarray(resized).astype(np.float32) / 255.0  
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    nir = np.clip(0.6 * green + 0.4 * red, 0.0, 1.0)
    swir = np.clip(0.7 * red + 0.3 * blue, 0.0, 1.0)
    coastal = np.clip(0.6 * blue + 0.4 * green, 0.0, 1.0)
    red_edge_1 = np.clip(0.8 * red + 0.2 * nir, 0.0, 1.0)
    red_edge_2 = np.clip(0.6 * red + 0.4 * nir, 0.0, 1.0)
    red_edge_3 = np.clip(0.4 * red + 0.6 * nir, 0.0, 1.0)
    nir_narrow = np.clip(0.9 * nir + 0.1 * red, 0.0, 1.0)
    water_vapor = np.clip(0.5 * nir + 0.5 * swir, 0.0, 1.0)
    cirrus = np.clip(0.3 * nir + 0.7 * swir, 0.0, 1.0)
    swir2 = np.clip(0.9 * swir + 0.1 * red, 0.0, 1.0)

    bands = np.stack(
        [
            coastal,   
            blue,      
            green,     
            red,       
            red_edge_1,  
            red_edge_2,  
            red_edge_3,  
            nir,       
            nir_narrow,  
            water_vapor,  
            cirrus,    
            swir,      
            swir2,     
        ],
        axis=0,
    )
    return torch.from_numpy(bands).float()

def _safe_div(num: float, den: float, eps: float = 1e-6) -> float:
    return float(num / (den + eps))

def _spectral_indices(image_13_band: torch.Tensor) -> dict:
    
    arr = image_13_band.cpu().numpy()
    red = float(arr[3].mean())   
    green = float(arr[2].mean()) 
    nir = float(arr[7].mean())   
    swir = float(arr[11].mean()) 

    ndvi = _safe_div(nir - red, nir + red)
    ndwi = _safe_div(green - nir, green + nir)
    ndbi = _safe_div(swir - nir, swir + nir)
    return {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi}

def _classify_scene(indices: dict) -> dict:
    
    ndvi = float(indices["NDVI"])
    ndwi = float(indices["NDWI"])
    ndbi = float(indices["NDBI"])

    if ndwi > 0.2:
        label = "water_dominant"
        confidence = min(0.98, 0.6 + abs(ndwi - 0.2))
    elif ndvi > 0.45 and ndbi < 0.0:
        label = "vegetation_dominant"
        confidence = min(0.98, 0.6 + abs(ndvi - 0.45))
    elif ndbi > 0.2 and ndvi < 0.3:
        label = "urban_builtup"
        confidence = min(0.98, 0.6 + abs(ndbi - 0.2))
    elif ndvi < 0.15 and ndwi < 0.0 and ndbi < 0.05:
        label = "barren_or_sparse"
        confidence = 0.72
    else:
        label = "mixed_landcover"
        confidence = 0.64

    return {"label": label, "confidence": round(float(confidence), 3)}

def _layer_summary(image_13_band: torch.Tensor) -> list:
    
    band_ids = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    arr = image_13_band.cpu().numpy()
    layers = []
    for idx, band_id in enumerate(band_ids):
        band = arr[idx]
        # Create a dictionary of statistics for each spectral band
        layers.append(
            {
                "band": band_id,
                "mean": round(float(band.mean()), 5),
                "std": round(float(band.std()), 5),
                "min": round(float(band.min()), 5),
                "max": round(float(band.max()), 5),
            }
        )
    return layers

def _build_analysis_metadata(image_13_band: torch.Tensor, pil_image: Image.Image) -> dict:
    indices = _spectral_indices(image_13_band)
    # Bundle all analysis results into a structured metadata dictionary
    return {
        "dimensions": list(pil_image.size),
        "model": "Terra Sight SpectralVLM",
        "device": str(device),
        "classification": _classify_scene(indices),
        "spectral_indices": {k: round(float(v), 5) for k, v in indices.items()},
        "layers": _layer_summary(image_13_band),
    }

def _build_change_report(meta_a: dict, meta_b: dict) -> dict:
    idx_a = meta_a["spectral_indices"]
    idx_b = meta_b["spectral_indices"]
    delta = {k: round(float(idx_b[k] - idx_a[k]), 5) for k in idx_a.keys()}
    magnitude = abs(delta["NDVI"]) + abs(delta["NDWI"]) + abs(delta["NDBI"])

    if magnitude < 0.05:
        change_label = "low_change"
    elif magnitude < 0.15:
        change_label = "moderate_change"
    else:
        change_label = "high_change"

    # Report the differences and labels for the two images
    return {
        "change_magnitude": change_label,
        "spectral_delta": delta,
        "initial_label": meta_a["classification"]["label"],
        "final_label": meta_b["classification"]["label"],
    }

def _generate_response(
    loaded_model: MultispectralVLM,
    image_13_band: torch.Tensor,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    
    loaded_model.eval()
    if not question.strip():
        question = "Describe this satellite image."

    with torch.no_grad():
        visual_embeddings = loaded_model.encode_image(image_13_band.unsqueeze(0).to(device))
        prompt_tokens = loaded_model.tokenizer(question, return_tensors="pt").to(device)
        generated_ids = prompt_tokens.input_ids.clone()

        for _ in range(max_new_tokens):
            text_embeddings = loaded_model.language_model.get_input_embeddings()(generated_ids)
            combined = torch.cat([visual_embeddings, text_embeddings], dim=1)

            outputs = loaded_model.language_model(
                inputs_embeds=combined,
                attention_mask=torch.ones(combined.shape[:2], dtype=torch.long, device=device),
            )
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0:
                next_token_logits = next_token_logits / max(temperature, 1e-3)
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == loaded_model.tokenizer.eos_token_id:
                break

        return loaded_model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def _build_chat_prompt(history: List[dict], user_message: str) -> str:
    lines = [f"System: {RESEARCH_SYSTEM_PROMPT}"]
    for turn in history[-MAX_CHAT_TURNS:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)

def _generate_text_response(
    loaded_model: MultispectralVLM, prompt: str, max_new_tokens: int, temperature: float
) -> str:
    loaded_model.eval()
    tokenizer = loaded_model.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        output_ids = loaded_model.language_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

@app.on_event("startup")
async def load_model() -> None:
    global model
    try:
        model = MultispectralVLM(use_lora=True, lora_rank=8)
        # Define potential paths for the model checkpoint
        checkpoint_candidates = [
            "checkpoints/isro_eo_enhanced_best.pt",
            "checkpoints/best_model.pt",
            "checkpoints/multispectral_vlm_latest.pt",
            "models/multispectral_vlm_latest.pt",
        ]
        checkpoint = None
        for ckpt_path in checkpoint_candidates:
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                print(f"Loaded checkpoint: {ckpt_path}")
                break
            except FileNotFoundError:
                continue

        if checkpoint is None:
            raise FileNotFoundError("No checkpoint found in checkpoints/")

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load model/checkpoint: {exc}") from exc

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    max_length: int = Form(default=128),
    temperature: float = Form(default=0.2),
) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        multispectral = _rgb_to_multispectral(pil_image)
        analysis = _generate_response(
            model, multispectral, question, max_new_tokens=max_length, temperature=temperature
        )
        metadata = _build_analysis_metadata(multispectral, pil_image)
        # Return the analysis results as JSON
        return JSONResponse(
            {
                "success": True,
                "question": question,
                "analysis": analysis,
                "metadata": metadata,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/batch_analyze")
async def batch_analyze(
    images: List[UploadFile] = File(...),
    questions: List[str] = Form(...),
    max_length: int = Form(default=128),
    temperature: float = Form(default=0.2),
) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    if len(images) != len(questions):
        raise HTTPException(status_code=400, detail="Number of images must match number of questions")

    results = []
    for upload, question in zip(images, questions):
        image_data = await upload.read()
        pil_image = Image.open(io.BytesIO(image_data))
        multispectral = _rgb_to_multispectral(pil_image)
        analysis = _generate_response(
            model, multispectral, question, max_new_tokens=max_length, temperature=temperature
        )
        metadata = _build_analysis_metadata(multispectral, pil_image)
        results.append({"image": upload.filename, "question": question, "analysis": analysis, "metadata": metadata})

    return JSONResponse({"success": True, "results": results})

@app.post("/analyze_dual")
async def analyze_dual_images(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    question: str = Form(default="Compare these two EO images and summarize the changes."),
    max_length: int = Form(default=128),
    temperature: float = Form(default=0.2),
) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        image_data_a = await image_a.read()
        image_data_b = await image_b.read()

        pil_a = Image.open(io.BytesIO(image_data_a))
        pil_b = Image.open(io.BytesIO(image_data_b))
        multi_a = _rgb_to_multispectral(pil_a)
        multi_b = _rgb_to_multispectral(pil_b)

        analysis_a = _generate_response(
            model, multi_a, f"[Image A] {question}", max_new_tokens=max_length, temperature=temperature
        )
        analysis_b = _generate_response(
            model, multi_b, f"[Image B] {question}", max_new_tokens=max_length, temperature=temperature
        )

        meta_a = _build_analysis_metadata(multi_a, pil_a)
        meta_b = _build_analysis_metadata(multi_b, pil_b)
        change_report = _build_change_report(meta_a, meta_b)

        # Compile and return dual-image comparison results
        return JSONResponse(
            {
                "success": True,
                "question": question,
                "image_a": {"name": image_a.filename, "analysis": analysis_a, "metadata": meta_a},
                "image_b": {"name": image_b.filename, "analysis": analysis_b, "metadata": meta_b},
                "change_report": change_report,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "model_loaded": model is not None, "device": str(device)}

@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    session_id = request.session_id or str(uuid.uuid4())
    history = chat_sessions.get(session_id, [])
    prompt = _build_chat_prompt(history, request.message)

    try:
        answer = _generate_text_response(
            model, prompt, max_new_tokens=request.max_tokens, temperature=request.temperature
        )
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": answer})
        chat_sessions[session_id] = history[-(MAX_CHAT_TURNS * 2) :]
        # Respond with the generated chat message and session info
        return JSONResponse(
            {
                "success": True,
                "session_id": session_id,
                "answer": answer,
                "history_length": len(chat_sessions[session_id]),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/chat/reset")
async def chat_reset(request: ChatResetRequest) -> JSONResponse:
    chat_sessions.pop(request.session_id, None)
    return JSONResponse({"success": True, "session_id": request.session_id, "reset": True})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

