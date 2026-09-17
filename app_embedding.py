from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from torch.nn import functional as F

import app_preview as base
from fireredaudio.utils.audio import UNDERSTAND_SAMPLE_RATE, read_audio


MODEL_ALIAS = os.getenv("MODEL_ALIAS", "firered-audio-encoder-meanpool-v1")
NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes", "on"}
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(32 * 1024 * 1024)))

# New routes are registered before the root mount so Gradio cannot shadow them.
# The mounted base app retains all existing FireRed UI and API capabilities.
app = FastAPI(title="TTD FireRed Dialogue Embedding Preview", version="0.1.0")

from multi_audio import multi_audio_router


def _understand_multiple(paths, prompt, enable_thinking):
    # Reuse the same engine and lock as all existing Gradio inference routes.
    result = base._run(lambda engine: engine.understand(
        paths, prompt, task="understand", enable_thinking=enable_thinking,
        max_new_tokens=4096 if enable_thinking else 2048,
    ))
    return result.answer


def _understand_single(path, prompt, max_new_tokens):
    # Same upstream encoding/generation contract as inference.py:understand,
    # retaining token termination metadata that UnderstandOutput discards.
    from inference import build_understand_prompt, FEAT_TYPE_UNDERSTAND, split_thinking

    @torch.inference_mode()
    def run(engine):
        prompt_text = build_understand_prompt(prompt, 1, engine.model.config.audio_special_token, enable_thinking=False)
        batch = engine.encoder.encode(prompt_text, [{'feat_type': FEAT_TYPE_UNDERSTAND,
            'audio_understand': read_audio(path, UNDERSTAND_SAMPLE_RATE).numpy(), 'audio_generation': None, 'role': 'user'}])
        out = engine.model.generate(
            input_ids=batch['input_ids'].to(engine.device), attention_mask=batch['attention_mask'].to(engine.device),
            audio_features=batch['audio_features'].to(engine.device),
            audio_feature_attention_mask=batch['audio_feature_attention_mask'].to(engine.device),
            generation_config=engine._gen_config('understand', max_new_tokens))
        ids = out[0].tolist()
        _, answer = split_thinking(engine.tokenizer.decode(out[0], skip_special_tokens=True))
        return {'answer': answer, 'generated_tokens': len(ids),
                'finish_reason': 'stop' if ids and ids[-1] == engine._eos_id else 'length' if len(ids) >= max_new_tokens else 'unknown'}
    return base._run(run)


app.include_router(multi_audio_router(_understand_multiple, max_bytes=MAX_AUDIO_BYTES, infer_single=_understand_single))



def _embedding_status() -> dict[str, Any]:
    cuda = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        cuda.update(
            {
                "device": torch.cuda.get_device_name(0),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        )
    return {
        "ok": True,
        "model": MODEL_ALIAS,
        "base_model_loaded": base._engine is not None,
        "dimension": 4096,
        "pooling": "mean_over_valid_audio_frames",
        "normalized": NORMALIZE,
        "sample_rate_hz": UNDERSTAND_SAMPLE_RATE,
        "cuda": cuda,
    }


@torch.inference_mode()
def _embed_path(audio_path: Path) -> tuple[list[float], int]:
    engine = base._load_engine()
    waveform = read_audio(str(audio_path), UNDERSTAND_SAMPLE_RATE).numpy()
    processed = engine.processor([waveform])
    with base._inference_lock:
        features = engine.model.get_audio_features(
            input_features=processed["input_features"].to(engine.device),
            feature_attention_mask=processed["feature_attention_mask"].to(engine.device),
        ).float()
        if features.ndim != 2 or features.shape[0] == 0:
            raise RuntimeError(f"unexpected FireRed audio feature shape: {tuple(features.shape)}")
        pooled = features.mean(dim=0)
        if NORMALIZE:
            pooled = F.normalize(pooled, p=2, dim=0)
        vector = pooled.cpu().tolist()
    return vector, int(features.shape[0])


def _validate_model(requested: str) -> None:
    if requested != MODEL_ALIAS:
        raise HTTPException(status_code=400, detail=f"unsupported model: {requested}; supported: {MODEL_ALIAS}")


@app.get("/embedding/health")
def embedding_health() -> dict[str, Any]:
    return _embedding_status()


@app.get("/embedding/models")
def embedding_models() -> dict[str, Any]:
    return {"models": [_embedding_status()]}


@app.post("/embedding/load")
def embedding_load() -> dict[str, Any]:
    base._load_engine()
    return _embedding_status()


@app.post("/embedding/unload")
def embedding_unload() -> dict[str, Any]:
    unloaded = False
    with base._engine_lock:
        if base._engine is not None:
            base._engine = None
            unloaded = True
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"unloaded": unloaded, **_embedding_status()}


@app.post("/embed/audio")
async def embed_audio(
    audio: UploadFile = File(...),
    model: str = Form(default=MODEL_ALIAS),
) -> dict[str, Any]:
    _validate_model(model)
    payload = await audio.read(MAX_AUDIO_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(payload) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"audio exceeds {MAX_AUDIO_BYTES} bytes")
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)
        vector, frame_count = _embed_path(temporary_path)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - deployment diagnostics are returned to the caller
        raise HTTPException(status_code=422, detail=f"FireRed audio embedding failed: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "model": MODEL_ALIAS,
        "dimension": len(vector),
        "pooling": "mean_over_valid_audio_frames",
        "normalized": NORMALIZE,
        "sample_rate_hz": UNDERSTAND_SAMPLE_RATE,
        "frame_count": frame_count,
        "embedding": vector,
    }


app.mount("/", base.app)
