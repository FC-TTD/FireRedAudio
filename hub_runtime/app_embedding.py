from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from torch.nn import functional as F

from . import app_preview as base
from . import bridge
from fireredaudio.utils.audio import UNDERSTAND_SAMPLE_RATE, read_audio


MODEL_ALIAS = os.getenv("MODEL_ALIAS", "firered-audio-encoder-meanpool-v1")
NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes", "on"}
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(32 * 1024 * 1024)))

# New routes are registered before the root mount so Gradio cannot shadow them.
# The mounted base app retains all existing FireRed UI and API capabilities.
app = FastAPI(title="TTD FireRed Dialogue Embedding Preview", version="0.1.0")

from .multi_audio import multi_audio_router
from ttd_model_runtime.protocol import HubError


def _understand_multiple(paths, prompt, enable_thinking):
    # Reuse the same engine and lock as all existing Gradio inference routes.
    result = base._run(lambda engine: engine.understand(
        paths, prompt, task="understand", enable_thinking=enable_thinking,
        max_new_tokens=4096 if enable_thinking else 2048,
    ))
    return result.answer


def _understand_single(path, prompt, max_new_tokens):
    return bridge.call("understand_single", path, prompt, max_new_tokens)


app.include_router(multi_audio_router(_understand_multiple, max_bytes=MAX_AUDIO_BYTES, infer_single=_understand_single))



def _embedding_status() -> dict[str, Any]:
    current = bridge.snapshot()
    cuda = current.get("cuda", {"available": False})
    return {
        "ok": True,
        "model": MODEL_ALIAS,
        "base_model_loaded": current.get("base_model_loaded", False),
        "dimension": 4096,
        "pooling": "mean_over_valid_audio_frames",
        "normalized": NORMALIZE,
        "sample_rate_hz": UNDERSTAND_SAMPLE_RATE,
        "cuda": cuda,
    }


async def _embed_path(audio_path: Path) -> tuple[list[float], int]:
    return await bridge.acall("embed", str(audio_path), normalize=NORMALIZE)


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
    bridge.call("load")
    return _embedding_status()


@app.post("/embedding/unload")
def embedding_unload() -> dict[str, Any]:
    unloaded = bridge.unload()
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
        vector, frame_count = await _embed_path(temporary_path)
    except (HTTPException, HubError):
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


# Mounted ASGI applications do not receive lifespan automatically. Start the
# original Gradio queue with the outer app, before accepting queued callbacks.
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application):
    async with base.app.router.lifespan_context(base.app):
        yield

app.router.lifespan_context = lifespan
app.mount("/", base.app)
