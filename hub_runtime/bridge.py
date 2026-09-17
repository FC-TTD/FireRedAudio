"""CPU facade; observations are fenced to the current SDK engine generation."""
import asyncio
import threading
from types import SimpleNamespace

_runtime = None
_guard = threading.Lock()
_observation = None
_engine_id = None

def configure(runtime):
    global _runtime, _observation, _engine_id
    _runtime, _observation, _engine_id = runtime, None, None

def ready():
    if _runtime is None:
        return False
    status = _runtime.status()
    return bool(status.get("accepting") and status.get("healthy") and status.get("residency") != "unknown")

def snapshot():
    status = _runtime.status()
    with _guard:
        observation = dict(_observation or {})
        same = status.get("residency") == "ready" and status.get("engine", {}).get("id") == _engine_id
    if not same:
        observation = {"base_model_loaded": False, "cuda": {"available": False}}
    observation.pop("sequence", None)
    return observation

def _receive(envelope, engine_id):
    global _observation, _engine_id
    with _guard:
        observed = envelope["status"]
        if _engine_id != engine_id or observed["sequence"] >= (_observation or {}).get("sequence", -1):
            _observation, _engine_id = observed, engine_id
    if not envelope["ok"]:
        raise RuntimeError(envelope["error"])
    return envelope.get("value")

def _invoke(operation, args, kwargs):
    engine = _runtime.get()
    engine_id = _runtime.status().get("engine", {}).get("id")
    return _receive(engine.invoke(operation, args, kwargs), engine_id)

def call(operation, *args, **kwargs):
    @_runtime.task
    def owned():
        return _invoke(operation, args, kwargs)
    return owned()

async def acall(operation, *args, **kwargs):
    @_runtime.task
    async def owned():
        # Resolve the model in the owning asyncio task. A worker thread has a
        # different SDK owner even though contextvars are copied into it.
        engine = _runtime.get()
        engine_id = _runtime.status().get("engine", {}).get("id")
        envelope = await asyncio.to_thread(engine.invoke, operation, args, kwargs)
        return _receive(envelope, engine_id)
    return await owned()

def unload():
    status = _runtime.status()
    if status.get("residency") != "ready" and not status.get("active", 0):
        return False
    # Weights only. The SDK engine/lease remains until node-authorized release.
    return call("unload")

class EngineFacade:
    def __getattr__(self, operation):
        if operation not in ("understand", "tts", "voice_design", "edit"):
            raise AttributeError(operation)
        def invoke(*args, **kwargs):
            return SimpleNamespace(**call(operation, *args, **kwargs))
        return invoke
