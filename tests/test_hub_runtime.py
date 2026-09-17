"""Contract tests using real ASGI, original UI callbacks and SDK task settling."""
import asyncio
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
import io
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import httpx
from ttd_model_runtime.runtime import Runtime
from hub_runtime import bridge
from hub_runtime.native import NativeBackend

_owned = ContextVar("test_owned", default=False)

class FakeEngine:
    def __init__(self):
        self.calls=[]; self.loaded=False; self.sequence=0
        self.started=threading.Event(); self.release=None
    def invoke(self, op, args, kwargs):
        assert _owned.get(), "native call outside runtime activity"
        self.calls.append((op,args,kwargs)); self.started.set()
        if self.release is not None:
            assert self.release.wait(5)
        self.loaded=op != "unload"; self.sequence+=1
        value={"answer":"answer","reasoning":None}
        if op=="embed": value=[[0.0]*4096,12]
        elif op=="understand_single": value={"answer":"answer","generated_tokens":3,"finish_reason":"stop"}
        elif op in ("tts","voice_design","edit"): value={"audio":"/outputs/result.wav","text":"text"}
        elif op=="unload": value=True
        return {"ok":True,"value":value,"status":{"base_model_loaded":self.loaded,
            "cuda":{"available":True,"allocated_bytes":1234,"reserved_bytes":1500},"sequence":self.sequence}}

class FakeRuntime:
    task=Runtime.task
    def __init__(self):
        self.engine=FakeEngine();self.active=0;self.finished=0;self.residency="unloaded";self.engine_id="one"
    def _nested(self): return False
    def get(self):
        assert _owned.get()
        return self.engine
    @contextmanager
    def execution(self):
        self.active+=1;self.residency="ready";token=_owned.set(True)
        try: yield self
        finally: _owned.reset(token);self.active-=1;self.finished+=1
    @asynccontextmanager
    async def aexecution(self):
        with self.execution(): yield self
    def status(self):
        return {"accepting":True,"healthy":True,"active":self.active,"residency":self.residency,"engine":{"id":self.engine_id}}

class CpuNativeFixture(NativeBackend):
    def _load(self):
        from types import SimpleNamespace
        if self.engine is None:
            self.engine = SimpleNamespace(understand=lambda *a, **k: SimpleNamespace(answer="transport", reasoning=None))
        return self.engine

def process_backend(): return CpuNativeFixture()
def process_release(backend): backend.close()
def process_cleanup(): pass

class Transport(unittest.TestCase):
    def test_real_process_model_native_envelope_and_release(self):
        from ttd_model_runtime.engine import ProcessModel
        process=ProcessModel(process_backend,None,process_cleanup,process_release,
            {"gpu":"GPU-00000000-0000-0000-0000-000000000000","generation":1},timeout=30)
        try:
            proxy=process.start()
            result=proxy.invoke("understand",("a.wav","question"),{})
            self.assertEqual(result["value"],{"answer":"transport","reasoning":None})
            self.assertTrue(result["status"]["base_model_loaded"])
            unloaded=proxy.invoke("unload",(),{})
            self.assertFalse(unloaded["status"]["base_model_loaded"])
        finally: process.close()
        self.assertTrue(process.engine_status()["group_empty"])

class Contracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime=FakeRuntime();bridge.configure(self.runtime)
        from hub_runtime.app_embedding import app
        self.app=app
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://firered")
    async def asyncTearDown(self): await self.client.aclose()
    async def test_cold_health_and_eviction_do_not_load(self):
        for url in ("/health","/embedding/health","/embedding/models","/config"):
            self.assertEqual((await self.client.get(url)).status_code,200)
        self.assertFalse((await self.client.get("/health")).json()["model_loaded"])
        self.assertEqual(self.runtime.engine.calls,[])
        await self.client.post("/embedding/load")
        self.assertTrue((await self.client.get("/health")).json()["model_loaded"])
        self.runtime.residency="unloaded"
        self.assertFalse((await self.client.get("/health")).json()["model_loaded"])
        self.assertFalse((await self.client.post("/embedding/unload")).json()["unloaded"])
        self.assertEqual(len(self.runtime.engine.calls),1)
    async def test_runtime_ready_respects_local_failure_and_acceptance(self):
        self.assertTrue((await self.client.get("/health")).json()["runtime_ready"])
        for changes in ({"accepting":False},{"healthy":False},{"residency":"unknown"}):
            state={**self.runtime.status(),**changes}
            with patch.object(self.runtime,"status",return_value=state):
                self.assertFalse((await self.client.get("/health")).json()["runtime_ready"])
        self.assertEqual(self.runtime.engine.calls,[])
    async def test_async_bridge_obeys_real_sdk_owner_check(self):
        from types import SimpleNamespace
        class OwnedRuntime(FakeRuntime):
            _nested=Runtime._nested
            get=Runtime.get
            aexecution=Runtime.aexecution
            def __init__(self):
                super().__init__()
                self._guard=threading.RLock();self._started=True;self._active={}
                self.service='firered-audio';self.version='test'
                self.health=SimpleNamespace(record_error=lambda *args:None)
                self._model=SimpleNamespace(invoke=lambda *args:{"ok":True,"value":"owned",
                    "status":{"base_model_loaded":True,"cuda":{"available":False},"sequence":1}})
            def _begin(self,grant):
                self._active['real-scope']={};self.residency='ready'
                return {'id':'real-scope'}
            def _finish(self,activity,state):
                self._active.pop(activity['id']);self.finished+=1
        runtime=OwnedRuntime();bridge.configure(runtime)
        self.assertEqual(await bridge.acall('load'),'owned')
        self.assertEqual(runtime.finished,1)
    async def test_embedding_rest_and_validation(self):
        response=await self.client.post("/embed/audio",files={"audio":("x.wav",b"sample")})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()["dimension"],4096)
        op,args,kwargs=self.runtime.engine.calls[-1]
        self.assertEqual((op,kwargs),("embed",{"normalize":True}));self.assertFalse(Path(args[0]).exists())
        bad=await self.client.post("/embed/audio",files={"audio":("x.wav",b"sample")},data={"model":"bad"})
        self.assertEqual(bad.status_code,400);self.assertEqual(len(self.runtime.engine.calls),1)
    async def test_embedding_cancellation_settles_before_tempfile_cleanup(self):
        from fastapi import UploadFile
        from hub_runtime.app_embedding import embed_audio, MODEL_ALIAS
        self.runtime.engine.release=threading.Event()
        task=asyncio.create_task(embed_audio(UploadFile(io.BytesIO(b"audio"),filename="x.wav"),MODEL_ALIAS))
        await asyncio.to_thread(self.runtime.engine.started.wait,2)
        path=Path(self.runtime.engine.calls[-1][1][0]);task.cancel()
        await asyncio.sleep(.05)
        self.assertTrue(path.exists());self.assertEqual(self.runtime.active,1);self.assertFalse(task.done())
        self.runtime.engine.release.set()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(path.exists());self.assertEqual(self.runtime.active,0)
    async def test_understand_contracts_and_default_budgets(self):
        response=await self.client.post("/understand/single",files={"audio":("x.wav",b"a")},data={"prompt":"hi"})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()["finish_reason"],"stop")
        self.assertEqual(self.runtime.engine.calls[-1][1][2],4096)
        response=await self.client.post("/understand/multi",files=[("audio",("a.wav",b"a")),("audio",("b.wav",b"b"))],data={"prompt":"compare"})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()["audio_count"],2)
        self.assertEqual(self.runtime.engine.calls[-1][2]["max_new_tokens"],2048)
    async def test_original_ui_endpoints_and_callbacks(self):
        from hub_runtime import app_preview as ui
        config=(await self.client.get("/config")).json()
        self.assertEqual({x["api_name"] for x in config["dependencies"]},
            {"model_status","transcribe","long_transcribe","locate_content","understand_audio","clone_voice","design_voice","edit_speech"})
        self.assertEqual(ui.demo._queue.max_size,8)
        self.assertEqual(ui.demo._queue.default_concurrency_limit,1)
        self.assertEqual(ui.transcribe("a.wav"),"answer")
        self.assertEqual(self.runtime.engine.calls[-1][2],{"task":"asr","max_new_tokens":2048})
        ui.long_transcribe("a.wav","WebVTT 字幕")
        self.assertEqual(self.runtime.engine.calls[-1][2]["max_new_tokens"],8192)
        ui.locate_content("a.wav","words");ui.understand_audio("a.wav","why",True)
        self.assertEqual(self.runtime.engine.calls[-1][2]["max_new_tokens"],4096)
        self.assertEqual(ui.clone_voice("a.wav","ref","target","英文"),"/outputs/result.wav")
        self.assertEqual(self.runtime.engine.calls[-1][2]["language"],"en")
        self.assertEqual(ui.design_voice("voice","text"),("/outputs/result.wav","text"))
        ui.edit_speech("a.wav","change","声音属性编辑")
        self.assertEqual(self.runtime.engine.calls[-1][2]["edit_type"],"acoustic")
        self.assertEqual(self.runtime.finished,7)
    async def test_unload_weights_keeps_engine_lease(self):
        await self.client.post("/embedding/load")
        result=(await self.client.post("/embedding/unload")).json()
        self.assertTrue(result["unloaded"]);self.assertFalse(result["base_model_loaded"])
        self.assertEqual(self.runtime.residency,"ready")
    async def test_generation_fence_and_out_of_order_observations(self):
        bridge.call("load")
        bridge._receive({"ok":True,"status":{"sequence":0,"base_model_loaded":False}},"one")
        self.assertTrue(bridge.snapshot()["base_model_loaded"])
        self.runtime.engine_id="two"
        self.assertFalse(bridge.snapshot()["base_model_loaded"])
    async def test_gradio_queued_callback_owns_real_work(self):
        import wave
        payload=io.BytesIO()
        with wave.open(payload,"wb") as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b"\x00\x00"*1600)
        async with self.app.router.lifespan_context(self.app):
            uploaded=await self.client.post("/gradio_api/upload",files={"files":("sample.wav",payload.getvalue(),"audio/wav")})
            self.assertEqual(uploaded.status_code,200)
            response=await self.client.post("/gradio_api/call/transcribe",json={"data":[{"path":uploaded.json()[0],"meta":{"_type":"gradio.FileData"}}]})
            self.assertEqual(response.status_code,200)
            result=await asyncio.wait_for(self.client.get("/gradio_api/call/transcribe/"+response.json()["event_id"]),5)
            self.assertIn('event: complete',result.text);self.assertIn('answer',result.text)
            self.assertEqual(self.runtime.engine.calls[-1][0],"understand")
            self.assertEqual(self.runtime.active,0);self.assertEqual(self.runtime.finished,1)
    async def test_native_unload_waits_for_inference_lock(self):
        import torch
        native=NativeBackend();native.engine=object()
        with patch.object(torch.cuda,"is_initialized",return_value=False), patch.object(torch.cuda,"is_available",return_value=False):
            native.lock.acquire()
            task=asyncio.create_task(asyncio.to_thread(native.invoke,"unload",(),{}))
            await asyncio.sleep(.05);self.assertFalse(task.done());self.assertIsNotNone(native.engine)
            native.lock.release();result=await task
            self.assertTrue(result["value"]);self.assertFalse(result["status"]["base_model_loaded"])

if __name__=="__main__": unittest.main()
