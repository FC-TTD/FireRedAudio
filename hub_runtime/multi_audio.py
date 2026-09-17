"""Independent audio inputs for the existing FireRed understanding engine."""
from pathlib import Path
import tempfile
import time
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from ttd_model_runtime.protocol import HubError


def multi_audio_router(infer, *, max_bytes=32*1024*1024, infer_single=None):
    router=APIRouter()

    @router.post('/understand/single')
    def single(audio: UploadFile=File(...), prompt: str=Form(...), max_new_tokens: int=Form(4096)):
        if infer_single is None:
            raise HTTPException(503, '单音频理解尚未配置')
        if not 512 <= max_new_tokens <= 8192 or not prompt.strip() or len(prompt)>12000:
            raise HTTPException(422, '问题或输出预算不合法')
        with tempfile.TemporaryDirectory(prefix='firered-single-') as directory:
            path=Path(directory)/'audio.wav'
            payload=audio.file.read(max_bytes+1)
            if not payload or len(payload)>max_bytes:
                raise HTTPException(413, '音频为空或超过大小限制')
            path.write_bytes(payload)
            started=time.monotonic()
            try:
                answer=infer_single(str(path),prompt.strip(),max_new_tokens)
            except HubError:
                raise
            except Exception as exc:
                raise HTTPException(502, '声音表演分析失败，可重试') from exc
        result=answer if isinstance(answer,dict) else {'answer':answer,'finish_reason':None}
        return {**result,'model':'FireRedAudio','max_new_tokens':max_new_tokens,
                'enable_thinking':False,'elapsed_seconds':round(time.monotonic()-started,3),
                'completeness':'consumer_protocol_validation_required'}

    @router.get('/understand/multi/capabilities')
    def capabilities():
        return {'model':'FireRedAudio','input_mode':'independent_audio_paths','max_files':12,
                'max_total_bytes':max_bytes,'shared_model':True,'speaker_quality_accepted':False,
                'single_audio_budget':{'min':512,'max':8192} if infer_single else None}

    @router.post('/understand/multi')
    def understand(audio: list[UploadFile]=File(...), prompt: str=Form(...), enable_thinking: bool=Form(False)):
        if not 2<=len(audio)<=12:
            raise HTTPException(422,'请提供 2～12 段独立音频')
        if not prompt.strip() or len(prompt)>12000:
            raise HTTPException(422,'问题不能为空且不能超过 12000 字符')
        started=time.monotonic()
        with tempfile.TemporaryDirectory(prefix='firered-multi-') as directory:
            paths=[];total=0
            for index,item in enumerate(audio):
                path=Path(directory)/f'audio-{index}.wav'
                with path.open('wb') as target:
                    while chunk:=item.file.read(1024*1024):
                        total+=len(chunk)
                        if total>max_bytes: raise HTTPException(413,'音频总大小超过限制')
                        target.write(chunk)
                if not path.stat().st_size: raise HTTPException(422,'音频不能为空')
                paths.append(str(path))
            try:
                answer=infer(paths,prompt.strip(),enable_thinking)
            except HubError:
                raise
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception('FireRed multi-audio inference failed')
                raise HTTPException(502,'FireRed 多音频比对失败，请核对音频或稍后重试') from exc
        return {'model':'FireRedAudio','input_mode':'independent_audio_paths','audio_count':len(paths),
                'answer':answer,'elapsed_seconds':round(time.monotonic()-started,3)}
    return router
