"""Native inference owns the original single-engine lock and all CUDA tensors."""
import gc
import os
import threading
import uuid
from pathlib import Path

class NativeBackend:
    def __init__(self):
        self.engine = None
        self.lock = threading.RLock()
        self.sequence = 0

    def _load(self):
        if self.engine is None:
            from inference import FireRedAudioInference
            from fireredaudio.redae.decoder import PretrainedRedAEAudioDecoderV1, PretrainedRedAEDecoderConfig
            engine = FireRedAudioInference(model_path=os.getenv("MODEL_PATH", "/models/FireRedAudio"),
                vae_decoder_path=None, device=os.getenv("MODEL_DEVICE", "cuda:0"))
            engine.vae_decoder = PretrainedRedAEAudioDecoderV1.from_pretrained(
                config=PretrainedRedAEDecoderConfig(), ckpt_path=os.getenv("VAE_DECODER_PATH", "/models/RedAE_decoder/model.pt")
            ).to(os.getenv("DECODER_DEVICE", "cpu"))
            self.engine = engine
        return self.engine

    def status(self):
        import torch
        available = torch.cuda.is_available()
        cuda = {"available": available}
        if available:
            cuda.update(device=torch.cuda.get_device_name(0), allocated_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved())
        return {"base_model_loaded": self.engine is not None, "cuda": cuda, "sequence": self.sequence}

    def close(self):
        import torch
        with self.lock:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self.engine = None
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def invoke(self, operation, args, kwargs):
        """One shared native lock, including explicit load/unload; no replay."""
        import torch
        with self.lock:
            self.sequence += 1
            try:
                if operation == "unload":
                    value = self.engine is not None
                    self.close()
                elif operation == "load":
                    self._load()
                    value = None
                else:
                    engine = self._load()
                    with torch.inference_mode():
                        if operation == "embed":
                            value = self._embed(engine, *args, **kwargs)
                        elif operation == "understand_single":
                            value = self._single(engine, *args, **kwargs)
                        elif operation == "understand":
                            result = engine.understand(*args, **kwargs)
                            value = {"answer": result.answer, "reasoning": result.reasoning}
                        elif operation in ("tts", "voice_design", "edit"):
                            result = getattr(engine, operation)(*args, **kwargs)
                            import torchaudio
                            directory = Path(os.getenv("OUTPUT_DIR", "/outputs"))
                            directory.mkdir(parents=True, exist_ok=True)
                            prefix = {"tts": "clone", "voice_design": "design", "edit": "edit"}[operation]
                            path = directory / f"{prefix}-{uuid.uuid4().hex[:10]}.wav"
                            torchaudio.save(str(path), result.audio.detach().cpu().float(), 24000)
                            value = {"audio": str(path), "text": result.text}
                            del result
                        else:
                            raise ValueError(f"Unsupported native operation: {operation}")
                    # Avoid retaining CUDA-bearing result references past activity completion.
                    if operation == "understand":
                        del result
                    del engine
                envelope = {"ok": True, "value": value}
            except Exception as exc:
                envelope = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            finally:
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            return {**envelope, "status": self.status()}

    @staticmethod
    def _embed(engine, path, normalize=True):
        from fireredaudio.utils.audio import read_audio, UNDERSTAND_SAMPLE_RATE
        from torch.nn import functional as F
        waveform = read_audio(str(path), UNDERSTAND_SAMPLE_RATE).numpy()
        processed = engine.processor([waveform])
        features = engine.model.get_audio_features(input_features=processed["input_features"].to(engine.device),
            feature_attention_mask=processed["feature_attention_mask"].to(engine.device)).float()
        if features.ndim != 2 or features.shape[0] == 0:
            raise RuntimeError(f"unexpected FireRed audio feature shape: {tuple(features.shape)}")
        pooled = features.mean(dim=0)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=0)
        return [pooled.cpu().tolist(), int(features.shape[0])]

    @staticmethod
    def _single(engine, path, prompt, max_new_tokens):
        from inference import build_understand_prompt, FEAT_TYPE_UNDERSTAND, split_thinking
        from fireredaudio.utils.audio import read_audio, UNDERSTAND_SAMPLE_RATE
        prompt_text = build_understand_prompt(prompt, 1, engine.model.config.audio_special_token, enable_thinking=False)
        batch = engine.encoder.encode(prompt_text, [{'feat_type': FEAT_TYPE_UNDERSTAND,
            'audio_understand': read_audio(path, UNDERSTAND_SAMPLE_RATE).numpy(), 'audio_generation': None, 'role': 'user'}])
        out = engine.model.generate(input_ids=batch['input_ids'].to(engine.device),
            attention_mask=batch['attention_mask'].to(engine.device), audio_features=batch['audio_features'].to(engine.device),
            audio_feature_attention_mask=batch['audio_feature_attention_mask'].to(engine.device),
            generation_config=engine._gen_config('understand', max_new_tokens))
        ids = out[0].tolist()
        _, answer = split_thinking(engine.tokenizer.decode(out[0], skip_special_tokens=True))
        return {'answer': answer, 'generated_tokens': len(ids),
            'finish_reason': 'stop' if ids and ids[-1] == engine._eos_id else 'length' if len(ids) >= max_new_tokens else 'unknown'}
