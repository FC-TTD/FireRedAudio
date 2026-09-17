import gc
import os
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from fastapi import FastAPI

from . import bridge
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cuda:0")
DECODER_DEVICE = os.environ.get("DECODER_DEVICE", "cpu")


def _require_audio(audio_path):
    if not audio_path:
        raise gr.Error("请选择音频文件")
    return audio_path


def _run(callback):
    return callback(bridge.EngineFacade())


def _save_audio(audio, prefix):
    # The child saves the original 24kHz WAV before its execution scope closes.
    return audio


def transcribe(audio_path):
    audio_path = _require_audio(audio_path)
    return _run(
        lambda engine: engine.understand(
            audio_path,
            "Transcribe speech to text.",
            task="asr",
            max_new_tokens=2048,
        ).answer
    )


def long_transcribe(audio_path, output_style):
    audio_path = _require_audio(audio_path)
    prompts = {
        "带时间戳逐字稿": (
            "Transcribe this recording in full. Organize it chronologically with "
            "second-level start and end timestamps for every segment. Preserve the "
            "spoken language and return only a clean timestamped transcript."
        ),
        "WebVTT 字幕": (
            "Transcribe this recording and return valid WebVTT subtitle text only. "
            "Use second-level timestamps and split at natural sentence boundaries."
        ),
        "结构化摘要": (
            "Organize this recording into a chronological outline. For each item, "
            "give its start and end time, topic, speakers when distinguishable, and "
            "a concise grounded summary."
        ),
    }
    return _run(
        lambda engine: engine.understand(
            audio_path,
            prompts[output_style],
            task="understand",
            max_new_tokens=8192,
        ).answer
    )


def locate_content(audio_path, query):
    audio_path = _require_audio(audio_path)
    if not query or not query.strip():
        raise gr.Error("请输入要定位的内容或时间")
    prompt = (
        "Find the requested evidence in the recording. Return the precise start and "
        "end timestamps, the matching transcript, and one short explanation. If the "
        f"request is a time range, describe what occurs there. Request: {query.strip()}"
    )
    return _run(
        lambda engine: engine.understand(
            audio_path, prompt, task="understand", max_new_tokens=4096
        ).answer
    )


def understand_audio(audio_path, question, enable_thinking):
    audio_path = _require_audio(audio_path)
    if not question or not question.strip():
        raise gr.Error("请输入音频问题")
    result = _run(
        lambda engine: engine.understand(
            audio_path,
            question.strip(),
            task="understand",
            enable_thinking=enable_thinking,
            max_new_tokens=4096 if enable_thinking else 1024,
        )
    )
    if result.reasoning:
        return f"思考过程\n\n{result.reasoning}\n\n回答\n\n{result.answer}"
    return result.answer


def clone_voice(prompt_audio, prompt_text, target_text, language):
    prompt_audio = _require_audio(prompt_audio)
    if not prompt_text or not target_text:
        raise gr.Error("参考音频逐字稿和合成文本不能为空")
    result = _run(
        lambda engine: engine.tts(
            prompt_text.strip(),
            prompt_audio,
            target_text.strip(),
            language="zh" if language == "中文" else "en",
            max_new_audio_steps=750,
        )
    )
    return _save_audio(result.audio, "clone")


def design_voice(instruction, text):
    if not instruction or not text:
        raise gr.Error("声音描述和合成文本不能为空")
    result = _run(
        lambda engine: engine.voice_design(
            instruction.strip(), text.strip(), max_new_audio_steps=750
        )
    )
    return _save_audio(result.audio, "design"), result.text or ""


def edit_speech(audio_path, instruction, edit_type):
    audio_path = _require_audio(audio_path)
    if not instruction:
        raise gr.Error("请输入编辑指令")
    result = _run(
        lambda engine: engine.edit(
            audio_path,
            instruction.strip(),
            edit_type="semantic" if edit_type == "内容编辑" else "acoustic",
            max_new_audio_steps=750,
        )
    )
    return _save_audio(result.audio, "edit"), result.text or ""


def model_status():
    current = bridge.snapshot()
    if not current.get("base_model_loaded"):
        return "未加载"
    used = current.get("cuda", {}).get("allocated_bytes", 0) / 1024**3
    return f"已加载 · GPU {used:.1f} GiB · RedAE {DECODER_DEVICE}"


with gr.Blocks(
    title="FireRedAudio 全能力测试",
    theme=gr.themes.Soft(primary_hue="red", neutral_hue="gray", radius_size="sm"),
) as demo:
    with gr.Row():
        gr.Markdown("# FireRedAudio 全能力测试")
        status = gr.Textbox(value="未加载", label="模型状态", interactive=False)
        refresh = gr.Button("刷新状态", size="sm")
        refresh.click(model_status, outputs=status)

    with gr.Tabs():
        with gr.Tab("语音识别"):
            asr_audio = gr.Audio(type="filepath", label="音频")
            asr_button = gr.Button("开始识别", variant="primary")
            asr_text = gr.Textbox(label="逐字稿", lines=12)
            asr_button.click(transcribe, asr_audio, asr_text)

        with gr.Tab("长音频转写"):
            long_audio = gr.Audio(type="filepath", label="长音频")
            long_style = gr.Radio(
                ["带时间戳逐字稿", "WebVTT 字幕", "结构化摘要"],
                value="带时间戳逐字稿",
                label="输出形式",
            )
            long_button = gr.Button("开始处理", variant="primary")
            long_text = gr.Textbox(label="结果", lines=18)
            long_button.click(long_transcribe, [long_audio, long_style], long_text)

        with gr.Tab("内容定位"):
            locate_audio = gr.Audio(type="filepath", label="音频")
            locate_query = gr.Textbox(label="内容或时间", lines=3)
            locate_button = gr.Button("查找片段", variant="primary")
            locate_result = gr.Textbox(label="定位结果", lines=12)
            locate_button.click(locate_content, [locate_audio, locate_query], locate_result)

        with gr.Tab("音频理解"):
            qa_audio = gr.Audio(type="filepath", label="音频")
            qa_question = gr.Textbox(label="问题", lines=3)
            qa_thinking = gr.Checkbox(label="启用推理", value=False)
            qa_button = gr.Button("回答问题", variant="primary")
            qa_result = gr.Textbox(label="回答", lines=14)
            qa_button.click(
                understand_audio, [qa_audio, qa_question, qa_thinking], qa_result
            )

        with gr.Tab("声音克隆"):
            clone_audio = gr.Audio(type="filepath", label="授权参考录音")
            clone_prompt = gr.Textbox(label="参考音频逐字稿", lines=3)
            clone_target = gr.Textbox(label="合成文本", lines=5)
            clone_language = gr.Radio(["中文", "英文"], value="中文", label="语言")
            clone_button = gr.Button("生成声音", variant="primary")
            clone_output = gr.Audio(label="生成结果")
            clone_button.click(
                clone_voice,
                [clone_audio, clone_prompt, clone_target, clone_language],
                clone_output,
            )

        with gr.Tab("声音设计"):
            design_instruction = gr.Textbox(label="声音描述", lines=4)
            design_text = gr.Textbox(label="合成文本", lines=5)
            design_button = gr.Button("设计并生成", variant="primary")
            design_audio = gr.Audio(label="生成结果")
            design_plan = gr.Textbox(label="声音属性", lines=5)
            design_button.click(
                design_voice,
                [design_instruction, design_text],
                [design_audio, design_plan],
            )

        with gr.Tab("语音编辑"):
            edit_audio = gr.Audio(type="filepath", label="原始音频")
            edit_instruction = gr.Textbox(label="编辑指令", lines=3)
            edit_type = gr.Radio(
                ["内容编辑", "声音属性编辑"], value="内容编辑", label="编辑类型"
            )
            edit_button = gr.Button("执行编辑", variant="primary")
            edited_audio = gr.Audio(label="编辑结果")
            edited_text = gr.Textbox(label="编辑后文本", lines=4)
            edit_button.click(
                edit_speech,
                [edit_audio, edit_instruction, edit_type],
                [edited_audio, edited_text],
            )

demo.queue(default_concurrency_limit=1, max_size=8)

app = FastAPI(title="FireRedAudio Preview")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": bridge.snapshot().get("base_model_loaded", False),
        "runtime_ready": bridge.ready(),
        "model_device": MODEL_DEVICE,
        "decoder_device": DECODER_DEVICE,
    }


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
