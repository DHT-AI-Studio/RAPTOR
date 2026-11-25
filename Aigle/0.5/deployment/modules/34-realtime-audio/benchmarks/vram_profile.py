"""Profile simultaneous local models using the real Module 08 image.

Launch a fresh process per combination; NVML sampling includes non-Torch CUDA
allocators. Local Paddle detector/recognizer directories are mandatory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.model_benchmark import gpu_peak


def main() -> None:
    """Load, warm up and retain every requested model before taking the peak."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="comma-separated stt,kokoro,whisperx,paddle")
    parser.add_argument("--stt-dir", required=True)
    parser.add_argument("--kokoro-dir", required=True)
    parser.add_argument("--whisperx-vad", help="WhisperX VAD segmentation weights (default: the packaged asset)")
    parser.add_argument("--paddle-det", required=True, help="model directory named after the model")
    parser.add_argument("--paddle-rec", required=True, help="model directory named after the model")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="您好，請問今天能幫您處理什麼問題？")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    selected = args.models.split(",")
    if not set(selected) <= {"stt", "kokoro", "whisperx", "paddle"}:
        raise ValueError("unknown model combination")
    import torch
    torch.cuda.set_device(args.gpu)
    retained = []
    with gpu_peak(args.gpu) as memory:
        if "stt" in selected:
            from faster_whisper import WhisperModel
            stt = WhisperModel(args.stt_dir, device="cuda", device_index=args.gpu,
                               compute_type="float16", local_files_only=True)
            import soundfile as sf
            audio, _ = sf.read(args.audio, dtype="float32")
            segments, _ = stt.transcribe(audio if audio.ndim == 1 else audio[:, 0], language="zh")
            list(segments)
            retained.append(stt)
        if "kokoro" in selected:
            from kokoro import KModel, KPipeline
            directory = Path(args.kokoro_dir)
            model = KModel(config=str(directory / "config.json"),
                           model=str(next(directory.glob("kokoro-*.pth")))).cuda().eval()
            pipeline = KPipeline(lang_code="z", model=model)
            list(pipeline(args.text, voice=str(directory / "voices/zf_001.pt")))
            retained.extend([model, pipeline])
        if "whisperx" in selected:
            import whisperx
            from whisperx.vad import load_vad_model
            vad = load_vad_model("cuda", model_fp=args.whisperx_vad)
            model = whisperx.load_model(args.stt_dir, "cuda", language="zh", compute_type="float16",
                                        device_index=args.gpu, vad_model=vad, local_files_only=True)
            model.transcribe(whisperx.load_audio(args.audio), batch_size=1)
            retained.extend([model, vad])
        if "paddle" in selected:
            from paddleocr import PaddleOCR
            # PaddleOCR 3.x checks the name against the directory's config (Module 08: PP-OCRv5_mobile_*)
            ocr = PaddleOCR(text_detection_model_name=Path(args.paddle_det).name,
                            text_detection_model_dir=args.paddle_det,
                            text_recognition_model_name=Path(args.paddle_rec).name,
                            text_recognition_model_dir=args.paddle_rec,
                            use_doc_orientation_classify=False, use_doc_unwarping=False,
                            use_textline_orientation=False, device=f"gpu:{args.gpu}")
            list(ocr.predict(args.image))
            retained.append(ocr)
        torch.cuda.synchronize(args.gpu)
    report = {"models": selected, "gpu": torch.cuda.get_device_name(args.gpu),
              "compute_capability": list(torch.cuda.get_device_capability(args.gpu)),
              "retained_models": len(retained), "sampled_memory": memory,
              "vram_total_mib": torch.cuda.get_device_properties(args.gpu).total_memory / 1024**2}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
