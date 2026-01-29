import os
import re
import shutil
from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from pydantic import BaseModel
from scripts.model_engine import QwenVisionEngine

router = APIRouter()
engine = None

@router.on_event("startup")
def load_model():
    global engine
    engine = QwenVisionEngine()

def format_time(seconds: float) -> str:
    """將秒數轉為 MM:SS 格式"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def process_time_alignment(text: str, start_time: float):
    """
    將模型輸出的 [+XXs] 替換為絕對時間 [MM:SS]
    """
    def replace_match(match):
        # 抓出模型說的相對秒數 (例如 "03" 或 "1.5")
        offset = float(match.group(1))
        # 計算絕對時間
        abs_time = start_time + offset
        # 轉成文字格式
        return f"[{format_time(abs_time)}]"

    # Regex 尋找 [+12s] 或 [+1.5s] 的模式
    # 這裡會把 `[+03s]` 變成 `[05:15]` (假設 start_time 是 312秒)
    updated_text = re.sub(r"\[\+(\d+\.?\d*)s\]", replace_match, text)
    return updated_text

@router.post("/analyze/video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    prompt: str = Form(None)
):
    """
    上傳影片並直接取得分析結果
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. 取得原始分析結果 (裡面會有 [+XXs])
        raw_result = engine.analyze(file_path, prompt_text=prompt, fps=2.0)
        
        # 2. 進行時間對齊處理 (變成 [MM:SS])
        final_result = process_time_alignment(raw_result, start_time)
        
        return {
            "status": "success", 
            "meta": {
                "filename": file.filename,
                "clip_start_time": format_time(start_time)
            },
            "analysis": final_result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

