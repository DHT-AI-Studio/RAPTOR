from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
import httpx
from pathlib import Path
import shutil
import uuid

router = APIRouter()

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# docker-compose service name
VISION_SERVICE_URL = "http://192.168.157.161:8000/analyze/video"


@router.post("/video")
async def analyze_video(
    file: UploadFile = File(...),
    start_time: int = Form(...),
    prompt: str = Form(...),
    request: Request = None
):
    try:
        # 暫存檔案
        suffix = Path(file.filename).suffix
        temp_name = f"{uuid.uuid4()}{suffix}"
        save_path = UPLOAD_DIR / temp_name

        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 轉發給 Vision Service
        async with httpx.AsyncClient(timeout=300) as client:
            with open(save_path, "rb") as f:
                files = {
                    "file": (file.filename, f, "video/mp4")
                }
                data = {
                    "start_time": str(start_time),
                    "prompt": prompt
                }

                headers = {}
                # 如果你要把 JWT 也轉發過去
                auth = request.headers.get("authorization")
                if auth:
                    headers["Authorization"] = auth

                resp = await client.post(
                    VISION_SERVICE_URL,
                    files=files,
                    data=data,
                    headers=headers
                )

        return resp.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
