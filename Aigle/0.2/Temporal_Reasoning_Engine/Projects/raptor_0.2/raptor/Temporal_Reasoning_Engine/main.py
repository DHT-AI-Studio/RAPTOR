import os
import re
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from scripts.model_engine import QwenVisionEngine

app = FastAPI(title="RAPTOR Vision API", version="1.0.0")
# engine = None


# import FastAPI router
from app.routers import vision_analyze_Qwen
app.include_router(vision_analyze_Qwen.router)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)