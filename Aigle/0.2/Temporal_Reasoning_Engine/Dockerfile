FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安裝系統依賴 (為了 OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 暴露兩個 Port: 8000 (API), 8501 (Demo UI)
EXPOSE 8000 8501

# 啟動腳本：同時跑 API 和 Streamlit
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run demo_ui.py --server.port 8501 --server.address 0.0.0.0"]