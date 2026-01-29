# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# WORKDIR /app

# # 安裝系統依賴 (為了 OpenCV)
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# # 先安裝 wheel
# RUN pip install --upgrade pip setuptools wheel

# # 生成 wheels
# RUN mkdir -p /wheels && pip wheel -w /wheels -r requirements.txt

# # 後續安裝依賴
# RUN pip install --no-index --find-links=/wheels -r requirements.txt

# COPY . .

# # 暴露兩個 Port: 8000 (API), 8501 (Demo UI)
# # EXPOSE 8000 8501
# EXPOSE 8000

# ENV PYTHONPATH=/app

# # 啟動腳本：同時跑 API 和 Streamlit
# # CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run demo_ui.py --server.port 8501 --server.address 0.0.0.0"]
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python env
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# 預先建 wheels（加速重 build）
RUN mkdir -p /wheels && pip wheel -w /wheels -r requirements.txt
RUN pip install --no-index --find-links=/wheels -r requirements.txt

# 專案
COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
