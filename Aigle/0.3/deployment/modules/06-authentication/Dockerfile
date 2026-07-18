# 選擇官方 Python 3.12 slim 版本
FROM python:3.10-slim

# 設定工作目錄
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

# 安裝依賴
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY ./app app

# 暴露 port
EXPOSE 8800

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8800"]