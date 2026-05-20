FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124

COPY ./app app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]