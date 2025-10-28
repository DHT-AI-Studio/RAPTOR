FROM python:3.10-slim

WORKDIR /app

COPY ./asset_management ./asset_management
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    libmagic1 \
    # file \
    # build-essential \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "asset_management.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
