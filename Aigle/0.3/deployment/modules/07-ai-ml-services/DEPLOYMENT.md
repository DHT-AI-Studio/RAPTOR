## **Using uv to manage python packages**
```bash
conda create -n raptor python=3.10 -y
conda activate raptor
pip install uv
uv pip install -r requirements.txt
```
**Uvicorn run command example**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8009 --reload
```

## **Deploy with Docker Compose**
```bash
docker compose up --build -d
```
**this will start the following services:**
- mlflow: MLflow 服務器，監聽在 `http://localhost:5000`
- AiModelLifecycle-api: AiModelLifecycle API 服務，監聽在 `http://localhost:8009`