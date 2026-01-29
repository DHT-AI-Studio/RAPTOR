# Joe 時序推理引擎程式碼

### 創建 conda 環境
```bash
conda create -n TRE -y
```

```bash
conda activate TRE
```

```bash
pip install -r requirements.txt
```


### 原始檔案結構
```bash
.
├── demo_ui.py
├── docker-compose_all.yaml                 # raptor all service contianer
├── docker-compose.yaml
├── Dockerfile
├── main.py
├── model_engine.py                         # Qwen Vision Engine – User-Defined Module
├── README_Minnie.md
├── requirements.txt
└── vision_service                          # empty directory 
```


### 新檔案結構
Adjusted by minnie on 2026-01-08

```bash
.
├── app                                 # 用來放與 FastAPI 相關的 code
│   ├── __init__.py
│   └── routers                         # 將 FastAPI 的服改用router來寫
│       ├── __init__.py
│       └── vision_analyze_Qwen.py      # 將原本 main.py 中的 FastAPI() 改成 APIRouter()
├── demo_ui.py
├── docker-compose_all.yaml
├── docker-compose.yaml
├── Dockerfile
├── main.py                             # main.py 從 ./app/routers import APIRouter()
├── README_Minnie.md
├── requirements.txt
├── scripts                             # 各種功能的程式碼腳本
│   ├── __init__.py
│   ├── model_engine.py                 # Qwen Vision Engine – User-Defined Module
└── vision_service
```