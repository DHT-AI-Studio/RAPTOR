# Ollama 模型 MLflow 註冊快速指南

## 問題場景

當你在 MLflow 中註冊 Ollama 模型時，可能會遇到以下情況：

- **MLflow 註冊名稱**: `llama2-7b-chat` （更友好的名稱）
- **本地 Ollama 名稱**: `llama2:7b` （Ollama 服務中的實際名稱）

如果沒有正確配置，API 推理時會找不到模型。

## 解決方案

### 步驟 1: 註冊模型時添加 tag

使用 Python 腳本註冊模型：

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 方法 1: 註冊新模型版本時添加 tag
model_version = client.create_model_version(
    name="llama2-7b-chat",
    source="models:/llama2/1",
    tags={
        "ollama_model_name": "llama2:7b",  # 👈 關鍵配置
        "engine": "ollama",
        "task": "text-generation"
    }
)

# 方法 2: 為現有模型版本添加 tag
client.set_model_version_tag(
    name="llama2-7b-chat",
    version="1",
    key="ollama_model_name",
    value="llama2:7b"  # 👈 關鍵配置
)
```

### 步驟 2: 驗證配置

```python
# 檢查 tag 是否正確設置
model_version = client.get_model_version(
    name="llama2-7b-chat",
    version="1"
)

print(f"Tags: {model_version.tags}")
# 輸出: {'ollama_model_name': 'llama2:7b', ...}
```

### 步驟 3: 使用 API 進行推理

```python
import requests

response = requests.post(
    "http://localhost:8000/inference/infer",
    json={
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2-7b-chat",  # 使用 MLflow 註冊名稱
        "data": {"inputs": "你好"},
        "options": {"max_length": 100}
    }
)

print(response.json())
```

## 完整示例腳本

```python
#!/usr/bin/env python3
"""
註冊 Ollama 模型到 MLflow 的完整示例
"""

import mlflow
from mlflow.tracking import MlflowClient

# 配置 MLflow
mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

# 要註冊的模型列表
models_to_register = [
    {
        "mlflow_name": "llama2-7b-chat",
        "ollama_name": "llama2:7b",
        "description": "Llama 2 7B Chat 模型"
    },
    {
        "mlflow_name": "mistral-7b-instruct",
        "ollama_name": "mistral:7b-instruct",
        "description": "Mistral 7B Instruct 模型"
    },
    {
        "mlflow_name": "codellama-13b",
        "ollama_name": "codellama:13b",
        "description": "Code Llama 13B 模型"
    }
]

for model_info in models_to_register:
    try:
        print(f"\n註冊模型: {model_info['mlflow_name']}")
        
        # 檢查模型是否已存在
        try:
            existing_model = client.get_registered_model(model_info['mlflow_name'])
            print(f"  模型已存在，添加新版本...")
            
            # 創建新版本
            model_version = client.create_model_version(
                name=model_info['mlflow_name'],
                source=f"models:/{model_info['mlflow_name']}/latest",
                description=model_info['description']
            )
            
        except:
            print(f"  創建新模型...")
            # 創建新模型
            client.create_registered_model(
                name=model_info['mlflow_name'],
                description=model_info['description']
            )
            
            model_version = client.create_model_version(
                name=model_info['mlflow_name'],
                source="none",  # Ollama 模型不需要 source
                description=model_info['description']
            )
        
        # 添加關鍵 tag
        client.set_model_version_tag(
            name=model_info['mlflow_name'],
            version=str(model_version.version),
            key="ollama_model_name",
            value=model_info['ollama_name']
        )
        
        # 添加其他 tags
        tags = {
            "engine": "ollama",
            "task": "text-generation",
            "framework": "ollama"
        }
        
        for key, value in tags.items():
            client.set_model_version_tag(
                name=model_info['mlflow_name'],
                version=str(model_version.version),
                key=key,
                value=value
            )
        
        print(f"✅ 成功註冊: {model_info['mlflow_name']} -> {model_info['ollama_name']}")
        
    except Exception as e:
        print(f"❌ 註冊失敗: {e}")

print("\n所有模型註冊完成！")
```

## 檢查 Ollama 可用模型

```bash
# 列出本地 Ollama 所有可用模型
curl http://localhost:11434/api/tags | jq '.models[].name'
```

## 常見問題

### Q1: 忘記添加 `ollama_model_name` tag 怎麼辦？

**A**: 可以隨時補充：

```python
client.set_model_version_tag(
    name="your-model-name",
    version="1",
    key="ollama_model_name",
    value="actual-ollama-name"
)
```

### Q2: 如何批量更新現有模型？

**A**: 使用腳本批量處理：

```python
# 獲取所有使用 Ollama 引擎的模型
all_models = client.search_registered_models()

for model in all_models:
    for version in client.search_model_versions(f"name='{model.name}'"):
        # 檢查是否需要更新
        if "engine" in version.tags and version.tags["engine"] == "ollama":
            if "ollama_model_name" not in version.tags:
                # 需要手動指定正確的 ollama 模型名稱
                ollama_name = input(f"輸入 {model.name} 的 Ollama 模型名稱: ")
                client.set_model_version_tag(
                    name=model.name,
                    version=version.version,
                    key="ollama_model_name",
                    value=ollama_name
                )
```

### Q3: 不使用 MLflow，直接用 Ollama 名稱可以嗎？

**A**: 可以！系統支持兩種方式：

```python
# 方式 1: 使用 MLflow 名稱（推薦，便於管理）
{
    "model_name": "llama2-7b-chat"
}

# 方式 2: 直接使用 Ollama 名稱
{
    "model_name": "llama2:7b"
}
```

## 測試驗證

```bash
# 運行測試腳本
cd /opt/home/george/george-test/AiModelLifecycle/VIE01/AiModelLifecycle
python test/test_ollama_mlflow_mapping.py
```

## 相關文檔

- [OLLAMA_MLFLOW_MAPPING_FIX.md](./OLLAMA_MLFLOW_MAPPING_FIX.md) - 技術細節和修復說明
- [MLflow Models](https://mlflow.org/docs/latest/models.html) - MLflow 官方文檔
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) - Ollama API 文檔
