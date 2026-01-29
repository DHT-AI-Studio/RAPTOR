#!/usr/bin/env python3
"""
錯誤處理重構驗證腳本

測試新的異常處理機制，確保：
1. 異常正確拋出和傳播
2. API 返回適當的 HTTP 狀態碼
3. 錯誤響應格式統一
"""

import requests
import json
from typing import Dict, Any

# API 基礎 URL
BASE_URL = "http://localhost:8009"

def print_result(test_name: str, response: requests.Response):
    """打印測試結果"""
    print(f"\n{'='*60}")
    print(f"測試: {test_name}")
    print(f"{'='*60}")
    print(f"狀態碼: {response.status_code}")
    print(f"響應: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_valid_request():
    """測試 1: 正常請求（如果有可用的模型）"""
    url = f"{BASE_URL}/inference/infer"
    payload = {
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "llama2-7b",  # 請根據實際情況修改
        "data": {
            "inputs": "Hello, world!"
        },
        "options": {
            "max_length": 50
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        print_result("正常請求", response)
        
        # 驗證成功響應
        if response.status_code == 200:
            data = response.json()
            assert 'success' in data, "響應缺少 success 字段"
            assert 'result' in data, "響應缺少 result 字段"
            print("✅ 成功響應格式正確")
        else:
            print(f"⚠️ 狀態碼不是 200，而是 {response.status_code}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_missing_parameters():
    """測試 2: 缺少必需參數 - 應返回 400"""
    url = f"{BASE_URL}/inference/infer"
    payload = {
        "task": "",  # 空任務名稱
        "engine": "ollama",
        "model_name": "test-model",
        "data": {"inputs": "test"}
    }
    
    try:
        response = requests.post(url, json=payload)
        print_result("缺少必需參數", response)
        
        # 驗證錯誤響應
        assert response.status_code == 400, f"應返回 400，實際返回 {response.status_code}"
        
        data = response.json()
        assert 'detail' in data, "錯誤響應缺少 detail 字段"
        
        detail = data['detail']
        if isinstance(detail, dict):
            assert 'error_type' in detail, "錯誤詳情缺少 error_type"
            assert detail['error_type'] == 'ValidationError', f"錯誤類型應為 ValidationError，實際為 {detail['error_type']}"
            print("✅ 400 錯誤響應格式正確")
        else:
            print(f"⚠️ detail 不是字典格式: {detail}")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_unsupported_task():
    """測試 3: 不支持的任務類型 - 應返回 400"""
    url = f"{BASE_URL}/inference/infer"
    payload = {
        "task": "unknown-task",
        "engine": "ollama",
        "model_name": "test-model",
        "data": {"inputs": "test"}
    }
    
    try:
        response = requests.post(url, json=payload)
        print_result("不支持的任務類型", response)
        
        # 驗證錯誤響應
        assert response.status_code == 400, f"應返回 400，實際返回 {response.status_code}"
        
        data = response.json()
        detail = data['detail']
        if isinstance(detail, dict):
            assert detail['error_type'] == 'UnsupportedTaskError', f"錯誤類型應為 UnsupportedTaskError"
            print("✅ 400 錯誤響應格式正確")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_invalid_engine_task_combination():
    """測試 4: 引擎與任務不兼容 - 應返回 400"""
    url = f"{BASE_URL}/inference/infer"
    payload = {
        "task": "vlm",  # VLM 任務不支持 ollama 引擎
        "engine": "ollama",
        "model_name": "test-model",
        "data": {"image": "test.jpg", "prompt": "test"}
    }
    
    try:
        response = requests.post(url, json=payload)
        print_result("引擎與任務不兼容", response)
        
        # 驗證錯誤響應
        assert response.status_code == 400, f"應返回 400，實際返回 {response.status_code}"
        
        data = response.json()
        detail = data['detail']
        if isinstance(detail, dict):
            assert detail['error_type'] == 'UnsupportedTaskError', f"錯誤類型應為 UnsupportedTaskError"
            print("✅ 400 錯誤響應格式正確")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_missing_data_fields():
    """測試 5: 缺少必需的數據字段 - 應返回 400"""
    url = f"{BASE_URL}/inference/infer"
    payload = {
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "test-model",
        "data": {}  # 缺少 inputs 字段
    }
    
    try:
        response = requests.post(url, json=payload)
        print_result("缺少必需的數據字段", response)
        
        # 驗證錯誤響應
        assert response.status_code == 400, f"應返回 400，實際返回 {response.status_code}"
        
        data = response.json()
        detail = data['detail']
        if isinstance(detail, dict):
            assert detail['error_type'] == 'ValidationError', f"錯誤類型應為 ValidationError"
            print("✅ 400 錯誤響應格式正確")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_model_not_found():
    """測試 6: 模型未找到 - 應返回 404 或 500（取決於實現）"""
    url = f"{BASE_URL}/inference/infer"
    payload = {
        "task": "text-generation",
        "engine": "ollama",
        "model_name": "non-existent-model-12345",
        "data": {"inputs": "test"}
    }
    
    try:
        response = requests.post(url, json=payload)
        print_result("模型未找到", response)
        
        # 可能返回 404 或 500，取決於何時檢測到模型不存在
        assert response.status_code in [404, 500], f"應返回 404 或 500，實際返回 {response.status_code}"
        
        data = response.json()
        print(f"✅ 錯誤狀態碼: {response.status_code}")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_health_check():
    """測試 7: 健康檢查端點"""
    url = f"{BASE_URL}/inference/health"
    
    try:
        response = requests.get(url)
        print_result("健康檢查", response)
        
        assert response.status_code == 200, f"健康檢查應返回 200，實際返回 {response.status_code}"
        
        data = response.json()
        assert 'status' in data, "健康檢查響應缺少 status 字段"
        print("✅ 健康檢查正常")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def test_supported_tasks():
    """測試 8: 獲取支持的任務"""
    url = f"{BASE_URL}/inference/supported-tasks"
    
    try:
        response = requests.get(url)
        print_result("獲取支持的任務", response)
        
        assert response.status_code == 200, f"應返回 200，實際返回 {response.status_code}"
        
        data = response.json()
        assert 'tasks' in data, "響應缺少 tasks 字段"
        print("✅ 支持的任務列表正常")
    except AssertionError as e:
        print(f"❌ 斷言失敗: {e}")
    except Exception as e:
        print(f"❌ 測試失敗: {e}")

def main():
    """運行所有測試"""
    print("="*60)
    print("開始錯誤處理重構驗證測試")
    print("="*60)
    
    # 先測試基礎端點
    print("\n【基礎端點測試】")
    test_health_check()
    test_supported_tasks()
    
    # 測試錯誤處理
    print("\n【錯誤處理測試】")
    test_missing_parameters()
    test_unsupported_task()
    test_invalid_engine_task_combination()
    test_missing_data_fields()
    test_model_not_found()
    
    # 最後測試正常請求（如果有可用模型）
    print("\n【正常請求測試】")
    print("⚠️ 此測試需要有可用的模型，如果沒有會失敗")
    test_valid_request()
    
    print("\n" + "="*60)
    print("測試完成")
    print("="*60)

if __name__ == "__main__":
    main()
