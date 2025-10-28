# src/core/config.py

import os
import re
import yaml
from pathlib import Path  # 1. 匯入 pathlib，用於處理路徑，比 os.path 更現代化
from typing import Any
from dotenv import load_dotenv # 2. 匯入 dotenv

# 可靠地定位專案根目錄
# __file__ 是當前檔案 (config.py) 的路徑
# .parent 會回傳上一層目錄
# src/core/config.py -> .parent -> src/core -> .parent -> src -> .parent -> 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 將環境變數替換的邏輯抽成一個輔助函數
def _substitute_vars_and_resolve_paths(config_value: Any) -> Any:
    """
    遞迴地替換設定值中的佔位符，並將特定路徑字串轉換為絕對路徑的 Path 物件。
    """
    if isinstance(config_value, dict):
        return {k: _substitute_vars_and_resolve_paths(v) for k, v in config_value.items()}
    elif isinstance(config_value, list):
        return [_substitute_vars_and_resolve_paths(i) for i in config_value]
    elif isinstance(config_value, str):
        # 步驟 2a：首先，替換 ${PROJECT_ROOT} 佔位符
        # str(PROJECT_ROOT) 確保路徑是字串格式，以便進行替換
        value_str = config_value.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))

        # 步驟 2b：接著，替換環境變數 ${VAR_NAME}
        pattern = re.compile(r"\$\{(.+?)\}")
        def replace_env(match):
            var_name = match.group(1)
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Configuration Error: Environment variable '{var_name}' not found.")
            return value
        
        final_str = pattern.sub(replace_env, value_str)

        # 步驟 2c (✨ 關鍵新增功能):
        # 如果原始設定的 key 包含 'path' 或 'root'，我們就將其視為路徑
        # 並轉換成 Path 物件，同時自動建立目錄
        # 注意：這個檢查是在遞迴的上一層進行的（見下方 dict comprehension）
        return final_str
    else:
        return config_value
    
# 在字典處理中加入路徑轉換和目錄建立的邏輯
def _process_config_dict(config_dict: dict) -> dict:
    processed = {}
    for k, v in config_dict.items():
        # 遞迴處理子項目
        processed_v = _substitute_vars_and_resolve_paths(v)

        # ✨ 如果 key 暗示這是一個路徑，並且值是字串，則轉換為 Path 物件並建立目錄
        if isinstance(processed_v, str) and ('path' in k or 'root' in k):
            path_obj = Path(processed_v)
            # 自動建立目錄，parents=True 允許建立多層目錄，exist_ok=True 表示如果目錄已存在則不報錯
            path_obj.mkdir(parents=True, exist_ok=True)
            processed[k] = path_obj # 儲存為 Path 物件
        else:
            processed[k] = processed_v
            
    return processed

# 設定管理器 (Singleton 單例模式)
class ConfigurationManager:
    _instance = None
    _config = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            env_path = PROJECT_ROOT / '.env'
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
            
            config_template_path = PROJECT_ROOT / "src" / "core" / "configs" / "base.yaml"
            if not config_template_path.exists():
                raise FileNotFoundError(f"Configuration file not found at {config_template_path}")
            
            with open(config_template_path, 'r') as f:
                base_config = yaml.safe_load(f)

            # ✨ 新增：載入 inference.yaml
            inference_config_path = PROJECT_ROOT / "src" / "core" / "configs" / "inference.yaml"
            inference_config = {}
            if inference_config_path.exists():
                with open(inference_config_path, 'r') as f:
                    inference_config = yaml.safe_load(f)
            
            # 合併設定，並將推理設定放在 'inference' key 下
            merged_config = base_config
            if inference_config:
                merged_config['inference'] = inference_config
            
            # 使用新的處理函數
            self._config = _process_config_dict(merged_config)

    def get_config(self, *keys: str) -> Any:
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                 return None
            value = value.get(key)
            if value is None:
                return None
        return value

# 建立全域唯一的設定管理器實例
# 當任何其他檔案 `from src.core.config import config` 時，
# __init__ 方法將會被執行一次，並完成所有設定的載入。
config = ConfigurationManager()