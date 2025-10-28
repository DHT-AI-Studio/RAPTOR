# src/inference/engines/ollama.py
"""
Ollama 引擎實現

專門用於 Ollama 服務的推理引擎。
僅支持文本生成任務（text-generation），所有 Ollama 模型使用統一的推理流程。

特點：
- 通過 HTTP API 與 Ollama 服務通信
- 模型由 Ollama 服務端管理，無需顯式加載到內存
- 支持自動拉取模型（如果本地不存在）
- 與 MLflow 模型註冊中心整合
"""

import logging
import requests
from typing import Any, Dict, Optional
from .base import BaseEngine

# 核心依賴導入
try:
    from ...core.model_manager import model_manager
    from ...core.config import config as app_config
except ImportError:
    try:
        from src.core.model_manager import model_manager
        from src.core.config import config as app_config
    except ImportError:
        model_manager = None
        app_config = None

logger = logging.getLogger(__name__)

class OllamaEngine(BaseEngine):
    """
    Ollama 引擎實現
    
    通過 HTTP API 與 Ollama 服務通信。
    僅支持文本生成任務，所有模型使用統一的推理流程。
    
    Attributes:
        base_url (str): Ollama 服務地址
        timeout (int): 請求超時時間（秒）
        auto_pull (bool): 是否自動拉取不存在的模型
        stream (bool): 是否使用流式輸出（默認 False）
        supported_tasks (list): 支持的任務類型 ['text-generation']
        version (str): 引擎版本號
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Ollama 引擎
        
        Args:
            config (Dict[str, Any], optional): 引擎配置
                可選配置項：
                - base_url (str): Ollama 服務地址（默認: 'http://localhost:11434'）
                - timeout (int): 請求超時時間（默認: 300）
                - auto_pull (bool): 自動拉取模型（默認: True）
                - stream (bool): 流式輸出（默認: False）
        """
        super().__init__(config)
        
        # 從配置獲取 Ollama 設置
        # 優先使用傳入的 config 參數（由 router 傳遞）
        if self.config and ('api_base' in self.config or 'base_url' in self.config):
            # 如果 router 已經傳遞了配置，直接使用
            self.base_url = self.config.get('base_url') or self.config.get('api_base', 'http://localhost:11434')
            self.timeout = self.config.get('timeout', 300)
        else:
            # 否則從全局配置讀取
            if app_config:
                try:
                    ollama_config = app_config.get_config("ollama") or {}
                    self.base_url = ollama_config.get('api_base', 'http://localhost:11434')
                    self.timeout = ollama_config.get('timeout', 300)
                except Exception as e:
                    logger.warning(f"無法讀取 Ollama 配置: {e}，使用默認值")
                    self.base_url = 'http://localhost:11434'
                    self.timeout = 300
            else:
                self.base_url = 'http://localhost:11434'
                self.timeout = 300
        
        self.auto_pull = self.config.get('auto_pull', True)
        self.stream = self.config.get('stream', False)
        
        logger.info(f"Ollama 引擎初始化: base_url={self.base_url}, timeout={self.timeout}")
        
        # 支持的任務類型
        self.supported_tasks = ['text-generation']
        self.version = '2.0.0'
        
        # 檢查 Ollama 服務是否可用
        self._check_service_availability()
        
        self.is_initialized = True
        logger.info(f"Ollama 引擎初始化完成 - 服務地址: {self.base_url}")
    
    def _check_service_availability(self):
        """
        檢查 Ollama 服務是否可用
        
        通過調用 /api/tags 端點來驗證服務連接。
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama 服務連接正常")
            else:
                logger.warning(f"Ollama 服務響應異常: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"無法連接到 Ollama 服務 ({self.base_url}): {e}")
            logger.warning("Ollama 引擎將繼續初始化，但推理可能失敗")
    
    def load_model(self, model_name: str, **kwargs) -> str:
        """
        加載模型（對於 Ollama，這主要是驗證和準備）
        
        Ollama 模型由服務端管理，不需要顯式加載到內存。
        此方法主要用於：
        1. 從 MLflow 獲取模型信息（如果已註冊）
        2. 檢查模型是否在 Ollama 中可用
        3. 自動拉取模型（如果啟用且模型不存在）
        
        Args:
            model_name (str): MLflow 中註冊的模型名稱或 Ollama 模型名稱
            **kwargs: 額外參數
                - force_pull (bool): 強制重新拉取模型
                
        Returns:
            str: Ollama 模型名稱（用於後續推理）
            
        Raises:
            RuntimeError: 模型加載失敗
            ValueError: 模型不支持或配置無效
            
        Example:
            >>> engine = OllamaEngine()
            >>> model = engine.load_model("llama2-7b")
            >>> # model = "llama2-7b" (Ollama 模型名稱)
        """
        try:
            logger.info(f"加載 Ollama 模型: {model_name}")
            
            # 1. 嘗試從 MLflow 獲取模型信息
            ollama_model_name = model_name
            
            # 嘗試在運行時導入 model_manager (避免循環導入和模塊加載順序問題)
            mm = model_manager
            if mm is None:
                try:
                    from ...core.model_manager import model_manager as mm
                except ImportError:
                    try:
                        from src.core.model_manager import model_manager as mm
                    except ImportError:
                        mm = None
            
            logger.info(f"model_manager 可用性: {mm is not None}")
            if mm:
                try:
                    logger.info(f"嘗試從 MLflow 獲取模型信息: {ollama_model_name}")
                    model_info = mm.get_model_details_from_mlflow(ollama_model_name)
                    logger.info(f"MLflow 返回的模型信息: {model_info is not None}")
                    if model_info and 'tags' in model_info:
                        logger.info(f"MLflow 模型標籤: {model_info['tags']}")
                        # 如果 MLflow 中有 Ollama 模型名稱標籤，使用它
                        if 'ollama_model_name' in model_info['tags']:
                            ollama_model_name = model_info['tags']['ollama_model_name']
                            logger.info(f"從 MLflow 獲取 Ollama 模型名稱: {ollama_model_name}")
                except Exception as e:
                    logger.warning(f"從 MLflow 獲取模型信息失敗（將直接使用模型名稱）: {e}")
            else:
                logger.warning(f"model_manager 未初始化，將直接使用模型名稱: {model_name}")
            
            # 2. 檢查模型是否在 Ollama 中可用
            force_pull = kwargs.get('force_pull', False)
            if force_pull or not self._is_model_available(ollama_model_name):
                if self.auto_pull or force_pull:
                    logger.info(f"模型 {ollama_model_name} 不可用，開始拉取...")
                    self._pull_model(ollama_model_name)
                else:
                    available_models = self.get_available_models()
                    raise RuntimeError(
                        f"模型 '{ollama_model_name}' 在 Ollama 中不可用。"
                        f"\n可用模型: {available_models}"
                        f"\n提示: 設置 auto_pull=True 以自動拉取模型"
                    )
            
            # 3. 測試模型（可選）
            if kwargs.get('test_model', False):
                if not self._test_model(ollama_model_name):
                    raise RuntimeError(f"模型 '{ollama_model_name}' 測試失敗")
            
            logger.info(f"Ollama 模型加載成功: {ollama_model_name}")
            return ollama_model_name
            
        except Exception as e:
            logger.error(f"加載 Ollama 模型失敗: {e}")
            raise RuntimeError(f"Ollama 模型加載失敗: {e}")
    
    def infer(self, model: str, inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        執行推理
        
        Args:
            model (str): Ollama 模型名稱（由 load_model 返回）
            inputs (Dict[str, Any]): 輸入數據
                必需字段：
                - inputs (str): 輸入文本/提示詞
            options (Dict[str, Any]): 推理選項
                通用選項：
                - max_length/num_predict (int): 最大生成長度
                - temperature (float): 生成溫度
                - top_p (float): nucleus sampling
                - top_k (int): top-k sampling
                Ollama 特定選項：
                - repeat_penalty (float): 重複懲罰
                - stop (list): 停止序列
                - mirostat (int): Mirostat 採樣模式
                
        Returns:
            Dict[str, Any]: 推理結果
                {
                    'response': str,              # 生成的文本
                    'model': str,                 # 使用的模型名稱
                    'done': bool,                 # 是否完成
                    'metadata': {                 # 元數據
                        'total_duration': int,    # 總耗時（納秒）
                        'load_duration': int,     # 加載耗時
                        'prompt_eval_count': int, # 提示詞 token 數
                        'eval_count': int         # 生成 token 數
                    }
                }
            
        Raises:
            ValueError: 輸入數據格式無效
            RuntimeError: 推理執行失敗
        """
        try:
            # 驗證輸入
            self.validate_inputs(inputs, ['inputs'])
            
            # 準備請求數據
            request_data = {
                'model': model,
                'prompt': inputs['inputs'],
                'stream': self.stream,
                'options': self._prepare_ollama_options(options)
            }
            
            logger.debug(f"發送 Ollama 推理請求: {model}")
            
            # 發送請求
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API 錯誤: {response.status_code}\n{response.text}"
                )
            
            # 解析響應
            result = response.json()
            
            # 格式化返回結果（統一格式）
            return {
                'response': result.get('response', ''),
                'model': model,
                'done': result.get('done', True),
                'metadata': {
                    'total_duration': result.get('total_duration', 0),
                    'load_duration': result.get('load_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'prompt_eval_duration': result.get('prompt_eval_duration', 0),
                    'eval_count': result.get('eval_count', 0),
                    'eval_duration': result.get('eval_duration', 0),
                    'context': result.get('context', [])
                }
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama 推理超時（{self.timeout}秒）")
            raise RuntimeError(f"推理超時（{self.timeout}秒），請考慮增加 timeout 設置")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama 請求失敗: {e}")
            raise RuntimeError(f"Ollama 服務請求失敗: {e}")
        except Exception as e:
            logger.error(f"Ollama 推理失敗: {e}")
            raise RuntimeError(f"推理執行失敗: {e}")
    
    def _is_model_available(self, model_name: str) -> bool:
        """
        檢查模型是否在 Ollama 中可用
        
        Args:
            model_name (str): 模型名稱
            
        Returns:
            bool: 是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                model_names = [model['name'] for model in models_data.get('models', [])]
                # 支持精確匹配和部分匹配
                return model_name in model_names or any(
                    model_name in name for name in model_names
                )
            return False
        except Exception as e:
            logger.warning(f"檢查模型可用性失敗: {e}")
            return False
    
    def _pull_model(self, model_name: str):
        """
        拉取模型到 Ollama
        
        Args:
            model_name (str): 模型名稱
            
        Raises:
            RuntimeError: 拉取失敗
        """
        try:
            logger.info(f"開始拉取 Ollama 模型: {model_name}（這可能需要幾分鐘）")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={'name': model_name},
                timeout=1200  # 拉取模型可能需要較長時間（20分鐘）
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"模型拉取失敗: {response.status_code}\n{response.text}"
                )
            
            logger.info(f"Ollama 模型拉取成功: {model_name}")
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"模型拉取超時，模型可能較大或網絡較慢")
        except Exception as e:
            logger.error(f"模型拉取失敗: {e}")
            raise RuntimeError(f"模型拉取失敗: {e}")
    
    def _test_model(self, model_name: str) -> bool:
        """
        測試模型是否正常工作
        
        Args:
            model_name (str): 模型名稱
            
        Returns:
            bool: 是否正常
        """
        try:
            logger.debug(f"測試 Ollama 模型: {model_name}")
            test_request = {
                'model': model_name,
                'prompt': 'Hello',
                'stream': False,
                'options': {'num_predict': 5}
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_request,
                timeout=30
            )
            
            success = response.status_code == 200
            if success:
                logger.debug(f"模型測試成功: {model_name}")
            else:
                logger.warning(f"模型測試失敗: {model_name}, 狀態碼: {response.status_code}")
            return success
            
        except Exception as e:
            logger.warning(f"模型測試失敗: {e}")
            return False
    
    def _prepare_ollama_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        將通用選項轉換為 Ollama 特定選項格式
        
        Args:
            options (Dict[str, Any]): 通用推理選項
            
        Returns:
            Dict[str, Any]: Ollama 選項格式
            
        Note:
            通用選項映射：
            - max_length/max_tokens -> num_predict
            - temperature -> temperature
            - top_p -> top_p
            - top_k -> top_k
        """
        ollama_options = {}
        
        # 映射通用選項到 Ollama 選項
        option_mapping = {
            'max_length': 'num_predict',
            'max_tokens': 'num_predict',
            'temperature': 'temperature',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'repeat_penalty': 'repeat_penalty',
            'stop': 'stop'
        }
        
        for generic_key, ollama_key in option_mapping.items():
            if generic_key in options:
                ollama_options[ollama_key] = options[generic_key]
        
        # 添加 Ollama 特定選項（直接傳遞）
        ollama_specific_keys = [
            'mirostat', 'mirostat_eta', 'mirostat_tau', 
            'num_ctx', 'num_batch', 'num_gqa', 'num_gpu',
            'main_gpu', 'low_vram', 'f16_kv', 'vocab_only',
            'use_mmap', 'use_mlock', 'embedding_only',
            'rope_frequency_base', 'rope_frequency_scale',
            'num_thread'
        ]
        
        for key in ollama_specific_keys:
            if key in options:
                ollama_options[key] = options[key]
        
        return ollama_options
    
    def get_available_models(self) -> list:
        """
        獲取 Ollama 服務中可用的模型列表
        
        Returns:
            list: 模型名稱列表
            
        Example:
            >>> engine = OllamaEngine()
            >>> models = engine.get_available_models()
            >>> print(models)
            ['llama2:7b', 'mistral:latest', 'codellama:13b']
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"獲取 Ollama 模型列表失敗: {e}")
            return []
    
    def get_info(self) -> Dict[str, Any]:
        """
        獲取引擎詳細信息
        
        Returns:
            Dict[str, Any]: 引擎信息，包括：
                - name: 引擎名稱
                - version: 版本號
                - supported_tasks: 支持的任務
                - base_url: Ollama 服務地址
                - timeout: 超時設置
                - auto_pull: 是否自動拉取
                - available_models: 可用模型列表
                - service_status: 服務狀態
        """
        base_info = super().get_info()
        base_info.update({
            'base_url': self.base_url,
            'timeout': self.timeout,
            'auto_pull': self.auto_pull,
            'stream': self.stream,
            'available_models': self.get_available_models(),
            'service_status': self._check_service_status()
        })
        return base_info
    
    def _check_service_status(self) -> Dict[str, Any]:
        """
        檢查 Ollama 服務狀態
        
        Returns:
            Dict[str, Any]: 服務狀態信息
        """
        try:
            import time
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            return {
                'available': response.status_code == 200,
                'status_code': response.status_code,
                'response_time_ms': round(response_time, 2)
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'response_time_ms': None
            }