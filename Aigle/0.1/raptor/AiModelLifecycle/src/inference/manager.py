# src/inference/manager.py
"""
重構後的推理管理器 - 簡化版本

統一的推理管理器，提供單一的推理接口，支持 Ollama 和 Transformers 引擎。
簡化了原有的複雜架構，提供清晰的任務路由和模型執行機制。
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager

# 核心依賴導入
try:
    from ..core.model_manager import model_manager
    from ..core.gpu_manager import gpu_manager
    from ..core.config import config
except ImportError:
    try:
        from src.core.model_manager import model_manager
        from src.core.gpu_manager import gpu_manager  
        from src.core.config import config
    except ImportError:
        model_manager = None
        gpu_manager = None
        config = None

# 模組內部導入
from .router import TaskRouter
from .executor import ModelExecutor
from .registry import ModelRegistry
from .cache import ModelCache

logger = logging.getLogger(__name__)

class InferenceError(Exception):
    """推理相關異常基類"""
    pass

class ModelNotFoundError(InferenceError):
    """模型未找到異常"""
    pass

class UnsupportedTaskError(InferenceError):
    """不支持的任務類型異常"""
    pass

class InferenceManager:
    """
    重構後的推理管理器
    
    提供統一的推理接口，支持以下功能：
    - 統一的 API 入口
    - 任務類型路由
    - 引擎選擇和管理
    - 模型緩存和資源管理
    
    Args:
        None
        
    Returns:
        Dict: 推理結果，包含以下字段：
            - success (bool): 推理是否成功
            - result (Any): 推理結果數據
            - task (str): 執行的任務類型
            - engine (str): 使用的引擎類型
            - model_name (str): 使用的模型名稱
            - processing_time (float): 處理時間（秒）
            - timestamp (float): 處理時間戳
    """
    
    _instance: Optional['InferenceManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'InferenceManager':
        """單例模式實現"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化推理管理器"""
        if hasattr(self, 'initialized'):
            return
            
        try:
            # 初始化核心組件
            self.router = TaskRouter()
            self.registry = ModelRegistry()
            self.cache = ModelCache()
            
            # 統計信息
            self.stats = {
                'total_inferences': 0,
                'successful_inferences': 0,
                'failed_inferences': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            
            self.initialized = True
            logger.info("推理管理器初始化完成")
            
        except Exception as e:
            logger.error(f"推理管理器初始化失敗: {e}")
            raise InferenceError(f"初始化失敗: {e}")
    
    def infer(self, 
              task: str, 
              engine: str, 
              model_name: str,
              data: Dict[str, Any],
              options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        統一推理接口
        
        Args:
            task (str): 任務類型，支持的值：
                - text-generation: 文本生成
                - vlm: 視覺語言模型
                - asr: 語音識別  
                - ocr: 光學字符識別
                - audio-classification: 音頻分類
                - video-analysis: 視頻分析
                - document-analysis: 文檔分析
            engine (str): 引擎類型，支持的值：
                - ollama: 僅支持 text-generation
                - transformers: 支持所有其他任務
            model_name (str): MLflow 中註冊的模型名稱
            data (Dict[str, Any]): 輸入數據，格式根據任務類型而異：
                - text-generation: {"inputs": "輸入文本"}
                - vlm: {"image": "base64圖像", "prompt": "提示詞"}
                - asr: {"audio": "音頻文件路徑"}
                - ocr: {"image": "圖像文件路徑"}
            options (Dict[str, Any], optional): 推理選項，如：
                - max_length: 最大生成長度
                - temperature: 生成溫度
                - top_p: nucleus sampling 參數
                
        Returns:
            Dict[str, Any]: 推理結果，包含：
                - success (bool): 是否成功
                - result (Any): 推理結果
                - task (str): 任務類型
                - engine (str): 引擎類型
                - model_name (str): 模型名稱
                - processing_time (float): 處理時間
                - timestamp (float): 時間戳
                
        Raises:
            UnsupportedTaskError: 不支持的任務類型
            ModelNotFoundError: 模型未找到
            InferenceError: 推理過程中的其他錯誤
        """
        start_time = time.time()
        self.stats['total_inferences'] += 1
        
        try:
            logger.info(f"開始推理 - 任務: {task}, 引擎: {engine}, 模型: {model_name}")
            
            # 參數驗證
            self._validate_parameters(task, engine, model_name, data)
            
            # 任務路由，獲取執行器
            executor = self.router.route(task, engine, model_name)
            
            # 執行推理
            result = executor.execute(model_name, data, options or {})
            
            # 更新統計
            self.stats['successful_inferences'] += 1
            processing_time = time.time() - start_time
            
            logger.info(f"推理完成 - 用時: {processing_time:.2f}秒")
            
            return {
                'success': True,
                'result': result,
                'task': task,
                'engine': engine,
                'model_name': model_name,
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.stats['failed_inferences'] += 1
            logger.error(f"推理失敗: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'task': task,
                'engine': engine,
                'model_name': model_name,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
    
    def _validate_parameters(self, task: str, engine: str, model_name: str, data: Dict) -> None:
        """
        驗證推理參數
        
        Args:
            task (str): 任務類型
            engine (str): 引擎類型
            model_name (str): 模型名稱
            data (Dict): 輸入數據
            
        Raises:
            UnsupportedTaskError: 不支持的任務或引擎組合
            ValueError: 參數無效
        """
        # 檢查參數完整性
        if not all([task, engine, model_name]):
            raise ValueError("task, engine, model_name 不能為空")
        
        # 檢查任務類型
        supported_tasks = {
            'text-generation', 'text-generation-ollama', 'text-generation-hf',
            'vlm', 'asr', 'asr-hf', 'vad-hf', 'ocr', 'ocr-hf',
            'audio-classification', 'video-analysis', 'scene-detection',
            'document-analysis', 'image-captioning', 'video-summary',
            'audio-transcription'
        }
        if task not in supported_tasks:
            raise UnsupportedTaskError(f"不支持的任務類型: {task}，支持的任務: {supported_tasks}")
        
        # 檢查引擎類型
        supported_engines = {'ollama', 'transformers'}
        if engine not in supported_engines:
            raise UnsupportedTaskError(f"不支持的引擎類型: {engine}")
        
        # 檢查引擎與任務的兼容性
        ollama_tasks = {'text-generation', 'text-generation-ollama'}
        if engine == 'ollama' and task not in ollama_tasks:
            raise UnsupportedTaskError(f"Ollama 引擎僅支持 {ollama_tasks} 任務，當前任務: {task}")
        
        # 檢查輸入數據
        if not isinstance(data, dict) or not data:
            raise ValueError("data 必須是非空字典")
        
        # 根據任務類型檢查必需的數據字段
        required_fields = {
            'text-generation': ['inputs'],
            'text-generation-ollama': ['inputs'],
            'text-generation-hf': ['inputs'],
            'vlm': ['image', 'prompt'],
            'asr': ['audio'],
            'asr-hf': ['audio'],
            'vad-hf': ['audio'],
            'ocr': ['image'],
            'ocr-hf': ['image'],
            'audio-classification': ['audio'],
            'video-analysis': ['video'],
            'scene-detection': ['video'],
            'document-analysis': ['document'],
            'image-captioning': ['image'],
            'video-summary': ['video'],
            'audio-transcription': ['audio']
        }
        
        if task in required_fields:
            missing_fields = [field for field in required_fields[task] if field not in data]
            if missing_fields:
                raise ValueError(f"任務 {task} 缺少必需的數據字段: {missing_fields}")
    
    def get_supported_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        獲取支持的任務類型和對應的引擎
        
        Returns:
            Dict[str, Dict[str, Any]]: 支持的任務配置
        """
        return {
            'text-generation-ollama': {
                'engines': ['ollama'],
                'description': '文本生成任務 (Ollama 引擎)',
                'input_format': {'inputs': '輸入文本'},
                'examples': ['llama2-7b', 'mistral-7b']
            },
            'text-generation-hf': {
                'engines': ['transformers'],
                'description': '文本生成任務 (HuggingFace 引擎)',
                'input_format': {'inputs': '輸入文本'},
                'examples': ['gpt2', 'bloom-560m']
            },
            'text-generation': {
                'engines': ['ollama', 'transformers'],
                'description': '文本生成任務（通用，向下兼容）',
                'input_format': {'inputs': '輸入文本'},
                'examples': ['gpt-3.5-turbo', 'llama2-7b']
            },
            'vlm': {
                'engines': ['transformers'],
                'description': '視覺語言模型任務',
                'input_format': {'image': 'base64圖像', 'prompt': '提示詞'},
                'examples': ['llava-1.5-7b', 'blip2-t5']
            },
            'asr-hf': {
                'engines': ['transformers'],
                'description': '自動語音識別任務 (HuggingFace)',
                'input_format': {'audio': '音頻文件路徑'},
                'examples': ['whisper-large', 'wav2vec2-large']
            },
            'asr': {
                'engines': ['transformers'],
                'description': '自動語音識別任務（通用）',
                'input_format': {'audio': '音頻文件路徑'},
                'examples': ['whisper-large', 'wav2vec2-large']
            },
            'vad-hf': {
                'engines': ['transformers'],
                'description': '語音活動檢測任務 (HuggingFace)',
                'input_format': {'audio': '音頻文件路徑'},
                'examples': ['silero-vad']
            },
            'ocr-hf': {
                'engines': ['transformers'],
                'description': '光學字符識別任務 (HuggingFace)',
                'input_format': {'image': '圖像文件路徑'},
                'examples': ['trocr-base', 'trocr-large']
            },
            'ocr': {
                'engines': ['transformers'],
                'description': '光學字符識別任務（通用）',
                'input_format': {'image': '圖像文件路徑'},
                'examples': ['trocr-base', 'paddleocr']
            },
            'audio-classification': {
                'engines': ['transformers'],
                'description': '音頻分類任務',
                'input_format': {'audio': '音頻文件路徑'},
                'examples': ['ast-finetuned', 'wav2vec2-audio']
            },
            'video-analysis': {
                'engines': ['transformers'],
                'description': '視頻分析任務',
                'input_format': {'video': '視頻文件路徑'},
                'examples': ['videomae-large', 'vivit-base']
            },
            'scene-detection': {
                'engines': ['transformers'],
                'description': '場景檢測任務',
                'input_format': {'video': '視頻文件路徑'},
                'examples': ['scene-detection-model']
            },
            'document-analysis': {
                'engines': ['transformers'],
                'description': '文檔分析任務',
                'input_format': {'document': '文檔文件路徑'},
                'examples': ['layoutlm-large', 'donut-base']
            },
            'image-captioning': {
                'engines': ['transformers'],
                'description': '圖像標題生成任務',
                'input_format': {'image': '圖像文件路徑'},
                'examples': ['blip-image-captioning']
            },
            'video-summary': {
                'engines': ['transformers'],
                'description': '視頻摘要任務',
                'input_format': {'video': '視頻文件路徑'},
                'examples': ['video-summary-model']
            },
            'audio-transcription': {
                'engines': ['transformers'],
                'description': '音頻轉錄任務',
                'input_format': {'audio': '音頻文件路徑'},
                'examples': ['whisper-large-v2']
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取推理管理器統計信息
        
        Returns:
            Dict[str, Any]: 統計信息
        """
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_inferences'] / max(self.stats['total_inferences'], 1)
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(
                    self.stats['cache_hits'] + self.stats['cache_misses'], 1
                )
            ),
            'cached_models': self.cache.get_cached_models(),
            'supported_tasks': list(self.get_supported_tasks().keys())
        }
    
    def clear_cache(self) -> None:
        """清理模型緩存"""
        self.cache.clear()
        logger.info("模型緩存已清理")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康檢查
        
        Returns:
            Dict[str, Any]: 健康狀態信息
        """
        try:
            # 檢查核心組件
            components_status = {
                'router': self.router is not None,
                'registry': self.registry is not None,
                'cache': self.cache is not None,
                'model_manager': model_manager is not None,
                'gpu_manager': gpu_manager is not None
            }
            
            all_healthy = all(components_status.values())
            
            return {
                'status': 'healthy' if all_healthy else 'unhealthy',
                'components': components_status,
                'stats': self.get_stats(),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"健康檢查失敗: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

# 創建全局實例
inference_manager = InferenceManager()