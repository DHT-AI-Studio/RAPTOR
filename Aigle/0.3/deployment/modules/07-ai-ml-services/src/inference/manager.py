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

# 導入統一的異常定義
from .exceptions import (
    InferenceError,
    ValidationError,
    UnsupportedTaskError,
    ModelNotFoundError,
    ResourceNotFoundError,
    ModelLoadError,
    InferenceExecutionError,
    EngineError,
    ResourceExhaustedError
)

logger = logging.getLogger(__name__)

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
        """單例模式實現 - 使用雙重檢查鎖定"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初始化推理管理器
        
        使用類級別的鎖來確保初始化過程的執行緒安全。
        只有第一次調用會執行初始化邏輯。
        """
        # 使用類級別的鎖來保護初始化檢查和設置
        with self.__class__._lock:
            if hasattr(self, '_initialized') and self._initialized:
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
                
                # 統計數據的執行緒鎖 - 保護 stats 字典的並發訪問
                self._stats_lock = threading.Lock()
                
                # 標記已初始化（使用 _ 前綴表示這是內部屬性）
                self._initialized = True
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
                - result (Any): 推理結果
                - task (str): 任務類型
                - engine (str): 引擎類型
                - model_name (str): 模型名稱
                - processing_time (float): 處理時間
                - timestamp (float): 時間戳
                
        Raises:
            ValidationError: 參數驗證失敗
            UnsupportedTaskError: 不支持的任務類型或引擎組合
            ModelNotFoundError: 模型未找到
            InferenceExecutionError: 推理執行失敗
            ResourceExhaustedError: 資源耗盡
            InferenceError: 推理過程中的其他錯誤
        """
        start_time = time.time()
        
        # 執行緒安全地更新統計計數
        with self._stats_lock:
            self.stats['total_inferences'] += 1
        
        try:
            logger.info(f"開始推理 - 任務: {task}, 引擎: {engine}, 模型: {model_name}")
            
            # 參數驗證 - 可能拋出 ValidationError 或 UnsupportedTaskError
            self._validate_parameters(task, engine, model_name, data)
            
            # 任務路由，獲取執行器 - 可能拋出 UnsupportedTaskError
            executor = self.router.route(task, engine, model_name)
            
            # 執行推理 - 可能拋出 ModelNotFoundError, InferenceExecutionError 等
            result = executor.execute(model_name, data, options or {})
            
            # 執行緒安全地更新成功統計
            with self._stats_lock:
                self.stats['successful_inferences'] += 1
            
            processing_time = time.time() - start_time
            
            logger.info(f"推理完成 - 用時: {processing_time:.2f}秒")
            
            # 只返回成功結果，不包含 success 標誌
            # 如果有錯誤，會通過異常拋出
            return {
                'result': result,
                'task': task,
                'engine': engine,
                'model_name': model_name,
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
        except (ValidationError, UnsupportedTaskError, ModelNotFoundError, 
                ResourceNotFoundError, ModelLoadError, InferenceExecutionError,
                EngineError, ResourceExhaustedError) as e:
            # 已知的推理異常，執行緒安全地更新失敗統計後直接向上傳播
            with self._stats_lock:
                self.stats['failed_inferences'] += 1
            
            logger.error(f"推理失敗 ({type(e).__name__}): {e}")
            raise
            
        except Exception as e:
            # 未預期的異常，包裝為 InferenceExecutionError 後向上傳播
            with self._stats_lock:
                self.stats['failed_inferences'] += 1
            
            logger.error(f"推理執行時發生未預期錯誤: {e}", exc_info=True)
            raise InferenceExecutionError(f"推理執行失敗: {str(e)}") from e
    
    def _validate_parameters(self, task: str, engine: str, model_name: str, data: Dict) -> None:
        """
        驗證推理參數
        
        Args:
            task (str): 任務類型
            engine (str): 引擎類型
            model_name (str): 模型名稱
            data (Dict): 輸入數據
            
        Raises:
            ValidationError: 參數無效
            UnsupportedTaskError: 不支持的任務或引擎組合
        """
        # 檢查參數完整性
        if not all([task, engine, model_name]):
            raise ValidationError("task, engine, model_name 不能為空")
        
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
            raise UnsupportedTaskError(f"不支持的引擎類型: {engine}，支持的引擎: {supported_engines}")
        
        # 檢查引擎與任務的兼容性
        ollama_tasks = {'text-generation', 'text-generation-ollama'}
        if engine == 'ollama' and task not in ollama_tasks:
            raise UnsupportedTaskError(f"Ollama 引擎僅支持 {ollama_tasks} 任務，當前任務: {task}")
        
        # 檢查輸入數據
        if not isinstance(data, dict) or not data:
            raise ValidationError("data 必須是非空字典")
        
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
                raise ValidationError(f"任務 {task} 缺少必需的數據字段: {missing_fields}")
    
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
        
        執行緒安全地讀取統計數據，確保返回的數據是一致的快照。
        
        Returns:
            Dict[str, Any]: 統計信息
        """
        # 執行緒安全地讀取統計數據，創建一個一致的快照
        with self._stats_lock:
            stats_snapshot = self.stats.copy()
            total = stats_snapshot['total_inferences']
            successful = stats_snapshot['successful_inferences']
            cache_hits = stats_snapshot['cache_hits']
            cache_misses = stats_snapshot['cache_misses']
        
        # 在鎖外計算衍生統計（避免長時間持有鎖）
        return {
            **stats_snapshot,
            'success_rate': successful / max(total, 1),
            'cache_hit_rate': cache_hits / max(cache_hits + cache_misses, 1),
            'cached_models': self.cache.get_cached_models(),
            'supported_tasks': list(self.get_supported_tasks().keys())
        }
    
    def clear_cache(self) -> None:
        """清理模型緩存"""
        self.cache.clear()
        logger.info("模型緩存已清理")
    
    def unload_model(self, model_name: str, task: Optional[str] = None, engine: Optional[str] = None) -> Dict[str, Any]:
        """
        卸載指定模型並釋放 GPU 內存
        
        Args:
            model_name (str): 模型名稱
            task (str, optional): 任務類型（如果知道，可加速查找）
            engine (str, optional): 引擎類型（如果知道，可加速查找）
            
        Returns:
            Dict[str, Any]: 卸載結果
        """
        try:
            unloaded_count = 0
            executors_found = []
            
            # 遍歷所有執行器，查找並卸載模型
            for executor_key, executor in self.router._executors.items():
                executor_task, executor_engine = executor_key
                
                # 如果指定了 task 或 engine，進行過濾
                if task and executor_task != task:
                    continue
                if engine and executor_engine != engine:
                    continue
                
                # 檢查該執行器是否加載了該模型
                if model_name in executor.get_loaded_models():
                    if executor.unload_model(model_name):
                        unloaded_count += 1
                        executors_found.append({
                            'task': executor_task,
                            'engine': executor_engine
                        })
            
            # 清理 GPU 緩存
            gpu_memory_freed = self._clear_gpu_cache()
            
            return {
                'success': True,
                'model_name': model_name,
                'unloaded_count': unloaded_count,
                'executors': executors_found,
                'gpu_memory_freed': gpu_memory_freed,
                'message': f'成功卸載模型 {model_name}，釋放了 {unloaded_count} 個實例'
            }
            
        except Exception as e:
            logger.error(f"卸載模型失敗: {e}", exc_info=True)
            return {
                'success': False,
                'model_name': model_name,
                'error': str(e),
                'message': f'卸載模型失敗: {str(e)}'
            }
    
    def unload_all_models(self) -> Dict[str, Any]:
        """
        卸載所有已加載的模型並釋放 GPU 內存
        
        Returns:
            Dict[str, Any]: 卸載結果
        """
        try:
            total_unloaded = 0
            models_unloaded = []
            
            # 遍歷所有執行器，卸載所有模型
            for executor_key, executor in self.router._executors.items():
                executor_task, executor_engine = executor_key
                loaded_models = executor.get_loaded_models()
                
                for model_name in loaded_models:
                    if executor.unload_model(model_name):
                        total_unloaded += 1
                        models_unloaded.append({
                            'model_name': model_name,
                            'task': executor_task,
                            'engine': executor_engine
                        })
            
            # 清理 GPU 緩存
            gpu_memory_freed = self._clear_gpu_cache()
            
            return {
                'success': True,
                'total_unloaded': total_unloaded,
                'models_unloaded': models_unloaded,
                'gpu_memory_freed': gpu_memory_freed,
                'message': f'成功卸載 {total_unloaded} 個模型實例'
            }
            
        except Exception as e:
            logger.error(f"卸載所有模型失敗: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'message': f'卸載所有模型失敗: {str(e)}'
            }
    
    def _clear_gpu_cache(self) -> Dict[str, Any]:
        """
        清理 GPU 緩存
        
        Returns:
            Dict[str, Any]: GPU 內存清理結果
        """
        try:
            import torch
            import gc
            
            if not torch.cuda.is_available():
                return {
                    'cuda_available': False,
                    'memory_freed': 0,
                    'message': 'CUDA 不可用'
                }
            
            # 獲取清理前的內存使用情況
            memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            # 清理 GPU 緩存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 獲取清理後的內存使用情況
            memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_freed = memory_before - memory_after
            
            return {
                'cuda_available': True,
                'memory_before_gb': round(memory_before, 2),
                'memory_after_gb': round(memory_after, 2),
                'memory_freed_gb': round(memory_freed, 2),
                'message': f'GPU 內存清理完成，釋放 {memory_freed:.2f} GB'
            }
            
        except ImportError:
            return {
                'cuda_available': False,
                'memory_freed': 0,
                'message': 'PyTorch 未安裝'
            }
        except Exception as e:
            logger.error(f"清理 GPU 緩存失敗: {e}")
            return {
                'cuda_available': True,
                'memory_freed': 0,
                'error': str(e),
                'message': f'清理 GPU 緩存失敗: {str(e)}'
            }
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """
        獲取所有已加載的模型列表
        
        Returns:
            Dict[str, Any]: 已加載的模型信息
        """
        try:
            loaded_models = {}
            
            for executor_key, executor in self.router._executors.items():
                executor_task, executor_engine = executor_key
                models = executor.get_loaded_models()
                
                for model_name in models:
                    if model_name not in loaded_models:
                        loaded_models[model_name] = {
                            'model_name': model_name,
                            'instances': []
                        }
                    
                    loaded_models[model_name]['instances'].append({
                        'task': executor_task,
                        'engine': executor_engine
                    })
            
            return {
                'total_models': len(loaded_models),
                'total_instances': sum(len(info['instances']) for info in loaded_models.values()),
                'models': loaded_models
            }
            
        except Exception as e:
            logger.error(f"獲取已加載模型列表失敗: {e}")
            return {
                'total_models': 0,
                'total_instances': 0,
                'models': {},
                'error': str(e)
            }
    
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