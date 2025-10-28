# src/inference/engines/transformers.py
"""
Transformers 引擎實現

基於 HuggingFace Transformers 庫的推理引擎。
支持多種任務類型，包括 text-generation、VLM、ASR、OCR 等多模態任務。

特點：
- 與 MLflow 模型註冊中心深度整合
- 支持從 MLflow 獲取模型物理路徑
- 自動設備管理（CPU/CUDA）
- 支持多種多模態任務
- 可擴展的任務處理架構
"""

import logging
import os
import torch
import gc
from typing import Any, Dict, Optional
from pathlib import Path
from .base import BaseEngine
logger = logging.getLogger(__name__)


# HuggingFace 相關導入
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoProcessor,
        pipeline as hf_pipeline
    )
    from transformers import logging as transformers_logging
    HF_AVAILABLE = True
except ImportError:
    logger.warning("transformers 庫未安裝")
    HF_AVAILABLE = False

# PIL 圖像處理
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 核心依賴導入
try:
    from ...core.model_manager import model_manager
    from ...core.gpu_manager import gpu_manager
    from ...core.config import config as app_config
except ImportError:
    try:
        from src.core.model_manager import model_manager
        from src.core.gpu_manager import gpu_manager
        from src.core.config import config as app_config
    except ImportError:
        model_manager = None
        gpu_manager = None
        app_config = None


class TransformersEngine(BaseEngine):
    """
    Transformers 引擎實現
    
    基於 HuggingFace Transformers 庫，支持多種 AI 任務。
    與 MLflow 深度整合，自動從註冊中心獲取模型。
    
    Attributes:
        device (str): 計算設備（'cuda', 'cpu'）
        torch_dtype (torch.dtype): PyTorch 數據類型
        trust_remote_code (bool): 是否信任遠程代碼
        supported_tasks (list): 支持的任務類型
        version (str): 引擎版本號
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Transformers 引擎
        
        Args:
            config (Dict[str, Any], optional): 引擎配置
                可選配置項：
                - device (str): 計算設備（'cuda', 'cpu', 'auto'）
                - torch_dtype (str): 數據類型（'fp16', 'fp32', 'auto'）
                - trust_remote_code (bool): 是否信任遠程代碼（默認: True）
                - low_cpu_mem_usage (bool): 低內存模式（默認: True）
        """
        super().__init__(config)
        
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers 庫未安裝。請運行: pip install transformers"
            )
        
        # 設備配置
        self.device = self.config.get('device', 'auto')
        self._setup_device()
        
        # 數據類型配置
        torch_dtype_str = self.config.get('torch_dtype', 'auto')
        self.torch_dtype = self._get_torch_dtype(torch_dtype_str)
        
        # 其他配置
        self.trust_remote_code = self.config.get('trust_remote_code', True)
        self.low_cpu_mem_usage = self.config.get('low_cpu_mem_usage', True)
        
        # 支持的任務類型
        self.supported_tasks = [
            'text-generation',
            'vlm',
            'asr',
            'ocr',
            'audio-classification',
            'video-analysis',
            'document-analysis'
        ]
        self.version = '2.0.0'
        
        # 任務類型到 pipeline 任務的映射
        self._task_mapping = {
            'text-generation': 'text-generation',
            'asr': 'automatic-speech-recognition',
            'audio-classification': 'audio-classification',
            'ocr': 'image-to-text',
        }
        
        # 已加載的模型緩存
        self._loaded_models: Dict[str, Any] = {}
        
        self.is_initialized = True
        logger.info(
            f"Transformers 引擎初始化完成 - "
            f"設備: {self.device}, 類型: {self.torch_dtype}"
        )
    
    def _setup_device(self):
        """設置計算設備"""
        if self.device == 'auto':
            if torch.cuda.is_available():
                # 檢查 GPU 可用性
                if gpu_manager:
                    try:
                        gpu_info = gpu_manager.get_gpu_info()
                        if gpu_info.get('available_gpus', 0) > 0:
                            self.device = 'cuda'
                        else:
                            self.device = 'cpu'
                    except:
                        self.device = 'cuda'  # 回退到 CUDA
                else:
                    self.device = 'cuda'
            else:
                self.device = 'cpu'
        
        logger.info(f"使用計算設備: {self.device}")
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """
        獲取 PyTorch 數據類型
        
        Args:
            dtype_str (str): 數據類型字符串
            
        Returns:
            torch.dtype: PyTorch 數據類型
        """
        if dtype_str == 'auto':
            # 自動選擇：CUDA 用 fp16，CPU 用 fp32
            return torch.float16 if self.device == 'cuda' else torch.float32
        elif dtype_str == 'fp16':
            return torch.float16
        elif dtype_str == 'fp32':
            return torch.float32
        elif dtype_str == 'int8':
            return torch.int8
        else:
            logger.warning(f"未知的數據類型: {dtype_str}，使用 auto")
            return torch.float16 if self.device == 'cuda' else torch.float32
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """
        加載模型
        
        從 MLflow 註冊中心獲取模型路徑並加載到內存。
        支持多種任務類型的模型。
        
        Args:
            model_name (str): MLflow 中註冊的模型名稱
            **kwargs: 額外參數
                - model_path (str, optional): 模型物理路徑（覆蓋 MLflow 查找）
                - task (str, optional): 任務類型提示
                - device (str, optional): 設備覆蓋
                - force_reload (bool): 強制重新加載
                
        Returns:
            Any: 加載的模型對象（pipeline 或模型實例）
            
        Raises:
            FileNotFoundError: 模型路徑不存在
            RuntimeError: 模型加載失敗
            
        Example:
            >>> engine = TransformersEngine()
            >>> model = engine.load_model("llama2-7b", task="text-generation")
        """
        try:
            logger.info(f"加載 Transformers 模型: {model_name}")
            
            # 檢查是否已加載（且不強制重新加載）
            force_reload = kwargs.get('force_reload', False)
            if model_name in self._loaded_models and not force_reload:
                logger.info(f"重用已加載的模型: {model_name}")
                return self._loaded_models[model_name]
            
            # 1. 獲取模型路徑
            model_path = self._get_model_path(model_name, **kwargs)
            
            # 2. 獲取任務類型
            task = kwargs.get('task', 'text-generation')
            
            # 3. 加載模型
            model = self._load_model_by_task(model_path, task, **kwargs)
            
            # 4. 緩存模型
            self._loaded_models[model_name] = model
            
            logger.info(f"Transformers 模型加載成功: {model_name}")
            return model
            
        except FileNotFoundError as e:
            logger.error(f"模型文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"加載 Transformers 模型失敗: {e}")
            raise RuntimeError(f"模型加載失敗: {e}")
    
    def _get_model_path(self, model_name: str, **kwargs) -> str:
        """
        獲取模型物理路徑
        
        優先級：
        1. kwargs 中的 model_path
        2. MLflow 註冊的模型路徑
        3. 直接使用 model_name（假設是 HuggingFace Hub 模型）
        
        Args:
            model_name (str): 模型名稱
            **kwargs: 額外參數
            
        Returns:
            str: 模型物理路徑
        """
        # 優先使用直接指定的路徑
        if 'model_path' in kwargs:
            model_path = kwargs['model_path']
            logger.info(f"使用指定的模型路徑: {model_path}")
            return model_path
        
        # 嘗試從 MLflow 獲取
        if model_manager:
            try:
                model_info = model_manager.get_model_details_from_mlflow(model_name)
                if model_info and 'tags' in model_info:
                    tags = model_info['tags']
                    
                    # 檢查物理路徑
                    if 'physical_path' in tags:
                        physical_path = tags['physical_path']
                        
                        # 處理 lakeFS 路徑
                        if physical_path.startswith('lakefs://'):
                            logger.info(f"檢測到 lakeFS 路徑: {physical_path}")
                            # 嘗試下載或轉換路徑
                            local_path = self._handle_lakefs_path(physical_path, model_name)
                            if local_path:
                                return local_path
                        else:
                            # 本地路徑
                            if os.path.exists(physical_path):
                                logger.info(f"從 MLflow 獲取模型路徑: {physical_path}")
                                return physical_path
                            else:
                                logger.warning(f"MLflow 路徑不存在: {physical_path}")
                    
                    # 檢查 HuggingFace repo_id
                    if 'repo_id' in tags:
                        repo_id = tags['repo_id']
                        logger.info(f"使用 HuggingFace repo: {repo_id}")
                        return repo_id
                        
            except Exception as e:
                logger.warning(f"從 MLflow 獲取模型路徑失敗: {e}")
        
        # 回退：假設是 HuggingFace Hub 模型
        logger.info(f"使用模型名稱作為 HuggingFace Hub ID: {model_name}")
        return model_name
    
    def _handle_lakefs_path(self, lakefs_path: str, model_name: str) -> Optional[str]:
        """
        處理 lakeFS 路徑
        
        Args:
            lakefs_path (str): lakeFS 路徑
            model_name (str): 模型名稱
            
        Returns:
            Optional[str]: 本地路徑（如果下載成功）
        """
        try:
            if model_manager and hasattr(model_manager, 'download_model_from_lakefs'):
                logger.info(f"從 lakeFS 下載模型: {model_name}")
                local_path = model_manager.download_model_from_lakefs(
                    model_name,
                    lakefs_path
                )
                return local_path
        except Exception as e:
            logger.warning(f"從 lakeFS 下載失敗: {e}")
        return None
    
    def _load_model_by_task(self, model_path: str, task: str, **kwargs) -> Any:
        """
        根據任務類型加載模型
        
        Args:
            model_path (str): 模型路徑
            task (str): 任務類型
            **kwargs: 額外參數
            
        Returns:
            Any: 加載的模型對象
        """
        # 獲取設備（允許覆蓋）
        device = kwargs.get('device', self.device)
        
        # 對於某些任務，使用 pipeline
        if task in self._task_mapping:
            pipeline_task = self._task_mapping[task]
            logger.debug(f"使用 pipeline 加載模型: {pipeline_task}")
            
            model = hf_pipeline(
                pipeline_task,
                model=model_path,
                device=0 if device == 'cuda' else -1,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
            )
            return model
        
        # 對於 VLM、video、document 等任務，需要特殊處理
        elif task in ['vlm', 'video-analysis', 'document-analysis']:
            logger.debug(f"加載多模態模型: {task}")
            return self._load_multimodal_model(model_path, task, device)
        
        else:
            # 默認：加載通用模型
            logger.debug("加載通用 Transformers 模型")
            return self._load_generic_model(model_path, device)
    
    def _load_multimodal_model(self, model_path: str, task: str, device: str) -> Dict[str, Any]:
        """
        加載多模態模型
        
        Args:
            model_path (str): 模型路徑
            task (str): 任務類型
            device (str): 設備
            
        Returns:
            Dict[str, Any]: 包含模型和處理器的字典
        """
        try:
            # 加載處理器
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code
            )
            
            # 加載模型
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=self.low_cpu_mem_usage
            )
            
            # 移動到設備
            if device == 'cuda':
                model = model.to(device)
            
            return {
                'model': model,
                'processor': processor,
                'task': task
            }
            
        except Exception as e:
            logger.error(f"加載多模態模型失敗: {e}")
            raise
    
    def _load_generic_model(self, model_path: str, device: str) -> Dict[str, Any]:
        """
        加載通用模型
        
        Args:
            model_path (str): 模型路徑
            device (str): 設備
            
        Returns:
            Dict[str, Any]: 包含模型和分詞器的字典
        """
        try:
            # 加載分詞器
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code
            )
            
            # 嘗試加載因果語言模型
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=self.low_cpu_mem_usage
                )
            except:
                # 回退到通用模型
                model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=self.trust_remote_code,
                    low_cpu_mem_usage=self.low_cpu_mem_usage
                )
            
            # 移動到設備
            if device == 'cuda':
                model = model.to(device)
            
            return {
                'model': model,
                'tokenizer': tokenizer
            }
            
        except Exception as e:
            logger.error(f"加載通用模型失敗: {e}")
            raise
    
    def infer(self, model: Any, inputs: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """
        執行推理
        
        Args:
            model (Any): 由 load_model 返回的模型對象
            inputs (Dict[str, Any]): 輸入數據（已預處理）
                格式根據任務類型而異
            options (Dict[str, Any]): 推理選項
                通用選項：
                - max_length/max_new_tokens (int): 最大生成長度
                - temperature (float): 生成溫度
                - top_p (float): nucleus sampling
                - top_k (int): top-k sampling
                - do_sample (bool): 是否採樣
                - num_beams (int): beam search 數量
                
        Returns:
            Any: 推理結果（格式根據模型類型而異）
            
        Raises:
            ValueError: 輸入數據格式無效
            RuntimeError: 推理執行失敗
        """
        try:
            logger.debug("開始執行 Transformers 推理")
            
            # 如果是 pipeline 對象
            if hasattr(model, '__call__') and hasattr(model, 'task'):
                return self._infer_with_pipeline(model, inputs, options)
            
            # 如果是字典（包含模型和處理器）
            elif isinstance(model, dict):
                if 'task' in model:
                    task = model['task']
                    if task == 'vlm':
                        return self._infer_vlm(model, inputs, options)
                    elif task in ['video-analysis', 'document-analysis']:
                        return self._infer_multimodal(model, inputs, options)
                else:
                    return self._infer_generic(model, inputs, options)
            
            else:
                raise ValueError(f"不支持的模型類型: {type(model)}")
                
        except Exception as e:
            logger.error(f"Transformers 推理失敗: {e}")
            raise RuntimeError(f"推理執行失敗: {e}")
    
    def _infer_with_pipeline(self, pipeline, inputs: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """使用 pipeline 執行推理"""
        try:
            # 準備生成參數
            gen_kwargs = self._prepare_generation_kwargs(options)
            
            # 執行推理
            if pipeline.task == 'text-generation':
                result = pipeline(
                    inputs['inputs'],
                    **gen_kwargs
                )
                return {'response': result[0]['generated_text'], 'metadata': {}}
            
            elif pipeline.task == 'automatic-speech-recognition':
                result = pipeline(inputs['audio'])
                return {'text': result['text'], 'metadata': {}}
            
            elif pipeline.task == 'audio-classification':
                result = pipeline(inputs['audio'])
                return {'classifications': result, 'metadata': {}}
            
            elif pipeline.task == 'image-to-text':
                result = pipeline(inputs['image'])
                return {'text': result[0]['generated_text'], 'metadata': {}}
            
            else:
                # 通用處理
                result = pipeline(inputs['inputs'] if 'inputs' in inputs else inputs)
                return {'response': str(result), 'metadata': {}}
                
        except Exception as e:
            logger.error(f"Pipeline 推理失敗: {e}")
            raise
    
    def _infer_vlm(self, model_dict: Dict[str, Any], inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """VLM 推理"""
        try:
            model = model_dict['model']
            processor = model_dict['processor']
            
            # 準備輸入
            image = inputs.get('image')
            prompt = inputs.get('prompt', '')
            
            # 處理輸入
            processed_inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # 移動到設備
            if self.device == 'cuda':
                processed_inputs = {k: v.to(self.device) for k, v in processed_inputs.items()}
            
            # 生成參數
            gen_kwargs = self._prepare_generation_kwargs(options)
            
            # 執行推理
            with torch.no_grad():
                outputs = model.generate(**processed_inputs, **gen_kwargs)
            
            # 解碼輸出
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            return {
                'response': response,
                'metadata': {
                    'input_length': len(processed_inputs['input_ids'][0]),
                    'output_length': len(outputs[0])
                }
            }
            
        except Exception as e:
            logger.error(f"VLM 推理失敗: {e}")
            raise
    
    def _infer_multimodal(self, model_dict: Dict[str, Any], inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """多模態推理"""
        # 類似 VLM，但可能需要不同的處理邏輯
        return self._infer_vlm(model_dict, inputs, options)
    
    def _infer_generic(self, model_dict: Dict[str, Any], inputs: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """通用模型推理"""
        try:
            model = model_dict['model']
            tokenizer = model_dict['tokenizer']
            
            # 分詞
            input_text = inputs.get('inputs', '')
            encoded = tokenizer(input_text, return_tensors="pt")
            
            # 移動到設備
            if self.device == 'cuda':
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # 生成參數
            gen_kwargs = self._prepare_generation_kwargs(options)
            
            # 執行推理
            with torch.no_grad():
                outputs = model.generate(**encoded, **gen_kwargs)
            
            # 解碼
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'response': response,
                'metadata': {
                    'input_length': len(encoded['input_ids'][0]),
                    'output_length': len(outputs[0])
                }
            }
            
        except Exception as e:
            logger.error(f"通用模型推理失敗: {e}")
            raise
    
    def _prepare_generation_kwargs(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """準備生成參數"""
        gen_kwargs = {}
        
        # 映射通用選項
        option_mapping = {
            'max_length': 'max_length',
            'max_new_tokens': 'max_new_tokens',
            'temperature': 'temperature',
            'top_p': 'top_p',
            'top_k': 'top_k',
            'do_sample': 'do_sample',
            'num_beams': 'num_beams',
            'repetition_penalty': 'repetition_penalty',
            'length_penalty': 'length_penalty',
            'early_stopping': 'early_stopping',
        }
        
        for key, gen_key in option_mapping.items():
            if key in options:
                gen_kwargs[gen_key] = options[key]
        
        # 默認值
        if 'do_sample' not in gen_kwargs:
            gen_kwargs['do_sample'] = True
        if 'max_new_tokens' not in gen_kwargs and 'max_length' not in gen_kwargs:
            gen_kwargs['max_new_tokens'] = 512
        
        return gen_kwargs
    
    def unload_model(self, model: Any) -> bool:
        """
        卸載模型並釋放資源
        
        Args:
            model (Any): 要卸載的模型對象
            
        Returns:
            bool: 是否成功卸載
        """
        try:
            # 查找並移除緩存
            for model_name, cached_model in list(self._loaded_models.items()):
                if cached_model is model:
                    del self._loaded_models[model_name]
                    logger.info(f"從緩存中移除模型: {model_name}")
                    break
            
            # 清理模型
            if isinstance(model, dict):
                if 'model' in model:
                    del model['model']
                if 'tokenizer' in model:
                    del model['tokenizer']
                if 'processor' in model:
                    del model['processor']
            
            # 清理 GPU 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("模型卸載完成")
            return True
            
        except Exception as e:
            logger.error(f"模型卸載失敗: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        獲取引擎詳細信息
        
        Returns:
            Dict[str, Any]: 引擎信息
        """
        base_info = super().get_info()
        base_info.update({
            'device': self.device,
            'torch_dtype': str(self.torch_dtype),
            'trust_remote_code': self.trust_remote_code,
            'hf_available': HF_AVAILABLE,
            'cuda_available': torch.cuda.is_available(),
            'loaded_models': list(self._loaded_models.keys()),
            'loaded_models_count': len(self._loaded_models)
        })
        
        # GPU 信息
        if torch.cuda.is_available():
            base_info['gpu_info'] = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            }
        
        return base_info
