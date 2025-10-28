# src/inference/engines/base.py
"""
推理引擎抽象基類

定義所有推理引擎必須實現的核心接口。
簡化設計，移除不必要的抽象層，專注於實際使用場景。

設計原則：
- 簡單直接：方法簽名與實際使用一致
- 單一職責：每個方法只做一件事  
- 類型安全：完整的類型提示
- 與 MLflow 整合：支持從 MLflow 註冊中心加載模型
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseEngine(ABC):
    """
    推理引擎抽象基類
    
    所有推理引擎必須繼承此類並實現其抽象方法。
    
    生命週期：
    1. __init__() - 初始化引擎配置
    2. load_model() - 加載模型（可選，某些引擎如 Ollama 不需要顯式加載）
    3. infer() - 執行推理（可多次調用）
    4. unload_model() - 卸載模型（可選）
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化引擎
        
        Args:
            config (Dict[str, Any], optional): 引擎配置
                通用配置項：
                - device (str): 計算設備（'cuda', 'cpu', 'auto'）
                - precision (str): 精度設定（'fp16', 'fp32', 'int8'）
                - trust_remote_code (bool): 是否信任遠程代碼
                - timeout (int): 超時時間（秒）
                引擎特定配置會在子類中處理
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.is_initialized = False
        logger.debug(f"初始化引擎: {self.name}")
    
    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Any:
        """
        加載模型
        
        從 MLflow 註冊中心或本地路徑加載模型。
        引擎會自動處理：
        1. 從 MLflow 獲取模型的物理路徑
        2. 加載模型到內存/GPU
        3. 返回可用於推理的模型對象
        
        Args:
            model_name (str): 在 MLflow 中註冊的模型名稱
            **kwargs: 額外參數
                - model_path (str, optional): 模型的物理路徑（覆蓋 MLflow 查找）
                - task (str, optional): 任務類型提示
                - device (str, optional): 設備覆蓋
                - 其他引擎特定參數
                
        Returns:
            Any: 加載的模型對象（格式取決於引擎）
                - Ollama: 返回模型名稱字符串  
                - Transformers: 返回 pipeline 或模型實例
                
        Raises:
            FileNotFoundError: 模型路徑不存在
            RuntimeError: 模型加載失敗
            ValueError: 配置參數無效
            
        Example:
            >>> engine = TransformersEngine()
            >>> model = engine.load_model("llama2-7b")
            >>> # 或指定路徑
            >>> model = engine.load_model("custom-model", model_path="/path/to/model")
        """
        pass

    @abstractmethod
    def infer(self, model: Any, inputs: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """
        執行推理
        
        Args:
            model (Any): 由 load_model() 返回的模型對象
            inputs (Dict[str, Any]): 預處理後的輸入數據
                格式根據任務類型而異：
                - text-generation: {'inputs': str}
                - vlm: {'image': PIL.Image, 'prompt': str}
                - asr: {'audio': np.ndarray 或 str(文件路徑)}
                - ocr: {'image': PIL.Image 或 str(文件路徑)}
                - audio-classification: {'audio': np.ndarray 或 str}
                - video-analysis: {'video': str(文件路徑)}
                - document-analysis: {'document': str(文件路徑)}
                
            options (Dict[str, Any]): 推理選項
                通用選項：
                - max_length (int): 最大生成長度
                - temperature (float): 生成溫度 (0.0-1.0)
                - top_p (float): nucleus sampling 參數
                - top_k (int): top-k sampling 參數
                - do_sample (bool): 是否採樣
                引擎特定選項在子類中處理
                
        Returns:
            Any: 推理結果（格式取決於引擎和任務）
                推薦返回格式：
                {
                    'response': str,           # 主要結果
                    'metadata': dict,          # 元數據（如耗時、token數等）
                    # 其他任務特定字段
                }
                
        Raises:
            ValueError: 輸入數據格式無效
            RuntimeError: 推理執行失敗
            
        Example:
            >>> result = engine.infer(
            ...     model=model,
            ...     inputs={'inputs': 'Hello, how are you?'},
            ...     options={'max_length': 100, 'temperature': 0.7}
            ... )
        """
        pass
    
    def unload_model(self, model: Any) -> bool:
        """
        卸載模型並釋放資源
        
        Args:
            model (Any): 要卸載的模型對象
            
        Returns:
            bool: 是否成功卸載
            
        Note:
            - 某些引擎（如 Ollama）可能不需要顯式卸載
            - 應該清理 GPU 緩存和臨時文件
            - 默認實現返回 True，子類可以重寫以實現特定邏輯
        """
        logger.debug(f"卸載模型: {self.name}")
        return True
    
    def validate_inputs(self, inputs: Dict[str, Any], required_keys: list) -> bool:
        """
        驗證輸入數據
        
        Args:
            inputs (Dict[str, Any]): 輸入數據
            required_keys (list): 必需的鍵列表
            
        Returns:
            bool: 是否通過驗證
            
        Raises:
            ValueError: 驗證失敗
        """
        if not isinstance(inputs, dict):
            raise ValueError(f"輸入必須是字典類型，收到: {type(inputs)}")
        
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"缺少必需的輸入鍵: {missing_keys}")
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        獲取引擎信息
        
        Returns:
            Dict[str, Any]: 引擎信息
                - name (str): 引擎名稱
                - version (str): 版本號
                - supported_tasks (list): 支持的任務列表
                - is_initialized (bool): 是否已初始化
                - config (dict): 當前配置
        """
        return {
            'name': self.name,
            'version': getattr(self, 'version', '1.0.0'),
            'supported_tasks': getattr(self, 'supported_tasks', []),
            'is_initialized': self.is_initialized,
            'config': self.config
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}(initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        """詳細字符串表示"""
        return f"{self.name}(config={self.config}, initialized={self.is_initialized})"