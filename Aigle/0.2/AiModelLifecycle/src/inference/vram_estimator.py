# src/inference/vram_estimator.py
"""
模型資源估算器 (Model Resource Estimator)

這個類提供了一套工具，用於近似估算 AI 模型在推理時的 VRAM 需求、
潛在延遲和吞吐量。估算結果越準確，需要提供的輸入參數越多。

主要功能：
1. 根據模型參數、類型和精度估算 VRAM。
2. 根據 FLOPs 和 GPU 性能估算延遲。
3. 提供一個選擇器，用於在多個模型元數據中，根據當前資源限制選擇最優模型。
"""

from typing import Optional, Dict, Any, List, Tuple

class ModelResourceEstimator:
    """
    一個用於估算模型運行時資源需求的類。
    """
    
    # 將精度常量作為類屬性，方便管理
    PRECISION_BYTES: Dict[str, int] = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1
    }

    def _safe_get(self, d: Dict, k: str, default: Any = None) -> Any:
        """
        安全的從字典中獲取值，處理鍵不存在或值為 None 的情況。

        Args:
            d (Dict): 目標字典。
            k (str): 要獲取的鍵。
            default (Any, optional): 如果鍵不存在或值為 None，返回的默認值。

        Returns:
            Any: 獲取到的值或默認值。
        """
        return d[k] if k in d and d[k] is not None else default

    def _get_ollama_precision_bytes(self, quantization_level: str, fallback_precision: str) -> int:
        """
        根據 Ollama 量化級別獲取精確的位元組數。
        
        Args:
            quantization_level (str): Ollama 量化級別，如 "Q4_0", "Q8_0" 等
            fallback_precision (str): 備用精度，當無法識別量化級別時使用
            
        Returns:
            int: 每個參數的位元組數
        """
        if not quantization_level:
            return self.PRECISION_BYTES.get(fallback_precision.lower(), 2)
            
        quant = quantization_level.upper()
        
        # Ollama 量化級別到位元組的精確映射
        if quant in ['Q2_K']:
            return 0.25  # ~2 bits per parameter
        elif quant in ['Q3_K_S', 'Q3_K_M', 'Q3_K_L']:
            return 0.375  # ~3 bits per parameter
        elif quant in ['Q4_0', 'Q4_1', 'Q4_K_S', 'Q4_K_M']:
            return 0.5  # ~4 bits per parameter
        elif quant in ['Q5_0', 'Q5_1', 'Q5_K_S', 'Q5_K_M']:
            return 0.625  # ~5 bits per parameter
        elif quant in ['Q6_K']:
            return 0.75  # ~6 bits per parameter
        elif quant in ['Q8_0', 'Q8_1']:
            return 1  # 8 bits = 1 byte per parameter
        elif quant in ['F16', 'FP16']:
            return 2  # 16 bits = 2 bytes per parameter
        elif quant in ['F32', 'FP32']:
            return 4  # 32 bits = 4 bytes per parameter
        else:
            # 未知量化級別，使用備用精度
            return self.PRECISION_BYTES.get(fallback_precision.lower(), 2)

    def explain_quantization_calculation(self, quantization_level: str) -> dict:
        """
        解釋量化級別的計算方式和來源。
        
        Args:
            quantization_level (str): Ollama 量化級別
            
        Returns:
            dict: 包含計算詳情的字典
        """
        quant = quantization_level.upper()
        
        explanations = {
            'Q2_K': {
                'theoretical_bits': 2,
                'actual_bits_per_param': 2.2,
                'bytes_per_param': 0.25,
                'explanation': '2-bit 量化 + K-means 分組開銷',
                'source': 'GGML Q2_K 格式規範'
            },
            'Q3_K_S': {
                'theoretical_bits': 3,
                'actual_bits_per_param': 3.1,
                'bytes_per_param': 0.375,
                'explanation': '3-bit 量化 + 小型 K-means 分組',
                'source': 'GGML Q3_K 格式，Small variant'
            },
            'Q4_0': {
                'theoretical_bits': 4,
                'actual_bits_per_param': 4.5,
                'bytes_per_param': 0.5,
                'explanation': '4-bit 量化 + 32個參數共享一個 scale',
                'source': 'GGML Q4_0 格式：每32個參數一個 FP16 scale'
            },
            'Q4_K_M': {
                'theoretical_bits': 4,
                'actual_bits_per_param': 4.75,
                'bytes_per_param': 0.5,
                'explanation': '4-bit 量化 + K-means 分組，中等精度',
                'source': 'GGML Q4_K 格式，Medium variant'
            },
            'Q8_0': {
                'theoretical_bits': 8,
                'actual_bits_per_param': 8.2,
                'bytes_per_param': 1.0,
                'explanation': '8-bit 量化 + 少量 scale/zero-point 開銷',
                'source': 'GGML Q8_0 格式規範'
            },
            'F16': {
                'theoretical_bits': 16,
                'actual_bits_per_param': 16,
                'bytes_per_param': 2.0,
                'explanation': '標準 IEEE 754 half precision',
                'source': 'IEEE 754-2008 binary16 格式'
            }
        }
        
        if quant in explanations:
            result = explanations[quant].copy()
            result['quantization_level'] = quant
            return result
        else:
            return {
                'quantization_level': quant,
                'error': f'未知的量化級別: {quant}',
                'available_levels': list(explanations.keys())
            }

    def estimate_resources(
        self,
        *,
        params: Optional[int] = None,
        flops_per_inference: Optional[float] = None,
        model_type: str = "text-generation-hf",
        precision: str = "fp16",
        context_tokens: int = 2048,
        output_tokens: int = 512,
        use_kv_cache: bool = True,
        image_resolution: Optional[Tuple[int, int]] = None,
        batch_size: int = 1,
        gpu_peak_tflops: float = 40.0,
        supports_int8: bool = False,
        measured_weights_bytes: Optional[int] = None,
        measured_activations_bytes: Optional[int] = None,
        notes: Optional[List[str]] = None,
        # 新增智能推斷參數
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        engine_type: Optional[str] = None,
        auto_infer: bool = True,
        # 新增 Ollama 量化支持
        quantization_level: Optional[str] = None
    ) -> dict[str, Any]:
        """
        估算一個模型的資源需求，並返回一個包含估計值的字典。
        支持智能參數推斷：當 auto_infer=True 且提供 model_name 時，會自動推斷缺失的參數。

        Args:
            params (Optional[int]): 模型的參數數量 (例如, 70 * 10^9)。若未提供且 auto_infer=True，會從 model_name 推斷。
            flops_per_inference (Optional[float]): 每次推理的浮點運算次數 (FLOPs)。若提供，會比內部估算更準確。
            model_type (str): 模型類型，影響 activation 估算。可選: "text-generation-hf", "text-generation-ollama", "vlm", "asr-hf", "vad-hf", "ocr-hf", "cnn", "detection", "other"。
            precision (str): 模型權重和計算的精度。可選: "fp32", "fp16", "bf16", "int8"。
            context_tokens (int): Transformer 模型處理的上下文 token 數量。
            output_tokens (int): Transformer 模型生成的 token 數量。
            use_kv_cache (bool): Transformer 是否使用 KV 緩存，會顯著增加 activation 內存。
            image_resolution (Optional[Tuple[int, int]]): CNN 或檢測模型的輸入圖像解析度 (H, W)。
            batch_size (int): 推理時的批次大小。
            gpu_peak_tflops (float): 目標 GPU 的理論峰值 TFLOPS，用於估算延遲。
            supports_int8 (bool): 模型是否支持 INT8 量化。
            measured_weights_bytes (Optional[int]): 如果已知模型權重的精確大小 (bytes)，可直接傳入以覆蓋估算。
            measured_activations_bytes (Optional[int]): 如果已知 activation 的精確大小 (bytes)，可直接傳入以覆蓋估算。
            notes (Optional[List[str]]): 任何需要附加到結果中的額外註記。
            # 智能推斷參數
            model_name (Optional[str]): 模型名稱，用於自動推斷參數（當 auto_infer=True 時）。
            task (Optional[str]): 任務類型，用於推斷 model_type（當 auto_infer=True 時）。
            engine_type (Optional[str]): 引擎類型，僅用於附加信息。
            auto_infer (bool): 是否啟用智能參數推斷。預設為 True。
        Returns:
            Dict: 一個包含所有估算資源指標的字典，可以作為模型的元數據或標籤。
            example:
            {
                "params": 70000000000,
                "flops_per_inference": 1.5e14,
                "precision": "fp16",
                "weights_vram_gb": 28.0,
                "activations_vram_gb": 12.5,
                "runtime_overhead_gb": 0.5,
                "estimated_total_vram_gb": 41.0,
                "estimated_latency_s": 1.5,
                "estimated_throughput_qps": 0.67,
                "recommended_batch_size": 1,
                "notes": ["Supports INT8 quantization"]
            }
        """
        # 智能參數推斷邏輯
        notes = notes or []
        assumptions = []
        
        if auto_infer:
            # 1) 推斷參數數量
            if params is None and model_name:
                params = self._infer_params_from_name(model_name)
                if params is None:
                    assumptions.append("未能從名稱推斷參數，使用預設值 1B")
                    params = int(1e9)
                else:
                    assumptions.append(f"從名稱推斷參數約為 {params/1e9:.2f}B")
            
            # 2) 推斷 model_type (僅當使用預設值且提供了 task 時)
            if task and model_type == "text-generation-hf":
                # 簡化的任務到模型類型映射
                if task in ["text-generation-ollama", "vlm", "asr-hf", "vad-hf", "ocr-hf"]:
                    model_type = task
                    assumptions.append(f"根據任務 '{task}' 推斷模型類型為 '{model_type}'")
        
        # 將推斷假設添加到註記中
        if assumptions:
            notes.extend(assumptions)
        
        precision = precision.lower()
        
        # 對於 Ollama 模型，根據量化級別調整精度位元組數
        if model_type == "text-generation-ollama" and quantization_level:
            pbytes = self._get_ollama_precision_bytes(quantization_level, precision)
            if quantization_level:
                notes.append(f"使用 Ollama 量化級別 '{quantization_level}' 調整精度為 {pbytes} bytes")
        else:
            pbytes = self.PRECISION_BYTES.get(precision, 2)

        # 1) 權重 (weights) 的 VRAM 估算 (bytes)
        if measured_weights_bytes is not None:
            weights_bytes = measured_weights_bytes
        elif params is not None:
            weights_bytes = int(params * pbytes)
        else:
            weights_bytes = None

        # 2) 每次推理的 FLOPs 估算
        if flops_per_inference is not None:
            flops = float(flops_per_inference)
        else:
            # 文字生成任務（HF Transformers 和 Ollama）
            if model_type in ("text-generation-hf", "text-generation-ollama", "vlm") and params is not None:
                flops = 2.0 * params * (context_tokens + output_tokens) * batch_size
            # 語音相關任務
            elif model_type in ("asr-hf", "vad-hf") and params is not None:
                # ASR/VAD 通常處理音頻序列，估算基於音頻長度
                audio_frames = context_tokens  # 假設 context_tokens 代表音頻幀數
                flops = 1.5 * params * audio_frames * batch_size
            # OCR 任務
            elif model_type == "ocr-hf" and image_resolution and params:
                H, W = image_resolution
                flops = params * (H * W) * 0.8 * batch_size  # OCR 係數
            # 傳統 CNN 和檢測模型
            elif model_type in ("cnn", "detection") and image_resolution and params:
                H, W = image_resolution
                k = 0.5  # Heuristic coefficient
                flops = params * (H * W) * k * batch_size
            else:
                est_tokens = max(context_tokens + output_tokens, 1024)
                flops = (2.0 * (params or 1) * est_tokens * batch_size) if params else None

        # 3) Activations 的 VRAM 估算 (bytes)
        if measured_activations_bytes is not None:
            activations_bytes = measured_activations_bytes
        else:
            # 文字生成任務（包括 HF 和 Ollama）和 VLM 任務
            if model_type in ("text-generation-hf", "text-generation-ollama", "vlm") and params is not None:
                hidden_dim = max(64, int((params ** 0.5) * 0.7))
                activations_bytes = int(batch_size * (context_tokens + output_tokens) * hidden_dim * pbytes)
                if use_kv_cache:
                    kv_extra = int(batch_size * context_tokens * hidden_dim * pbytes * 2)
                    activations_bytes += kv_extra
            # 語音任務 (ASR/VAD)
            elif model_type in ("asr-hf", "vad-hf") and params is not None:
                # 語音模型通常有較小的 activation
                hidden_dim = max(32, int((params ** 0.4) * 0.5))
                audio_frames = context_tokens
                activations_bytes = int(batch_size * audio_frames * hidden_dim * pbytes)
            # OCR 任務
            elif model_type == "ocr-hf" and image_resolution and params:
                H, W = image_resolution
                # OCR 模型通常需要處理圖像特徵
                feature_channels = max(128, int((params ** 0.3) * 100))
                activations_bytes = int(batch_size * (H // 4) * (W // 4) * feature_channels * pbytes)
            # 傳統 CNN 和檢測模型
            elif model_type in ("cnn", "detection") and image_resolution:
                H, W = image_resolution
                avg_channels = 256
                activations_bytes = int(batch_size * H * W * avg_channels * pbytes)
            else:
                activations_bytes = int(50 * 1024 * 1024)  # 50 MB fallback

        # 4) 總 VRAM 估算 (bytes)
        runtime_overhead_bytes = int(500 * 1024 * 1024)  # 預留 500 MB for runtime buffers
        total_vram_bytes = None
        if weights_bytes is not None and activations_bytes is not None:
            total_vram_bytes = weights_bytes + activations_bytes + runtime_overhead_bytes
            
        # 為 Ollama 模型添加說明註釋，但仍然計算 VRAM 佔用
        if model_type == "text-generation-ollama":
            notes.append("Ollama 模型運行在外部服務，但仍需佔用 GPU VRAM")

        # 5) 延遲與吞吐量估算
        estimated_latency_s = None
        estimated_throughput_qps = None
        if flops is not None and gpu_peak_tflops > 0:
            theoretical_s = flops / (gpu_peak_tflops * 1e12)
            # Ollama 模型需要考慮額外的網路通信延遲
            if model_type == "text-generation-ollama":
                slowdown_factor = 7.0  # 更高的延遲因子，考慮網路通信
                network_latency = 0.05  # 假設 50ms 的網路延遲
                estimated_latency_s = theoretical_s * slowdown_factor + network_latency
            else:
                slowdown_factor = 5.0  # Conservative slowdown factor
                estimated_latency_s = theoretical_s * slowdown_factor
            
            if estimated_latency_s > 0:
                estimated_throughput_qps = batch_size / estimated_latency_s

        # 6) 輸出組裝
        def _to_gb(b: Optional[int]) -> Optional[float]:
            return (b / (1024**3)) if b is not None else None

        result = {
            "params": params,
            "flops_per_inference": flops,
            "precision": precision,
            "weights_vram_gb": _to_gb(weights_bytes),
            "activations_vram_gb": _to_gb(activations_bytes),
            "runtime_overhead_gb": _to_gb(runtime_overhead_bytes),
            "estimated_total_vram_gb": _to_gb(total_vram_bytes),
            "estimated_latency_s": estimated_latency_s,
            "estimated_throughput_qps": estimated_throughput_qps,
            "recommended_batch_size": batch_size,
            "notes": notes
        }
        
        # 添加額外信息（如果提供）
        if model_name:
            result["model_name"] = model_name
        if task:
            result["task"] = task
        if engine_type:
            result["engine_type"] = engine_type
        if auto_infer and assumptions:
            result["input_assumptions"] = assumptions
            
        return result

    def select_best_model(
        self, 
        models_meta: List[Dict[str, Any]], 
        *,
        available_gpu_vram_gb: float,
        max_latency_s: Optional[float] = None,
        min_throughput_qps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        從一組模型元數據中，根據當前資源限制選擇最優模型。

        Args:
            models_meta (List[Dict[str, Any]]): 一個字典列表，每個字典是 `estimate_resources` 的輸出，
                                                且必須額外包含一個 `model_id` 鍵。
            available_gpu_vram_gb (float): 當前 GPU 可用的 VRAM (GB)。
            max_latency_s (Optional[float]): 可接受的最大延遲 (秒)。
            min_throughput_qps (Optional[float]): 要求的最小吞吐量 (每秒查詢數)。

        Returns:
            Dict[str, Any]: 得分最高的模型的元數據字典，如果沒有符合條件的模型，則返回一個包含錯誤信息的字典。
        """
        candidates = []
        for m in models_meta:
            total_vram = self._safe_get(m, "estimated_total_vram_gb")
            if total_vram is not None and total_vram > (available_gpu_vram_gb * 0.95): # 留 5% margin
                continue

            # Latency score
            lat = self._safe_get(m, "estimated_latency_s", 9999.0)
            if max_latency_s is not None and lat > max_latency_s:
                continue # 硬性排除
            latency_score = 1.0 / (lat + 0.1) # 簡單的反比分數

            # Throughput score
            th = self._safe_get(m, "estimated_throughput_qps", 0.0)
            if min_throughput_qps is not None and th < min_throughput_qps:
                continue # 硬性排除
            throughput_score = th

            # VRAM score (越少越好)
            vram_score = 1.0 / (total_vram + 0.1) if total_vram is not None else 0

            # 權重可根據業務需求調整
            w_v, w_l, w_t = 0.4, 0.4, 0.2
            score = w_v * vram_score + w_l * latency_score + w_t * throughput_score
            candidates.append((score, m))

        if not candidates:
            return {"error": "No viable models found for the current resource constraints."}

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_model = candidates[0]
        
        result = dict(best_model)
        result["_selection_score"] = best_score
        return result

    # === 簡化接口：僅提供模型名稱與少量參數時的 VRAM 估算 ===
    def _infer_params_from_name(self, model_name: str) -> Optional[int]:
        """根據常見命名慣例從模型名稱中推斷參數量 (非常粗略)。

        規則：
        - 7b / 7B -> 7 * 1e9
        - 13b -> 13 * 1e9
        - 70b -> 70 * 1e9
        - 1.5b -> 1.5 * 1e9
        - 270m / 270M -> 270 * 1e6
        - fastvlm-0.5b 等亦適用
        回傳 None 表示無法推斷。
        """
        name = model_name.lower()
        
        # 遍歷字符串，尋找數字和單位的組合
        i = 0
        while i < len(name):
            # 跳過非數字字符
            if not name[i].isdigit():
                i += 1
                continue
            
            # 提取數字部分（只支持標準的小數點格式）
            number_str = ""
            j = i
            has_decimal = False
            
            while j < len(name):
                char = name[j]
                if char.isdigit():
                    number_str += char
                elif char == '.' and not has_decimal and j + 1 < len(name) and name[j + 1].isdigit():
                    # 允許一個小數點，後面必須跟數字（如 1.5b）
                    number_str += '.'
                    has_decimal = True
                elif char == ',':
                    # 跳過千位分隔符（如果有）
                    pass
                else:
                    # 遇到其他字符（包括下劃線），停止數字提取
                    break
                j += 1
            
            # 檢查數字後面是否緊接著單位 b 或 m
            if number_str and j < len(name):
                unit = name[j]
                
                try:
                    number = float(number_str)
                    
                    # B 單位 (十億)
                    if unit == 'b':
                        # 驗證數字範圍合理性（0.1B - 999B）
                        if 0.1 <= number <= 999:
                            return int(number * 1e9)
                    
                    # M 單位 (百萬)
                    elif unit == 'm':
                        # 驗證數字範圍合理性（10M - 9999M）
                        if 10 <= number <= 9999:
                            return int(number * 1e6)
                    
                except ValueError:
                    pass
            
            # 繼續搜索下一個數字
            i = j if j > i else i + 1
        
        return None

    # 注意：estimate_model_vram 函數已整合到 estimate_resources 中
    # 使用 estimate_resources(model_name="模型名稱", auto_infer=True, ...) 來獲得相同功能

    def estimate_and_prepare_tags(self, model_info: Dict[str, Any]) -> Dict[str, str]:
        """
        接收模型的基本信息，調用估算器，並將結果格式化為適合存儲為 MLflow 標籤的字符串字典。

        Args:
            model_info (Dict[str, Any]): 一個包含模型基本信息的字典，例如:
                {
                    "params": 70000000000,
                    "model_type": "text-generation-hf",
                    "precision": "fp16",
                    ...
                }
                此處的鍵應與 `estimate_resources` 的參數對應。

        Returns:
            Dict[str, str]: 一個鍵值均為字符串的字典，可直接用於 MLflow 的 `tags` 參數。
        """
        # 從 model_info 中提取 `estimate_resources` 所需的參數
        estimation_args = {
            k: self._safe_get(model_info, k) for k in [
                "params", "flops_per_inference", "model_type", "precision",
                "context_tokens", "output_tokens", "use_kv_cache", "image_resolution",
                "batch_size", "gpu_peak_tflops", "supports_int8", 
                "measured_weights_bytes", "measured_activations_bytes", "notes",
                "quantization_level"
            ]
        }
        # 過濾掉值為 None 的參數，避免傳遞給估算器
        estimation_args = {k: v for k, v in estimation_args.items() if v is not None}
        
        meta = self.estimate_resources(**estimation_args)
        
        # 將所有估算結果轉換為字符串，並加上前綴，以便存儲為標籤
        tags = {}
        for key, value in meta.items():
            if value is not None:
                # 對於浮點數，格式化為小數點後4位
                if isinstance(value, float):
                    tags[f"est_{key}"] = f"{value:.4f}"
                # 對於列表，轉換為逗號分隔的字符串
                elif isinstance(value, list):
                     tags[f"est_{key}"] = ",".join(map(str, value))
                else:
                    tags[f"est_{key}"] = str(value)
        
        return tags

# --- 使用範例 ---
if __name__ == '__main__':
    estimator = ModelResourceEstimator()

    # 範例 1: 直接估算（提供完整參數）
    direct_estimate = estimator.estimate_resources(
        params=7 * 10**9,
        model_type="text-generation-hf",
        precision="fp16",
        context_tokens=4096,
        output_tokens=512,
        gpu_peak_tflops=90.0,  # 假設在一張 A100 上
        auto_infer=False  # 關閉自動推斷
    )
    
    print("--- 直接估算結果 ---")
    import json
    print(json.dumps(direct_estimate, indent=2))
    
    # 範例 1.5: 智能推斷估算（僅提供模型名稱）
    smart_estimate = estimator.estimate_resources(
        model_name="llama2-7b-chat",
        task="text-generation-hf",
        precision="fp16",
        auto_infer=True  # 啟用智能推斷
    )
    
    print("\n--- 智能推斷估算結果 ---")
    print(json.dumps(smart_estimate, indent=2))
    
    # 範例 1.6: 準備 MLflow 標籤
    llm_7b_info = {
        "params": 7 * 10**9,
        "model_type": "text-generation-hf",
        "precision": "fp16",
        "context_tokens": 4096,
        "output_tokens": 512,
        "gpu_peak_tflops": 90.0
    }
    
    llm_7b_tags = estimator.estimate_and_prepare_tags(llm_7b_info)
    
    print("\n--- 7B LLM MLflow 標籤 ---")
    print(json.dumps(llm_7b_tags, indent=2))
    # 這些標籤可以直接在註冊模型時傳入 inference_metadata

    print("\n" + "="*30 + "\n")
    
    # 範例 2: 智能推斷 Ollama 模型（外部服務但仍佔用 GPU VRAM）
    ollama_estimate = estimator.estimate_resources(
        model_name="llama2-7b-chat",
        task="text-generation-ollama",
        engine_type="ollama",
        precision="fp16",
        auto_infer=True
    )
    
    print("--- Ollama 智能推斷結果 ---")
    print(json.dumps(ollama_estimate, indent=2))
    
    # 範例 2.5: 準備 Ollama MLflow 標籤
    ollama_7b_info = {
        "params": 7 * 10**9,
        "model_type": "text-generation-ollama",
        "precision": "fp16",
        "context_tokens": 4096,
        "output_tokens": 512
    }
    
    ollama_7b_tags = estimator.estimate_and_prepare_tags(ollama_7b_info)
    
    print("\n--- Ollama 7B LLM MLflow 標籤 ---")
    print(json.dumps(ollama_7b_tags, indent=2))
    
    print("\n" + "="*30 + "\n")
    
    # 範例 3: 從多個模型中選擇最優
    model_A_meta = {
        "model_id": "model_A_fast",
        "estimated_total_vram_gb": 14.0,
        "estimated_latency_s": 0.5,
        "estimated_throughput_qps": 2.0
    }
    model_B_meta = {
        "model_id": "model_B_large_accurate",
        "estimated_total_vram_gb": 18.0,
        "estimated_latency_s": 1.5,
        "estimated_throughput_qps": 0.6
    }
    model_C_meta = {
        "model_id": "model_C_super_large",
        "estimated_total_vram_gb": 30.0,
        "estimated_latency_s": 3.0,
        "estimated_throughput_qps": 0.3
    }
    
    available_vram = 20.0 # GB
    print(f"--- 模型選擇 (可用 VRAM: {available_vram} GB) ---")
    
    best_choice = estimator.select_best_model(
        [model_A_meta, model_B_meta, model_C_meta],
        available_gpu_vram_gb=available_vram
    )
    
    print("最佳選擇:")
    print(json.dumps(best_choice, indent=2))
    # model_C 會因為 VRAM 超出而被排除
    # model_A 和 model_B 會根據分數選擇，通常延遲低、VRAM 佔用少的 model_A 分數會更高