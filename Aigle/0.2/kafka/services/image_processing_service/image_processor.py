# services/image_processing_service/image_processor.py

import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import logging
import json
from typing import Optional
from config import (
    MODEL_PATH, CUDA_VISIBLE_DEVICES, MAX_MEMORY_PER_GPU, 
    DESCRIPTION_PROMPT, OCR_PROMPT, TEMP_FILE_DIR
)
import opencc
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.dtype = None
        self.generation_config = dict(max_new_tokens=2048, do_sample=True)
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.converter = opencc.OpenCC('s2t')
        
    def initialize_model(self):
        """初始化模型 - 只初始化一次，供兩種功能共用"""
        try:
            # 設置 CUDA 設備
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
            
            logger.info(f"Loading shared model from: {MODEL_PATH}")
            logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
            
            # 設置記憶體配置
            num_gpus = torch.cuda.device_count()
            max_memory = {i: MAX_MEMORY_PER_GPU for i in range(num_gpus)}
            
            # 載入模型
            self.model = AutoModel.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory
            ).eval()
            
            # 載入 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            # 獲取模型的設備和數據類型
            self.device = next(self.model.parameters()).device
            self.dtype = next(self.model.parameters()).dtype
            
            logger.info(f"Shared model initialized successfully on device: {self.device}, dtype: {self.dtype}")
            
        except Exception as e:
            logger.error(f"Failed to initialize shared model: {e}")
            raise
    
    def build_transform(self, input_size: int):
        """建立圖片轉換"""
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """找到最接近的長寬比"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """動態預處理圖片"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image_file: str, input_size: int = 448, max_num: int = 12):
        """載入並處理圖片"""
        try:
            image = Image.open(image_file).convert('RGB')
            transform = self.build_transform(input_size=input_size)
            images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            
            # 確保張量移動到正確的設備和數據類型
            if self.device is not None and self.dtype is not None:
                pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
                logger.debug(f"Moved pixel_values to device: {self.device}, dtype: {self.dtype}")
            else:
                logger.warning("Model device/dtype not available, using default")
                if torch.cuda.is_available():
                    pixel_values = pixel_values.cuda().to(dtype=torch.bfloat16)
            
            return pixel_values
        except Exception as e:
            logger.error(f"Error loading image {image_file}: {e}")
            raise
    
    def generate_description(self, image_path: str, prompt: str = None) -> str:
        """生成圖片描述"""
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not initialized")
            
            # 使用預設 prompt 如果沒有提供
            if not prompt:
                prompt = DESCRIPTION_PROMPT
            
            logger.info(f"Generating description for: {image_path}")
            # 載入圖片
            pixel_values = self.load_image(image_path)
            
            logger.debug(f"Processing with pixel_values shape: {pixel_values.shape}")
            
            # 使用 torch.no_grad() 來節省記憶體並確保推理模式
            with torch.no_grad():
                # 清理 GPU 記憶體
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 生成描述
                response = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    prompt, 
                    self.generation_config
                )
                response = self.converter.convert(response)
            
            logger.info(f"Generated description for image: {image_path}")
            logger.debug(f"Description preview: {response[:100]}...")
            
            # 創建結果數據
            result_data = {
                "description": response,
                "image_path": image_path,
                "prompt": prompt,
                "timestamp": self._get_current_timestamp(),
                "action": "image_description"
            }
            
            # 保存結果到臨時檔案
            result_file_path = self._save_result(result_data, image_path, "description")
            
            # 清理記憶體
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result_file_path
            
        except Exception as e:
            logger.error(f"Error generating description for {image_path}: {e}")
            # 清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def extract_text(self, image_path: str, prompt: str = None) -> str:
        """提取圖片中的文字"""
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not initialized")
            
            # 使用預設 OCR prompt 如果沒有提供
            if not prompt:
                prompt = OCR_PROMPT
            
            logger.info(f"Extracting text from: {image_path}")
            # 載入圖片
            pixel_values = self.load_image(image_path)
            
            logger.debug(f"Processing OCR with pixel_values shape: {pixel_values.shape}")
            
            # 使用 torch.no_grad() 來節省記憶體並確保推理模式
            with torch.no_grad():
                # 清理 GPU 記憶體
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 提取文字
                response = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    prompt, 
                    self.generation_config
                )
                response = self.converter.convert(response)
            
            logger.info(f"Extracted text from image: {image_path}")
            logger.debug(f"Text preview: {response[:100]}...")
            
            # 創建結果數據
            result_data = {
                "extracted_text": response,
                "image_path": image_path,
                "prompt": prompt,
                "timestamp": self._get_current_timestamp(),
                "action": "image_ocr"
            }
            
            # 保存結果到臨時檔案
            result_file_path = self._save_result(result_data, image_path, "ocr")
            
            # 清理記憶體
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result_file_path
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            # 清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    
    def _save_result(self, result_data: dict, image_path: str, result_type: str) -> str:
        """保存處理結果到檔案"""
        try:
            # 從圖片路徑提取檔案名
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # 創建結果檔案路徑
            result_filename = f"{name_without_ext}_{result_type}_result.json"
            result_path = os.path.join(TEMP_FILE_DIR, result_filename)
            
            # 確保目錄存在
            os.makedirs(TEMP_FILE_DIR, exist_ok=True)
            
            # 保存結果
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"{result_type.title()} result saved to: {result_path}")
            return result_path
            
        except Exception as e:
            logger.error(f"Error saving {result_type} result: {e}")
            raise
    
    def _get_current_timestamp(self) -> str:
        """獲取當前時間戳"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
