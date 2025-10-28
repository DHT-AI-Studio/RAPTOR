import fitz  # PyMuPDF
import uuid
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import logging
import io
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import os

logger = logging.getLogger(__name__)

class PDFOCRProcessor:
    def __init__(self, vlm_annotator=None, max_chunk_tokens: int = 400):
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
        # 使用共享的 VLM 實例或創建新的
        if vlm_annotator is not None:
            logger.info("使用共享的 VLM 實例")
            self.vlm_annotator = vlm_annotator
            # 從共享實例獲取模型相關屬性
            self.model = self.vlm_annotator.model
            self.model_tokenizer = self.vlm_annotator.model_tokenizer
            self.generation_config = self.vlm_annotator.generation_config
            self.IMAGENET_MEAN = self.vlm_annotator.IMAGENET_MEAN
            self.IMAGENET_STD = self.vlm_annotator.IMAGENET_STD
        else:
            logger.info("創建新的 VLM 實例")
            # 為了向後兼容，如果沒有傳入 vlm_annotator，就創建一個
            from processors.vlm_annotator import VLMAnnotator
            self.vlm_annotator = VLMAnnotator()
            self.model = self.vlm_annotator.model
            self.model_tokenizer = self.vlm_annotator.model_tokenizer
            self.generation_config = self.vlm_annotator.generation_config
            self.IMAGENET_MEAN = self.vlm_annotator.IMAGENET_MEAN
            self.IMAGENET_STD = self.vlm_annotator.IMAGENET_STD
        
        logger.info("PDF處理器初始化完成")
    
    def _build_transform(self, input_size=448):
        """建立圖片變換"""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
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
    
    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """動態預處理圖片 - 適用於 OCR"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 計算目標比例
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # 找到最接近的長寬比
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # 計算目標寬度和高度
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # 調整圖片大小
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # 分割圖片
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == blocks
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images
    
    def _call_local_model(self, image, prompt):
        """調用本地模型進行 OCR - 現在使用共享的 VLM 實例"""
        try:
            logger.info("正在調用本地模型...")
            
            # 確保圖片是 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 使用動態預處理
            input_size = 448
            max_num = 12  # 對於 OCR，限制分割數量
            
            transform = self._build_transform(input_size=input_size)
            images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)
            
            # 移動到正確的設備和數據類型
            pixel_values = pixel_values.to(self.model.device).to(torch.bfloat16)
            
            # 計算 num_patches_list
            num_patches_list = [pixel_values.shape[0]]
            
            # 調用模型的 chat 方法
            response, history = self.model.chat(
                self.model_tokenizer, 
                pixel_values, 
                prompt, 
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None, 
                return_history=True
            )
            
            logger.info(f"本地模型回應長度: {len(response)}")
            return response
            
        except Exception as e:
            logger.error(f"本地模型調用時發生錯誤: {e}")
            logger.error(f"錯誤詳情: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"完整錯誤堆疊: {traceback.format_exc()}")
            return f"OCR處理失敗: {str(e)}"
    
    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        if not text:
            return 0
            
        # max_chars = 1500
        # if len(text) > max_chars:
        #     text = text[:max_chars]
        
        tokens = self.tokenizer.encode(
            text, 
            truncation=True, 
            max_length=512,
            add_special_tokens=False
        )
        return len(tokens)
        
    def _extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        """從圖片字節數據中提取文字"""

        prompt = """請提取圖片中的所有可見文字，包括：
- 標題、內文、小字
- 表格內容
- 頁眉、頁腳文字
- 任何角落的小字
- 無視重複的浮水印內容

保持原始排版結構，使用 Markdown 格式：
- 如果為標題用 # ## ###
- 如果為表格用 | 分隔
- 如果為列表用 - 或數字
- 如果是小字直接顯示

Remind: Extract ALL visible text from this image

如果完全沒有文字，回覆：'NO_TEXT_FOUND'"""
        
# """Extract ALL visible text from this image, including small text in headers, footers, and margins. preserving the original layout and formatting.

# # Requirements:
# # - Maintain original text structure and spacing
# # - For large, bold, or prominent text that serves as titles/headers, format as:
# #   - Main titles: # Title
# #   - Section headers: ## Header  
# #   - Sub-headers: ### Sub-header
# # - For tables, use simple Markdown table format: | Column | format
# # - For lists, use - or numbered format
# # - Extract ALL visible text from this image
# # 並且無視重複的浮水印內容。
# # 避免在輸出中不必要地重複使用符號，但保留必要的 Markdown 格式符號（如 #、|、-）用於結構化文字。避免重複換行符號(\n)、多餘空格、製表符(\t)等。
# # No additional explanation. 
# # If there is absolutely no readable text in the image, respond with exactly: 'NO_TEXT_FOUND'"""
        
        # 將字節數據轉換為 PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        response = self._call_local_model(image, prompt)
        
        if response.strip() == "NO_TEXT_FOUND":
            return ""
        return response
    
    def _describe_image_bytes(self, image_bytes: bytes) -> str:
        """描述圖片內容"""
        prompt = "Briefly describe the visual layout and key graphical elements of this image. Focus on: images, charts, diagrams, colors, and overall design. Maximum 20 words in Traditional Chinese. No additional explanation."
        
        # 將字節數據轉換為 PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return self._call_local_model(image, prompt)
    
    def _pdf_to_image_bytes(self, pdf_path: Path) -> list:
        """將PDF每頁轉換為圖片字節數據，不保存到磁碟"""
        image_bytes_list = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # 轉換為圖片
                mat = fitz.Matrix(2.0, 2.0)  # 放大倍數
                pix = page.get_pixmap(matrix=mat)
                
                # 轉換為字節數據，不保存到磁碟
                image_bytes = pix.tobytes("png")
                image_bytes_list.append({
                    "page_num": page_num + 1,
                    "image_bytes": image_bytes
                })
                
                logger.info(f"已轉換第 {page_num + 1} 頁")
            
            pdf_document.close()
            return image_bytes_list
            
        except Exception as e:
            logger.error(f"PDF轉換失敗: {str(e)}")
            return []
    
    def _process_single_page_bytes(self, image_bytes: bytes, page_num: int) -> dict:
        """處理單一頁面的圖片字節數據"""
        logger.info(f"正在處理第 {page_num} 頁...")
        
        # 第一次調用：提取文字
        extracted_text = self._extract_text_from_image_bytes(image_bytes)
        
        # 第二次調用：描述圖片
        description = self._describe_image_bytes(image_bytes)
        
        # 合併內容
        combined_text = f"{extracted_text} <IMAGE>{description}</IMAGE>".strip()
        
        return {
            "page_number": page_num,
            "content": combined_text,
            "token_count": self.count_tokens(combined_text)
        }
    
    def _merge_pages_by_tokens(self, page_data_list: list) -> list:
        """根據token數量合併頁面"""
        chunks = []
        current_chunk = {
            "content": "",
            "page_numbers": [],
            "token_count": 0
        }
        
        for page_data in page_data_list:
            page_tokens = page_data["token_count"]
            
            if current_chunk["token_count"] + page_tokens <= self.max_chunk_tokens:
                if current_chunk["content"]:
                    current_chunk["content"] += " " + page_data["content"]
                else:
                    current_chunk["content"] = page_data["content"]
                
                current_chunk["page_numbers"].append(page_data["page_number"])
                current_chunk["token_count"] += page_tokens
            else:
                if current_chunk["content"]:
                    chunks.append(current_chunk.copy())
                
                current_chunk = {
                    "content": page_data["content"],
                    "page_numbers": [page_data["page_number"]],
                    "token_count": page_tokens
                }
        
        if current_chunk["content"]:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_document(self, pdf_path: str) -> dict:
        """處理PDF文件，返回JSON格式結果，不創建臨時文件"""
        pdf_path = Path(pdf_path)
        filename = pdf_path.stem
        upload_time = datetime.now().isoformat()
        
        try:
            # 1. PDF轉圖片字節數據（不保存到磁碟）
            logger.info("開始轉換PDF為圖片...")
            image_data_list = self._pdf_to_image_bytes(pdf_path)
            
            if not image_data_list:
                return {"error": "PDF轉換失敗"}
            
            # 2. 處理每一頁
            logger.info("開始處理每一頁...")
            page_data_list = []
            for image_data in image_data_list:
                page_data = self._process_single_page_bytes(
                    image_data["image_bytes"], 
                    image_data["page_num"]
                )
                page_data_list.append(page_data)
            
            # 3. 根據token數量合併頁面
            logger.info("開始合併頁面...")
            merged_chunks = self._merge_pages_by_tokens(page_data_list)
            
            # 4. 生成最終JSON格式
            chunks = []
            document_id = pdf_path.stem  
            filename = pdf_path.name     
            source = filename.split('.')[-1].lower()
            upload_time = datetime.now().isoformat()
            for chunk_index, chunk_data in enumerate(merged_chunks):
                chunk = {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "document_id": f"{filename}_chunk_{chunk_index}",
                        "type": "documents",
                        "text": chunk_data["content"],
                        "filename": filename,
                        "source": "pdf",
                        "chunk_index": chunk_index,
                        "page_numbers": chunk_data["page_numbers"],
                        "element_types": ["text", "image"],
                        "char_count": len(chunk_data["content"]),
                        "upload_time": upload_time,
                        "embedding_type": "text"
                    }
                }
                chunks.append(chunk)
            
            logger.info(f"處理完成，共生成 {len(chunks)} 個chunks")
            return {"chunk": chunks}
            
        except Exception as e:
            logger.error(f"處理PDF時發生錯誤: {str(e)}")
            return {"error": str(e)}
