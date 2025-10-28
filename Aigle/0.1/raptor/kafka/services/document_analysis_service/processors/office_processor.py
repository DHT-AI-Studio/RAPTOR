
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from io import BytesIO
from PIL import Image
import re
import os
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat


logger = logging.getLogger(__name__)

class VLMAnnotator:
    def __init__(self):
        self._init_local_model()
    
    def _init_local_model(self):
        """初始化本地 InternVL 模型"""
        try:
            # 設置 CUDA 設備
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            # 模型路徑和配置
            path = 'OpenGVLab/InternVL3_5-4B'
            num_gpus = torch.cuda.device_count()
            max_memory = {i: "36GiB" for i in range(num_gpus)}
            
            # 載入模型
            self.model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map="auto",
                max_memory=max_memory
            ).eval()
            
            self.model_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            
            # 生成配置
            self.generation_config = dict(
                max_new_tokens=2048, 
                do_sample=True,
                temperature=0.7
            )
            
            # 定義常數
            self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
            self.IMAGENET_STD = (0.229, 0.224, 0.225)
            
            logger.info("本地 InternVL 模型初始化完成")
            
        except Exception as e:
            logger.info(f"本地模型初始化失敗: {e}")
            raise
    
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
        """動態預處理圖片"""
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
    
    def describe_image(self, image):
        """使用本地模型描述圖片"""
        try:
            # 確保圖片是 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 使用動態預處理
            input_size = 448
            max_num = 12
            
            transform = self._build_transform(input_size=input_size)
            images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)
            
            # 移動到正確的設備和數據類型
            pixel_values = pixel_values.to(self.model.device).to(torch.bfloat16)
            
            # 計算 num_patches_list
            num_patches_list = [pixel_values.shape[0]]
            
            # 調用模型的 chat 方法
            prompt = "請只根據圖片，簡單一句話說明圖片內容，不要給額外解釋。"
            response, history = self.model.chat(
                self.model_tokenizer, 
                pixel_values, 
                prompt, 
                self.generation_config,
                num_patches_list=num_patches_list,
                history=None, 
                return_history=True
            )
            
            return response
            
        except Exception as e:
            logger.info(f"圖片描述生成失敗: {e}")
            return None

class OfficeDocumentProcessor:
    def __init__(self, vlm_annotator: VLMAnnotator = None, max_chunk_tokens: int = 400):
        self.vlm_annotator = vlm_annotator or VLMAnnotator()
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.setup_docling()
        
    def setup_docling(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        if not text:
            return 0
            
        # 預先截斷，避免超過模型限制
        max_chars = 1500  # 大約對應400-500 tokens
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # 使用 truncation 確保不超過限制
        tokens = self.tokenizer.encode(
            text, 
            truncation=True, 
            max_length=512,
            add_special_tokens=False
        )
        return len(tokens)
    
    def generate_context(self, element: Dict, document_title: str = None) -> str:
        """為元素生成上下文訊息"""
        return ""
    
    def extract_page_numbers(self, item: Dict) -> List[int]:
        page_numbers = []
        if "prov" in item:
            for prov in item["prov"]:
                if "page_no" in prov:
                    page_numbers.append(prov["page_no"])
        return list(set(page_numbers)) if page_numbers else [1]
        
    def extract_y_position(self, element: Dict) -> float:
        """提取元素的Y座標，返回用來排序的值"""
        try:
            item = element.get("_raw_item", element)  
            if "prov" in item:
                for prov in item["prov"]:
                    if "bbox" in prov:
                        bbox = prov["bbox"]
                        
                        if isinstance(bbox, dict) and "t" in bbox:
                            coord_origin = bbox.get("coord_origin", "TOPLEFT")
                            y_value = float(bbox["t"])
                            
                            if coord_origin == "BOTTOMLEFT":
                                return -y_value  # 反轉以實現從上到下排序
                            else:
                                return y_value
            
            return 0.0
        except Exception as e:
            logger.info(f"警告：提取Y座標時出錯: {e}")
            return 0.0
    
    def has_valid_grid(self, table_data: Dict) -> bool:
        """檢查docling處理後是否有包含grid"""
        try:
            data = table_data.get("data", {})
            grid = data.get("grid", [])
            return (
                isinstance(grid, list) and 
                len(grid) > 0 and 
                isinstance(grid[0], list) and
                len(grid[0]) > 0
            )
        except:
            return False

    def has_valid_cells(self, table_data: Dict) -> bool:
        """檢查docling處理後是否有包含cells"""
        try:
            data = table_data.get("data", {})
            return (
                "table_cells" in data and 
                isinstance(data["table_cells"], list) and
                len(data["table_cells"]) > 0 and
                "num_rows" in data and 
                "num_cols" in data and
                data["num_rows"] > 0 and 
                data["num_cols"] > 0
            )
        except:
            return False

    def extract_markdown_tables(self, markdown_text: str) -> List[str]:
        """從 Markdown 中提取表格"""
        table_pattern = r'(\|[^\n]*\|(?:\n\|[^\n]*\|)*)'
        tables = re.findall(table_pattern, markdown_text, re.MULTILINE)
        return [table.strip() for table in tables if table.strip() and '|' in table]
    
    def extract_tables_from_structured_data(self, doc_data: Dict, markdown_text: str) -> List[Dict]:
        """根據docling提取表格的資訊重建表格"""
        table_elements = []
        
        if "tables" not in doc_data:
            return table_elements
        
        markdown_tables = self.extract_markdown_tables(markdown_text)
        
        for i, table_data in enumerate(doc_data["tables"]):
            page_numbers = self.extract_page_numbers(table_data)
            formatted_table = ""
            
            # 嘗試從 grid 重建表格
            if self.has_valid_grid(table_data):
                try:
                    markdown_table = self.rebuild_table_from_grid(table_data["data"]["grid"])
                    formatted_table = f"<TABLE>\n{markdown_table.strip()}\n</TABLE>"
                except Exception:
                    pass
            
            # 如果 grid 失敗，嘗試從 cells 重建
            if not formatted_table and self.has_valid_cells(table_data):
                try:
                    markdown_table = self.rebuild_table_from_cells(table_data["data"])
                    formatted_table = f"<TABLE>\n{markdown_table.strip()}\n</TABLE>"
                except Exception:
                    pass
            
            # 如果都失敗，使用 markdown 中提取的表格
            if not formatted_table and i < len(markdown_tables):
                formatted_table = f"<TABLE>\n{markdown_tables[i].strip()}\n</TABLE>"
            
            if formatted_table and formatted_table.strip():
                table_elements.append({
                    "content": formatted_table,
                    "element_type": "table", 
                    "page_numbers": page_numbers,
                    "_raw_item": table_data
                })
        
        return table_elements
    
    def rebuild_table_from_grid(self, grid: List[List[Dict]]) -> str:
        markdown_lines = []
        
        for row_idx, row in enumerate(grid):
            row_texts = []
            for cell in row:
                text = cell.get("text", "").strip()
                text = self.clean_cell_text(text)
                row_texts.append(text)
            
            markdown_row = "| " + " | ".join(row_texts) + " |"
            markdown_lines.append(markdown_row)
            
            if row_idx == 0:
                separator = "| " + " | ".join(["---"] * len(row_texts)) + " |"
                markdown_lines.append(separator)
        
        return "\n".join(markdown_lines)

    def rebuild_table_from_cells(self, table_data: Dict) -> str:
        cells = table_data.get("table_cells", [])
        num_rows = table_data.get("num_rows", 0)
        num_cols = table_data.get("num_cols", 0)
        
        if not cells or num_rows == 0 or num_cols == 0:
            return ""
        
        table_matrix = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        
        for cell in cells:
            row_idx = cell.get("start_row_offset_idx", 0)
            col_idx = cell.get("start_col_offset_idx", 0)
            text = self.clean_cell_text(cell.get("text", ""))
            
            if 0 <= row_idx < num_rows and 0 <= col_idx < num_cols:
                table_matrix[row_idx][col_idx] = text
        
        markdown_lines = []
        for row_idx, row in enumerate(table_matrix):
            markdown_row = "| " + " | ".join(row) + " |"
            markdown_lines.append(markdown_row)
            
            if row_idx == 0:
                separator = "| " + " | ".join(["---"] * len(row)) + " |"
                markdown_lines.append(separator)
        
        return "\n".join(markdown_lines)
    
    def clean_cell_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace("|", "\\|")
        
        return text
    
    def format_image_description(self, description: str) -> str:
        """format vlm 生成的圖片描述"""
        if description and description.strip():
            return f"<IMAGE>\n{description.strip()}\n</IMAGE>"
        return ""
    
    def decode_base64_image(self, base64_data: str) -> Image.Image:
        try:
            if base64_data.startswith('data:image/'):
                base64_data = base64_data.split(',', 1)[1]
            
            import base64
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes))
            return image
        except Exception:
            return None
    
    def smart_chunk_text(self, elements: List[Dict], document_title: str = None) -> List[Dict]:
        chunks = []
        current_chunk_content = []
        current_chunk_types = []
        current_chunk_pages = []
        current_token_count = 0
        
        for element in elements:
            # 添加上下文
            context = self.generate_context(element, document_title)
            content_with_context = element["content"]
            
            element_type = element["element_type"]
            page_numbers = element["page_numbers"]
            
            # 使用 token 計數
            separator = "\n\n" if current_chunk_content else ""
            element_tokens = self.count_tokens(content_with_context)
            separator_tokens = self.count_tokens(separator) if separator else 0
            new_token_count = current_token_count + separator_tokens + element_tokens
            
            if new_token_count > self.max_chunk_tokens and current_chunk_content:
                # 若超過最大token即完成當前 chunk
                chunks.append({
                    "content": "\n\n".join(current_chunk_content),
                    "element_types": list(set(current_chunk_types)),
                    "page_numbers": sorted(set(current_chunk_pages)),
                    "token_count": current_token_count
                })
                
                # 開始新 chunk
                current_chunk_content = [content_with_context]
                current_chunk_types = [element_type]
                current_chunk_pages = page_numbers.copy()
                current_token_count = element_tokens
            else:
                current_chunk_content.append(content_with_context)
                current_chunk_types.append(element_type)
                current_chunk_pages.extend(page_numbers)
                current_token_count = new_token_count
        
        # 處理最後一個 chunk
        if current_chunk_content:
            chunks.append({
                "content": "\n\n".join(current_chunk_content),
                "element_types": list(set(current_chunk_types)),
                "page_numbers": sorted(set(current_chunk_pages)),
                "token_count": current_token_count
            })
        
        return chunks
    
    def process_document(self, file_path: Path) -> Dict[str, List[Dict]]:
        """處理 PDF/PPT/DOCX 文件"""
        conv_result = self.doc_converter.convert(file_path)
        doc_data = conv_result.document.export_to_dict()
        markdown_text = conv_result.document.export_to_markdown()
        
        # 提取文檔標題
        document_title = file_path.stem
        
        all_elements = []
        
        # 處理文字
        if "texts" in doc_data:
            for text_item in doc_data["texts"]:
                if text_item.get("text", "").strip():
                    all_elements.append({
                        "content": text_item["text"].strip(),
                        "element_type": "text",
                        "page_numbers": self.extract_page_numbers(text_item),
                        "_raw_item": text_item
                    })
        
        # 處理表格
        table_elements = self.extract_tables_from_structured_data(doc_data, markdown_text)
        all_elements.extend(table_elements)
        
        # 處理圖片
        if "pictures" in doc_data:
            for pic_item in doc_data["pictures"]:
                pil_image = None
                
                if "image" in pic_item and "uri" in pic_item["image"]:
                    pil_image = self.decode_base64_image(pic_item["image"]["uri"])
                
                if pil_image:
                    description = self.vlm_annotator.describe_image(pil_image)
                    if description:
                        formatted_description = self.format_image_description(description)
                        all_elements.append({
                            "content": formatted_description,
                            "element_type": "image",
                            "page_numbers": self.extract_page_numbers(pic_item),
                            "_raw_item": pic_item
                        })
        
        # 按頁碼和Y座標排序
        all_elements.sort(key=lambda x: (
            min(x["page_numbers"]) if x["page_numbers"] else 0,
            self.extract_y_position(x)
        ))
        
        # 進行上下文化分塊
        text_chunks = self.smart_chunk_text(all_elements, document_title)
        
        # 生成最終的 chunk 結構
        chunks = []
        document_id = file_path.stem
        filename = file_path.name
        source = filename.split('.')[-1].lower()
        upload_time = datetime.now().isoformat()
        
        for chunk_index, chunk_data in enumerate(text_chunks):
            chunk = {
                "id": str(uuid.uuid4()),
                "payload": {
                    "document_id": f"{document_id}_chunk_{chunk_index}",
                    "type": "documents", 
                    "text": chunk_data["content"],
                    "filename": filename,
                    "source": source,
                    "chunk_index": chunk_index,
                    "page_numbers": chunk_data["page_numbers"],
                    "element_types": chunk_data["element_types"],
                    "char_count": len(chunk_data["content"]),
                    "upload_time": upload_time,
                    "embedding_type": "text"
                }
            }
            chunks.append(chunk)
        
        return {"chunk": chunks}
