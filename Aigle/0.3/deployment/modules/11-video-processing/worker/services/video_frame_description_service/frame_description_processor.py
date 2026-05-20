# services/video_frame_description_service/frame_description_processor.py

import os
import opencc
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import logging
import json
import aiofiles
import glob
from typing import Dict, Any, List, Tuple, Optional
from decord import VideoReader, cpu
import numpy as np
from config import (
    MODEL_PATH,
    MAX_MEMORY_PER_GPU,
    GENERATION_CONFIG,
    INPUT_SIZE,
    MAX_NUM_PATCHES,
    FRAME_DESCRIPTION_RESULTS_DIR,
    CUDA_VISIBLE_DEVICES,
    VLM_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class FrameDescriptionProcessor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        self.cc = opencc.OpenCC('s2tw')
        self._initialize_model()

        # 確保輸出目錄存在
        os.makedirs(FRAME_DESCRIPTION_RESULTS_DIR, exist_ok=True)
    
    def _initialize_model(self):
        """初始化模型和分詞器"""
        try:
            logger.info(f"Initializing VLM model: {MODEL_PATH}")

            # 清理 CUDA 快取
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 模型路徑和配置（由 config.MODEL_PATH 讀取，docker-compose env 控制）
            path = MODEL_PATH
            num_gpus = torch.cuda.device_count()
            max_memory = {i: MAX_MEMORY_PER_GPU for i in range(num_gpus)}
            
            logger.info(f"Available GPUs: {num_gpus}")
            logger.info(f"Memory allocation per GPU: {max_memory}")
            
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
            
            # 載入分詞器
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            # 檢查模型載入後的記憶體使用
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Model loaded - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def build_transform(self, input_size):
        """建立圖像轉換管道"""
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """尋找最接近的長寬比"""
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
        """動態預處理圖像，將圖像分割成多個區塊"""
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

    def load_image(self, image_file, input_size=448, max_num=12):
        """載入單張圖片"""
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        """獲取視頻幀索引"""
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        """載入視頻幀"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def load_video_with_scene_frames(self, video_path, scene_frames, input_size=448, max_num=1):
        """
        Load video frames specifically at scene change points.
        
        Args:
            video_path: Path to video file
            scene_frames: List of scene change frame data
            input_size: Input size for processing
            max_num: Maximum number of patches per frame
        
        Returns:
            Tuple of (pixel_values, num_patches_list)
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        
        # Extract frame indices from scene change data
        frame_indices = []
        for scene_frame in scene_frames:
            frame_idx = scene_frame.get('frame_index')
            if frame_idx is not None and 0 <= frame_idx <= max_frame:
                frame_indices.append(frame_idx)
        
        # Limit to reasonable number of frames to avoid memory issues
        frame_indices = frame_indices[:16]  # Max 16 frames
        
        # If no valid frame indices, fall back to temporal sampling
        if not frame_indices:
            return self.load_video(video_path, num_segments=8, max_num=max_num)
        
        for frame_index in frame_indices:
            try:
                img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
                img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
                pixel_values = [transform(tile) for tile in img]
                pixel_values = torch.stack(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                pixel_values_list.append(pixel_values)
            except Exception as e:
                logger.warning(f"Error processing frame {frame_index}: {e}")
                continue
        
        # If no frames were successfully processed, fall back to temporal sampling
        if not pixel_values_list:
            return self.load_video(video_path, num_segments=8, max_num=max_num)
        
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def generate_text_with_visual(self, video_path, question='請詳細的描述這段影片的內容。', scene_frames=None):
        """
        Generate text response for video using VLM.
        
        Args:
            video_path: Path to video file
            question: Question about the video
            scene_frames: Optional list of scene change frames for targeted analysis
        
        Returns:
            Generated text response
        """
        if scene_frames and len(scene_frames) > 0:
            # Use scene change frames for more targeted analysis
            pixel_values, num_patches_list = self.load_video_with_scene_frames(
                video_path, scene_frames, max_num=1
            )
        else:
            # Use default temporal sampling
            pixel_values, num_patches_list = self.load_video(video_path, num_segments=8, max_num=1)
        
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        prompt = video_prefix + question
        response, history = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config,
                                   num_patches_list=num_patches_list, history=None, return_history=True)
        return response

    def generate_text(self, prompt):
        """
        Generate text response using VLM.
        
        Args:
            prompt: Input text prompt
        
        Returns:
            Generated text response
        """
        response, history = self.model.chat(self.tokenizer, None, prompt, self.generation_config,
                                   num_patches_list=None, history=None, return_history=True)
        return response

    def load_images_from_directory(
        self, frame_dir: str, max_frames: int = 16, input_size: int = 448, max_num: int = 1
    ):
        """Load pre-extracted keyframe images from scene directory (replaces decord video reading)."""
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "scene_frame_*.jpg")))

        if not frame_files:
            raise ValueError(f"No scene_frame_*.jpg found in {frame_dir}")

        if len(frame_files) > max_frames:
            indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]

        pixel_values_list, num_patches_list = [], []
        for frame_file in frame_files:
            try:
                pv = self.load_image(frame_file, input_size=input_size, max_num=max_num)
                num_patches_list.append(pv.shape[0])
                pixel_values_list.append(pv)
            except Exception as e:
                logger.warning(f"Skipping frame {frame_file}: {e}")

        if not pixel_values_list:
            raise ValueError(f"No frames could be loaded from {frame_dir}")

        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def generate_vlm_response(
        self,
        frame_dir: str,
        scene_frames: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate event summary using pre-extracted keyframe images from scene directory.
        Uses frame_dir instead of decord video reading.
        """
        prompt = '''
        <background>
        你現在是一位媒體編輯
        </background> \n
        <Goal>
        任務是從影片中提取「事件摘要」。
        事件摘要是對影片中發生的主要事件或核心陳述的描述。
        </Goal> \n
        <Rules>
        1. 只描述影片的主要事件或核心陳述。
        2. 不要提及人物外貌、服裝、顏色、背景細節。
        3. 專注於事件本身，避免描述次要細節。
        4. 請用繁體中文回答，可以有多個事件摘要，依照重要程度排序。
        </Rules> \n
        '''
        try:
            if not frame_dir:
                raise ValueError("frame_dir cannot be empty or None")

            logger.info(f"Generating VLM response from keyframes in: {frame_dir}")

            pixel_values, num_patches_list = self.load_images_from_directory(frame_dir)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            full_prompt = video_prefix + prompt

            logger.info(f"Running VLM inference on {len(num_patches_list)} keyframes")
            response, _ = self.model.chat(
                self.tokenizer, pixel_values, full_prompt, self.generation_config,
                num_patches_list=num_patches_list, history=None, return_history=True
            )

            logger.info("VLM response generated successfully")
            metadata = {
                'response_generated': True,
                'keyframes_used': len(num_patches_list),
            }
            return response, metadata

        except Exception as e:
            error_msg = f"VLM generation failed: {str(e)}"
            logger.error(f"Error in VLM generation: {e}")
            raise RuntimeError(error_msg)

    async def process_frame_descriptions(
        self,
        scene_json_path: str,
        video_path: str,
        scene_output_directory: str,
        request_id: str,
        primary_filename: str
    ) -> str:
        """
        處理影片事件摘要
        使用場景幀數據生成事件摘要
        """
        try:
            # 讀取場景檢測結果
            async with aiofiles.open(scene_json_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                scene_data = json.loads(content)
            
            if not isinstance(scene_data, list):
                raise ValueError(f"Scene data should be a list, got {type(scene_data)}")
            
            logger.info(f"Processing event summary from {len(scene_data)} scene frames")
            
            if not os.path.exists(scene_output_directory):
                raise ValueError(f"Scene output directory not found: {scene_output_directory}")

            # Use pre-extracted keyframes from scene directory (no decord needed)
            response, metadata = self.generate_vlm_response(scene_output_directory, scene_data)
            
            # 準備結果數據
            result_data = {
                "event_summary": self.cc.convert(response),
                "scene_frames_data": scene_data
                # "processing_info": {
                #     "model": MODEL_PATH,
                #     "input_size": INPUT_SIZE,
                #     "max_patches": MAX_NUM_PATCHES,
                #     "scene_frames_processed": len(scene_data),
                #     "video_path": video_path,
                #     "processing_method": "video_frames_with_scene_data",
                #     "prompt_type": "event_summary"
                # },
                # "metadata": metadata
            }
            
            # 保存結果
            result_filename = f"{request_id}_event_summary.json"
            result_file_path = os.path.join(FRAME_DESCRIPTION_RESULTS_DIR, result_filename)
            
            async with aiofiles.open(result_file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result_data, indent=2, ensure_ascii=False))
            
            logger.info(f"Event summary processing completed: {result_file_path}")
            logger.info(f"Processing method: video_frames_with_scene_data")
            logger.info(f"Processed {len(scene_data)} scene frames")

            return result_file_path

        except Exception as e:
            logger.error(f"Event summary processing failed: {e}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Per-moment VLM methods (Raptor 0.3)
    # ──────────────────────────────────────────────────────────────────────────

    def load_images_from_paths(
        self, keyframe_paths: List[str], input_size: int = 448, max_num: int = 1
    ):
        """Load images from an explicit list of file paths."""
        pixel_values_list, num_patches_list = [], []
        for path in keyframe_paths:
            if not os.path.exists(path):
                logger.warning(f"Keyframe not found: {path}")
                continue
            try:
                pv = self.load_image(path, input_size=input_size, max_num=max_num)
                num_patches_list.append(pv.shape[0])
                pixel_values_list.append(pv)
            except Exception as e:
                logger.warning(f"Skipping keyframe {path}: {e}")
        if not pixel_values_list:
            raise ValueError("No keyframes could be loaded from the provided paths")
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def _moment_prompt(self, start_sec: float, end_sec: float) -> str:
        return f"""
<background>
你現在是一位媒體編輯
</background>
<Goal>
任務是描述這段影片片段（{start_sec:.1f}秒到{end_sec:.1f}秒）的核心內容。
</Goal>
<Rules>
1. 只描述影片片段的主要事件或核心陳述。
2. 不要提及人物外貌、服裝、顏色、背景細節。
3. 專注於事件本身，避免描述次要細節。
4. 請用繁體中文回答，簡短精確。
</Rules>
"""

    def generate_moment_descriptions_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Run InternVL batch_chat on a list of moments.
        Each item: {"keyframe_paths": [...], "start_time": float, "end_time": float}
        Returns list of description strings in the same order.
        """
        pixel_values_list = []
        num_patches_list_per_moment = []
        questions = []
        valid_indices = []

        for i, item in enumerate(batch):
            try:
                pv, npl = self.load_images_from_paths(item["keyframe_paths"])
                pixel_values_list.append(pv.to(torch.bfloat16).cuda())
                num_patches_list_per_moment.append(npl)
                video_prefix = "".join([f"Frame{j+1}: <image>\n" for j in range(len(npl))])
                questions.append(video_prefix + self._moment_prompt(item["start_time"], item["end_time"]))
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Skipping moment in batch (index {i}): {e}")

        if not pixel_values_list:
            return [""] * len(batch)

        # InternVL batch_chat: concatenate all frames, one total-patches count per sample
        pixel_values_cat = torch.cat(pixel_values_list, dim=0)
        num_patches_per_sample = [sum(npl) for npl in num_patches_list_per_moment]

        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values_cat,
            num_patches_list=num_patches_per_sample,
            questions=questions,
            generation_config=self.generation_config,
        )

        results = [""] * len(batch)
        for resp, idx in zip(responses, valid_indices):
            results[idx] = resp
        return results

    async def process_moments_descriptions(
        self,
        moments: List[Dict[str, Any]],
        request_id: str,
        primary_filename: str,
    ) -> str:
        """
        Generate per-moment VLM descriptions for all moments using batch_chat.
        Saves a JSON file compatible with result_merger format.
        Returns path to the saved JSON file.
        """
        import asyncio

        moment_descriptions: List[Dict[str, Any]] = []

        loop = asyncio.get_event_loop()
        for batch_start in range(0, len(moments), VLM_BATCH_SIZE):
            batch_moments = moments[batch_start:batch_start + VLM_BATCH_SIZE]

            batch_items = []
            no_keyframe_indices = set()
            for i, moment in enumerate(batch_moments):
                kf = moment.get("keyframe_paths", [])
                if not kf:
                    logger.warning(f"No keyframes for moment {moment.get('moment_index', 0)}, skipping VLM")
                    no_keyframe_indices.add(i)
                    batch_items.append({"keyframe_paths": [], "start_time": 0, "end_time": 0})
                else:
                    batch_items.append({
                        "keyframe_paths": kf,
                        "start_time": float(moment.get("start_time", 0.0)),
                        "end_time": float(moment.get("end_time", 0.0)),
                    })

            try:
                responses = await loop.run_in_executor(
                    None, self.generate_moment_descriptions_batch, batch_items
                )
            except Exception as e:
                logger.warning(f"Batch VLM failed (batch {batch_start}): {e}")
                responses = [""] * len(batch_moments)

            for i, (moment, response) in enumerate(zip(batch_moments, responses)):
                moment_index = moment.get("moment_index", 0)
                start_sec = float(moment.get("start_time", 0.0))
                end_sec = float(moment.get("end_time", 0.0))
                description = self.cc.convert(response) if response else ""
                moment_descriptions.append({
                    "moment_index": moment_index,
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "lvlm_description": description,
                })
                logger.info(f"VLM description for moment {moment_index} ({start_sec:.1f}s-{end_sec:.1f}s)")

        # event_summary is generated by video_summary_service (OCR+ASR+LVLM); skip here
        result_data = {
            "event_summary": "",
            "moment_descriptions": moment_descriptions,
            "scene_frames_data": [],
        }

        result_filename = f"{request_id}_event_summary.json"
        result_file_path = os.path.join(FRAME_DESCRIPTION_RESULTS_DIR, result_filename)

        async with aiofiles.open(result_file_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(result_data, indent=2, ensure_ascii=False))

        logger.info(
            f"Per-moment VLM complete: {len(moment_descriptions)} moments → {result_file_path}"
        )
        return result_file_path
