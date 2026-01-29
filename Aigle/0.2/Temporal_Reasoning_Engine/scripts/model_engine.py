import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class QwenVisionEngine:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"🔧 [VisionEngine] Initializing model: {model_id} (4-bit)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except ImportError:
            print("⚠️ Flash Attention not found, falling back to default.")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("✅ [VisionEngine] Model loaded successfully.")

    def analyze(self, video_path, prompt_text, fps=1.0, max_pixels=360*420):
        if not prompt_text:
            prompt_text = (
                "You are an expert video analyst. Analyze strictly based on visual evidence.\n"
                "Report in Traditional Chinese (繁體中文). Do NOT output markdown code blocks (```json), just plain text.\n\n"
                
                "1. **關鍵物件與狀態 (Key Objects & States)**:\n"
                "   - List active agents/objects. Do not count unless relevant.\n\n"
                
                "2. **微觀動作分解 (Micro-Action Breakdown)**:\n"
                "   - Chronological steps. Note hand interactions.\n\n"
                
                "3. **總結 (Summary)**:\n"
                "   - Synthesize the event.\n\n"
                
                "4. **前因後果 (Causal Analysis)**:\n"
                "   - Explain logic.\n\n"
                
                "5. **精準時序紀錄 (Precise Event Log)**:\n"
                "   - List key actions with their **relative timestamp** from the start of this clip.\n"
                "   - Strict Format: `[+seconds] Action description`\n"
                "   - Example:\n"
                "     [+01s] The woman reaches for the bottle.\n"
                "     [+03s] She pours water into the glass."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": max_pixels, 
                        "fps": fps, 
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]