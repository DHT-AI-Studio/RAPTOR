from transformers import AutoModelForCausalLM
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)

# 檢查 GPU 是否支援 FP16
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("FP16 supported:", torch.cuda.get_device_capability(0)[0] >= 7)

# 嘗試將模型轉 FP16
try:
    model.half().to(device)
    print("Model successfully converted to FP16")
except Exception as e:
    print("FP16 conversion failed:", e)
