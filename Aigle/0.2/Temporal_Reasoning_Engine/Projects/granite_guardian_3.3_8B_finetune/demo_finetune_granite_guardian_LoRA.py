from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch

# 選擇模型
model_name = "ibm-granite/granite-guardian-3.2-5b"

# Tokenizer 與模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 適用大部分 Transformer LM
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 小型示範資料集
data = [
    {"prompt": "Classify the policy: This document violates internal data policy.", 
     "completion": " [Data Privacy]"},
    {"prompt": "Classify the policy: All employees must follow HR onboarding procedures.", 
     "completion": " [HR Policy]"},
    {"prompt": "Classify the policy: Server access should be logged and audited.", 
     "completion": " [Security Policy]"},
]

dataset = Dataset.from_list(data)

# Tokenize
def tokenize_fn(example):
    # 將 prompt 與 completion 合併，轉成模型輸入
    full_input = example["prompt"] + example["completion"]
    inputs = tokenizer(full_input, truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # Causal LM 的 label 與 input_ids 相同
    return inputs

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

# Trainer 設定
training_args = TrainingArguments(
    output_dir="./granite_guardian_lora_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=5,
    #fp16=True,             # 使用半精度
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# 開始訓練
trainer.train()
