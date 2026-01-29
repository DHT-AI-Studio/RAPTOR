"""
沒有使用 LoRA 記憶體會爆
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch

model_name = "ibm-granite/granite-guardian-3.2-5b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

data = [
    {"prompt": "Classify the policy: This document violates internal data policy.", "completion": " [Data Privacy]"},
    {"prompt": "Classify the policy: All employees must follow HR onboarding procedures.", "completion": " [HR Policy]"},
    {"prompt": "Classify the policy: Server access should be logged and audited.", "completion": " [Security Policy]"},
]

dataset = Dataset.from_list(data)

def tokenize_fn(example):
    full_input = example["prompt"] + example["completion"]
    inputs = tokenizer(full_input, truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

training_args = TrainingArguments(
    output_dir="./granite_guardian_full_finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=5,
    # fp16=True, # 模型或訓練環境與 FP16 不完全兼容
    fp16=False,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
