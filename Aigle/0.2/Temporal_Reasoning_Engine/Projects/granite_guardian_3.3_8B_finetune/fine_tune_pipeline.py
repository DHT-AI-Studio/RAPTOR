# fine_tune_pipeline.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

def main():
    # 模型名稱
    model_name = "ibm-granite/granite-guardian-3.2-5b"

    # Tokenizer 與模型載入（使用 LoRA 微調）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 設定 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 資料載入：假設你有 CSV/JSON 格式，包含 prompt + completion 欄位
    dataset_path = "path/to/your_dataset.json"  # 修改為你的資料集路徑
    data = load_dataset("json", data_files={"train": dataset_path, "validation": dataset_path})  # 可分 train/val
    def tokenize_fn(example):
        full_input = example["prompt"] + example["completion"]
        inputs = tokenizer(full_input, truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized = data.map(tokenize_fn, batched=True)

    # 訓練參數
    training_args = TrainingArguments(
        output_dir="./granite_guardian_lora_finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"]
    )

    # 開始訓練
    trainer.train()

    # 儲存模型與 tokenizer
    trainer.save_model("./granite_guardian_lora_finetuned")
    tokenizer.save_pretrained("./granite_guardian_lora_finetuned")

    # 測試推論
    prompt = "Classify the policy: All employees must change passwords every 90 days."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generation = model.generate(**inputs, max_new_tokens=20)
    print("Prediction:", tokenizer.decode(generation[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
