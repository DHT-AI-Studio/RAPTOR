"""
Compare Original vs Fine-tuned Model Outputs
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Configuration - paths to original and fine-tuned models
MODEL_DIR = "/opt/home/dht/mnt/aiml/tmp/models/"

ORIGINAL_MODEL = MODEL_DIR + "google_gemma-3-1b-it"
FINETUNED_MODEL = MODEL_DIR + "google_gemma-3-1b-it_gemma_finetune_07947b98f3a64d"


def load_models():
    """Load both original and fine-tuned models"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL,
        trust_remote_code=True
    )

    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL,
        dtype=torch.bfloat16,
        # device_map="auto",
        device_map={"": 0},
        trust_remote_code=True
    )

    print("Loading fine-tuned model...")
    # Model was saved as merged full model, load directly
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL,
        dtype=torch.bfloat16,
        # device_map="auto",
        device_map={"": 0},
        trust_remote_code=True
    )

    print("Models loaded successfully!\n")
    return tokenizer, original_model, finetuned_model


def verify_parameters(model_a, model_b):
    """
    確定模型是否訓練成功：詳細對比權重維度與數值
    """
    print("\n" + "=" * 60)
    print(" 🔎 [模型訓練成果校驗報告] ")
    print("=" * 60)
    
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    
    # 定義檢查點
    test_layers = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.15.mlp.gate_proj.weight",
        "lm_head.weight"
    ]
    
    print(f"{'檢查目標層級':<40} | {'比對結果'}")
    print("-" * 75)

    diff_count = 0
    for layer in test_layers:
        # 取得權重物件
        w_a = sd_a.get(layer)
        w_b = sd_b.get(layer)

        if w_a is None or w_b is None:
            # 嘗試模糊匹配 (有些微調會加上 base_model 前綴)
            layer_alt = next((k for k in sd_b.keys() if layer in k), None)
            w_b = sd_b.get(layer_alt) if layer_alt else None

        if w_a is not None and w_b is not None:
            shape_a = list(w_a.shape)
            shape_b = list(w_b.shape)

            if shape_a != shape_b:
                print(f"{layer:<40} | 🚀 【結構已改變】")
                print(f"  └─ Original: {shape_a}")
                print(f"  └─ Fine-tuned: {shape_b}")
                diff_count += 1
            else:
                # 形狀相同則比對數值
                mae = torch.abs(w_a.float() - w_b.float()).mean().item()
                if mae > 1e-9:
                    print(f"{layer:<40} | ✅ 【數值已優化】(MAE: {mae:.2e})")
                    diff_count += 1
                else:
                    print(f"{layer:<40} | 🧊 【保持凍結】 (相同)")
        else:
            print(f"{layer:<40} | ❓ 找不到此層")

    print("-" * 75)
    if diff_count > 0:
        print(f"🎉 驗證完成：偵測到 {diff_count} 處關鍵差異。")
        print("💡 結論：微調模型『確實不同』於原始模型，存檔正確！")
    else:
        print("⚠️ 警告：未偵測到任何差異，請重新檢查訓練流程。")
    print("=" * 60 + "\n")


def generate_response(model, tokenizer, prompt, max_new_tokens=2048):
    """Generate response from a model"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(prompt)}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=True,
            add_generation_prompt=True
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"[Error generating response: {e}]"


def compare_outputs(tokenizer, original_model, finetuned_model, prompt):
    """Compare outputs from both models"""
    print("=" * 60)
    print(f"INPUT: {prompt}")
    print("=" * 60)

    print("\n[ORIGINAL MODEL]:")
    print("-" * 40)
    original_response = generate_response(original_model, tokenizer, prompt)
    print(original_response)

    print("\n[FINE-TUNED MODEL]:")
    print("-" * 40)
    finetuned_response = generate_response(finetuned_model, tokenizer, prompt)
    print(finetuned_response)

    print("\n")
    return original_response, finetuned_response


def main():
    print("=" * 60)
    print("Model Comparison: Original vs Fine-tuned")
    print("=" * 60 + "\n")

    tokenizer, original_model, finetuned_model = load_models()

    verify_parameters(original_model, finetuned_model)

    # Test prompts
    # test_prompts = [
    #     "What is machine learning?",
    #     "How do I make fried rice?",
    # ]
    test_prompts = []

    print("Running comparison with test prompts...\n")
    for prompt in test_prompts:
        compare_outputs(tokenizer, original_model, finetuned_model, prompt)

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Your prompt: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input:
                compare_outputs(tokenizer, original_model, finetuned_model, user_input)
        except KeyboardInterrupt:
            break

    print("Goodbye!")


if __name__ == "__main__":
    main()
