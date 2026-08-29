"""
Train LLM using RAPTOR Framework API

This script implements the training workflow using RAPTOR's REST API
instead of direct transformers/peft calls.

API endpoints: http://<raptor_host_ip>:8009/docs
MLflow tracking: http://<raptor_host_ip>:5000/
Redis cache: http://<raptor_host_ip>:5540/
TensorBoard: http://<raptor_host_ip>:6006/
"""

import requests
import time

# Configuration
AIML_BASE_URL = "http://<raptor_host_ip>:8010"
TRAINING_BASE_URL = "http://<raptor_host_ip>:8009"

# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-1.7B"
# MODEL_NAME = "google/gemma-3-270m-it"
MODEL_NAME = "google/gemma-3-1b-it"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

DATASET_NAME = "yentinglin/TaiwanChat"  # Has standard 'train' split
# DATASET_NAME = "Mxode/Chinese-Reasoning-Distil-Data"
# DATASET_NAME = "yentinglin/twllm-data"

# EXPERIMENT_NAME = "llama-training-test"
# EXPERIMENT_NAME = "qwen2.5-training-test"
# EXPERIMENT_NAME = "qwen3-training-test"
EXPERIMENT_NAME = "gemma3-training-test"

MAX_SEQ_LENGTH = 1024
VRAM_BUDGET_GB = 25


def download_model():
    """Step 1: Download model from HuggingFace"""
    print("Downloading model...")
    response = requests.post(
        f"{AIML_BASE_URL}/models/download",
        json={
            "model_source": "huggingface",
            "model_name": MODEL_NAME
        }
    )
    response.raise_for_status()
    result = response.json()
    print(f"Model downloaded: {result}")
    # API returns path in 'details' field
    return result.get("details") or result.get("model_path", MODEL_NAME)

def download_dataset():
    """Step 2: Download dataset from HuggingFace"""
    print("Downloading dataset...")
    response = requests.post(
        f"{AIML_BASE_URL}/datasets/download_from_network",
        json={
            "dataset_name": DATASET_NAME,
            "dataset_source": "huggingface",
            "extract_multimedia": False
        }
    )
    response.raise_for_status()
    result = response.json()
    print(f"Dataset downloaded: {result}")
    # API returns path in 'local_path' field
    return result.get("local_path") or result.get("dataset_path", DATASET_NAME)

def submit_training_job(model_path: str, dataset_path: str):
    """Step 3: Submit training job with LoRA configuration"""
    print("Submitting training job...")

    submission = {
        "task_type": "instruction", # "instruction" or "text"
        "select_multiple_gpus": True,
        "vram_budget_gb": VRAM_BUDGET_GB,
        "training_config": {
            "model_name_or_path": model_path,
            "use_bfloat16": True,
            "use_flash_attn": False,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "max_epochs": 50,
            "batch_size": 1,
            "learning_rate": 2e-5,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "logging_steps": 25,
            "val_check_interval": 1.0, # 每 1.0 epoch 驗證一次
            "gradient_checkpointing": True,
            "max_grad_norm": 0.3,
            "use_8bit_adamw": True, # 使用 8-bit AdamW
            "experiment_name": EXPERIMENT_NAME,
            "lora_config": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "modules_to_save": None
            },
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",        # 通常 nf4 比 fp4 效果好
                "bnb_4bit_use_double_quant": True,   # 雙重量化，進一步節省記憶體
                "bnb_4bit_compute_dtype": "bfloat16" # 計算時使用的精度 (建議與 use_bfloat16 一致)
            },
            "deepspeed_config": {
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                    "stage3_gather_16bit_weights_on_model_save": False
                },
                "fp16": {
                    "enabled": False
                },
                "bf16": {
                    "enabled": True
                },
                "activation_checkpointing": {
                    "enabled": True,
                    "contiguous_memory_optimization": True,
                    "cpu_checkpointing": False,
                    "partition_activations": True
                }
            },
        },
        "dataset_config": {
            "dataset_name_or_path": DATASET_NAME,  # Use HuggingFace name
            "default_system_prompt": None,
            "train_size": 200,
            "val_ratio": None,
            "val_size": 50,
            "max_length": MAX_SEQ_LENGTH,
            "cache_dir": dataset_path,  # Downloaded path as cache
            "train_split_name": "train",        # 指定split
            # "val_split_name": "validation",   # 若無指定，則會根據val_ratio或val_size自動切割train_split。（優先級：val_ratio > val_size）
            "column_mapping": {
                "messages": "messages",    # 用於instruction，完整對話欄位，格式如 [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "response"}]。
                # "context": "context",			# 用於instruction，如文章或背景資訊，模型在回答前會先讀這段內容。
                # "reasoning": "reasoning",     # 用於instruction，如模型思考過程。
                # "input": "prompt",			# 用於instruction，如提問或指令。
                # "output": "response",			# 用於instruction，如模型期望的回答。
                # "text": "text"				# 用於text generation
            }
        }
    }

    response = requests.post(
        f"{TRAINING_BASE_URL}/api/v1/training/submit",
        json=submission
    )
    response.raise_for_status()
    result = response.json()
    job_id = result.get("job_id")
    print(f"Training job submitted: {job_id}")
    return job_id


def monitor_training(job_id: str):
    """Step 4: Monitor training progress"""
    print(f"\nMonitoring training job: {job_id}")
    print("=" * 50)
    print(f"MLflow UI: http://<raptor_host_ip>:5000/")
    print(f"Redis cache: http://<raptor_host_ip>:5540/")
    print("=" * 50 + "\n")

    while True:
        response = requests.get(f"{TRAINING_BASE_URL}/api/v1/training/status/{job_id}")
        response.raise_for_status()
        status = response.json()

        state = status.get("status", "unknown")
        metrics = status.get("metrics") or {}

        progress = metrics.get("progress_percentage", 0)
        train_loss = metrics.get("train_loss_epoch") or metrics.get("train_loss") or "N/A"
        val_loss = metrics.get("val_loss", "N/A")

        if state == "running":
            print(f"Status: {state} | Progress: {progress}% | Train Loss: {train_loss} | Val Loss: {val_loss}")
        elif state == "completed":
            print(f"Status: {state} | Train Loss: {train_loss} | Val Loss: {val_loss}")

        if state in ["completed", "failed", "cancelled"]:
            print(f"\nTraining {state}!")
            if state == "completed":
                model_path = status.get('model_path') or metrics.get('final_model_path', 'N/A')
                print(f"Model saved to: {model_path}")
            elif state == "failed":
                error_msg = metrics.get('error', 'Unknown error')
                print(f"Error: {error_msg}")
            break

        time.sleep(30)  # Check every 30 seconds

    return status


def list_jobs():
    """List all training jobs"""
    response = requests.get(f"{TRAINING_BASE_URL}/api/v1/training/list")
    response.raise_for_status()
    return response.json()


def cancel_job(job_id: str):
    """Cancel a running training job"""
    response = requests.post(f"{TRAINING_BASE_URL}/api/v1/training/cancel/{job_id}")
    response.raise_for_status()
    return response.json()


def main():
    """Main training workflow"""
    print("=" * 50)
    print(f"RAPTOR Training: {MODEL_NAME} Fine-tuning")
    print("=" * 50 + "\n")

    try:
        # Step 1: Download model
        model_path = download_model()

        # Step 2: Download dataset
        dataset_path = download_dataset()

        # Step 3: Submit training job
        job_id = submit_training_job(model_path, dataset_path)

        # Step 4: Monitor training
        final_status = monitor_training(job_id)

        print("\n" + "=" * 50)
        print("Training workflow complete!")
        print("=" * 50)

        return final_status

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        raise
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        raise


if __name__ == "__main__":
    main()
