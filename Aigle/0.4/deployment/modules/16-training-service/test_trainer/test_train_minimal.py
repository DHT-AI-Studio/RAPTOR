"""Minimal training smoke-test — confirm this machine can actually fine-tune.

Deliberately tiny + safe for a low-RAM box (no DeepSpeed CPU offload):
  - google/gemma-3-270m-it (already on the shared NFS)
  - LoRA + 4-bit quantization
  - 1 epoch, 30 train / 6 val samples, max_length 512
Mirrors train_with_raptor_api.py but points at localhost.
"""

import time

import requests

AIML_BASE_URL = "http://localhost:8010"
TRAINING_BASE_URL = "http://localhost:8009"

MODEL_PATH = "/app/tmp/models/google_gemma-3-270m-it"   # shared aiml NFS, already downloaded
DATASET_NAME = "philschmid/dolly-15k-oai-style"          # small, OpenAI "messages" format
EXPERIMENT_NAME = "gemma270m-smoketest"


def download_dataset() -> str:
    print(f"Downloading dataset {DATASET_NAME} ...")
    r = requests.post(
        f"{AIML_BASE_URL}/datasets/download_from_network",
        json={"dataset_name": DATASET_NAME, "dataset_source": "huggingface",
              "extract_multimedia": False},
        timeout=600,
    )
    r.raise_for_status()
    result = r.json()
    print(f"  → {result}")
    return result.get("local_path") or result.get("dataset_path", DATASET_NAME)


def submit(dataset_path: str) -> str:
    submission = {
        "task_type": "instruction",
        "select_multiple_gpus": False,
        "vram_budget_gb": 12,
        "training_config": {
            "model_name_or_path": MODEL_PATH,
            "use_bfloat16": True,
            "use_flash_attn": False,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "max_epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 2,
            "warmup_steps": 2,
            "logging_steps": 2,
            "val_check_interval": 1.0,
            "gradient_checkpointing": True,
            "max_grad_norm": 0.3,
            "use_8bit_adamw": True,
            "experiment_name": EXPERIMENT_NAME,
            "lora_config": {
                "r": 8, "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.05, "bias": "none", "modules_to_save": None,
            },
            "quantization_config": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            # NOTE: deepspeed_config deliberately omitted — CPU offload would
            # exhaust this box's limited RAM; a 270M LoRA run does not need it.
        },
        "dataset_config": {
            "dataset_name_or_path": DATASET_NAME,
            "default_system_prompt": None,
            "train_size": 30,
            "val_ratio": None,
            "val_size": 6,
            "max_length": 512,
            "cache_dir": dataset_path,
            "train_split_name": "train",
            "column_mapping": {"messages": "messages"},
        },
    }
    print("Submitting training job ...")
    r = requests.post(f"{TRAINING_BASE_URL}/api/v1/training/submit", json=submission, timeout=120)
    r.raise_for_status()
    job_id = r.json().get("job_id")
    print(f"  → job_id = {job_id}")
    return job_id


def monitor(job_id: str, max_minutes: int = 20):
    print(f"\nMonitoring {job_id} (up to {max_minutes} min) ...")
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        r = requests.get(f"{TRAINING_BASE_URL}/api/v1/training/status/{job_id}", timeout=30)
        r.raise_for_status()
        st = r.json()
        state = st.get("status", "unknown")
        m = st.get("metrics") or {}
        print(f"  status={state} | progress={m.get('progress_percentage','?')}% "
              f"| train_loss={m.get('train_loss_epoch') or m.get('train_loss','?')} "
              f"| val_loss={m.get('val_loss','?')}")
        if state in ("completed", "failed", "cancelled"):
            if state == "completed":
                print(f"\n✅ TRAINING COMPLETED — model: {st.get('model_path') or m.get('final_model_path')}")
            else:
                print(f"\n❌ TRAINING {state.upper()} — error: {m.get('error','?')}")
            return st
        time.sleep(15)
    print("\n⏳ timed out waiting (still running — check status manually)")
    return None


if __name__ == "__main__":
    ds = download_dataset()
    jid = submit(ds)
    monitor(jid)
