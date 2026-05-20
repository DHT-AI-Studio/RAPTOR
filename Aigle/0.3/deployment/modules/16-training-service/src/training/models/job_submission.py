from pydantic import BaseModel, Field
from typing import Literal
from ..models.trainer_config import TrainerConfig, DatasetConfig


class TrainingJobSubmission(BaseModel):
    """
    Training job submission model.
    """
    task_type: Literal["instruction", "text"] = Field(
        ...,
        description="Task type: 'instruction' (Instruction Generation) or 'text' (Text Generation)"
    )

    select_multiple_gpus: bool = Field(
        ...,
        description="Whether to select multiple GPUs for training"
    )

    vram_budget_gb: float = Field(
        ...,
        gt=0,
        description="Estimated VRAM requirement (GB) for training and resource management"
    )

    training_config: TrainerConfig = Field(
        ...,
        description="Complete training configuration"
    )

    dataset_config: DatasetConfig = Field(
        ...,
        description="Complete dataset configuration"
    )

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "task_type": "instruction",
                "vram_budget_gb": 25,
                "select_multiple_gpus": True,
                "training_config": {
                    "model_name_or_path": "/app/tmp/models/google_gemma-3-1b-it",
                    "use_bfloat16": True,
                    "use_flash_attn": False,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.03,
                    "quantization_config": {
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_compute_dtype": "bfloat16"
                    },
                    "lora_config": {
                        "r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                        "bias": "none",
                        "modules_to_save": None
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
                    "max_epochs": 50,
                    "batch_size": 1,
                    "learning_rate": 2e-5,
                    "gradient_accumulation_steps": 4,
                    "warmup_steps": 100,
                    "use_8bit_adamw": True,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_epsilon": 1e-8,
                    "logging_steps": 5,
                    "val_check_interval": 0.5,
                    "experiment_name": "gemma_finetune"
                },
                "dataset_config": {
                    "dataset_name_or_path": "yentinglin/TaiwanChat",
                    "default_system_prompt": None,
                    "train_size": 200,
                    "val_ratio": None,
                    "val_size": 50,
                    "max_length": 1024,
                    "cache_dir": "/app/tmp/datasets/yentinglin_TaiwanChat",
                    "column_mapping": {
                        "messages": "messages"      
                    },
                    "train_split_name": "train",
                    "val_split_name": None
                }
            }
        }