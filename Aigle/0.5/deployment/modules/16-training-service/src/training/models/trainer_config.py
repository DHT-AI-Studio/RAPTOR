from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainerConfig:
    """Configuration for trainer initialization."""

    model_name_or_path: str = "google/gemma-3-1b-it"
    use_bfloat16: bool = True
    use_flash_attn: bool = False
    weight_decay: float = 0.01
    warmup_ratio: float | None = None
    seed: int = 42  # seeds all RNGs (weights init, dropout, dataloader) for reproducible runs

    quantization_config: dict[str, Any] | None = field(default_factory=dict) 
    lora_config: dict[str, Any] | None = field(default_factory=dict)
    
    # Basic training settings
    max_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"  # linear | cosine | constant (warmup still applies)
    optimizer_type: str = "adamw"  # adamw | lion | sgd (deepspeed paths stay Adam)
    use_8bit_adamw: bool = False  # Use bitsandbytes.optim.AdamW8bit
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Logging and checkpointing
    logging_steps: int = 10
    val_check_interval: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # GPU settings
    accelerator: str = "auto"
    devices: int = -1
    strategy: str = "auto"
    precision: str = "bf16-mixed"

    # Experiment tracking
    experiment_name: str = "llm-training"
    run_name: str | None = None

    # Advanced settings
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    deepspeed_stage: int | None = None  # 例如: 2 或 3
    deepspeed_offload_optimizer: bool = False # 是否將優化器狀態卸載到 CPU (節省 VRAM)
    deepspeed_offload_parameters: bool = False # 是否將模型參數卸載到 CPU (僅 Stage 3 有效)
    deepspeed_config: str | dict[str, Any] | None = None

@dataclass
class DatasetConfig:
    """Configuration for Dataset initialization."""
    dataset_name_or_path: str
    default_system_prompt: str | None = None
    max_length: int = 2048
    train_size: int | None = None
    val_size: int | None = None
    val_ratio: float | None = None
    cache_dir: str | None = None
    batched_map: bool = True
    num_workers: int = 4
    seed: int = 42
    column_mapping: dict[str, str] = field(default_factory=lambda: {
        "text": "text",
        "messages": "messages",
        "reasoning": "reasoning",
        "context": "context",       
        "input": "instruction",     
        "output": "output",         
    })
    
    train_split_name: str = "train"
    val_split_name: str | None = None