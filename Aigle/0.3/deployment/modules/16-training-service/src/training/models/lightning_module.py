"""
PyTorch Lightning Module wrapper for LLM training.
"""
from pathlib import Path
import torch
import lightning as L
from typing import Any, Optional, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from lightning.pytorch.utilities import rank_zero_only
import logging
import time

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

from ..trainers.base_trainer import TrainerConfig


logger = logging.getLogger(__name__)


class LightningLLMModule(L.LightningModule):
    """
    PyTorch Lightning Module for LLM fine-tuning with LoRA support.
    """

    def __init__(
        self, 
        config: TrainerConfig,
        cancel_token: Any = None,
        cancel_check_interval: int = 3
    ):
        super().__init__()
        self.config = config
        self.cancel_token = cancel_token
        self._cancel_check_interval = cancel_check_interval
        self._last_cancel_check_time = 0

        # --- Quantization Setup ---
        bnb_config = None
        if self.config.quantization_config:
            logger.info(f"Initializing model with quantization config: {self.config.quantization_config}")
            q_config = self.config.quantization_config.copy()

            if q_config.get("bnb_4bit_compute_dtype") == "bfloat16":
                q_config["bnb_4bit_compute_dtype"] = torch.bfloat16
            elif q_config.get("bnb_4bit_compute_dtype") == "float16":
                q_config["bnb_4bit_compute_dtype"] = torch.float16

            bnb_config = BitsAndBytesConfig(**q_config)

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, # Use config field
            dtype=torch.bfloat16 if config.use_bfloat16 else torch.float16,
            use_cache=False,
            # device_map="auto",
            device_map=None,
            attn_implementation="flash_attention_2" if config.use_flash_attn else None,
            quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            logger.warning("Pad token not found in tokenizer. It will add a new pad token '<PAD>'.")
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('<PAD>')

        self.tokenizer.padding_side = "right"

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id 

        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # --- QLoRA Preparation ---
        if config.quantization_config and config.lora_config:
            logger.info("Preparing model for k-bit training (QLoRA setup)...")
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=config.gradient_checkpointing
            )

        # Apply LoRA if configuration provided
        if config.lora_config:
            self._apply_lora(config.lora_config)

        # Save hyperparameters
        try:
            self.save_hyperparameters(ignore=['model', 'cancel_token'])
        except TypeError:
            from copy import copy
            self.save_hyperparameters(copy(vars(self.config)))

    def _check_for_cancellation(self):
        """Check if the cancellation token is set."""
        now = time.time()
        if now - self._last_cancel_check_time < self._cancel_check_interval:
            return

        self._last_cancel_check_time = now
        if self.cancel_token is not None and \
           hasattr(self.cancel_token, 'is_cancelled') and \
           callable(getattr(self.cancel_token, 'is_cancelled')):
            if self.cancel_token.is_cancelled():
                logger.info("Training canceled via cancel token.")
                if self.trainer is not None:
                    self.trainer.should_stop = True

    def _handle_non_finite_loss(self, loss: torch.Tensor, batch: Dict[str, torch.Tensor], batch_idx: int, stage: str):
        """
        Handles non-finite loss (NaN/Inf) by logging the error and the problematic sample, 
        and then raising a RuntimeError to stop the trainer.
        """
        if torch.isfinite(loss):
            return

        # 1. Get the first sample's input_ids
        first_input_ids = batch["input_ids"][0].cpu().tolist()
        
        # 2. Use tokenizer to decode
        problem_text_preview = "Decoding failed."
        try:
            problem_text = self.tokenizer.decode(
                first_input_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            # Truncate to 500 characters
            problem_text_preview = problem_text[:500] + ('...' if len(problem_text) > 500 else '')
        except Exception as e:
            problem_text_preview = f"Decoding failed: {e}"
        
        # 3. Log the error
        #  Ensure trainer is not None and optimizers are available 
        current_lr = "N/A"
        try:
            if self.trainer and self.trainer.optimizers:
                current_lr = f"{self.trainer.optimizers[0].param_groups[0]['lr']:.2e}"
        except Exception:
            pass # 忽略獲取 LR 的錯誤

        error_msg = (
            f"FATAL: Non-finite loss (NAN/INF) detected in {stage}_step at global step {self.global_step}, batch_idx {batch_idx}.\n"
            f"Current LR: {current_lr}. Please check learning rate and precision settings (e.g., use 'bf16-mixed').\n"
            f"--- Problematic Sample Preview (First in Batch) ---\n"
            f"{problem_text_preview}"
        )
        logger.error(error_msg)
        
        # 4. Raise RuntimeError to stop training
        raise RuntimeError(error_msg)

    def _apply_lora(self, lora_config: Dict[str, Any]) -> None:        
        """Apply LoRA adapters to the model."""
        logger.info(f"Applying LoRA with config: {lora_config}")

        peft_config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=lora_config.get('modules_to_save', None)
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("Trainable parameters information logged above.")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, use_cache: bool = False) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        self._check_for_cancellation()

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            use_cache=False
        )

        loss = outputs.loss
        self._handle_non_finite_loss(loss, batch, batch_idx, stage="training")

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        self._check_for_cancellation()
       
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
            use_cache=True
        )
                
        loss = outputs.loss
        self._handle_non_finite_loss(loss, batch, batch_idx, stage="validation")

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["input_ids"].size(0), sync_dist=True)
        return loss

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        # Optional: Log learning rate
        if hasattr(self.trainer.optimizers[0], "param_groups"):
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, on_epoch=True, prog_bar=True)

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        epoch_loss = self.trainer.callback_metrics.get('train_loss_epoch', -1)
        logger.info(f"Epoch {self.current_epoch} completed. Train loss: {epoch_loss:.4f}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and LR schedulers."""

        # Filter parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = None

        is_deepspeed = self.trainer.strategy.__class__.__name__ == "DeepSpeedStrategy"
        ds_offload = False

        if is_deepspeed and HAS_DEEPSPEED:
            # 檢查是否開啟了 CPU Offload
            # Lightning 的 DeepSpeedStrategy 會把 config 存在 config 屬性中
            ds_config = getattr(self.trainer.strategy, "config", {})
            if ds_config:
                # 檢查 zero_optimization.offload_optimizer.device
                zero_opt = ds_config.get("zero_optimization", {})
                offload_opt = zero_opt.get("offload_optimizer", {})
                if offload_opt.get("device") == "cpu":
                    ds_offload = True
                    
        if is_deepspeed and ds_offload:
            logger.info("Using DeepSpeedCPUAdam (CPU Offload enabled).")
            optimizer = DeepSpeedCPUAdam(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                adamw_mode=True,
                weight_decay=self.config.weight_decay
            )
        elif is_deepspeed and HAS_DEEPSPEED:
            logger.info("Using DeepSpeed FusedAdam.")
            optimizer = FusedAdam(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay
            )
        elif self.config.use_8bit_adamw:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
                logger.info("Using bnb.optim.AdamW8bit...")
                optimizer = optimizer_class(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed. Fallback to torch.optim.AdamW.")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay, 
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay, 
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )

        # Use estimated stepping batches (handles DDP + accumulation)
        total_steps = self.trainer.estimated_stepping_batches

        # Calculate warmup steps based on ratio or fixed value
        warmup_steps = 0
        if self.config.warmup_ratio is not None:
             warmup_steps = int(self.config.warmup_ratio * total_steps)
        else:
             warmup_steps = self.config.warmup_steps

        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self.model.eval()
        return self.model.generate(*args, **kwargs)

    @rank_zero_only
    @torch.no_grad()
    def save_hf_model(self, output_dir: str) -> None:
        """
        Merge trained LoRA weights into the base model and save the full Hugging Face model and tokenizer.
        """        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Merging LoRA weights and saving full model to {output_dir}")
        self.model.eval()

        try:
            if self.config.lora_config:
                try:
                    logger.info("Merging LoRA weights...")
                    merged_model = self.model.merge_and_unload()
                except Exception as merge_err:
                    if self.config.quantization_config:
                        logger.warning(f"Could not merge_and_unload (likely due to quantization): {merge_err}. Saving adapters only.")
                        self.model.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        logger.info("Adapters saved successfully (base model not merged).")
                        return
                    else:
                        raise merge_err
            else:
                merged_model = self.model
            
            merged_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            logger.info("Full Hugging Face model saved successfully.")

        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}", exc_info=True)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            keys_to_remove = [
                k for k in state_dict.keys() 
                if any(x in k for x in [
                    ".absmax", ".quant_map", ".nested_absmax", 
                    ".nested_quant_map", ".quant_state", "bitsandbytes", ".SCB"
                ])
            ]

            if keys_to_remove:
                logger.info(f"Filtering out {len(keys_to_remove)} quantization keys (4-bit/8-bit artifacts).")
                for k in keys_to_remove:
                    del state_dict[k]