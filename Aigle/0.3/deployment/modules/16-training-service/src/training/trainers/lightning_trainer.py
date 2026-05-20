"""
PyTorch Lightning-based trainer implementation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import shutil

from .base_trainer import BaseTrainer
from ..models.trainer_config import TrainerConfig
from ..models.lightning_module import LightningLLMModule
from ..callbacks.ttc_progress_callback import TotalStepsLoggerCallback, TTCEWMAProgressCallback

logger = logging.getLogger(__name__) 


class LightningTrainer(BaseTrainer):
    """
    PyTorch Lightning implementation of the BaseTrainer interface.
    """
    def __init__(self, config: TrainerConfig, cancel_token: Any = None, mlflow_uri: Optional[str] = None):
        super().__init__(config)
        self.trainer: Optional[Trainer] = None
        self.module: Optional[L.LightningModule] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.cancel_token = cancel_token
        self.mlflow_uri = mlflow_uri
        self.checkpoint_path = Path(self.config.checkpoint_dir) / self.config.experiment_name / (self.config.run_name or "version_0")
        self._initialize_trainer()
        self._initialize_module()

    def _initialize_trainer(self) -> None:
        """Initializes the PyTorch Lightning Trainer instance and callbacks."""
        callbacks = self._create_callbacks()
        loggers = self._create_loggers()

        strategy = self.config.strategy

        if self.config.deepspeed_config or self.config.deepspeed_stage:
            logger.info("Configuring DeepSpeed Strategy...")
            
            # 如果提供了完整的字典或 json 路徑
            if self.config.deepspeed_config:
                keys_to_remove = [
                    "gradient_accumulation_steps", 
                    "train_batch_size", 
                    "train_micro_batch_size_per_gpu",
                    "gradient_clipping"
                ]

                for key in keys_to_remove:
                    if key in self.config.deepspeed_config:
                        logger.warning(f"Removing '{key}' from DeepSpeed config to allow Lightning to manage it.")
                        self.config.deepspeed_config.pop(key)

                strategy = DeepSpeedStrategy(config=self.config.deepspeed_config)
            
            # 否則使用簡化參數構建
            elif self.config.deepspeed_stage:
                stage = self.config.deepspeed_stage
                offload_optimizer = self.config.deepspeed_offload_optimizer
                offload_parameters = self.config.deepspeed_offload_parameters
                
                logger.info(f"DeepSpeed Stage: {stage}, Offload Optimizer: {offload_optimizer}")

                strategy = DeepSpeedStrategy(
                    stage=stage,
                    offload_optimizer=offload_optimizer,
                    offload_parameters=offload_parameters,
                    load_full_weights=True 
                )

        self.trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            strategy=strategy,
            precision=self.config.precision,
            callbacks=callbacks,
            logger=loggers,
            accumulate_grad_batches=self.config.gradient_accumulation_steps,
            gradient_clip_val=self.config.max_grad_norm,
            log_every_n_steps=self.config.logging_steps,
            val_check_interval=self.config.val_check_interval,
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=False,
            # use_distributed_sampler=True 
        )

        logger.info(f"LightningTrainer initialized with {self.config.devices} device(s)")

    def _initialize_module(self) -> None:
        """Initializes the LightningLLMModule."""
        self.module = LightningLLMModule(config=self.config, cancel_token=self.cancel_token)

        self.model = self.module.model
        self.tokenizer = self.module.tokenizer
        logger.info("LightningLLMModule model initialized.")

    def _create_callbacks(self) -> List[Callback]:
        """Creates the list of standard callbacks."""
        callbacks = []

        callbacks.append(TotalStepsLoggerCallback(self.training_state))
        callbacks.append(TTCEWMAProgressCallback(self.training_state))
        
        # LR monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Last Checkpoint
        last_checkpoint = ModelCheckpoint(
            dirpath=self.checkpoint_path,
            filename="last-manual-{epoch:02d}-{step}",
            monitor=None,
            save_top_k=0,
            save_last=True,
        )
        callbacks.append(last_checkpoint)

        return callbacks

    def _create_loggers(self) -> List:
        """Creates the list of loggers (TensorBoard, MLflow)."""
        loggers = []

        # TensorBoard
        tb_logger = TensorBoardLogger(
            save_dir=self.config.log_dir,
            name=self.config.experiment_name,
            version=self.config.run_name or "version_0",
            default_hp_metric=False,
        )
        loggers.append(tb_logger)

        if self.mlflow_uri:
            try:
                from lightning.pytorch.loggers import MLFlowLogger 
                
                mlflow_logger = MLFlowLogger(
                    experiment_name=self.config.experiment_name,
                    tracking_uri=self.mlflow_uri,
                    run_name=self.config.run_name,
                )
                loggers.append(mlflow_logger)
                logger.info(f"MLflow logging enabled, tracking URI: {self.mlflow_uri}")
                
            except ImportError:
                logger.warning("MLflow logging configured but package not installed. Skipping MLflowLogger.")
            except Exception as e:
                logger.warning(f"MLflow logging setup failed (URI: {self.mlflow_uri}): {e}")
        return loggers

    def train(
        self,
        train_dataset: Any, 
        eval_dataset: Optional[Any] = None, 
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Execute training loop using PyTorch Lightning's Trainer.fit().

        Args:
            model: Model path string (from config) or an initialized L.LightningModule.
            train_dataset: Training DataLoader or DataModule.
            eval_dataset: Optional validation DataLoader or DataModule.
            callbacks: Optional list of callback functions.

        Returns:
            Dictionary containing training results and metrics.
        """

        # DataLoader Check
        train_dl = train_dataset if hasattr(train_dataset, "__iter__") else None
        val_dl = eval_dataset if eval_dataset and hasattr(eval_dataset, "__iter__") else None

        if train_dl is None:
            raise ValueError("train_dataset must be a DataLoader, LightningDataModule, or other iterable.")

        # Trainer Configuration
        if val_dl is not None:
            self.trainer.limit_val_batches = 1.0

            logger.info("Validation DataLoader detected. Adding val_loss dependent callbacks (ModelCheckpoint, EarlyStopping).")
        
            best_checkpoint = ModelCheckpoint(
                dirpath=self.checkpoint_path,
                filename="best-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=False
            )
            self.trainer.callbacks.append(best_checkpoint)

            early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
            self.trainer.callbacks.append(early_stop)

        else:
            self.trainer.limit_val_batches = 0
            logger.warning("No Validation DataLoader provided. Validation loop and val_loss callbacks are disabled.")
        
        if callbacks:
            self.trainer.callbacks.extend(callbacks)

        # Check for resumption path
        resume_path = self.training_state.get("resume_checkpoint_path")
        if resume_path:
            if not Path(resume_path).exists():
                 logger.warning(f"Resume checkpoint path not found: {resume_path}. Starting from scratch.")
                 resume_path = None
            else:
                 logger.info(f"Resuming training from checkpoint: {resume_path}")
        
        self.training_state["is_training"] = True

        try:
            logger.info("Starting training...")
            self.trainer.fit(
                model=self.module,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                ckpt_path=resume_path,
            )
            
            if self.trainer.is_global_zero:
                self.training_state["resume_checkpoint_path"] = None

                best_path = None
                last_path = None

                for cb in self.trainer.callbacks:
                    if isinstance(cb, ModelCheckpoint):
                        if getattr(cb, "monitor", None) == "val_loss":
                            best_path = getattr(cb, "best_model_path", None)
                        if getattr(cb, "save_last", False):
                            last_path = getattr(cb, "last_model_path", None)

                final_model_path_to_load = best_path if best_path else last_path
                final_output_path = Path(f"{self.config.model_name_or_path}_{self.config.experiment_name}_{self.config.run_name}")

                if final_model_path_to_load and Path(final_model_path_to_load).exists():
                    logger.info(f"Loading final model from {final_model_path_to_load}")
                    
                    # 檢查是否為 DeepSpeed Checkpoint (資料夾)
                    if Path(final_model_path_to_load).is_dir():
                        logger.info("DeepSpeed checkpoint detected (directory).")
                        try:
                            # 轉換 Checkpoint (比較耗時與吃硬碟空間)
                            temp_state_path = f"{final_model_path_to_load}_consolidated.pt"
                            logger.info(f"Converting ZeRO checkpoint to single file: {temp_state_path}")
                            
                            convert_zero_checkpoint_to_fp32_state_dict(
                                final_model_path_to_load, 
                                temp_state_path
                            )
                            
                            # 載入轉換後的權重
                            final_model = LightningLLMModule.load_from_checkpoint(temp_state_path)
                            self.module = final_model
                            logger.info("DeepSpeed checkpoint loaded successfully.")
                            
                        except Exception as e:
                            logger.warning(f"Could not convert/load DeepSpeed checkpoint: {e}")
                            logger.warning("Falling back to saving current model state (Last Epoch).")
                            # 如果轉換失敗，直接使用記憶體中目前的模型 (通常是最後一個 Epoch)
                    else:
                        # 標準 .ckpt 檔案
                        final_model = LightningLLMModule.load_from_checkpoint(final_model_path_to_load)
                        self.module = final_model

                    self.module.save_hf_model(final_output_path)
                    logger.info(f"Final model saved to {final_output_path}")

                else:
                    logger.warning("No best or last checkpoint found. Saving current module state instead.")
                    self.module.save_hf_model(final_output_path)

                metrics = {
                    "train_loss": float(self.trainer.logged_metrics.get("train_loss_epoch", 0.0)),
                    "val_loss": float(self.trainer.logged_metrics.get("val_loss", 0.0)),
                    "current_epoch": self.trainer.current_epoch,
                    "global_step": self.trainer.global_step,
                    "final_model_path": str(final_output_path)
                }

                logger.info(f"Training completed: {metrics}")
                return metrics
            
            else:
                return {}
        
        except KeyboardInterrupt:
            logger.info("Training process successfully intercepted cancellation request.")
            return {"status": "cancelled", "message": "Job manually cancelled."}

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
        finally:
            self.training_state["is_training"] = False
            if self.trainer.is_global_zero and Path(self.checkpoint_path).exists():
                try:
                    # ignore_errors=True 防止 FileNotFoundError 導致崩潰
                    shutil.rmtree(self.checkpoint_path, ignore_errors=True)
                    logger.info(f"Cleaned up checkpoint directory: {self.checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint directory: {e}")

    def validate(self, eval_dataset: Any) -> Dict[str, float]:
        """
        Run validation loop using PyTorch Lightning's Trainer.validate().

        Args:
            eval_dataset: Validation DataLoader or DataModule.

        Returns:
            Dictionary of validation metrics.
        """           
        eval_dl = eval_dataset if hasattr(eval_dataset, "__iter__") else None
        if not eval_dl:
            raise ValueError("eval_dataset must be a DataLoader or iterable.")

        results = self.trainer.validate(model=self.module, dataloaders=eval_dl)
        return results[0] if results else {}

    def save_checkpoint(self, output_dir: str, epoch: int = None, step: int = None) -> str:
        """
        Save the current state as a checkpoint manually.

        Args:
            output_dir: Directory to save checkpoint.
            epoch: Current epoch number.
            step: Current training step.

        Returns:
            Path to saved checkpoint.
        """
        if not self.trainer:
            raise RuntimeError("Trainer is not initialized. Cannot save checkpoint.")
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        name = f"manual-epoch{epoch or self.trainer.current_epoch}-step{step or self.trainer.global_step}.ckpt"
        path = Path(output_dir) / name
        
        self.trainer.save_checkpoint(path, weights_only=False)
        logger.info(f"Manual checkpoint saved: {path}")
        return path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Set the checkpoint path for training resumption.
        The actual loading happens during the next call to train().

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        super().load_checkpoint(checkpoint_path) 
        logger.info(f"Checkpoint path set for resume: {checkpoint_path}")

    def get_training_state(self) -> Dict[str, Any]:
        """
        Get current training state including epoch, step, and best metric.

        Returns:
            Dictionary with current epoch, step, and metrics.
        """
        if self.trainer:
            current_best_metric = self.trainer.callback_metrics.get("val_loss", self.training_state["best_metric"])
            self.training_state.update({
                "epoch": self.trainer.current_epoch,
                "global_step": self.trainer.global_step,
                "best_metric": float(current_best_metric),
            })
        return self.training_state

    def stop_training(self) -> None:
        """
        Training cancellation is handled by cancel_token and PyTorch Lightning's Module.
        """
        pass