import time
from datetime import timedelta
import logging
from typing import Dict, Any, List, Tuple


import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only

logger = logging.getLogger(__name__)


class TotalStepsLoggerCallback(Callback):
    """
    Callback to calculate the total number of training steps and log it to MLflow and training_state.
    This runs at the very beginning of the training process.
    """
    def __init__(self, training_state: Dict[str, Any]):
        super().__init__()
        self.training_state = training_state
    
    @rank_zero_only
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called when fit begins."""
        
        # estimated_stepping_batches 在 on_fit_start 時已被計算
        total_steps = trainer.estimated_stepping_batches
        
        if total_steps is not None and total_steps > 0:
            total_steps = int(total_steps)
            self.training_state["total_steps"] = total_steps
            logger.info(f"Total estimated training steps: {total_steps}")
            
            # 記錄到 MLflow Parameters
            mlf_logger = next((log for log in trainer.loggers if isinstance(log, MLFlowLogger)), None)
            if mlf_logger and mlf_logger.experiment_id and mlf_logger.run_id:
                try:
                    mlf_logger.experiment.log_param(mlf_logger.run_id, "total_steps", total_steps)
                except Exception as e:
                    logger.warning(f"Failed to log total_steps to MLflow: {e}")


class TTCProgressCallback(Callback):
    """
    Callback to calculate and log the Estimated Time to Completion (TTC).
    Calculations happen every `log_every_n_steps`.
    """
    def __init__(self, training_state: Dict[str, Any]):
        super().__init__()
        self.training_state = training_state
        # 歷史記錄: 儲存 (global_step, timestamp)
        self.history: List[Tuple[int, float]] = [] 
        # 只保留最近 N 個點來計算平滑速率
        self.history_window_size = 10 

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx: int) -> None:
        # 僅在達到 log_every_n_steps 時執行計算
        if trainer.global_step == 0 or (trainer.global_step % trainer.log_every_n_steps) != 0:
            return
        
        current_time = time.time()
        current_step = trainer.global_step

        # 1. 記錄當前數據點
        self.history.append((current_step, current_time))
        if len(self.history) > self.history_window_size:
            self.history.pop(0)

        # 2. 檢查是否有足夠的數據點進行計算
        if len(self.history) < 2:
            return

        # 3. 獲取總 Step 數 (從 TotalStepsLoggerCallback 寫入的 State 中獲取)
        total_steps = self.training_state.get("total_steps")
        if total_steps is None or total_steps <= 0:
            return 
        
        # 4. 計算平滑平均速率
        start_step, start_time = self.history[0]
        step_diff = current_step - start_step
        time_diff = current_time - start_time
        
        avg_rate_steps_per_second = 0
        if time_diff > 0 and step_diff > 0:
            avg_rate_steps_per_second = step_diff / time_diff

        # 5. 預估剩餘時間 (TTC) 和進度
        remaining_steps = total_steps - current_step
        
        if avg_rate_steps_per_second > 0 and remaining_steps > 0:
            time_remaining_seconds = int(remaining_steps / avg_rate_steps_per_second)
            time_remaining_human = str(timedelta(seconds=int(time_remaining_seconds)))
            
            progress_percentage = (current_step / total_steps) * 100
            
            # 6. 記錄進度到 Lightning Log (會同步到 MLflow/TensorBoard)
            pl_module.log("estimated_time_remaining_seconds", time_remaining_seconds, on_step=True, on_epoch=False)
            pl_module.log("progress_percentage", progress_percentage, on_step=True, on_epoch=False)
            
            # 7. 更新 Trainer 外部的 State (供外部 API 查詢)
            # self.training_state["estimated_time_remaining_seconds"] = time_remaining_seconds
            # self.training_state["progress_percentage"] = progress_percentage
            
            logger.debug(f"TTC: {time_remaining_human} (Progress: {progress_percentage:.2f}%)")
    

class TTCEWMAProgressCallback(L.Callback):
    def __init__(self, training_state, alpha=0.3, min_steps=10):
        self.training_state = training_state
        self.alpha = alpha
        self.min_steps = min_steps
        self.train_history = []
        self.train_ewma_rate = None
        self.val_ewma_time = None
        self.val_start_time = None

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()
        current_step = trainer.global_step
        self.train_history.append((current_step, current_time))
        if len(self.train_history) < self.min_steps:
            return

        start_step, start_time = self.train_history[0]
        step_diff = current_step - start_step
        time_diff = current_time - start_time
        if step_diff <= 0 or time_diff <= 0:
            return

        instant_rate = step_diff / time_diff
        if self.train_ewma_rate is None:
            self.train_ewma_rate = instant_rate
        else:
            self.train_ewma_rate = self.alpha * instant_rate + (1 - self.alpha) * self.train_ewma_rate

        total_steps = self.training_state.get("total_steps")
        if total_steps is None:
            return
        remaining_steps = total_steps - current_step

        # 用秒估算
        avg_train_step_time = 1 / self.train_ewma_rate
        remaining_val_time = self.val_ewma_time if self.val_ewma_time else 0
        ttc_seconds = remaining_steps * avg_train_step_time + remaining_val_time
        progress = (current_step / total_steps) * 100

        pl_module.log("estimated_time_remaining_seconds", int(ttc_seconds), on_step=True, on_epoch=False)
        pl_module.log("progress_percentage", progress, on_step=True, on_epoch=False)

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        self.val_start_time = time.time()

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.val_start_time is None:
            return
        val_time = time.time() - self.val_start_time
        if self.val_ewma_time is None:
            self.val_ewma_time = val_time
        else:
            self.val_ewma_time = self.alpha * val_time + (1 - self.alpha) * self.val_ewma_time