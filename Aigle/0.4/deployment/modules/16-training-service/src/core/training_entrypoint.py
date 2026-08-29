import argparse
import json
import sys
import os
import subprocess
import time
import traceback
import logging
from pathlib import Path

sys.path.append(os.getcwd())

import torch

from src.training.models.trainer_config import TrainerConfig, DatasetConfig
from src.core.training_orchestrator import TrainingOrchestrator
from src.core.training_job_manager import CancelToken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

def _try_nvml_wakeup() -> bool:
    """Call nvidia-smi -pm 1 to attempt to refresh a stale NVML/driver connection.

    When a Docker container runs for days, NVML inside a spawned subprocess may
    lose its connection to the NVIDIA kernel driver even though the parent process
    (uvicorn) is unaffected.  Running nvidia-smi from inside the container can
    reinitialise the driver interface in some borderline cases.  If this still
    doesn't help, the container healthcheck will detect the failure and restart
    the container automatically.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-pm", "1"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            logger.info("nvidia-smi -pm 1 wakeup succeeded")
            return True
        logger.warning(f"nvidia-smi -pm 1 returned non-zero: {result.stderr.strip()}")
    except Exception as e:
        logger.warning(f"nvidia-smi wakeup call failed: {e}")
    return False

def _check_cuda_available():
    """Verify CUDA is accessible in this subprocess before starting training."""
    if not torch.cuda.is_available():
        # Attempt to wake up a stale NVML connection before giving up.
        logger.warning("CUDA not initially available in subprocess; attempting nvidia-smi wakeup...")
        _try_nvml_wakeup()
        time.sleep(3)  # Give the driver a moment to reinitialise

        if not torch.cuda.is_available():
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
            raise RuntimeError(
                f"CUDA is not available in the training subprocess "
                f"(CUDA_VISIBLE_DEVICES={cuda_devices}). "
                f"NVML failed to initialize — the container's /dev/nvidia* device "
                f"references have become stale after extended uptime. "
                f"The container healthcheck will detect this and restart automatically. "
                f"To recover immediately: run 'sudo nvidia-smi -pm 1' on the host "
                f"then 'docker restart aigle-training-service'."
            )
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    logger.info(f"CUDA available: {gpu_count} GPU(s) detected: {gpu_names}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--redis_url", type=str, required=True)
    parser.add_argument("--redis_password", type=str, default="")
    parser.add_argument("--mlflow_uri", type=str, required=True)
    
    args = parser.parse_args()

    try:
        # 1. 讀取配置
        with open(args.config_path, "r") as f:
            job_config = json.load(f)

        trainer_config = TrainerConfig(**job_config["trainer_config"])
        dataset_config = DatasetConfig(**job_config["dataset_config"])
        task_type = job_config["task_type"]

        logger.info(f"Starting job {args.job_id} task {task_type}")

        # 1.5 驗證 CUDA 在此子進程中可用
        _check_cuda_available()

        # 2. 初始化 Cancel Token
        token = CancelToken(args.job_id, args.redis_url, args.redis_password)

        # 3. 啟動 Orchestrator
        orchestrator = TrainingOrchestrator(
            task_type=task_type,
            config=trainer_config,
            dataset_config=dataset_config,
            cancel_token=token,
            mlflow_uri=args.mlflow_uri,
        )

        metrics = orchestrator.run()

        # 4. 寫入成功結果
        result = {
            "status": "completed",
            "metrics": metrics,
            "final_model_path": metrics.get("final_model_path")
        }

    except Exception as e:
        logger.error(f"Training script failed: {e}", exc_info=True)
        result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    # 5. 將結果寫回 JSON 檔案，供主進程讀取
    with open(args.result_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()