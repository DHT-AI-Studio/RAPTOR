import subprocess
import json
import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_in_dedicated_process_and_terminate(
    job_config: Dict[str, Any], 
    job_id: str, 
    cuda_devices: str, 
    mlflow_uri: str, 
    redis_url: str, 
    redis_password: str
) -> Dict[str, Any]:
    """
    使用 subprocess.Popen 啟動獨立的訓練腳本。
    """
    
    # 1. 準備暫存檔案來傳遞 Config 和接收 Result
    # 使用 tempfile 確保路徑唯一
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as config_file:
        json.dump(job_config, config_file)
        config_path = config_file.name
    
    # 建立一個空的結果檔案路徑
    result_path = config_path.replace(".json", "_result.json")

    # 2. 設定環境變數 (CUDA Isolation)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # 確保 PYTHONPATH 包含當前專案根目錄，這樣腳本才能 import src
    env["PYTHONPATH"] = os.getcwd() 
    
    # 強制設定為單機多卡模式所需的變數，避免 DeepSpeed 混淆
    # 如果是單機多卡，MASTER_ADDR 等通常由 Lightning 自動處理，
    # 但為了保險，可以在此處不設定，讓 Lightning 決定。
    
    cmd = [
        sys.executable, # 使用當前的 python 解譯器路徑
        "src/core/training_entrypoint.py",
        "--job_id", job_id,
        "--config_path", config_path,
        "--result_path", result_path,
        "--redis_url", redis_url,
        "--mlflow_uri", mlflow_uri
    ]
    if redis_password:
        cmd.extend(["--redis_password", redis_password])

    logger.info(f"Job {job_id}: Launching subprocess: {' '.join(cmd)}")
    logger.info(f"Job {job_id}: CUDA_VISIBLE_DEVICES={cuda_devices}")

    process = None
    try:
        # 3. 啟動子進程
        process = subprocess.Popen(
            cmd, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                clean_line = line.strip()
                output_lines.append(clean_line)
                logger.info(f"[Job {job_id} Worker] {clean_line}")

        # 等待進程結束
        return_code = process.wait()
        
        if return_code != 0:
            error = "\n".join(output_lines)
            raise RuntimeError(f"Training script exited with code {return_code}. \n{error}")
        
        # 4. 讀取結果
        if not os.path.exists(result_path):
            raise RuntimeError("Result file was not created by the training script.")
            
        with open(result_path, "r") as f:
            result = json.load(f)
            
        if result.get("status") == "failed":
            raise RuntimeError(f"Training failed: {result.get('error')}\n{result.get('traceback')}")
            
        return result.get("metrics", {})

    except Exception as e:
        logger.error(f"Subprocess execution failed: {e}")
        # 如果發生異常且進程還在跑，強制殺掉
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise e
        
    finally:
        # 清理暫存檔
        if os.path.exists(config_path):
            os.remove(config_path)
        if os.path.exists(result_path):
            os.remove(result_path)