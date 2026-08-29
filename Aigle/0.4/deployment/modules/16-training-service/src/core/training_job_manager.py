import uuid
import logging
import threading
import traceback
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict
from redis.exceptions import RedisClusterException, ConnectionError
# Redis client construction (auto-detects standalone vs cluster) lives in redis_factory.
from .redis_factory import make_async_redis_client, make_redis_client, probe_cluster_enabled
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import concurrent.futures
import multiprocessing
from multiprocessing import Process, Pipe
import shutil

from ..training.models.job_submission import TrainingJobSubmission
from .training_worker import run_in_dedicated_process_and_terminate
from ..config import settings
from .gpu_manager import get_gpu_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# multiprocessing.set_start_method('spawn', force=True)

gpu_manager = get_gpu_manager()

def _redis_key(job_id: str) -> str:
    return f"training_job:{job_id}"

class CancelToken:
    def __init__(self, job_id: str, redis_url: str, redis_password: str):
        self.job_id = job_id
        self.redis_client = make_redis_client(redis_url, redis_password)

    def is_cancelled(self) -> bool:
        key = _redis_key(self.job_id)
        job = self.redis_client.get(key)
        data = json.loads(job) if job else {}
        return data.get("status") == "cancelled"


class TrainingJob:
    def __init__(self, job_id: str, config: Dict[str, Any], gpu_id: Union[int, List[int]] = -1):
        self.job_id = job_id
        self.config = config
        self.gpu_id = gpu_id
        self.status = "queued"
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "gpu_id": self.gpu_id,
            "status": self.status,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "model_path": self.model_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        job = cls(data["job_id"], data["config"], data.get("gpu_id", -1))
        job.status = data.get("status") or "unknown"
        job.start_time = data.get("start_time")
        job.end_time = data.get("end_time")
        job.metrics = data.get("metrics")
        job.model_path = data.get("model_path")
        return job


class TrainingJobManager:
    SYNC_CALL_TIMEOUT = 30

    def __init__(
        self,
        job_ttl_seconds: int = 7 * 24 * 3600,
        backup_dir: Optional[str] = None,
        gpu_ids: List[int] = None,
        vram_overhead_gb: float = 1.0,
    ):
        self.gpu_ids = sorted(gpu_ids or [0])
        self.vram_overhead_bytes = int(settings.vram_overhead_gb * 1024**3) or int(vram_overhead_gb * 1024**3)

        # Redis
        self.redis_url = settings.redis_url
        self.redis_password = settings.redis_password
        # Detect standalone vs cluster ONCE at construction (sync, before the loop
        # serves requests), so the async client can be built without a probe delay.
        try:
            self._redis_is_cluster = probe_cluster_enabled(self.redis_url, self.redis_password)
            logger.info("Redis mode auto-detected: %s", "cluster" if self._redis_is_cluster else "standalone")
        except Exception as exc:  # noqa: BLE001 — fall back to standalone if probe fails
            logger.warning("Redis cluster probe failed (%s); assuming standalone.", exc)
            self._redis_is_cluster = False
        self.job_ttl = job_ttl_seconds
        self.backup_dir = Path(backup_dir) if backup_dir else None
        if self.backup_dir:
            self.backup_dir.mkdir(exist_ok=True)

        # MLflow
        self.mlflow_uri = settings.mlflow_uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient()

        # self.process_pool = ProcessPoolExecutor(max_workers=4)

        # 狀態
        self.pending_queue: Dict[str, TrainingJob] = {}
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.lock = threading.Lock()

        # 事件驅動
        self._scheduler_wakeup = asyncio.Event()

        # 異步 I/O 迴圈
        self._async_loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_async_loop, name="AsyncIOLoop", daemon=True)
        self._async_thread.start()

        # 背景 Redis 儲存（完全非阻塞）
        self._redis_save_queue = asyncio.Queue()
        self._redis_worker_task = None

        # 排程器
        self._scheduler_stop_event = asyncio.Event()
        self._scheduler_task = None

        # 啟動背景任務
        self._start_background_tasks()
        
        # # 初始化
        # self._sync_call(self._init_redis_client())

        # 恢復
        self._sync_call(self._recover_jobs_from_redis())

        logger.info(f"TrainingJobManager initialized | GPUs: {self.gpu_ids}")

    def _run_async_loop(self):
        asyncio.set_event_loop(self._async_loop)
        self._async_loop.run_forever()

    def _sync_call(self, coro, timeout: int = None):
        if threading.current_thread() == self._async_thread:
            raise RuntimeError("_sync_call cannot be called from async loop thread")
        future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
        try:
            result = future.result(timeout=timeout or self.SYNC_CALL_TIMEOUT)
            return result
        except concurrent.futures.TimeoutError:
            logger.error(
                f"_sync_call timed out after {timeout or self.SYNC_CALL_TIMEOUT}s. "
                f"Cancelling coroutine."
            )
            future.cancel()
            raise TimeoutError(
                f"Operation timed out after {timeout or self.SYNC_CALL_TIMEOUT} seconds"
            )
        except Exception as e:
            logger.error(f"_sync_call failed: {e}", exc_info=True)
            raise

    def _start_background_tasks(self):
        asyncio.run_coroutine_threadsafe(self._init_redis_client(), self._async_loop)
        self._redis_worker_task = asyncio.run_coroutine_threadsafe(self._redis_save_worker(), self._async_loop)
        self._scheduler_task = asyncio.run_coroutine_threadsafe(self._run_scheduler(), self._async_loop)
        logger.info("Background tasks started.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(2),
        retry=retry_if_exception_type((ConnectionError, RedisClusterException)),
        reraise=True
    )
    async def _init_redis_client(self):
        self.redis_client = make_async_redis_client(
            self.redis_url,
            self.redis_password,
            self._redis_is_cluster,
            socket_timeout=5.0,
            max_connections=100
        )
        # await self.redis_client.ping()
        logger.info("Async Redis client initialized (cluster=%s).", self._redis_is_cluster)

    async def _redis_save_worker(self):
        while True:
            try:
                job = await self._redis_save_queue.get()
                if job is None:
                    break
                key = _redis_key(job.job_id)
                data = json.dumps(job.to_dict())
                try:
                    await self.redis_client.set(key, data, ex=self.job_ttl)
                    if self.backup_dir:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, (self.backup_dir / f"{job.job_id}.json").write_bytes, data)
                except Exception as e:
                    logger.error(f"Redis save failed for {job.job_id}: {e}", exc_info=True)
                self._redis_save_queue.task_done()
            except Exception as e:
                logger.error(f"Redis worker error: {e}", exc_info=True)

    async def _save_job_to_redis_immediate_async(self, job: TrainingJob):
        key = _redis_key(job.job_id)
        data = json.dumps(job.to_dict())
        try:
            await self.redis_client.set(key, data, ex=self.job_ttl)
            logger.debug(f"Job {job.job_id} saved to Redis.")
            if self.backup_dir:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, (self.backup_dir / f"{job.job_id}.json").write_bytes, data)
                logger.debug(f"Job {job.job_id} backup written to {self.backup_dir / f'{job.job_id}.json'}.")
        except Exception as e:
            logger.error(f"Immediate Redis save failed for {job.job_id}: {e}", exc_info=True)
            raise

    def _save_job_to_redis(self, job: TrainingJob):
        asyncio.run_coroutine_threadsafe(self._redis_save_queue.put(job), self._async_loop)

    def _get_vram_usage_per_gpu(self) -> Dict[int, Dict[str, int]]:
        all_gpu_info = gpu_manager.get_gpu_info()
        internal_budgets = {}
        with self.lock: 
            # 複製 self.running_jobs.values() 到一個 list，確保遍歷時字典不會變動。
            running_jobs_snapshot = list(self.running_jobs.values())

        for job in running_jobs_snapshot:
            budget_gb = job.config.get('vram_budget_gb', 0)
            budget_bytes = int(budget_gb * 1024**3)

            if isinstance(job.gpu_id, list):
                assigned_gpus = job.gpu_id
            else:
                assigned_gpus = [job.gpu_id]

            num_gpus = len(assigned_gpus)

            if num_gpus > 0:
                budget_per_gpu = budget_bytes / num_gpus
            else:
                budget_per_gpu = 0

            for gpu_id in assigned_gpus:
                budget_value = int(round(budget_per_gpu))
                current_budget = internal_budgets.get(gpu_id, 0)
                internal_budgets[gpu_id] = current_budget + budget_value

        usage = {}
        for info in all_gpu_info:
            gpu_id = info['id']
            # 從 GPUManager 獲取數據 (都是以 GB 為單位)
            total_gb = info.get('total_memory_gb', 0.0) or 0.0
            used_gb = info.get('used_memory_gb', 0.0) or 0.0 # NVML 報告的實際已用量
            
            total_bytes = int(total_gb * 1024**3)
            driver_used_bytes = int(used_gb * 1024**3)
            
            # 3. 執行調度邏輯：取 Driver 報告的已用量 和 內部預算 中較高的作為排程基礎
            internal_used_bytes = internal_budgets.get(gpu_id, 0)
            used_vram_for_scheduling = max(driver_used_bytes, internal_used_bytes)
            
            # 4. 計算可用空間
            available = max(0, total_bytes - used_vram_for_scheduling)
            
            usage[gpu_id] = {
                "total": total_bytes,
                "used": used_vram_for_scheduling,
                "available": available
            }
            
        # 如果 gpu_manager 沒找到任何 GPU（例如：沒有 NVIDIA GPU），則用 self.gpu_ids 確保結構一致
        for gid in self.gpu_ids:
            if gid not in usage:
                usage[gid] = {"total": 0, "used": 10**18, "available": 0} # 確保它不會被選中

        logger.info(f"VRAM usage per GPU: {usage}")
                
        return usage

    def _select_best_gpus(self, required_gb: float, select_multiple_gpus: bool = False) -> Optional[List[int]]:
        """根據 VRAM 需求選擇一張或多張 GPU，直到達到總 VRAM 需求。"""
        
        required_bytes = int(required_gb * 1024**3)
        vram_usage = self._get_vram_usage_per_gpu()
        
        # 1. 過濾滿足 VRAM 需求的候選 GPU (單/多模式分開)
        
        if select_multiple_gpus:
            # 多 GPU 模式：只需要選擇扣除 overhead 後，尚有可用 VRAM 的 GPU
            candidates = [
                gpu_id for gpu_id, info in vram_usage.items()
                if info["available"] - self.vram_overhead_bytes > 0
            ]
        else:
            # 單 GPU 模式：GPU 必須單獨滿足所需的 VRAM + overhead
            candidates = [
                gpu_id for gpu_id, info in vram_usage.items()
                if info["available"] - self.vram_overhead_bytes >= required_bytes
            ]
        
        if not candidates:
            logger.warning("No GPU available for training.")
            return None
        
        # 2. 按可用記憶體降序排列 (Worst-Fit)
        candidates.sort(key=lambda gid: vram_usage[gid]["available"], reverse=True)
        
        # 3. 根據選擇單張或多張 GPU
        if select_multiple_gpus:
            # --- 多 GPU 組合 Worst-Fit ---
            selected_gpus = []
            total_vram_available = 0
            
            for gpu_id in candidates:
                # 關鍵修正：計算每張 GPU 實際可貢獻的 VRAM (扣除 overhead)
                contributable_vram = vram_usage[gpu_id]["available"] - self.vram_overhead_bytes
                
                # 由於 candidates 已經篩選過 (> 0)，但為了安全再次檢查
                if contributable_vram <= 0: 
                    continue
                    
                total_vram_available += contributable_vram
                selected_gpus.append(gpu_id)
                
                if total_vram_available >= required_bytes:
                    break
            
            # 如果無法達到需求，返回 None
            if total_vram_available < required_bytes:
                return None
            
            logger.info(f"_select_best_gpus (multiple/worst-fit) ({required_gb}GB) -> {selected_gpus}")
            return selected_gpus
        else:
            # --- 單 GPU  Worst-Fit ---
            # 由於 candidates 已經降序排列，最好的選擇 (VRAM 最多) 總是在第一個
            chosen_id = candidates[0]
            logger.info(f"_select_best_gpus (single/worst-fit) ({required_gb}GB) -> {chosen_id}")
            # 返回單一 GPU 的 ID 列表 (符合 List[int] 的簽名)
            return [chosen_id]
    
    # --- Redis 操作 ---
    async def _load_job_from_redis_async(self, job_id: str) -> Optional[TrainingJob]:
        key = _redis_key(job_id)
        try:
            data = await self.redis_client.get(key)
            if not data:
                return None
            raw = json.loads(data)
            if not isinstance(raw, dict) or raw.get("job_id") != job_id:
                raise ValueError("Invalid job payload")
            return TrainingJob.from_dict(raw)
        except Exception as e:
            logger.error(f"Load failed {job_id}: {e}")
            return None

    async def _recover_jobs_from_redis(self):
        logger.info("Starting job recovery from Redis...")
        pattern = _redis_key("*")
        failed_count = 0
        try:
            async for key in self.redis_client.scan_iter(match=pattern, count=100):
                job_id = key.split(":", 1)[1]
                job = await self._load_job_from_redis_async(job_id)
                if not job:
                    logger.warning(f"Failed to load job {job_id} from Redis.")
                    continue
                if job.status == "running":
                    job.status = "failed"
                    job.metrics = {"error": "Process died during restart."}
                    await self._save_job_to_redis_immediate_async(job)
                    failed_count += 1
                elif job.status == "queued":
                    self.pending_queue[job_id] = job
        except Exception as e:
            logger.error(f"Recovery failed: {e}", exc_info=True)
        logger.info(f"Recovered {len(self.pending_queue)} queued jobs from Redis.")
        if failed_count:
            logger.warning(f"Reset {failed_count} interrupted jobs to 'failed'.")

    def _load_job_from_redis(self, job_id: str) -> Optional[TrainingJob]:
        return self._sync_call(self._load_job_from_redis_async(job_id))

    async def _run_scheduler(self):
        logger.info("Scheduler started.")
        while not self._scheduler_stop_event.is_set():
            await self._scheduler_wakeup.wait()
            self._scheduler_wakeup.clear()

            jobs_to_schedule = []
            with self.lock:
                if not self.pending_queue:
                    logger.debug("No pending jobs in queue.")
                    continue
                jobs_to_schedule = list(self.pending_queue.values())

            for job in jobs_to_schedule:
                job_id = job.job_id
                with self.lock:
                    if job_id not in self.pending_queue:
                        continue

                logger.info(f"Scheduler checking pending job {job_id}")
                required_gb = job.config.get("vram_budget_gb", 0)
                select_multiple_gpus = job.config.get("select_multiple_gpus", False)
                gpu_id = self._select_best_gpus(required_gb, select_multiple_gpus)
                logger.info(f"GPU selection for job {job_id}: {gpu_id}")
                if gpu_id is not None:
                    with self.lock:
                        if job_id not in self.pending_queue:
                            continue
                        job.gpu_id = gpu_id
                        job = self.pending_queue.pop(job_id)
                        self.running_jobs[job_id] = job
                        job.status = "running"
                        job.start_time = datetime.now().isoformat()
                        self._save_job_to_redis(job)
                        logger.info(f"Job {job_id} scheduled on GPU {gpu_id}.")
                    task = asyncio.run_coroutine_threadsafe(self._execute_job(job_id), self._async_loop)
                    # task = asyncio.create_task(self._execute_job(job_id))
                    self.running_tasks[job_id] = task
                else:
                    logger.info(f"Job {job_id} waiting for GPU. Required: {required_gb}GB")

    async def _execute_job(self, job_id: str):
        job = self.running_jobs.get(job_id)
        if not job:
            logger.warning(f"_execute_job: Job {job_id} not found.")
            return
        logger.info(f"Job {job_id} execution started on GPU {job.gpu_id}.")

        loop = asyncio.get_running_loop()

        if isinstance(job.gpu_id, list):
            cuda_devices = ",".join(map(str, job.gpu_id))
        elif isinstance(job.gpu_id, int):
            cuda_devices = str(job.gpu_id)
        else:
             cuda_devices = ""

        try:
            if job.status == "cancelled":
                logger.info(f"Job {job_id} already cancelled before start.")
                return
            logger.info(f"Job {job_id} moved to ThreadPoolExecutor for training.")
            metrics = await loop.run_in_executor(
                # self.process_pool,
                None,
                run_in_dedicated_process_and_terminate, 
                job.config, 
                job_id,
                cuda_devices,
                self.mlflow_uri,
                self.redis_url,
                self.redis_password,
            )

            if job.status != "cancelled":
                job.status = "completed"
                job.metrics = metrics
                job.model_path = metrics.get("final_model_path")
                logger.info(f"Job {job_id} completed successfully.")
        except Exception as e:
            if job.status != "cancelled":
                job.status = "failed"
                job.metrics = {"error": str(e), "traceback": traceback.format_exc()}
                logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        finally:
            job.end_time = datetime.now().isoformat()
            with self.lock:
                self.running_jobs.pop(job_id, None)
                self.running_tasks.pop(job_id, None)
            self._save_job_to_redis(job)
            logger.info(f"Job {job_id} state saved to Redis.")
            self._scheduler_wakeup.set()

    def submit_job(self, submission: TrainingJobSubmission) -> str:
        return self._sync_call(self.async_submit_job(submission))

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._load_job_from_redis(job_id)

    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._sync_call(self.async_list_jobs(status))

    def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._sync_call(self.async_get_job_progress(job_id))
    
    def delete_job(self, job_id: str, force_cancel: bool = False) -> bool:
        return self._sync_call(self.async_delete_job(job_id, force_cancel))

    def cancel_job(self, job_id: str) -> bool:
        with self.lock:
            if job_id in self.pending_queue:
                del self.pending_queue[job_id]
                job = self._load_job_from_redis(job_id)
                if job:
                    job.status = "cancelled"
                    self._save_job_to_redis(job)
                self._scheduler_wakeup.set()
                return True

            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                job.status = "cancelled"
                self._save_job_to_redis(job)
                self._scheduler_wakeup.set()
                logger.warning(f"Job {job_id} marked as cancelled.")
                return True
        return False

    async def async_submit_job(self, submission: TrainingJobSubmission) -> str:
        job_id = str(uuid.uuid4())[:16].replace("-", "")
        logger.info(f"Submitting new job {job_id}.")
        training_config = submission.training_config
        if not training_config.run_name:
            training_config.run_name = job_id
        if not training_config.experiment_name:
            training_config.experiment_name = f"{submission.task_type}_experiments"
        job_config = {
            "task_type": submission.task_type,
            "vram_budget_gb": submission.vram_budget_gb,
            "select_multiple_gpus": submission.select_multiple_gpus,
            "trainer_config": asdict(training_config),
            "dataset_config": asdict(submission.dataset_config),
            # "submission_args": submission.model_dump(),
        }
        job = TrainingJob(job_id, job_config, gpu_id=-1)
        await self._save_job_to_redis_immediate_async(job)
        with self.lock:
            self.pending_queue[job_id] = job
        self._scheduler_wakeup.set()
        logger.info(f"Job {job_id} submitted and added to pending_queue.")
        return job_id

    async def async_get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = await self._load_job_from_redis_async(job_id)
        if not job:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_get_job_progress, job)

    def _sync_get_job_progress(self, job: TrainingJob) -> Optional[Dict[str, Any]]:
        exp_name = job.config["trainer_config"].get("experiment_name")
        run_name = job.job_id
        
        STEP_LOSS_KEY = "train_loss_step" 
        
        try:
            exp = self.mlflow_client.get_experiment_by_name(exp_name)
            if not exp:
                return None
            
            runs = self.mlflow_client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                run_view_type=ViewType.ALL,
                max_results=1
            )
            if not runs:
                return None
            
            run_id = runs[0].info.run_id
            
            # 1. 獲取 Run 的完整快照數據（包含所有最終/聚合指標、Tags等）
            run = self.mlflow_client.get_run(run_id)
            
            # 以 Run 快照的 metrics 作為基礎，它包含 val_loss 等聚合指標
            progress_data = dict(run.data.metrics) 
            progress_data["mlflow_run_id"] = run_id
            progress_data["total_steps"] = int(run.data.params.get("total_steps", "-1"))
            if progress_data.get("progress_percentage"):
                progress_data["progress_percentage"] = round(progress_data["progress_percentage"], 2)
            else:
                progress_data["progress_percentage"] = 0
            
            if progress_data.get("estimated_time_remaining_seconds"):
                progress_data["estimated_time_remaining_seconds"] = int(progress_data["estimated_time_remaining_seconds"])
            else:
                progress_data["estimated_time_remaining_seconds"] = "unknown"

            # 2. 查詢 STEP_LOSS_KEY 的歷史記錄以獲取最新的 Step 和 Loss
            step_history = self.mlflow_client.get_metric_history(run_id, STEP_LOSS_KEY)
            
            if step_history:
                latest_metric = step_history[-1]
                
                # 3. 覆蓋：將歷史記錄中的最新 Step 和 Loss 覆蓋到快照數據中
                #    這確保了我們報告的是即時的 global_step 和 train_loss_step。
                progress_data["current_step"] = int(latest_metric.step)
                progress_data[STEP_LOSS_KEY] = latest_metric.value
                progress_data["progress_timestamp"] = datetime.fromtimestamp(latest_metric.timestamp/1000)
            else:
                progress_data["current_step"] = 0
                
            return progress_data
            
        except Exception as e:
            logger.warning(f"MLflow progress query failed: {e}", exc_info=True)
            return None

    async def async_list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        pattern = _redis_key("*")
        jobs_list = []
        try:
            async for key in self.redis_client.scan_iter(match=pattern, count=100):
                job_id = key.split(":", 1)[1]
                job = await self._load_job_from_redis_async(job_id)
                if not job:
                    continue
                if not status or job.status == status:
                    jobs_list.append({
                        "job_id": job.job_id,
                        "status": job.status,
                        "gpu_id": job.gpu_id,
                        "task_type": job.config.get("task_type"),
                        "model": job.config["trainer_config"].get("model_name_or_path"),
                        "vram_gb": job.config.get("vram_budget_gb"),
                        "submitted": job.start_time or job.end_time or "1970-01-01T00:00:00",
                    })
        except Exception as e:
            logger.error(f"List jobs failed: {e}", exc_info=True)
        return sorted(jobs_list, key=lambda x: x["submitted"], reverse=True)

    async def async_delete_job(self, job_id: str, force_cancel: bool = False) -> bool:
        job = await self._load_job_from_redis_async(job_id)
        if not job:
            logger.info(f"Job {job_id} not found in Redis.")
            return False

        if job.status in {"running", "queued"} and not force_cancel:
            logger.info(f"Job {job_id} is in {job.status} status and cannot be deleted. Use 'force_cancel=true' to cancel and delete.")
            return False

        if job.status in {"running", "queued"} and force_cancel:
            logger.info(f"Job {job_id} is in {job.status} status and will be canceled and deleted.")
            self.cancel_job(job_id)

        key = _redis_key(job_id)
        try:
            deleted = await self.redis_client.delete(key) > 0
            if self.backup_dir:
                (self.backup_dir / f"{job_id}.json").unlink(missing_ok=True)
            # 刪除模型
            if job.model_path:
                if Path(job.model_path).exists() and deleted:
                    shutil.rmtree(job.model_path)
            else:
                model_path = Path(f'{job.config["trainer_config"]["model_name_or_path"]}_{job.config["trainer_config"]["experiment_name"]}_{job.job_id}')
                if model_path.exists() and deleted:
                    shutil.rmtree(model_path)
            # 刪除log
            log_dir = Path(job.config["trainer_config"]["log_dir"]) / job.config["trainer_config"]["experiment_name"] / job.job_id
            if log_dir.exists() and deleted:
                shutil.rmtree(log_dir)
            return deleted

        except Exception as e:
            logger.error(f"Delete job {job_id} failed: {e}", exc_info=True)
            return False

    def shutdown(self):
        self._scheduler_stop_event.set()
        self._scheduler_wakeup.set()
        self.process_pool.shutdown(wait=False, cancel_futures=True)

        # 等待所有背景任務完成
        async def _shutdown():
            await self._redis_save_queue.join()
            await self._redis_save_queue.put(None)
            if self._redis_worker_task:
                await self._redis_worker_task
        self._sync_call(_shutdown())

        self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._async_thread.is_alive():
            self._async_thread.join()

        logger.info("TrainingJobManager shut down gracefully.")


_job_manager_instance = None

def get_job_manager():
    """獲取 TrainingJobManager 的單例實例。"""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = TrainingJobManager()
    return _job_manager_instance