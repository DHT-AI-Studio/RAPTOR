import threading
import time
from collections import OrderedDict

class TaskManager:
    """管理背景任務進度，限制任務數量避免記憶體無限制增長"""
    def __init__(self, max_tasks=100):
        self._tasks = OrderedDict()
        self._lock = threading.Lock()
        self.max_tasks = max_tasks

    def _trim_tasks(self):
        while len(self._tasks) > self.max_tasks:
            # 移除最舊的任務
            self._tasks.popitem(last=False)

    def create_task(self, name: str) -> str:
        import uuid
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "name": name,
                "done": 0,
                "total": None,
                "status": "running",
                "error": None,
                "created_at": time.time()
            }
            self._trim_tasks()
        return task_id

    def update_progress(self, task_id: str, done: int, total: int = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task["done"] = done
                if total is not None:
                    task["total"] = total

    def mark_done(self, task_id: str, total: int = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task["status"] = "done"
                task["total"] = total or task["done"]
                self._trim_tasks()

    def mark_failed(self, task_id: str, error: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task["status"] = "failed"
                task["error"] = error
                self._trim_tasks()

    def get_status(self, task_id: str):
        with self._lock:
            return self._tasks.get(task_id)
