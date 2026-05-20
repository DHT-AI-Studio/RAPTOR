# src/core/gpu_manager.py
"""
GPU 管理器

功能包含：
- 靜態查詢（是否有 GPU、數量、名稱、總記憶體）
- 即時監控（已用 / 可用記憶體、使用率、溫度、功耗、風扇）
- 智慧分配（round-robin、least_used、first_fit）
- 裝置上下文管理（with gpu_manager.use_device(id): ...）
- 清除 GPU 快取（避免 OOM）
- 可選的背景監控（start_monitoring / stop_monitoring）

說明：此檔案會優先使用 pynvml（NVML，全名 NVIDIA Management Library）取得詳細資訊；若不存在則嘗試使用 torch 或 nvidia-smi 作為 fallback。

註：檔案中大量註解為繁體中文（適合學習），每個步驟皆嘗試說明用途。
"""

from __future__ import annotations

import logging
import threading
import time
import subprocess
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

import torch

# 嘗試載入 pynvml（如果可用，會用來取得詳細 GPU 資訊）
try:
    import pynvml
    _HAS_PYNVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_PYNVML = False

_logger = logging.getLogger(__name__)


class GPUManager:
    """GPU 管理器類別。

    - 若系統沒有 NVIDIA GPU，仍可正常使用（回傳 cpu/devices count 0）。
    - 提供選擇/分配機制、監控、上下文管理與快取清理。
    """

    def __init__(self):
        # 初始化狀態
        self._use_pynvml = _HAS_PYNVML
        if self._use_pynvml:
            try:
                pynvml.nvmlInit()
                _logger.debug("pynvml 初始化成功")
            except Exception as e:
                _logger.warning("pynvml 初始化失敗，改用 fallback: %s", e)
                self._use_pynvml = False

        # round-robin 計數器
        self._rr_counter = 0

        # 背景監控 thread 與相關旗標
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()
        self._monitor_interval = 5.0  # 預設監控間隔（秒）
        self._last_stats: Dict[int, Dict] = {}

        # 建立最初狀態
        self.refresh()

    # ---------------------- 靜態查詢 ----------------------
    def refresh(self) -> None:
        """重新讀取 CUDA / GPU 基本狀態（數量等）。

        在系統資源變動時可以呼叫此方法以更新內部快取值。
        """
        self._is_available = torch.cuda.is_available()
        try:
            self._device_count = torch.cuda.device_count() if self._is_available else 0
        except Exception:
            # 在極少數情況下 torch.cuda.device_count() 可能丟出錯誤
            self._device_count = 0

    def is_gpu_available(self) -> bool:
        """檢查系統是否有可用的 GPU（透過 PyTorch 檢查）。"""
        return bool(self._is_available)

    def get_device_count(self) -> int:
        """取得 GPU 數量。"""
        return int(self._device_count)

    def get_device(self, device_id: int = 0) -> str:
        """取得指定設備的字串標示，例如 'cuda:0' 或 'cpu'。"""
        if self.is_gpu_available() and 0 <= device_id < self._device_count:
            return f"cuda:{device_id}"
        return "cpu"

    # ---------------------- 監控 / 資訊 ----------------------
    def _query_with_pynvml(self, idx: int) -> Dict:
        """使用 pynvml 查詢單張 GPU 的詳細資訊（NVML = NVIDIA Management Library）。"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # 溫度（攝氏）
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        # 功耗（毫瓦）
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        except Exception:
            power_mw = None
        # 風扇轉速（百分比，部份 GPU/驅動可能回傳 -1）
        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
        except Exception:
            fan = None

        return {
            "id": idx,
            "name": torch.cuda.get_device_name(idx) if torch.cuda.is_available() else None,
            "total_memory_gb": mem.total / (1024 ** 3),
            "used_memory_gb": mem.used / (1024 ** 3),
            "free_memory_gb": mem.free / (1024 ** 3),
            "utilization_percent": util.gpu if hasattr(util, "gpu") else util.gpu_utilization if hasattr(util, "gpu_utilization") else None,
            "memory_util_percent": util.memory if hasattr(util, "memory") else None,
            "temperature_c": temp,
            "power_draw_w": (power_mw / 1000) if power_mw is not None else None,
            "fan_speed_percent": fan,
        }

    def _query_with_torch(self, idx: int) -> Dict:
        """使用 PyTorch 的 API 取得較簡單的資訊（記憶體相關）。"""
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory
        # 嘗試用 torch 的 memory_allocated 取得已用量
        try:
            used = torch.cuda.memory_allocated(idx)
            reserved = torch.cuda.memory_reserved(idx) if hasattr(torch.cuda, "memory_reserved") else None
        except Exception:
            used = None
            reserved = None

        return {
            "id": idx,
            "name": torch.cuda.get_device_name(idx),
            "total_memory_gb": total / (1024 ** 3),
            "used_memory_gb": (used / (1024 ** 3)) if used is not None else None,
            "reserved_memory_gb": (reserved / (1024 ** 3)) if reserved is not None else None,
            "free_memory_gb": (total - used) / (1024 ** 3) if used is not None else None,
            "utilization_percent": None,
            "temperature_c": None,
            "power_draw_w": None,
            "fan_speed_percent": None,
        }

    def _query_with_nvidia_smi(self, idx: int) -> Dict:
        """使用 nvidia-smi 命令行作為 fallback（系統需可執行 nvidia-smi）。"""
        # 利用 nvidia-smi 查詢 CSV 格式
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,fan.speed",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(cmd, encoding="utf-8")
            lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
            # 找到對應 idx 的行
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if str(idx) == parts[0]:
                    # parts: index, name, total_mem, used_mem, util, temp, power, fan
                    total = float(parts[2])
                    used = float(parts[3])
                    util = float(parts[4]) if parts[4] != "N/A" else None
                    temp = float(parts[5]) if parts[5] != "N/A" else None
                    power = float(parts[6]) if parts[6] != "N/A" else None
                    fan = float(parts[7]) if parts[7] != "N/A" else None
                    return {
                        "id": idx,
                        "name": parts[1],
                        "total_memory_gb": total / 1024.0,
                        "used_memory_gb": used / 1024.0,
                        "free_memory_gb": (total - used) / 1024.0,
                        "utilization_percent": util,
                        "temperature_c": temp,
                        "power_draw_w": power,
                        "fan_speed_percent": fan,
                    }
        except Exception as e:
            _logger.debug("nvidia-smi fallback 失敗: %s", e)

        # 若無法解析則回傳簡單結構
        return {
            "id": idx,
            "name": None,
            "total_memory_gb": None,
            "used_memory_gb": None,
            "free_memory_gb": None,
            "utilization_percent": None,
            "temperature_c": None,
            "power_draw_w": None,
            "fan_speed_percent": None,
        }

    def get_gpu_info(self) -> List[Dict]:
        """取得所有 GPU 的詳細資訊，回傳 list。"""
        self.refresh()
        if not self.is_gpu_available():
            return []

        info = []
        for i in range(self._device_count):
            try:
                if self._use_pynvml:
                    stat = self._query_with_pynvml(i)
                else:
                    # 優先用 torch，若資訊不完整再用 nvidia-smi
                    stat = self._query_with_torch(i)
                    if stat.get("utilization_percent") is None:
                        # 嘗試 nvidia-smi
                        smi_stat = self._query_with_nvidia_smi(i)
                        # 合併有用的欄位
                        for k, v in smi_stat.items():
                            if v is not None and stat.get(k) is None:
                                stat[k] = v
                info.append(stat)
            except Exception as e:
                _logger.exception("取得 GPU 資訊時發生錯誤: %s", e)
                info.append({"id": i, "error": str(e)})

        return info
        # info 範例：
        # [{'id': 0,
        # 'name': 'NVIDIA RTX A6000',
        # 'total_memory_gb': 47.98828125,
        # 'used_memory_gb': 41.2662353515625,
        # 'free_memory_gb': 6.7220458984375,
        # 'utilization_percent': 0,
        # 'memory_util_percent': 0,
        # 'temperature_c': 56,
        # 'power_draw_w': 33.377,
        # 'fan_speed_percent': 30},
        # {'id': 1,
        # 'name': 'NVIDIA RTX A6000',
        # 'total_memory_gb': 47.98828125,
        # 'used_memory_gb': 0.47210693359375,
        # 'free_memory_gb': 47.51617431640625,
        # 'utilization_percent': 0,
        # 'memory_util_percent': 0,
        # 'temperature_c': 64,
        # 'power_draw_w': 44.853,
        # 'fan_speed_percent': 36}]

    # ---------------------- 選擇 / 分配 ----------------------
    def select_device(
        self,
        strategy: str = "least_used",
        min_free_memory_gb: float = 0.0,
        prefer_lower_util: bool = True,
    ) -> Optional[int]:
        """根據策略選擇一張 GPU，回傳 device id（找不到回傳 None，代表應使用 CPU）。
        Args:
            strategy: 選擇策略，可選：
             - "least_used" : 選擇剩餘記憶體最多的 GPU
             - "round_robin": 輪流分配
             - "first_fit": 找到第一張 free_memory >= min_free_memory_gb 的 GPU
            min_free_memory_gb: 最小可用記憶體（GB），預設 0.0（不限制）
            prefer_lower_util: 在 least_used 策略下，若多張 GPU 記憶體相同，是否偏好使用率較低的 GPU
        Returns:
            選中的 GPU device id（0, 1, ...），找不到則回傳 None（代表使用 CPU）
        Note:
            此方法會呼叫 self.refresh() 以確保資訊是最新的。
        """
        self.refresh()
        if not self.is_gpu_available() or self._device_count == 0:
            return None

        stats = self.get_gpu_info()

        # 過濾出滿足最小可用記憶體的 GPU
        candidates = [s for s in stats if s.get("free_memory_gb") is not None and s.get("free_memory_gb") >= min_free_memory_gb]
        if not candidates:
            # 若沒候選者，再放寬條件（允許任何 GPU）
            candidates = [s for s in stats if s.get("free_memory_gb") is not None]

        if not candidates:
            return None

        if strategy == "round_robin":
            idx = self._rr_counter % len(candidates)
            # 將候選者中的第 idx 個轉回其真實 device id
            chosen = candidates[idx]["id"]
            self._rr_counter += 1
            return chosen

        if strategy == "first_fit":
            return candidates[0]["id"]

        # 默認 least_used：選擇 free_memory 最大或 utilisation 最小
        if strategy == "least_used":
            # 先嘗試以 free_memory_gb 排序
            candidates_sorted = sorted(candidates, key=lambda x: (-(x.get("free_memory_gb") or 0), x.get("utilization_percent") or 100))
            return candidates_sorted[0]["id"]

        # 未知策略 -> 回傳 None
        _logger.warning("未知的選擇策略: %s", strategy)
        return None

    @contextmanager
    def use_device(self, device_id: Optional[int]):
        """裝置上下文管理器（context manager）。

        範例：
            with gpu_manager.use_device(0):
                model.to('cuda')

        呼叫時若 device_id 為 None 或不合法，會使用 'cpu'。
        """
        # 保存原始裝置（若是 GPU，PyTorch 會有 active device）
        try:
            orig_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        except Exception:
            orig_device = None

        try:
            if device_id is None or not self.is_gpu_available():
                yield "cpu"
            else:
                # 若 device_id 合法，設置並回傳字串
                if 0 <= device_id < self._device_count:
                    torch.cuda.set_device(device_id)
                    yield f"cuda:{device_id}"
                else:
                    yield "cpu"
        finally:
            # 嘗試回復到原本的 device（如果適用）
            try:
                if orig_device is not None:
                    torch.cuda.set_device(orig_device)
            except Exception as e:
                _logger.exception("回復原始裝置失敗: %s", e)

    # ---------------------- 快取 / 清理 ----------------------
    def clear_cache(self) -> None:
        """清除 PyTorch 的 CUDA 快取（避免 OOM）。"""
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            _logger.debug("清除 CUDA 快取失敗: %s", e)

    # ---------------------- 兼容性 Shims ----------------------
    # 為了兼容較舊的 API 呼叫（gpu_api / 測試腳本中引用的函式），提供輕量封裝。
    def get_memory_info(self, device_id: Optional[int] = None) -> Dict:
        """返回指定 GPU 或全部 GPU 已用/總記憶體資訊。"""
        info_list = self.get_gpu_info()
        if device_id is None:
            # 聚合
            total = sum(i.get('total_memory_gb', 0) or 0 for i in info_list)
            used = sum(i.get('used_memory_gb', 0) or 0 for i in info_list)
            return {
                'total_memory_gb': total,
                'used_memory_gb': used,
                'free_memory_gb': total - used
            }
        for i in info_list:
            if i.get('id') == device_id:
                return {
                    'total_memory_gb': i.get('total_memory_gb'),
                    'used_memory_gb': i.get('used_memory_gb'),
                    'free_memory_gb': i.get('free_memory_gb')
                }
        return {}

    def get_device_info(self, device_id: int) -> Dict:
        for i in self.get_gpu_info():
            if i.get('id') == device_id:
                return i
        return {}

    def get_utilization_info(self, device_id: Optional[int] = None) -> Dict:
        info_list = self.get_gpu_info()
        if device_id is not None:
            dev = self.get_device_info(device_id)
            return {'device_id': device_id, 'utilization_percent': dev.get('utilization_percent')}
        # 聚合平均
        if not info_list:
            return {'average_utilization_percent': 0}
        vals = [i.get('utilization_percent') for i in info_list if i.get('utilization_percent') is not None]
        avg = sum(vals)/len(vals) if vals else 0
        return {'average_utilization_percent': avg}

    def get_temperature_info(self, device_id: Optional[int] = None) -> Dict:
        info_list = self.get_gpu_info()
        if device_id is not None:
            dev = self.get_device_info(device_id)
            return {'device_id': device_id, 'temperature_c': dev.get('temperature_c')}
        temps = [i.get('temperature_c') for i in info_list if i.get('temperature_c') is not None]
        avg = sum(temps)/len(temps) if temps else None
        return {'average_temperature_c': avg}

    def is_device_available(self, device_id: int) -> bool:
        # 簡化：若能取得裝置資訊即視為可用
        return 0 <= device_id < self.get_device_count()

    # 舊策略接口映射到 select_device
    def select_device_least_used(self) -> Optional[int]:
        return self.select_device(strategy='least_used')

    def select_device_round_robin(self) -> Optional[int]:
        return self.select_device(strategy='round_robin')

    def select_device_first_fit(self) -> Optional[int]:
        return self.select_device(strategy='first_fit')

    # ---------------------- 背景監控 ----------------------
    def _monitor_loop(self, callback: Optional[Callable[[Dict[int, Dict]], None]] = None, interval: float = 5.0):
        """內部的背景監控迴圈，會定期更新 self._last_stats 並呼叫 callback。"""
        self._monitor_stop_event.clear()
        while not self._monitor_stop_event.is_set():
            try:
                stats = {s["id"]: s for s in self.get_gpu_info()}
                self._last_stats = stats
                if callback:
                    try:
                        callback(stats)
                    except Exception:
                        _logger.exception("monitor callback 執行失敗")
            except Exception:
                _logger.exception("背景監控讀取 GPU 資訊失敗")
            # 等待下一次
            self._monitor_stop_event.wait(interval)

    def start_monitoring(self, callback: Optional[Callable[[Dict[int, Dict]], None]] = None, interval: float = 5.0):
        """啟動背景監控（非同步 thread）。

        - callback: 若提供，會每次讀取 stats 時被呼叫，參數為 dict(device_id -> stat)
        - interval: 監控間隔（秒）

        注意：此方法會產生背景 thread，若在 production 或容器內需考慮 lifecycle 管理。
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            _logger.debug("monitor thread 已在執行")
            return
        self._monitor_interval = interval
        self._monitor_stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(callback, interval), daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """停止背景監控。"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_stop_event.set()
            self._monitor_thread.join(timeout=interval if (interval := self._monitor_interval) else 5.0)

    def get_last_stats(self) -> Dict[int, Dict]:
        """取得最後一次背景監控的統計資料（若未啟動監控則為空）。"""
        return self._last_stats.copy()


# 建立模組層級的全域 instance（便於在其他模組 import 使用）
gpu_manager = GPUManager()


# ---------------------- 當作腳本執行時的簡單示範 ----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GPU Available:", gpu_manager.is_gpu_available())
    print("Device count:", gpu_manager.get_device_count())
    print("GPU Info:")
    from pprint import pprint

    pprint(gpu_manager.get_gpu_info())

    # 範例：以 least_used 策略選一張 GPU
    chosen = gpu_manager.select_device(strategy="least_used", min_free_memory_gb=1.0)
    print("Chosen device:", chosen)

    # 範例：使用上下文管理器
    with gpu_manager.use_device(chosen) as dev:
        print("Using device:", dev)
        # 模擬清理
        gpu_manager.clear_cache()

    # 示範 background monitor（列印 callback）
    def cb(stats):
        print("monitor callback ->", stats)

    gpu_manager.start_monitoring(callback=cb, interval=3.0)
    time.sleep(6)
    gpu_manager.stop_monitoring()
