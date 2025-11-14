import os
import threading
import time

import psutil

from common.comm import GPUstats

from common.constants import NodeEnv, NodeType, AcceleratorType
from common.log import default_logger as logger
from common.singleton import Singleton

def get_process_cpu_percent():
    """Get the cpu percent of the current process."""
    try:
        procTotalPercent = 0
        result = {}
        proc_info = []
        for proc in psutil.process_iter(
            ["pid", "ppid", "name", "username", "cmdline"]
        ):
            proc_percent = proc.cpu_percent()
            procTotalPercent += proc_percent
            proc.info["cpu_percent"] = round(proc_percent, 2)
            proc_info.append(proc.info)
        result["proc_info"] = proc_info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = round(procTotalPercent / cpu_count, 2)
    except Exception:
        cpu_percent = 0.0
    return cpu_percent / 100.0

def get_used_memory():
    """Get the used memory of the container"""
    mem = psutil.virtual_memory()
    return int(mem.used / 1024 / 1024)

def get_gpu_stats(gpus=[]):
    """Get the used gpu info of the container"""

    try:
        import pynvml
    except ImportError:
        logger.warning("No pynvml is available, skip getting gpu stats.")
        return []

    try:
        pynvml.nvmlInit()

        if not gpus:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                logger.warning("No GPU is available.")
                device_count = 0
            gpus = list(range(device_count))
        gpu_stats: list[GPUStats] = []
        for i in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total / (1024**2)
            used_memory = memory_info.used / (1024**2)

            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu

            gpu_stats.append(
                GPUStats(
                    index=i,
                    total_memory_mb=total_memory,
                    used_memory_mb=used_memory,
                    gpu_utilization=gpu_utilization,
                )
            )
        return gpu_stats
    except Exception as e:
        logger.warning(f"Got unexpected error when getting gpu stats: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

class ResourceMonitor(Singleton):
    def __init__(self, gpu_type: str = AcceleratorType.NVIDIA_GPU, node_type: str = NodeType.TEMP_NODE):
        """
        The moniotr samples the used memory and cpu percent 
        reports the used memory and cpu percent
        """
        self._total_cpu = psutil.cpu_count(logical=True)
        self._gpu_type = gpu_type
        self._gpu_stats: list[GPUStats] = []
        self._node_type = node_type
        self._Node_Server = NodeServer.singleton_instance()

    def start(self):
        log.info(f"ResourceMonitor started for {self._node_type}")

        try:
            thread = threading.Thread(
                target = self._monitor_resource,
                name = "ResourceMonitor-thread"
                daemon = True
            )
            thread.start()
            if thread.is_alive():
                logger.info(f"ResourceMonitor stainitialized successfully for {self._node_type}")
            else:
                logger.error(f"ResourceMonitor failed to initialize for {self._node_type}")
        except Exception as e:
            logger.error(f"ResourceMonitor failed to initialize for {self._node_type}: {e}")

    def stop(self):
        pass

    def report_resource(self):
        used_mem = get_used_memory()
        cpu_percent = get_process_cpu_percent()

        if self._gpu_type == AcceleratorType.NVIDIA_GPU:
            self._gpu_stats = get_gpu_stats()
        else:
            #OPTIMIZE: not supported for other
            pass

        current_cpu = round(cpu_percent * self._total_cpu, 2)
        #FIXME 这里应该是report到上层的一个封装，构建一个消息队列以供Client消费
        self._collector_server._report(used_meme, current_cpu, self._gpu_stats)

        logger.debug(
            "Reported resource usage for {self._node_type}: used_mem={used_mem}, current_cpu={current_cpu}, gpu_stats={self._gpu_stats}"
        )

    def _monitor_resource(self):
        logger.info(f"ResourceMonitor started monitoring resource for {self._node_type}")
        while True:
            try:
                self._monitor_resource()
            except Exception as e:
                logger.debug(f"ResourceMonitor failed to monitor resource for {self._node_type}: {e}")
            time.sleep(30)