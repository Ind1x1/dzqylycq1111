import os
import threading
import time

import psutil

from common.comm import GPUstats

from common.constants import NodeEnv, NodeType, AcceleratorType
from common.log import default_logger as logger
from common.singleton import Singleton

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
        #FIXME
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