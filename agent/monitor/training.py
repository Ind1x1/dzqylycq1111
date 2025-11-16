import json
import os
import threading
import time

from common.log import default_logger as logger
from common.singleton import Singleton
from monitor.resource import ResourceMonitor
from common.constants import NodeType, AcceleratorType, NodeEnv
from common import env_utils

class RLHFTrainingMonitor(Singleton):
    def __init__(
        self, 
        node_type: NodeType, 
        node_uid: int, 
        gpu_type: AcceleratorType = AcceleratorType.NVIDIA_GPU,
        metrics_path: str = "",
    ):
        """
        The monitor samples the training log and reports the training log
        """
        self._last_timestamp = 0
        self._start_time = 0
        #FIXME: 这里应该是report到上层的一个封装，构建一个消息队列以供Client消费
        self._group_rank = env_utils.get_group_rank()
        self._node_type = node_type
        self._node_uid = node_uid
        self._gpu_type = gpu_type
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        self._metrics_path = metrics_path
        self._resource_monitor = ResourceMonitor(gpu_type=gpu_type, node_type=node_type)

    def start(self):
        if os.getenv(NodeEnv.MONITOR_ENABLE, "false") != "true":
            logger.info(
                f"RLHFTrainingMonitor is disabled by the environment variable {NodeEnv.MONITOR_ENABLE}"
            )
            return
        thread = threading.Thread(
            target = self._periodically_report,
            name = "node_reporter",
            daemon = True
        )
        thread.start()

    def stop(self):
        self._resource_monitor.stop()

    def report_step(self):
        if self._group_rank != 0:
            return
        try:
            if not os.path.exists(self._metrics_path):
                return
            with open(self._metrics_path, "r") as f:
                record = json.load(f)
                step = record.get("step", 0)
                timestamp = record.get("timestamp", 0)
            if step > 0 and timestamp - self._last_timestamp > 15:
                self._last_timestamp = timestamp
                #FIXME
        except Exception as e:
            logger.error(f"Failed to report step: {e}")

    def _periodically_report(self):
        logger.info(f"RLHFTrainingMonitor started reporting metrics for {self._node_type}")
        while True:
            if self._group_rank == 0:
                self.report_step()
            time.sleep(15)