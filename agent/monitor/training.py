import json
import os
import threading
import time


from agent.data_collector.collected_data import WorkerTrainingMetric
from agent.data_collector.constants import CollectedDataType
from common.log import default_logger as logger
from common.singleton import Singleton
from agent.monitor.resource import ResourceMonitor
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

    def report_step(self) -> WorkerTrainingMetric:
        """
        Report training step metrics from metrics file.
        
        Returns:
            WorkerTrainingMetric: Training metric data, or None if not available
        """
        if self._group_rank != 0:
            return None
        try:
            if not os.path.exists(self._metrics_path):
                return None
            with open(self._metrics_path, "r") as f:
                record = json.load(f)
                step = record.get("step", 0)
                timestamp = record.get("timestamp", 0)
            
            if step > 0 and timestamp - self._last_timestamp > 15:
                self._last_timestamp = timestamp
                
                # Convert record to JSON string for data_content
                data_content = json.dumps(record)
                
                # Create and return WorkerTrainingMetric
                metric = WorkerTrainingMetric(
                    timestamp=timestamp,
                    data_type=CollectedDataType.GENERIC,
                    data_content=data_content,
                    node_type=self._node_type,
                )
                logger.debug(f"Reported training step {step} at timestamp {timestamp}")
                return metric
            return None
        except Exception as e:
            logger.error(f"Failed to report step: {e}")
            return None

    def _periodically_report(self):
        logger.info(f"RLHFTrainingMonitor started reporting metrics for {self._node_type}")
        while True:
            if self._group_rank == 0:
                self.report_step()
            time.sleep(15)