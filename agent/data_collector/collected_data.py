import json
from abc import ABCMeta
from datetime import datetime
from typing import List, Optional

from agent.data_collector.constants import (
    CollectedDataType,
    CollectedNodeType,
)
from common import env_utils

class CollectedData(metaclass=ABCMeta):
    """
    Basic definition of diagnosis data.

    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        data_type (str): Type of metric. Defaults to "GENERIC".
        data_content (str): Content of the metric. Defaults to "".
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
    """

    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = CollectedDataType.GENERIC,
        data_content: str = "",
        node_id: int = -1,
        node_type: str = CollectedNodeType.TRAIN_NODE,
        node_rank: int = -1,
    ):
        if timestamp == 0:
            self._timestamp = int(round(datetime.now().timestamp()))
        else:
            self._timestamp = timestamp
        self._data_type = data_type
        self._data_content = data_content
        self._node_id = node_id
        self._node_type = node_type
        self._node_rank = node_rank

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def data_content(self) -> str:
        return self._data_content

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type(self):
        return self._node_type

    @property
    def node_rank(self):
        return self._node_rank

    def to_json(self):
        data = {k.lstrip("_"): v for k, v in self.__dict__.items()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))

    def is_from_worker(self):
        return self._node_id != -1

class WorkerTrainingMetric(CollectedData):
    """
    Worker's training metric.

    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        metric (dict): Metric content in dict format.
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
    """

    def __init__(
        self,
        timestamp: int = 0,
        data_type: str = CollectedDataType.GENERIC,
        data_content: str = "",
        node_id=env_utils.get_node_id(),
        node_type=env_utils.get_node_type(),
        node_rank=env_utils.get_node_rank(),
        is_final_result=False,
        need_report=False,
    ):
        super(WorkerTrainingMetric, self).__init__(
            timestamp, data_type, data_content, node_id, node_type, node_rank
        )
        self._is_final_result = is_final_result
        self._need_report = need_report

    @property
    def is_final_result(self):
        return self._is_final_result

    @property
    def need_report(self):
        return self._need_report

    def is_resolvable(self):
        # TODO: 需要定义 DiagnosisDataType 或使用 CollectedDataType
        # if self.data_type == DiagnosisDataType.XPU_TIMER_METRIC:
        #     return True
        # TODO: add more resolvable metric type later
        return False


class TrainingLog(CollectedData):
    """
    Worker's training log.

    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        logs (list): Log content in list format.
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
    """

    def __init__(
        self,
        timestamp: int = 0,
        logs: Optional[List[str]] = None,
        node_id=env_utils.get_node_id(),
        node_type=env_utils.get_node_type(),
        node_rank=env_utils.get_node_rank(),
    ):
        if logs is None:
            data_content = ""
        else:
            data_content = "\n".join(logs)

        super().__init__(
            timestamp,
            CollectedDataType.TRAINING_LOG,
            data_content,
            node_id,
            node_type,
            node_rank,
        )

    @property
    def logs(self) -> List[str]:
        if not self.data_content:
            return []
        return [line for line in self.data_content.splitlines()]


class ResourceData(CollectedData):
    """
    Worker's resource usage data.
    
    Args:
        timestamp (datetime): Timestamp of diagnosis data.
        used_memory_mb (int): Used memory in MB.
        cpu_percent (float): CPU usage percentage (0.0-1.0).
        current_cpu (float): Current CPU usage.
        gpu_stats (list): GPU statistics list.
        node_id (int): Node ID. Defaults to -1.
        node_type (str): Node type. Defaults to "".
        node_rank (int): Node rank. Defaults to -1.
    """

    def __init__(
        self,
        timestamp: int = 0,
        used_memory_mb: int = 0,
        cpu_percent: float = 0.0,
        current_cpu: float = 0.0,
        gpu_stats: Optional[List] = None,
        node_id=env_utils.get_node_id(),
        node_type=env_utils.get_node_type(),
        node_rank=env_utils.get_node_rank(),
    ):
        # Convert resource data to JSON string for data_content
        resource_dict = {
            "used_memory_mb": used_memory_mb,
            "cpu_percent": cpu_percent,
            "current_cpu": current_cpu,
            "gpu_stats": gpu_stats or [],
        }
        data_content = json.dumps(resource_dict)
        
        super().__init__(
            timestamp,
            CollectedDataType.HARDWARE_METRIC,
            data_content,
            node_id,
            node_type,
            node_rank,
        )
        self._used_memory_mb = used_memory_mb
        self._cpu_percent = cpu_percent
        self._current_cpu = current_cpu
        self._gpu_stats = gpu_stats or []

    @property
    def used_memory_mb(self) -> int:
        return self._used_memory_mb

    @property
    def cpu_percent(self) -> float:
        return self._cpu_percent

    @property
    def current_cpu(self) -> float:
        return self._current_cpu

    @property
    def gpu_stats(self) -> List:
        return self._gpu_stats

    @classmethod
    def from_resource_dict(cls, resource_dict: dict, **kwargs):
        """
        Create ResourceData from resource dictionary.
        
        Args:
            resource_dict: Dictionary containing resource data
            **kwargs: Additional arguments for CollectedData (node_id, node_type, etc.)
        """
        # Extract node_type from dict if present, otherwise use kwargs or default
        node_type = kwargs.pop("node_type", resource_dict.pop("node_type", None))
        
        return cls(
            used_memory_mb=resource_dict.get("used_memory_mb", 0),
            cpu_percent=resource_dict.get("cpu_percent", 0.0),
            current_cpu=resource_dict.get("current_cpu", 0.0),
            gpu_stats=resource_dict.get("gpu_stats", []),
            node_type=node_type,
            **kwargs
        )