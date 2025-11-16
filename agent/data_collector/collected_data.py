import json
from abc import ABCMeta
from datetime import datetime
from typing import List, Optional

from agent.data_collector.constants import (
    CollectedDataType,
    CollectedNodeType,
)
from common import env_util

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
        if self.data_type == DiagnosisDataType.XPU_TIMER_METRIC:
            return True
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