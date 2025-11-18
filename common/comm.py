import base64
import socket
from dataclasses import dataclass, field
from typing import Dict, List

#OPTIMIZE: built-in pickle 
import pickle
from common.log import default_logger as logger
from common.serialize import JsonSerializable
from common.constants import CollectorType


def deserialize_message(data: bytes):
    """The method will create a message instance with the content.
    Args:
        data: pickle bytes of a class instance.
    """
    message = None
    if data:
        try:
            message = pickle.loads(data)
        except Exception as e:
            logger.warning(f"Pickle failed to load {str(data)}", e)
    return message



class Message(JsonSerializable):
    def serialize(self):
        return pickle.dumps(self)

@dataclass
class BaseRequest(Message):
    node_id: int = -1
    #TODO: use CollectedNodeType
    # node_type: str = ""
    data: bytes = b""

    def to_json(self):
        return {
            "node_id": self.node_id,
            # "node_type": self.node_type,
            "data": base64.b64encode(self.data).decode("utf-8"),
        }

    @staticmethod
    def from_json(json_data):
        return BaseRequest(
            node_id=json_data.get("node_id"),
            # node_type=json_data.get("node_type"),
            data=base64.b64decode(json_data.get("data")),
        )


@dataclass
class BaseResponse(Message):
    success: bool = False
    data: bytes = b""

    def to_json(self):
        return {
            "success": self.success,
            "data": base64.b64encode(self.data).decode("utf-8"),
        }

    @staticmethod
    def from_json(json_data):
        return BaseResponse(
            success=bool(json_data.get("success")),
            data=base64.b64decode(json_data.get("data")),
        )

@dataclass
class GPUStats(Message):
    index: int = 0
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    gpu_utilization: float = 0

@dataclass
class CollectorRequest(Message):
    """Request to get state from a specific collector type.
    
    Args:
        collector_type: Type of collector to collect. Options: "log", "resource", "stack"
        node_id: Node ID (optional)
    """
    collector_type: CollectorType = CollectorType.DUMMY_COLLECTOR  # "log", "resource", "stack"....
    node_id: int = -1
    
    def to_json(self):
        return {
            "collector_type": self.collector_type,
            "node_id": self.node_id,
        }
    
    @staticmethod
    def from_json(json_data):
        collector_type = json_data.get("collector_type", CollectorType.DUMMY_COLLECTOR)
        # Ensure collector_type is a CollectorType value (string)
        if collector_type == "":
            collector_type = CollectorType.DUMMY_COLLECTOR
        return CollectorRequest(
            collector_type=collector_type,
            node_id=json_data.get("node_id", -1),
        )
