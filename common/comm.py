import base64
import socket
from dataclasses import dataclass, field
from typing import Dict, List

#OPTIMIZE: built-in pickle 
import pickle
from common.log import default_logger as logger
from common.serialize import JsonSerializable


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
    node_type: str = ""
    data: bytes = b""

    def to_json(self):
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "data": base64.b64encode(self.data).decode("utf-8"),
        }

    @staticmethod
    def from_json(json_data):
        return BaseRequest(
            node_id=json_data.get("node_id"),
            node_type=json_data.get("node_type"),
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
