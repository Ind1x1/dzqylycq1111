import json
import pickle
from dataclasses import dataclass, field
from typing import Dict


def to_dict(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    elif hasattr(o, "__dict__"):
        return o.__dict__
    else:
        return {}


class JsonSerializable(object):
    def to_json(self, indent=None):
        return json.dumps(
            self,
            default=to_dict,
            sort_keys=True,
            indent=indent,
        )


class PickleSerializable(object):
    def serialize(self):
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data):
        return pickle.loads(data)


@dataclass
class ClassMeta:
    module_path: str = ""
    class_name: str = ""
    kwargs: Dict[str, str] = field(default_factory=dict)