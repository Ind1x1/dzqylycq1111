import pickle
import threading
from abc import ABC, abstractmethod
from typing import Optional

import requests

from agent.data_collector.constants import CollectedNodeType
from common import comm
from common.comm import BaseRequest, BaseResponse
from common.log import default_logger as logger
from common.singleton import Singleton
from util import env_util
from util.comm_util import find_free_port
from util.func_util import retry


# OPTIMIZE: HTTPsClinet

class ControllerClinet(Singleton, ABC):
    """ControllerClient provides HTTP-based APIs to communicate with controller service."""

    _instance_lock = threading.Lock()

    def __init__(
        self,
        master_addr: str,
        node_id: int,
        node_type: CollectedNodeType,
        timeout: int = 10,
    ):
        logger.info(
            "ControllerClient initialized with master_addr=%s, node_id=%s, "
            "node_type=%s, timeout=%s",
            master_addr,
            node_id,
            node_type,
            timeout,
        )
        self._timeout = timeout
        self._master_addr = master_addr
        self._node_id = node_id
        self._node_type = node_type
        self._node_ip = env_util.get_node_ip()
        self._worker_local_process_id = env_util.get_worker_local_process_id()
        self._ddp_server_port = find_free_port()

    def _serialize_message(self, message: object) -> bytes:
        try:
            return pickle.dumps(message)
        except Exception as exc:
            raise RuntimeError(f"Failed to serialize message {message}: {exc}") from exc

    def _gen_request(self, message: object) -> BaseRequest:
        return BaseRequest(
            node_id=self._node_id,
            node_type=self._node_type,
            data=self._serialize_message(message),
        )

    def report(self, message: object) -> BaseResponse:
        return self._report(message)

    def get(self, message: object):
        return self._get(message)

    @retry()
    @abstractmethod
    def _report(self, message: object) -> BaseResponse:
        """Report the message to the controller service."""

    @retry()
    @abstractmethod
    def _get(self, message: object):
        """Get the message from the controller service."""

    # TODO: add more APIs here


class HttpControllerClient(ControllerClinet):
    def __init__(
        self,
        master_addr: str,
        node_id: int,
        node_type: CollectedNodeType,
        timeout: int = 10,
    ):
        super().__init__(master_addr, node_id, node_type, timeout)

    def _get_http_request_url(self, path: str) -> str:
        return "http://" + self._master_addr + path

    def _handle_response(self, response: requests.Response) -> BaseResponse:
        if response.status_code != 200:
            error_msg = (
                f"HTTP request failed with status={response.status_code}, "
                f"content={response.text}"
            )
            raise RuntimeError(error_msg)

        response_data: Optional[BaseResponse] = comm.deserialize_message(response.content)
        if response_data is None:
            raise RuntimeError("Failed to deserialize BaseResponse from controller.")
        return response_data

    @retry()
    def _report(self, message: object) -> BaseResponse:
        response = requests.post(
            self._get_http_request_url("/report"),
            json=self._gen_request(message).to_json(),
            timeout=self._timeout,
        )
        return self._handle_response(response)

    @retry()
    def _get(self, message: object):
        response = requests.post(
            self._get_http_request_url("/get"),
            json=self._gen_request(message).to_json(),
            timeout=self._timeout,
        )
        response_data = self._handle_response(response)
        if not response_data.data:
            return None
        return comm.deserialize_message(response_data.data)
 