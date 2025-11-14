import pickle
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests

from agent.data_collector.constants import CollectedNodeType
from common import comm
from common.comm import BaseRequest, BaseResponse, StateRequest
from common.log import default_logger as logger
from common.singleton import Singleton
from util import env_util
from util.comm_util import find_free_port
from util.func_util import retry


class ControllerClient(Singleton, ABC):
    """Controller Client Base class for worker nodes."""

    _instance_lock = threading.Lock()

    def __init__(
        self,
        servicers_addr: Dict[int, str],
        timeout: int = 10,
    ):
        if not servicers_addr:
            raise ValueError("servicers_addr is empty")

        self._timeout = timeout
        self._servicers_addr = servicers_addr

        # self._node_ip = env_util.get_node_ip()
        # self._worker_local_process_id = env_util.get_worker_local_process_id()
        # self._ddp_server_port = find_free_port()

        logger.info(
            "ControllerClient initialized with servicers_addr=%s, timeout=%s",
            servicers_addr,
            timeout,
        )

    def _resolve_servicer_id(self, servicer_id: Optional[int]) -> int:
        return servicer_id if servicer_id is not None else next(iter(self._servicers_addr.keys()))
        if target_id not in self._servicers_addr:
            raise ValueError(f"servicer {target_id} not found")
        return target_id

    def _serialize_message(self, message: object) -> bytes:
        try:
            return pickle.dumps(message)
        except Exception as exc:
            raise RuntimeError(f"Failed to serialize message: {message}") from exc

    @retry()
    @abstractmethod
    def get_state(self, state_type: str, servicer_id: Optional[int] = None):
        """Get state from a specific servicer."""
        raise NotImplementedError


class HttpControllerClient(ControllerClient):
    """HTTP Controller Client that manages multiple Worker nodes."""

    def __init__(
        self,
        servicers_addr: Dict[int, str],
        timeout: int = 10,
    ):
        super().__init__(servicers_addr, timeout)
        logger.info(
            "HttpControllerClient initialized with %s servicers", len(self._servicers_addr)
        )

    def _normalize_path(self, path: str) -> str:
        return path if path.startswith("/") else f"/{path}"

    def _get_http_request_url(self, servicer_id: int, path: str) -> str:
        normalized_path = self._normalize_path(path)
        return f"http://{self._servicers_addr[servicer_id]}{normalized_path}"

    def _handle_response(self, response: requests.Response) -> BaseResponse:
        if response.status_code != 200:
            error_msg = (
                f"HTTP request failed, status={response.status_code}, body={response.text}"
            )
            raise RuntimeError(error_msg)

        response_data = comm.deserialize_message(response.content)
        if not isinstance(response_data, BaseResponse):
            raise RuntimeError("Failed to deserialize response data, not BaseResponse instance")
        return response_data

    @retry()
    def get_state(
        self,
        state_type: str,
        servicer_id: Optional[int] = None,
    ) -> Optional[object]:
        target_id = self._resolve_servicer_id(servicer_id)
        state_request = StateRequest(state_type=state_type, node_id=target_id)
        request = BaseRequest(
            node_id=target_id,
            data=self._serialize_message(state_request),
        )

        try:
            response = requests.post(
                self._get_http_request_url(target_id, "/get_state"),
                json=request.to_json(),
                timeout=self._timeout,
            )
            response_data = self._handle_response(response)
            if not response_data.success or not response_data.data:
                logger.warning(
                    "Failed to get state from servicer %s for %s: success=%s, data_len=%s",
                    target_id,
                    state_type,
                    response_data.success,
                    len(response_data.data),
                )
                return None

            return comm.deserialize_message(response_data.data)
        except Exception as exc:
            logger.error(
                "Failed to get state from servicer %s for %s: %s", target_id, state_type, exc
            )
            return None