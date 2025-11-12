import json
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import tornado.web

from common import comm
from common.comm import BaseRequest, BaseResponse
from common.constants import CommunicationReqMeta
from common.http_server import TornadoHTTPServer
from common.log import default_logger as logger


class ControllerServicer(ABC):
    """ControllerServicer is the abstract base class for the controller service."""

    def __init__(self):
        self._lock = threading.Lock()

    def validate_request(self, headers: Dict[str, str]) -> bool:
        return True

    @abstractmethod
    def get_response(self, request: BaseRequest) -> BaseResponse:
        pass

    @abstractmethod
    def get_task_type(self, request: BaseRequest) -> BaseResponse:
        pass


class HttpControllerServicer(ControllerServicer):

    def __init__(self):
        super().__init__()
        self._reports: List[object] = []

    def _deserialize_report(self, request: BaseRequest) -> Optional[object]:
        message = None
        if request.data:
            try:
                message = pickle.loads(request.data)
            except Exception as exc:
                logger.error(f"Failed to deserialize the report data: {exc}")
        return message

    def get_response(self, request: BaseRequest) -> BaseResponse:
        message = self._deserialize_report(request)
        if message is None:
            logger.error("Received empty report message.")
            return BaseResponse(success=False)

        with self._lock:
            self._reports.append(message)

        print(f"[HttpControllerServicer] Received report: {message}")
        return BaseResponse(success=True)

    def get_task_type(self, request: BaseRequest) -> BaseResponse:
        with self._lock:
            if not self._reports:
                return BaseResponse(success=True)
            message = self._reports.pop(0)

        try:
            payload = pickle.dumps(message)
        except Exception as exc:
            logger.error(f"Failed to serialize the return data: {exc}")
            return BaseResponse(success=False)

        return BaseResponse(success=True, data=payload)

    def peek_latest_report(self) -> Optional[object]:
        """Only for testing, quickly view the latest report data."""
        with self._lock:
            return self._reports[-1] if self._reports else None


class HttpControllerHandler(tornado.web.RequestHandler):
    def initialize(self, servicer: HttpControllerServicer):
        self._servicer = servicer

    def get(self):
        self.write("ok")

    def post(self):
        try:
            headers = self.request.headers
            if not self._servicer.validate_request(headers):
                self.set_status(406)
                self.write(CommunicationReqMeta.COMM_META_JOB_UID_INVALID_MSG)
                return

            path = self.request.path
            request_body = json.loads(self.request.body.decode("utf-8"))
            request = BaseRequest.from_json(request_body)

            if path == "/get":
                response = self._servicer.get_task_type(request)
                self.write(response.serialize())
            elif path == "/report":
                response = self._servicer.get_response(request)
                self.write(response.serialize())
            else:
                self.set_status(404)
                logger.error(f"No service found for {path}.")
                self.write(b"")
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            self.set_status(500)
            self.write(str(exc))


def create_http_controller_handler(
    host: str,
    port: int,
    servicer: Optional[HttpControllerServicer] = None,
) -> Tuple[TornadoHTTPServer, HttpControllerServicer]:
    """Create and return the Tornado HTTP server and service implementation."""
    controller_servicer = servicer or HttpControllerServicer()
    server = TornadoHTTPServer(
        host,
        port,
        [
            (r"/", HttpControllerHandler, dict(servicer=controller_servicer)),
            (r"/get", HttpControllerHandler, dict(servicer=controller_servicer)),
            (r"/report", HttpControllerHandler, dict(servicer=controller_servicer)),
        ],
    )
    return server, controller_servicer