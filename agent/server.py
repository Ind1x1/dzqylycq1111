import json
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import tornado.web

from common import comm
from common.comm import BaseRequest, BaseResponse, CollectorRequest
from common.constants import CollectorType, CommunicationReqMeta
from common.http_server import TornadoHTTPServer
from common.log import default_logger as logger

from agent.data_collector.data_collector import DataCollector

class ControllerServicer(ABC):
    """Controller service base class."""
    def __init__(
        self, collectors: Optional[Dict[CollectorType, DataCollector]] = None
    ):
        self._collectors = collectors or {}
        self._lock = threading.Lock()
        self._reports: List[object] = []

    @abstractmethod
    def get_state(self, request: BaseRequest) -> BaseResponse:
        pass

    def get_response(self, request: BaseRequest) -> BaseResponse:
        pass

    def validate_request(self, headers: Dict[str, str]) -> bool:
        """Validate request headers."""
        return True
    
    def _deserialize_report(self, request: BaseRequest) -> Optional[object]:
        """Deserialize the report data from request."""
        message = None
        if request.data:
            try:
                message = pickle.loads(request.data)
            except Exception as exc:
                logger.error(f"Failed to deserialize the report data: {exc}")
        return message


class HttpControllerServicer(ControllerServicer):
    """HTTP Controller Servicer running on Worker that collects states from collectors."""
    
    def __init__(self, collectors: Optional[Dict[str, object]] = None):
        super().__init__(collectors)
    
    def get_response(self, request: BaseRequest) -> BaseResponse:
        """
        Handle report request from client (for storing data).
        
        Args:
            request: BaseRequest containing the reported data
            
        Returns:
            BaseResponse indicating success or failure
        """
        message = self._deserialize_report(request)
        if message is None:
            logger.error("Received empty report message.")
            return BaseResponse(success=False)
        
        with self._lock:
            self._reports.append(message)
        
        logger.info(f"[HttpControllerServicer] Received report: {type(message).__name__}")
        return BaseResponse(success=True)
    
    def get_state(self, request: BaseRequest) -> BaseResponse:
        """
        Get state from a specific collector based on request.
        
        Master client sends a request specifying which collector type to collect (log, resource, stack).
        This method routes to the appropriate collector and returns the state.
        
        Args:
            request: BaseRequest containing CollectorRequest in data field
            
        Returns:
            BaseResponse with collected state data, or error if collector not found/available
        """
        try:
            if not request.data:
                logger.error("CollectorRequest data is empty.")
                return BaseResponse(success=False)
            
            collector_request = pickle.loads(request.data)
            if not isinstance(collector_request, CollectorRequest):
                logger.error(f"Invalid CollectorRequest type: {type(collector_request)}")
                return BaseResponse(success=False)
            
            # collector_type is guaranteed to be CollectorType value (e.g., "LOG_COLLECTOR")
            collector_type = collector_request.collector_type
        except Exception as exc:
            logger.error(f"Failed to deserialize CollectorRequest: {exc}")
            return BaseResponse(success=False)
        
        collector = self._collectors.get(collector_type)
        if collector is None:
            logger.warning(f"Collector for collector_type '{collector_type}' is not configured.")
            return BaseResponse(success=False)
        
        try:
            state = collector.collect_data()
            if state is None:
                logger.warning(f"Collector '{collector_type}' returned None state.")
                return BaseResponse(success=False)
            
            state_data = pickle.dumps(state)
            logger.info(f"[HttpControllerServicer] Collected {collector_type} state successfully")
            return BaseResponse(success=True, data=state_data)
        except Exception as exc:
            logger.error(f"Failed to collect state from {collector_type} collector: {exc}")
            return BaseResponse(success=False)


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

            if path == "/check":
                # Health check endpoint
                self.write("ok")
            elif path == "/get_state":
                # Get state from collector
                response = self._servicer.get_state(request)
                self.write(response.serialize())
            elif path == "/report":
                # Report data to servicer
                pass
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
    collectors: Optional[Dict[str, object]] = None,
) -> Tuple[TornadoHTTPServer, HttpControllerServicer]:
    """
    Create and return the Tornado HTTP server and service implementation.
    
    Args:
        host: Server host address
        port: Server port
        servicer: Optional pre-configured servicer instance
        collectors: Dict mapping collector_type to collector, e.g., {"log": log_collector, "resource": resource_collector}
        
    Returns:
        Tuple of (TornadoHTTPServer, HttpControllerServicer)
    """
    controller_servicer = HttpControllerServicer(collectors=collectors)
    server = TornadoHTTPServer(
        host,
        port,
        [
            (r"/", HttpControllerHandler, dict(servicer=controller_servicer)),
            (r"/check", HttpControllerHandler, dict(servicer=controller_servicer)),
            (r"/get_state", HttpControllerHandler, dict(servicer=controller_servicer)),
            (r"/report", HttpControllerHandler, dict(servicer=controller_servicer)),
        ],
    )
    return server, controller_servicer