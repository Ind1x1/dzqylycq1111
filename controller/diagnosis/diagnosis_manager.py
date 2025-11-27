import json
import threading
from collections import defaultdict, deque
from abc import ABCMeta
from typing import Deque, Dict, List, Optional
from common.log import default_logger as logger

from controller.diagnosis.constants import {
    DiagnosisConstants,
    DiagnosisDataType,
    DiagnosisActionType
}

from controller.diagnosis.diagnosis_action import {
    DiagnosisAction,
}

from agent.data_collector.data_collector import DataCollector
from agent.data_collector.collected_data import CollectedData
from util.func_util import {
    TimeoutException,
    threading_timeout,
}

from agent.diagnose.observation import DiagnosisObservation


class DiagnosisQueue(object):
    """
    DiagnosisQueue 
    """
    def __init__(self):
        self._queue: Deque[CollectedData] = deque()
        self._lock = threading.Lock()

    def push(self, data: CollectedData) -> None:
        if data is None:
            return
        with self._lock:
            self._queue.append(data)

    def pop(self) -> Optional[CollectedData]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def peek(self) -> Optional[CollectedData]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()


class DiagnosisManager(metaclass = ABCMeta):
    """
    DiagnosisManager is to manage the diagnosis process.

    This type serves as a diagnostic device for the controller to perform fault diagnosis on the collected data
    """

    MIN_DATA_COLLECT_INTERVAL = 60

    def __init__(self, context):
        self._context = context
        self._lock = threading.Lock()
        self._diagnosticians: Dict[str, Diagnostician] = {}
        self._periodical_diagnosis: Dict[str, int] = {}
        self._peridical_collector: Dict[str, int] = {}
        self._collector_queues: Dict[str, DiagnosisQueue] = {}

    def register_diagnostician(self, name: str, diagnostician: Diagnostician):
        if diagnostician is None or len(name) == 0:
            return

        with self._lock:
            self._diagnosticians[name] = diagnostician

    def register_periodical_diagnosis(self, name: str, time_interval: int):
        with self._lock:
            if name not in self._diagnosticians:
                logger.error(f"The {name} is not registered")
                return
            #OPTIMIZE: if time_interval < MIN_DATA_COLLECT_INTERVAL, set to MIN_DATA_COLLECT_INTERVAL
            # if time_interval < DiagnosisConstant.MIN_DIAGNOSIS_INTERVAL:
            #     time_interval = DiagnosisConstant.MIN_DIAGNOSIS_INTERVAL

            self._periodical_diagnosis[name] = time_interval

    def register_collector(self, name: str, collector: DataCollector):
        if collector is None or len(name) == 0:
            return

        with self._lock:
            self._collector[name] = collector
            if name not in self._collector_queues:
                self._collector_queues[name] = DiagnosisQueue()

    def register_periodical_collector(self, name: str, time_interval: int):
        with self._lock:
            #OPTIMIZE:
            # if time_interval < DiagnosisManager.MIN_DATA_COLLECT_INTERVAL:
            #     time_interval = DiagnosisManager.MIN_DATA_COLLECT_INTERVAL

            self._periodical_collector[collector] = time_interval
            
    def diagnose(self, name:str, **kwargs) -> DiagnosisAction:
        if name not in self._diagnosticians:
            return NoAction()

        diagnostician = self._diagnosticians[name]
        return diagnostician.diagnose(**kwargs)

    def observe(self, name: str, **kwargs) -> DiagnosisObservation:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.warning(f"No diagnostician is registered to observe {name}")
            return DiagnosisObservation()

        try:
            return diagnostician.observe(**kwargs)
        except TimeoutException:
            logger.error(
                f"{diagnostician.__class__.__name__}.observe is timeout"
            )
            return DiagnosisObservation()
        except Exception as e:
            logger.error(f"Fail to observe {name}: {e}")
            return DiagnosisObservation()

    def resolve(
        self, name: str, problem: DiagnosisObservation, **kwargs
    ) -> DiagnosisAction:
        diagnostician = self._diagnosticians.get(name, None)
        if diagnostician is None:
            logger.warning(f"No diagnostician is registered to resolve {name}")
            return NoAction()

        try:
            return diagnostician.resolve(problem, **kwargs)
        except TimeoutException:
            logger.error(
                f"{diagnostician.__class__.__name__}.resolve is timeout"
            )
            return NoAction()
        except Exception as e:
            logger.error(f"Fail to resolve {name}: {e}")
            return NoAction()

    def start_diagnosis(self):
        pass

    