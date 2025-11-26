from abc import ABCMeta
from typing import Dict

from common.log import default_logger as logger

from controller.diagnosis.constants import DiagnosisConstants

from controller.diagnosis.diagnosis_action import {
    DiagnosisAction, 
    NoAction, 
    EventAction,
}

from util.func_util import {
    TimeoutException,
    threading_timeout,
}



class DiagnosisObservation(metaclass=ABCMeta):
    def __init__(self, observation:str = ""):
        self._observation = observation
        self.extra_infos: Dict[str, str] = {}

    @property
    def observation(self):
        return self._observation

class Diagnostician:
    """
    Diagnostician is to observe problems and resolve those problems.


    """
    def __init__(self):
        pass

    @threading_timeout(secs=DiagnosisConstants.MIN_DIAGNOSIS_INTERVAL)
    def observe(self, **kwargs): -> DiagnosisObservation:
        return DiagnosisObservation("unknown")

    @threading_timeout(secs=DiagnosisConstants.MIN_DIAGNOSIS_INTERVAL)
    def resovle(
        self,
        problem: DiagnosisObservation,**kwargs
    ) -> DiagnosisAction:
        return EventAction()

    def diagnose(self, **kwargs) -> DiagnosisAction:
        # define the diagnosis procedure
        try:
            ob = self.observe(**kwargs)
            return self.resolve(ob, **kwargs)
        except TimeoutException:
            logger.error(
                f"The diagnosis of {self.__class__.__name__} is timeout."
            )
            return NoAction()
        except Exception as e:
            logger.error(f"Fail to diagnose the problem: {e}")
            return NoAction()