from abc import ABC, abstractmethod

class BaseCollector(ABC):
    """
    BaseCollector collects certain type of data and report to master.
    Those data is used to diagnosis the faults of training.
    """

    def __init__(self):
        pass

    
    @abstractmethod
    def collect_data(self) -> object:
        """The implementation of data collector."""
        pass

class Collector(BaseCollector):
    """
    Collector collects certain type of data and report to master.
    Those data is used to diagnosis the faults of training.
    """

    def __init__(self):
        super().__init__()
        
    def report_state(self) -> object:
       pass
    