from abc import ABCMeta, abstractmethod

class DataCollector(metaclass=ABCMeta):
    """
    DataCollector collects certain type of data and report to master.
    Those data is used to diagnosis the faults of training.
    
    Each client request triggers a fresh data collection via collect_data().
    """

    def __init__(self):
        """Initialize DataCollector."""
        pass

    @abstractmethod
    def collect_data(self) -> object:
        """The implementation of data collector."""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether the collector is enabled."""
        return True


class SimpleDataCollector(DataCollector):
    """
    An simple implementation of data collector
    """

    def __init__(self):
        super().__init__()

    def is_enabled(self) -> bool:
        return True

    def collect_data(self) -> object:
        data = "data"
        return data