from abc import ABCMeta, abstractmethod
from collections import deque
from threading import Lock
from typing import Optional

class DataCollector(metaclass=ABCMeta):
    """
    DataCollector collects certain type of data and report to master.
    Those data is used to diagnosis the faults of training.
    
    Features:
    - Message queue with configurable max size
    - Thread-safe data storage and consumption
    - Automatically removes oldest data when queue is full
    """

    def __init__(self, queue_size: int = 10):
        """
        Initialize DataCollector with message queue.
        
        Args:
            queue_size: Maximum size of the message queue. Defaults to 10.
        """
        self._queue_size = queue_size
        self._message_queue: deque = deque(maxlen=queue_size)
        self._lock = Lock()

    @abstractmethod
    def collect_data(self) -> object:
        """The implementation of data collector."""
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether the collector is enabled."""
        return True

    def store_data(self, data: object) -> None:
        """
        Store data into queue.
        If queue exceeds max size, the oldest data is automatically removed.
        
        Args:
            data: The data object to store (can be any type).
        """
        with self._lock:
            # If queue is full, deque(maxlen=queue_size) automatically removes the oldest item
            self._message_queue.append(data)

    def consume_data(self) -> Optional[object]:
        """
        Consume (get and remove) the latest data from the queue.
        
        Returns:
            The latest data object from the queue, or None if queue is empty.
        """
        with self._lock:
            if not self._message_queue:
                return None
            # Remove and return the rightmost (latest) item
            return self._message_queue.pop()


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
        # Store collected data in queue
        self.store_data(data)
        return data