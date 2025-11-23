from pathlib import Path
from typing import List, Optional

from agent.data_collector.collected_data import WorkerTrainingMetric
from agent.data_collector.data_collector import DataCollector
from agent.monitor.training import RLHFTrainingMonitor
from common.constants import CollectorType

class MetricCollector(DataCollector):
    """
    MetricCollector collects the metric of the node.
    """

    def __init__(self, monitor: Optional[RLHFTrainingMonitor] = None):
        """
        Initialize MetricCollector.
        
        Args:
            monitor: Optional RLHFTrainingMonitor instance. If None, will try to get singleton instance.
        """
        super().__init__()
        if monitor is not None:
            self._monitor = monitor
        else:
            # Try to get singleton instance if it exists, otherwise it will raise error
            # The singleton should be initialized before creating MetricCollector
            self._monitor = RLHFTrainingMonitor.singleton_instance()
        self._collector_type = CollectorType.METRIC_COLLECTOR
        
    def collect_data(self) -> object:
        # Use report_step() to get training metrics
        metric_data = self._monitor.report_step()   
        return metric_data

    def is_enabled(self) -> bool:
        return True