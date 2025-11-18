from pathlib import Path
from typing import List

from agent.data_collector.collected_data import WorkerTrainingMetric
from agent.data_collector.data_collector import DataCollector
from common.constants import CollectorType

class MetricCollector(DataCollector):
    """
    MetricCollector collects the metric of the node.
    """

    def __init__(self):
        super().__init__()
        self._monitor = RLHFTrainingMonitor().singleton_instance()
        self._collector_type = CollectorType.METRIC_COLLECTOR
        
    def collect_data(self) -> object:
        metric_data = self._monitor.report_metric()
        # Get ResourceData object from monitor (may be running in subprocess)
        if metric_data:
            self.store_data(metric_data)
        
        return metric_data

    def is_enabled(self) -> bool:
        return True