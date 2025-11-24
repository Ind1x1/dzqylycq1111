from typing import Optional

from agent.data_collector.data_collector import DataCollector
from agent.data_collector.collected_data import ResourceData
from agent.monitor.resource import ResourceMonitor
from common.constants import CollectorType

class ResourceCollector(DataCollector):
    """
    ResourceCollector collects the resource usage of the node.
    """

    def __init__(self, monitor: Optional[ResourceMonitor] = None):
        super().__init__()
        if monitor is not None:
            self._monitor = monitor
        else:
            self._monitor = ResourceMonitor.singleton_instance()
        self._collector_type = CollectorType.RESOURCE_COLLECTOR

    def collect_data(self) -> object:
        # Get ResourceData object from monitor (may be running in subprocess)
        resource_data = self._monitor.report_resource()     
        return resource_data

    def is_enabled(self) -> bool:
        return True