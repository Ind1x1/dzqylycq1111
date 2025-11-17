from agent.data_collector.data_collector import DataCollector
from agent.data_collector.collected_data import ResourceData
from agent.monitor.resource import ResourceMonitor

class ResourceCollector(DataCollector):
    """
    ResourceCollector collects the resource usage of the node.
    """

    def __init__(self):
        super().__init__()
        self._monitor = ResourceMonitor().singleton_instance()

    def collect_data(self) -> object:
        # Get ResourceData object from monitor (may be running in subprocess)
        resource_data = self._monitor.report_resource()
        
        # Store collected resource data in queue
        if resource_data:
            self.store_data(resource_data)
        
        return resource_data

    def is_enabled(self) -> bool:
        return True