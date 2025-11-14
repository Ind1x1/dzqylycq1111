import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.data_collector.collected import Collector
from agent.data_collector.constants import CollectedNodeType
from agent.server import create_http_controller_handler
from controller.client import HttpControllerClient
from util.comm_util import find_free_port


class DummyCollector(Collector):
    """返回固定结果的 Collector，便于测试。"""

    def __init__(self, name: str, payload: object):
        super().__init__()
        self._name = name
        self._payload = payload

    def collect_data(self):
        return self._payload

    def report_state(self):
        return {
            "collector": self._name,
            "payload": self._payload,
            "timestamp": time.time(),
        }


def _start_server(state_type: str, payload: object):
    port = find_free_port()
    collectors = {
        state_type: DummyCollector(state_type, payload),
    }
    server, servicer = create_http_controller_handler(
        host="127.0.0.1",
        port=port,
        collectors=collectors,
    )
    server.start()
    # 留一点时间让 Tornado loop 启动
    time.sleep(0.2)
    return server, servicer, port


def test_httpsc_end_to_end():
    server_log, _, port_log = _start_server("log", "log-lines")
    server_usage, _, port_usage = _start_server("usage", {"cpu": 0.6, "mem": 0.4})

    try:
        addr_map = {
            0: f"127.0.0.1:{port_log}",
            1: f"127.0.0.1:{port_usage}",
        }
        client = HttpControllerClient(
            servicers_addr=addr_map,
            timeout=5,
        )

        log_state = client.get_state("log", servicer_id=0)
        usage_state = client.get_state("usage", servicer_id=1)

        assert log_state is not None and log_state["collector"] == "log"
        assert usage_state is not None and usage_state["collector"] == "usage"

        print("[Test] 从 server 0 获取 log 状态:", log_state)
        print("[Test] 从 server 1 获取 usage 状态:", usage_state)
    finally:
        server_log.stop()
        server_usage.stop()
        time.sleep(0.1)


if __name__ == "__main__":
    test_httpsc_end_to_end()
