import sys
import time
import pickle
import json
import os
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agent.data_collector.data_collector import DataCollector
from agent.data_collector.constants import CollectedNodeType
from agent.data_collector.metric_collector import MetricCollector
from agent.monitor.training import RLHFTrainingMonitor
from agent.server import create_http_controller_handler
from common import comm
from common.comm import BaseRequest, BaseResponse, CollectorRequest
from common.constants import CollectorType, NodeType, AcceleratorType
from util.comm_util import find_free_port

class DummyCollector(DataCollector):
    """返回固定结果的 Collector，便于测试。"""

    def __init__(self, name: str, payload: object):
        super().__init__()
        self._name = name
        self._payload = payload

    def collect_data(self):
        # 每次请求时直接采集并返回数据
        data = {
            "collector": self._name,
            "payload": self._payload,
            "timestamp": time.time(),
        }
        return data

    def is_enabled(self) -> bool:
        return True


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


def _get_state(port: int, collector_type: CollectorType):
    """从指定端口的服务器获取状态（直接采集数据）"""
    collector_request = CollectorRequest(collector_type=collector_type, node_id=0)
    request = BaseRequest(
        node_id=0,
        data=pickle.dumps(collector_request),
    )
    
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/get_state",
            json=request.to_json(),
            timeout=5,
        )
        if response.status_code != 200:
            print(f"HTTP request failed, status={response.status_code}, body={response.text}")
            return None
        
        # Server returns pickle-serialized BaseResponse, deserialize it
        response_data = comm.deserialize_message(response.content)
        if not isinstance(response_data, BaseResponse):
            print(f"Failed to deserialize response data, got {type(response_data)}")
            return None
        
        if response_data.success and response_data.data:
            return pickle.loads(response_data.data)
        return None
    except Exception as e:
        print(f"Error getting state: {e}")
        return None


def test_httpsc_end_to_end():
    # 设置环境变量，确保 group_rank 为 0，这样 report_step 才能返回数据
    import os
    original_group_rank = os.environ.get("AUTO_RL_GROUP_RANK")
    os.environ["AUTO_RL_GROUP_RANK"] = "0"
    
    try:
        server_log, servicer_log, port_log = _start_server(CollectorType.LOG_COLLECTOR, "log-lines")
        server_usage, servicer_usage, port_usage = _start_server(CollectorType.RESOURCE_COLLECTOR, {"cpu": 0.6, "mem": 0.4})
        
        # 创建训练指标文件
        metrics_file = Path(__file__).parent / "check_training_metric.txt"
        base_timestamp = int(time.time())
        
        # 初始化 RLHFTrainingMonitor 并设置 metrics_path
        monitor = RLHFTrainingMonitor(
            node_type=NodeType.TRAIN_NODE,
            node_uid=0,
            gpu_type=AcceleratorType.NVIDIA_GPU,
            metrics_path=str(metrics_file)
        )
        
        # 创建 MetricCollector 并传入 monitor 实例
        metric_collector = MetricCollector(monitor=monitor)
        port_metric = find_free_port()
        collectors_metric = {
            CollectorType.METRIC_COLLECTOR: metric_collector,
        }
        server_metric, servicer_metric = create_http_controller_handler(
            host="127.0.0.1",
            port=port_metric,
            collectors=collectors_metric,
        )
        server_metric.start()
        time.sleep(0.2)

        # 测试 1: 通过 HTTP get_state 端点直接采集数据
        # 每次请求都会触发一次新的数据采集
        
        # 准备 metric 数据 - 第一次收集
        metrics_data_1 = {
            "step": 100,
            "timestamp": base_timestamp,  # base_timestamp 是当前时间戳，肯定 > 15
            "loss": 0.5,
            "accuracy": 0.95,
            "learning_rate": 0.001
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data_1, f)
        
        # 通过 HTTP 获取状态（直接采集）
        log_data_1 = _get_state(port_log, CollectorType.LOG_COLLECTOR)
        resource_data_1 = _get_state(port_usage, CollectorType.RESOURCE_COLLECTOR)
        metric_data_1 = _get_state(port_metric, CollectorType.METRIC_COLLECTOR)
        
        print("[Test] 第一次获取 log 数据:", log_data_1)
        print("[Test] 第一次获取 resource 数据:", resource_data_1)
        print("[Test] 第一次获取 metric 数据:", metric_data_1)
        
        # 验证第一次采集
        assert log_data_1 is not None, "Log collector should return data"
        assert resource_data_1 is not None, "Resource collector should return data"
        assert metric_data_1 is not None, "Metric collector should return data"
        
        # 验证 metric 数据内容
        if metric_data_1:
            print(f"[Test] Metric data content: {metric_data_1.data_content}")
            metric_dict = json.loads(metric_data_1.data_content)
            assert "step" in metric_dict, "Metric should contain 'step'"
            assert "timestamp" in metric_dict, "Metric should contain 'timestamp'"
            assert metric_dict["step"] == 100, "First metric should have step 100"
        
        # 等待一下，然后更新 metric 数据并再次采集
        time.sleep(0.1)
        metrics_data_2 = {
            "step": 200,
            "timestamp": base_timestamp + 20,  # 确保时间戳差异大于15秒
            "loss": 0.4,
            "accuracy": 0.96,
            "learning_rate": 0.0009
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data_2, f)
        
        # 第二次通过 HTTP 获取状态（应该采集到新的数据）
        log_data_2 = _get_state(port_log, CollectorType.LOG_COLLECTOR)
        resource_data_2 = _get_state(port_usage, CollectorType.RESOURCE_COLLECTOR)
        metric_data_2 = _get_state(port_metric, CollectorType.METRIC_COLLECTOR)
        
        print("[Test] 第二次获取 log 数据:", log_data_2)
        print("[Test] 第二次获取 resource 数据:", resource_data_2)
        print("[Test] 第二次获取 metric 数据:", metric_data_2)
        
        # 验证第二次采集
        assert log_data_2 is not None, "Log collector should return data on second request"
        assert resource_data_2 is not None, "Resource collector should return data on second request"
        assert metric_data_2 is not None, "Metric collector should return data on second request"
        
        # 验证 metric 数据已更新
        if metric_data_2:
            metric_dict_2 = json.loads(metric_data_2.data_content)
            assert metric_dict_2["step"] == 200, "Second metric should have step 200"
        
        print("[Test] 直接采集测试通过!")
        
    finally:
        server_log.stop()
        server_usage.stop()
        server_metric.stop()
        # 清理测试文件
        if metrics_file.exists():
            os.remove(metrics_file)
        # 恢复环境变量
        if original_group_rank is not None:
            os.environ["AUTO_RL_GROUP_RANK"] = original_group_rank
        elif "AUTO_RL_GROUP_RANK" in os.environ:
            del os.environ["AUTO_RL_GROUP_RANK"]
        time.sleep(0.1)


if __name__ == "__main__":
    test_httpsc_end_to_end()
