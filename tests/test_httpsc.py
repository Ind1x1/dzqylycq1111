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
        super().__init__(queue_size=10)
        self._name = name
        self._payload = payload

    def collect_data(self):
        # 收集数据并自动存储到队列
        data = {
            "collector": self._name,
            "payload": self._payload,
            "timestamp": time.time(),
        }
        self.store_data(data)
        return data

    def is_enabled(self) -> bool:
        return True

    def report_state(self):
        """用于 get_state 端点，返回当前状态"""
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


def _consume_queue(port: int, collector_type: CollectorType):
    """从指定端口的服务器消费队列数据"""
    collector_request = CollectorRequest(collector_type=collector_type, node_id=0)
    request = BaseRequest(
        node_id=0,
        data=pickle.dumps(collector_request),
    )
    
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/consume_queue",
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
        print(f"Error consuming queue: {e}")
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

        # 测试 1: 通过 collector 收集数据（自动存储到队列）
        collector_log = servicer_log._collectors.get(CollectorType.LOG_COLLECTOR)
        collector_log.collect_data()
        collector_log.collect_data()  # 收集两次，测试队列
        
        collector_resource = servicer_usage._collectors.get(CollectorType.RESOURCE_COLLECTOR)
        collector_resource.collect_data()
        collector_resource.collect_data()  # 收集两次，测试队列
        
        # 测试 metric collector - 需要确保时间戳差异大于15秒
        collector_metric = servicer_metric._collectors.get(CollectorType.METRIC_COLLECTOR)
        
        # 第一次收集：写入第一个指标
        # 注意：_last_timestamp 初始为 0，所以第一次需要 timestamp > 15
        metrics_data_1 = {
            "step": 100,
            "timestamp": base_timestamp,  # base_timestamp 是当前时间戳，肯定 > 15
            "loss": 0.5,
            "accuracy": 0.95,
            "learning_rate": 0.001
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data_1, f)
        result1 = collector_metric.collect_data()
        print(f"[Test] First metric collection result: {result1 is not None}")
        
        # 等待一下，然后写入第二个指标（时间戳差异大于15秒）
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
        result2 = collector_metric.collect_data()
        print(f"[Test] Second metric collection result: {result2 is not None}")
        
        # 测试 2: 通过 HTTP 消费队列数据
        log_data_1 = _consume_queue(port_log, CollectorType.LOG_COLLECTOR)
        log_data_2 = _consume_queue(port_log, CollectorType.LOG_COLLECTOR)
        resource_data_1 = _consume_queue(port_usage, CollectorType.RESOURCE_COLLECTOR)
        resource_data_2 = _consume_queue(port_usage, CollectorType.RESOURCE_COLLECTOR)
        metric_data_1 = _consume_queue(port_metric, CollectorType.METRIC_COLLECTOR)
        metric_data_2 = _consume_queue(port_metric, CollectorType.METRIC_COLLECTOR)
        
        print("[Test] 从队列消费 log 数据 1:", log_data_1)
        print("[Test] 从队列消费 log 数据 2:", log_data_2)
        print("[Test] 从队列消费 resource 数据 1:", resource_data_1)
        print("[Test] 从队列消费 resource 数据 2:", resource_data_2)
        print("[Test] 从队列消费 metric 数据 1:", metric_data_1)
        print("[Test] 从队列消费 metric 数据 2:", metric_data_2)
        
        # 验证队列功能
        assert log_data_1 is not None, "Log queue should have data (first consume)"
        assert log_data_2 is not None, "Log queue should have data (second consume)"
        assert resource_data_1 is not None, "Resource queue should have data (first consume)"
        assert resource_data_2 is not None, "Resource queue should have data (second consume)"
        assert metric_data_1 is not None, "Metric queue should have data (first consume)"
        assert metric_data_2 is not None, "Metric queue should have data (second consume)"
        
        # 验证 metric 数据内容
        if metric_data_1:
            print(f"[Test] Metric data content: {metric_data_1.data_content}")
            metric_dict = json.loads(metric_data_1.data_content)
            assert "step" in metric_dict, "Metric should contain 'step'"
            assert "timestamp" in metric_dict, "Metric should contain 'timestamp'"
        
        # 验证消费后队列为空
        log_data_3 = _consume_queue(port_log, CollectorType.LOG_COLLECTOR)
        resource_data_3 = _consume_queue(port_usage, CollectorType.RESOURCE_COLLECTOR)
        metric_data_3 = _consume_queue(port_metric, CollectorType.METRIC_COLLECTOR)
        assert log_data_3 is None, "Log queue should be empty after consuming all data"
        assert resource_data_3 is None, "Resource queue should be empty after consuming all data"
        assert metric_data_3 is None, "Metric queue should be empty after consuming all data"
        
        print("[Test] 队列消费测试通过!")
        
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
