"""
多进程测试版本，模拟实际的分布式训练场景
每个进程启动独立的 HTTP 服务器，模拟不同的训练节点
适用于 Linux 环境
"""
import sys
import time
import pickle
import json
import os
import requests
import multiprocessing
from multiprocessing import Process, Queue, Manager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.data_collector.data_collector import DataCollector
from agent.data_collector.constants import CollectedNodeType
from agent.data_collector.metric_collector import MetricCollector
from agent.data_collector.stack_collector import StackTraceCollector
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


def worker_process(rank: int, world_size: int, port_queue: Queue, ready_queue: Queue, 
                   stop_event, test_type: str):
    """
    子进程工作函数，模拟一个训练节点
    
    Args:
        rank: 进程的 rank（0 到 world_size-1）
        world_size: 总进程数
        port_queue: 用于向主进程传递端口号的队列
        ready_queue: 用于同步启动的队列
        stop_event: 用于停止服务器的事件
        test_type: 测试类型 (log, resource, metric)
    """
    try:
        # 设置环境变量
        os.environ["AUTO_RL_GROUP_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        
        print(f"[Rank {rank}] Worker process started, PID: {os.getpid()}, test_type: {test_type}")
        
        # 根据测试类型创建不同的 collector
        port = find_free_port()
        
        if test_type == "log":
            collector = DummyCollector(
                f"log_collector_rank_{rank}", 
                f"log-lines-from-rank-{rank}"
            )
            collectors = {CollectorType.LOG_COLLECTOR: collector}
        elif test_type == "resource":
            collector = DummyCollector(
                f"resource_collector_rank_{rank}",
                {"cpu": 0.6 + rank * 0.05, "mem": 0.4 + rank * 0.05}
            )
            collectors = {CollectorType.RESOURCE_COLLECTOR: collector}
        elif test_type == "metric":
            # 创建训练指标文件
            metrics_file = Path(__file__).parent / f"check_training_metric_rank_{rank}.txt"
            base_timestamp = int(time.time())
            
            monitor = RLHFTrainingMonitor(
                node_type=NodeType.TRAIN_NODE,
                node_uid=rank,
                gpu_type=AcceleratorType.NVIDIA_GPU,
                metrics_path=str(metrics_file)
            )
            
            collector = MetricCollector(monitor=monitor)
            collectors = {CollectorType.METRIC_COLLECTOR: collector}
            
            # 写入初始指标数据（第一次）
            metrics_data_1 = {
                "step": 100 + rank * 10,
                "timestamp": base_timestamp + rank,
                "loss": 0.5 - rank * 0.01,
                "accuracy": 0.95 + rank * 0.01,
                "learning_rate": 0.001,
                "rank": rank
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics_data_1, f)
        elif test_type == "stack":
            # 创建堆栈收集器
            log_dir = Path(__file__).parent / "multiprocess_my_logs"
            collector = StackTraceCollector(log_dir=str(log_dir), n_line=100, rank=rank)
            collectors = {CollectorType.STACK_TRACE_COLLECTOR: collector}
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        # 创建 HTTP 服务器
        server, servicer = create_http_controller_handler(
            host="127.0.0.1",
            port=port,
            collectors=collectors,
        )
        server.start()
        
        # 等待服务器启动
        time.sleep(0.3)
        
        print(f"[Rank {rank}] Server started on port {port}")
        
        # 将端口号发送给主进程
        port_queue.put((rank, port, test_type))
        
        # 通知主进程已就绪
        ready_queue.put(rank)
        
        # 保持服务器运行，直到收到停止信号
        while not stop_event.is_set():
            time.sleep(0.1)
        
        print(f"[Rank {rank}] Stopping server...")
        server.stop()
        
        # 清理测试文件
        if test_type == "metric":
            if metrics_file.exists():
                os.remove(metrics_file)
        
        print(f"[Rank {rank}] Worker process finished")
        
    except Exception as e:
        print(f"[Rank {rank}] Error in worker process: {e}")
        import traceback
        traceback.print_exc()
        raise


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


def test_multiprocess_log_collector():
    """测试多进程 LOG 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess LOG Collector")
    print("="*60)
    
    world_size = 4
    manager = Manager()
    port_queue = manager.Queue()
    ready_queue = manager.Queue()
    stop_event = manager.Event()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, port_queue, ready_queue, stop_event, "log")
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程就绪并收集端口号
    port_map = {}
    for _ in range(world_size):
        ready_queue.get(timeout=10)
    
    for _ in range(world_size):
        rank, port, test_type = port_queue.get(timeout=10)
        port_map[rank] = port
    
    print(f"[Main] All workers ready. Port mapping: {port_map}")
    
    # 等待一下确保服务器完全启动
    time.sleep(0.5)
    
    # 从每个进程的服务器获取状态（直接采集数据）
    for rank in range(world_size):
        port = port_map[rank]
        print(f"\n[Main] Getting state from rank {rank} (port {port})...")
        
        # 每次请求都会触发一次新的数据采集
        data1 = _get_state(port, CollectorType.LOG_COLLECTOR)
        data2 = _get_state(port, CollectorType.LOG_COLLECTOR)
        
        print(f"[Main] Rank {rank} - Data 1: {data1}")
        print(f"[Main] Rank {rank} - Data 2: {data2}")
        
        assert data1 is not None, f"Rank {rank} should have data (first request)"
        assert data2 is not None, f"Rank {rank} should have data (second request)"
        assert f"rank-{rank}" in str(data1), f"Data should contain rank {rank}"
    
    print("\n[Main] LOG collector test passed!")
    
    # 停止所有进程
    stop_event.set()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    print("[Main] All worker processes stopped")


def test_multiprocess_resource_collector():
    """测试多进程 RESOURCE 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess RESOURCE Collector")
    print("="*60)
    
    world_size = 4
    manager = Manager()
    port_queue = manager.Queue()
    ready_queue = manager.Queue()
    stop_event = manager.Event()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, port_queue, ready_queue, stop_event, "resource")
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程就绪并收集端口号
    port_map = {}
    for _ in range(world_size):
        ready_queue.get(timeout=10)
    
    for _ in range(world_size):
        rank, port, test_type = port_queue.get(timeout=10)
        port_map[rank] = port
    
    print(f"[Main] All workers ready. Port mapping: {port_map}")
    
    # 等待一下确保服务器完全启动
    time.sleep(0.5)
    
    # 从每个进程的服务器获取状态（直接采集数据）
    for rank in range(world_size):
        port = port_map[rank]
        print(f"\n[Main] Getting state from rank {rank} (port {port})...")
        
        # 每次请求都会触发一次新的数据采集
        data1 = _get_state(port, CollectorType.RESOURCE_COLLECTOR)
        data2 = _get_state(port, CollectorType.RESOURCE_COLLECTOR)
        
        print(f"[Main] Rank {rank} - Data 1: {data1}")
        print(f"[Main] Rank {rank} - Data 2: {data2}")
        
        assert data1 is not None, f"Rank {rank} should have data (first request)"
        assert data2 is not None, f"Rank {rank} should have data (second request)"
        
        # 验证资源数据内容
        if data1:
            assert "cpu" in str(data1), "Resource should contain cpu info"
    
    print("\n[Main] RESOURCE collector test passed!")
    
    # 停止所有进程
    stop_event.set()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    print("[Main] All worker processes stopped")


def test_multiprocess_metric_collector():
    """测试多进程 METRIC 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess METRIC Collector")
    print("="*60)
    
    world_size = 3  # 使用3个进程测试指标收集
    manager = Manager()
    port_queue = manager.Queue()
    ready_queue = manager.Queue()
    stop_event = manager.Event()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, port_queue, ready_queue, stop_event, "metric")
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程就绪并收集端口号
    port_map = {}
    for _ in range(world_size):
        ready_queue.get(timeout=10)
    
    for _ in range(world_size):
        rank, port, test_type = port_queue.get(timeout=10)
        port_map[rank] = port
    
    print(f"[Main] All workers ready. Port mapping: {port_map}")
    
    # 等待一下确保服务器完全启动
    time.sleep(0.5)
    
    # 从每个进程的服务器获取状态（直接采集数据）
    for rank in range(world_size):
        port = port_map[rank]
        print(f"\n[Main] Getting state from rank {rank} (port {port})...")
        
        if rank == 0:
            # Rank 0: group_rank == 0，应该有数据
            # 第一次请求 - 采集当前的 metrics 数据
            data1 = _get_state(port, CollectorType.METRIC_COLLECTOR)
            
            print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
            assert data1 is not None, f"Rank {rank} should have data (first request)"
            
            # 验证第一次采集的数据
            if hasattr(data1, 'data_content'):
                metric_dict_1 = json.loads(data1.data_content)
                assert "step" in metric_dict_1, "Metric should contain 'step'"
                assert metric_dict_1["step"] == 100, "First request should get step 100"
                print(f"[Main] Rank {rank} metric 1: step={metric_dict_1.get('step')}, "
                      f"loss={metric_dict_1.get('loss')}, accuracy={metric_dict_1.get('accuracy')}")
            
            # 等待并更新 metrics 文件（模拟 15 秒时间间隔）
            time.sleep(0.1)
            metrics_file = Path(__file__).parent / f"check_training_metric_rank_{rank}.txt"
            base_timestamp = int(time.time())
            metrics_data_2 = {
                "step": 200 + rank * 10,
                "timestamp": base_timestamp + 20,  # 增加 20 秒（超过 15 秒限制）
                "loss": 0.4 - rank * 0.01,
                "accuracy": 0.96 + rank * 0.01,
                "learning_rate": 0.0009,
                "rank": rank
            }
            with open(metrics_file, "w") as f:
                json.dump(metrics_data_2, f)
            
            # 第二次请求 - 采集更新后的 metrics 数据
            data2 = _get_state(port, CollectorType.METRIC_COLLECTOR)
            
            print(f"[Main] Rank {rank} - Data 2 type: {type(data2)}")
            assert data2 is not None, f"Rank {rank} should have data (second request)"
            
            # 验证第二次采集的数据
            if hasattr(data2, 'data_content'):
                metric_dict_2 = json.loads(data2.data_content)
                assert "step" in metric_dict_2, "Metric should contain 'step'"
                assert metric_dict_2["step"] == 200, "Second request should get step 200"
                print(f"[Main] Rank {rank} metric 2: step={metric_dict_2.get('step')}, "
                      f"loss={metric_dict_2.get('loss')}, accuracy={metric_dict_2.get('accuracy')}")
            
        else:
            # Rank 1, 2: group_rank != 0，应该没有数据（因为 report_step 返回 None）
            data1 = _get_state(port, CollectorType.METRIC_COLLECTOR)
            
            print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
            print(f"[Main] Rank {rank} - As expected, no data because group_rank != 0")
            
            # 验证确实没有数据（这是符合设计的）
            assert data1 is None, f"Rank {rank} should have NO data (group_rank != 0)"
    
    print("\n[Main] METRIC collector test passed!")
    
    # 停止所有进程
    stop_event.set()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    print("[Main] All worker processes stopped")


def test_multiprocess_stack_collector():
    """测试多进程 STACK 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess STACK Collector")
    print("="*60)
    
    world_size = 4
    manager = Manager()
    port_queue = manager.Queue()
    ready_queue = manager.Queue()
    stop_event = manager.Event()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, port_queue, ready_queue, stop_event, "stack")
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程就绪并收集端口号
    port_map = {}
    for _ in range(world_size):
        ready_queue.get(timeout=10)
    
    for _ in range(world_size):
        rank, port, test_type = port_queue.get(timeout=10)
        port_map[rank] = port
    
    print(f"[Main] All workers ready. Port mapping: {port_map}")
    
    # 等待一下确保服务器完全启动
    time.sleep(0.5)
    
    # 预期的异常类型（根据 test_stack_collector.py）
    expected_exceptions = {
        0: "ValueError",
        1: "RuntimeError",
        2: "ZeroDivisionError",
        3: "TypeError"
    }
    
    expected_pids = {
        0: 3152,
        1: 36980,
        2: 33740,
        3: 13720
    }
    
    # 从每个进程的服务器获取状态（直接采集数据）
    for rank in range(world_size):
        port = port_map[rank]
        print(f"\n[Main] Getting stack trace from rank {rank} (port {port})...")
        
        # 获取堆栈数据
        stack_data = _get_state(port, CollectorType.STACK_TRACE_COLLECTOR)
        
        print(f"[Main] Rank {rank} - Stack data type: {type(stack_data)}")
        
        if stack_data is not None:
            print(f"[Main] Rank {rank} - has_exception: {stack_data.has_exception()}")
            print(f"[Main] Rank {rank} - exception_type: {stack_data.exception_type}")
            print(f"[Main] Rank {rank} - pid: {stack_data.pid}")
            print(f"[Main] Rank {rank} - stack_traces count: {len(stack_data.stack_traces)}")
            
            # 验证数据
            assert stack_data.has_exception(), f"Rank {rank} should have exception"
            assert stack_data.exception_type == expected_exceptions[rank], \
                f"Rank {rank} exception type should be {expected_exceptions[rank]}, got {stack_data.exception_type}"
            assert stack_data.pid == expected_pids[rank], \
                f"Rank {rank} PID should be {expected_pids[rank]}, got {stack_data.pid}"
            assert len(stack_data.stack_traces) > 0, f"Rank {rank} stack traces should not be empty"
        else:
            raise AssertionError(f"Rank {rank} should return stack data")
    
    print("\n[Main] STACK collector test passed!")
    
    # 停止所有进程
    stop_event.set()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    print("[Main] All worker processes stopped")


def main():
    """运行所有多进程测试"""
    print("\n" + "="*60)
    print("Starting Multiprocess HTTP Server Collector Tests")
    print("="*60)
    
    try:
        # 测试 LOG 收集器
        test_multiprocess_log_collector()
        time.sleep(1)
        
        # 测试 RESOURCE 收集器
        test_multiprocess_resource_collector()
        time.sleep(1)
        
        # 测试 METRIC 收集器
        test_multiprocess_metric_collector()
        time.sleep(1)
        
        # 测试 STACK 收集器
        test_multiprocess_stack_collector()
        
        print("\n" + "="*60)
        print("All multiprocess tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[Error] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 在 Linux 下使用 fork 模式（默认），在 Windows 下会自动使用 spawn
    if sys.platform != "win32":
        multiprocessing.set_start_method('fork', force=True)
    else:
        print("Warning: This test is designed for Linux. Windows may have issues.")
        multiprocessing.set_start_method('spawn', force=True)
    
    main()

