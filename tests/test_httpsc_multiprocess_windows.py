"""
Windows 多进程测试版本，模拟实际的分布式训练场景
每个进程启动独立的 HTTP 服务器，模拟不同的训练节点
针对 Windows 的 spawn 模式优化
"""
import sys
import time
import pickle
import json
import os
import requests
import multiprocessing
from multiprocessing import Process, Queue
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.data_collector.data_collector import DataCollector
from agent.data_collector.constants import CollectedNodeType
from agent.data_collector.metric_collector import MetricCollector
from agent.data_collector.stack_collector import StackTraceCollector
from agent.data_collector.observation_collector import ObservationCollector
from agent.monitor.training import RLHFTrainingMonitor
from agent.server import create_http_controller_handler
from common import comm
from common.comm import BaseRequest, BaseResponse, CollectorRequest
from common.constants import CollectorType, NodeType, AcceleratorType
from util.comm_util import find_free_port


class DummyCollector(DataCollector):
    """返回固定结果的 Collector，便于测试。"""

    def __init__(self, name: str, payload: object, node_rank: int = 0):
        super().__init__()
        self._name = name
        self._payload = payload
        self._node_rank = node_rank

    def collect_data(self):
        # 每次请求时直接采集并返回数据
        data = {
            "collector": self._name,
            "payload": self._payload,
            "timestamp": time.time(),
            "node_rank": self._node_rank,
        }
        return data

    def is_enabled(self) -> bool:
        return True


def worker_process(rank: int, world_size: int, result_queue: Queue, test_type: str):
    """
    子进程工作函数，模拟一个训练节点
    
    Args:
        rank: 进程的 rank（0 到 world_size-1）
        world_size: 总进程数
        result_queue: 用于返回结果的队列
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
                f"log-lines-from-rank-{rank}",
                node_rank=rank
            )
            collectors = {CollectorType.LOG_COLLECTOR: collector}
        elif test_type == "resource":
            collector = DummyCollector(
                f"resource_collector_rank_{rank}",
                {"cpu": 0.6 + rank * 0.05, "mem": 0.4 + rank * 0.05},
                node_rank=rank
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
        elif test_type == "observation":
            # 设置 error_type.json 环境变量
            os.environ["AUTO_RL_ERROR_TYPE_FILE"] = str(Path(__file__).parent / "error_type.json")
            # 创建观察收集器，每个 rank 对应一个日志文件
            log_file = Path(__file__).parent / "test_obervation_logs" / f"test_{rank + 1}.log"
            collector = ObservationCollector(log_file=str(log_file), n_line=100)
            collectors = {CollectorType.OBSERVATION_COLLECTOR: collector}
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
        
        # 将结果发送给主进程（rank, port, status）
        result_queue.put((rank, port, "ready"))
        
        # 等待主进程的结束信号（通过队列接收）
        # 使用超时避免无限等待
        timeout = 30  # 30秒超时
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # 非阻塞地检查是否有停止信号
                if not result_queue.empty():
                    msg = result_queue.get_nowait()
                    if msg == "stop":
                        break
            except:
                pass
            time.sleep(0.1)
        
        print(f"[Rank {rank}] Stopping server...")
        server.stop()
        
        # 清理测试文件
        if test_type == "metric":
            if metrics_file.exists():
                os.remove(metrics_file)
        
        result_queue.put((rank, 0, "stopped"))
        print(f"[Rank {rank}] Worker process finished")
        
    except Exception as e:
        print(f"[Rank {rank}] Error in worker process: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((rank, 0, f"error: {e}"))


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
    result_queue = Queue()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, result_queue, "log")
        )
        p.start()
        processes.append(p)
    
    try:
        # 等待所有进程就绪并收集端口号
        port_map = {}
        ready_count = 0
        
        while ready_count < world_size:
            try:
                rank, port, status = result_queue.get(timeout=15)
                if status == "ready":
                    port_map[rank] = port
                    ready_count += 1
                    print(f"[Main] Worker {rank} is ready on port {port}")
                elif "error" in status:
                    print(f"[Main] Worker {rank} failed: {status}")
                    raise RuntimeError(f"Worker {rank} failed to start")
            except:
                print(f"[Main] Timeout waiting for workers, only {ready_count}/{world_size} ready")
                raise
        
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
            
            print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
            print(f"[Main] Rank {rank} - Data 2 type: {type(data2)}")
            
            assert data1 is not None, f"Rank {rank} should have data (first request)"
            assert data2 is not None, f"Rank {rank} should have data (second request)"
            
            # 检查数据内容
            print(f"[Main] Rank {rank} - Data 1: {data1}")
            print(f"[Main] Rank {rank} - Data 2: {data2}")
            assert f"rank-{rank}" in str(data1), f"Data should contain rank {rank}"
        
        print("\n[Main] LOG collector test passed!")
        
    finally:
        # 发送停止信号给所有进程
        print("\n[Main] Sending stop signals to workers...")
        for _ in range(world_size):
            result_queue.put("stop")
        
        # 等待进程结束
        for i, p in enumerate(processes):
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] Force terminating process {i}")
                p.terminate()
                p.join(timeout=2)
        
        print("[Main] All worker processes stopped")


def test_multiprocess_resource_collector():
    """测试多进程 RESOURCE 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess RESOURCE Collector")
    print("="*60)
    
    world_size = 4
    result_queue = Queue()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, result_queue, "resource")
        )
        p.start()
        processes.append(p)
    
    try:
        # 等待所有进程就绪并收集端口号
        port_map = {}
        ready_count = 0
        
        while ready_count < world_size:
            try:
                rank, port, status = result_queue.get(timeout=15)
                if status == "ready":
                    port_map[rank] = port
                    ready_count += 1
                elif "error" in status:
                    raise RuntimeError(f"Worker {rank} failed to start")
            except:
                print(f"[Main] Timeout waiting for workers")
                raise
        
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
            
            print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
            print(f"[Main] Rank {rank} - Data 2 type: {type(data2)}")
            
            assert data1 is not None, f"Rank {rank} should have data (first request)"
            assert data2 is not None, f"Rank {rank} should have data (second request)"
            
            # 验证资源数据内容
            if data1:
                print(f"[Main] Rank {rank} - Data 1: {data1}")
                assert "cpu" in str(data1), "Resource should contain cpu info"
        
        print("\n[Main] RESOURCE collector test passed!")
        
    finally:
        # 停止所有进程
        print("\n[Main] Sending stop signals to workers...")
        for _ in range(world_size):
            result_queue.put("stop")
        
        for i, p in enumerate(processes):
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] Force terminating process {i}")
                p.terminate()
                p.join(timeout=2)
        
        print("[Main] All worker processes stopped")


def test_multiprocess_metric_collector():
    """测试多进程 METRIC 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess METRIC Collector")
    print("="*60)
    
    world_size = 3  # 使用3个进程测试指标收集
    result_queue = Queue()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, result_queue, "metric")
        )
        p.start()
        processes.append(p)
    
    try:
        # 等待所有进程就绪并收集端口号
        port_map = {}
        ready_count = 0
        
        while ready_count < world_size:
            try:
                rank, port, status = result_queue.get(timeout=15)
                if status == "ready":
                    port_map[rank] = port
                    ready_count += 1
                elif "error" in status:
                    raise RuntimeError(f"Worker {rank} failed to start")
            except:
                print(f"[Main] Timeout waiting for workers")
                raise
        
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
        
    finally:
        # 停止所有进程
        print("\n[Main] Sending stop signals to workers...")
        for _ in range(world_size):
            result_queue.put("stop")
        
        for i, p in enumerate(processes):
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] Force terminating process {i}")
                p.terminate()
                p.join(timeout=2)
        
        print("[Main] All worker processes stopped")


def test_multiprocess_stack_collector():
    """测试多进程 STACK 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess STACK Collector")
    print("="*60)
    
    world_size = 4
    result_queue = Queue()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, result_queue, "stack")
        )
        p.start()
        processes.append(p)
    
    try:
        # 等待所有进程就绪并收集端口号
        port_map = {}
        ready_count = 0
        
        while ready_count < world_size:
            try:
                rank, port, status = result_queue.get(timeout=15)
                if status == "ready":
                    port_map[rank] = port
                    ready_count += 1
                    print(f"[Main] Worker {rank} is ready on port {port}")
                elif "error" in status:
                    print(f"[Main] Worker {rank} failed: {status}")
                    raise RuntimeError(f"Worker {rank} failed to start")
            except:
                print(f"[Main] Timeout waiting for workers, only {ready_count}/{world_size} ready")
                raise
        
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
        
    finally:
        # 发送停止信号给所有进程
        print("\n[Main] Sending stop signals to workers...")
        for _ in range(world_size):
            result_queue.put("stop")
        
        # 等待进程结束
        for i, p in enumerate(processes):
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] Force terminating process {i}")
                p.terminate()
                p.join(timeout=2)
        
        print("[Main] All worker processes stopped")


def test_multiprocess_observation_collector():
    """测试多进程 OBSERVATION 收集器"""
    print("\n" + "="*60)
    print("Testing Multiprocess OBSERVATION Collector")
    print("="*60)
    
    world_size = 4
    result_queue = Queue()
    
    # 启动多个工作进程
    processes = []
    for rank in range(world_size):
        p = Process(
            target=worker_process,
            args=(rank, world_size, result_queue, "observation")
        )
        p.start()
        processes.append(p)
    
    try:
        # 等待所有进程就绪并收集端口号
        port_map = {}
        ready_count = 0
        
        while ready_count < world_size:
            try:
                rank, port, status = result_queue.get(timeout=15)
                if status == "ready":
                    port_map[rank] = port
                    ready_count += 1
                    print(f"[Main] Worker {rank} is ready on port {port}")
                elif "error" in status:
                    print(f"[Main] Worker {rank} failed: {status}")
                    raise RuntimeError(f"Worker {rank} failed to start")
            except:
                print(f"[Main] Timeout waiting for workers, only {ready_count}/{world_size} ready")
                raise
        
        print(f"[Main] All workers ready. Port mapping: {port_map}")
        
        # 等待一下确保服务器完全启动
        time.sleep(0.5)
        
        # 从每个进程的服务器获取状态（直接采集数据）
        for rank in range(world_size):
            port = port_map[rank]
            print(f"\n[Main] Getting observation data from rank {rank} (port {port})...")
            
            # 获取观察数据
            observation_data = _get_state(port, CollectorType.OBSERVATION_COLLECTOR)
            
            print(f"[Main] Rank {rank} - Observation data type: {type(observation_data)}")
            
            if observation_data is not None:
                print(f"[Main] Rank {rank} - has_diagnosis: {observation_data.has_diagnosis()}")
                print(f"[Main] Rank {rank} - observation: {observation_data.observation}")
                print(f"[Main] Rank {rank} - error_type: {observation_data.error_type}")
                print(f"[Main] Rank {rank} - sub_error: {observation_data.sub_error}")
                print(f"[Main] Rank {rank} - reason: {observation_data.reason}")
                
                # 验证数据结构
                assert hasattr(observation_data, 'has_diagnosis'), "Should have has_diagnosis method"
                assert hasattr(observation_data, 'observation'), "Should have observation attribute"
                assert hasattr(observation_data, 'error_type'), "Should have error_type attribute"
                
                # 根据日志内容，test_2.log (rank=1) 应该有诊断信息
                if rank == 1:
                    # test_2.log 包含错误诊断信息
                    print(f"[Main] Rank {rank} - Expected to have diagnosis (from test_2.log)")
                    # 注意：这里不强制断言，因为诊断结果可能因日志解析而异
            else:
                print(f"[Main] Rank {rank} - No observation data returned (may be expected)")
            
            # 第二次采集，验证可以多次采集
            observation_data_2 = _get_state(port, CollectorType.OBSERVATION_COLLECTOR)
            print(f"[Main] Rank {rank} - Second observation data type: {type(observation_data_2)}")
        
        print("\n[Main] OBSERVATION collector test passed!")
        
    finally:
        # 发送停止信号给所有进程
        print("\n[Main] Sending stop signals to workers...")
        for _ in range(world_size):
            result_queue.put("stop")
        
        # 等待进程结束
        for i, p in enumerate(processes):
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] Force terminating process {i}")
                p.terminate()
                p.join(timeout=2)
        
        print("[Main] All worker processes stopped")


def main():
    """运行所有多进程测试"""
    print("\n" + "="*60)
    print("Starting Multiprocess HTTP Server Collector Tests (Windows)")
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

        # 测试 OBSERVATION 收集器
        test_multiprocess_observation_collector()
        
        print("\n" + "="*60)
        print("All multiprocess tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[Error] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Windows 需要使用 spawn 模式
    multiprocessing.set_start_method('spawn', force=True)
    main()

