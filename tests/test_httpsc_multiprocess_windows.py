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
from agent.data_collector.collected_data import CollectedData
from agent.data_collector.constants import CollectedNodeType, CollectedDataType
from agent.data_collector.metric_collector import MetricCollector
from agent.monitor.training import RLHFTrainingMonitor
from agent.server import create_http_controller_handler
from common import comm
from common.comm import BaseRequest, BaseResponse, CollectorRequest
from common.constants import CollectorType, NodeType, AcceleratorType
from util.comm_util import find_free_port


class DummyCollector(DataCollector):
    """返回 CollectedData 对象的 Collector，用于测试。"""

    def __init__(self, name: str, payload: object, node_rank: int = 0):
        super().__init__(queue_size=10)
        self._name = name
        self._payload = payload
        self._node_rank = node_rank

    def collect_data(self):
        # 创建 CollectedData 对象
        data = CollectedData(
            timestamp=int(time.time()),
            data_type=CollectedDataType.GENERIC,
            data_content=json.dumps({
                "collector": self._name,
                "payload": self._payload,
                "timestamp": time.time(),
            }),
            node_id=self._node_rank,
            node_type=CollectedNodeType.TRAIN_NODE,
            node_rank=self._node_rank,
        )
        # 存储到队列
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
        
        # 收集一些数据到队列
        if test_type == "metric":
            # MetricCollector 有 15 秒时间间隔限制，需要模拟时间间隔
            for collector_type, collector in collectors.items():
                # 第一次收集
                collector.collect_data()
                
                # 模拟 15 秒时间间隔：修改 metrics 文件的 timestamp
                time.sleep(0.1)  # 短暂等待确保第一次收集完成
                metrics_data_2 = {
                    "step": 200 + rank * 10,
                    "timestamp": base_timestamp + rank + 20,  # 增加 20 秒（超过 15 秒限制）
                    "loss": 0.4 - rank * 0.01,
                    "accuracy": 0.96 + rank * 0.01,
                    "learning_rate": 0.0009,
                    "rank": rank
                }
                with open(metrics_file, "w") as f:
                    json.dump(metrics_data_2, f)
                
                # 第二次收集
                collector.collect_data()
        else:
            # LOG 和 RESOURCE 收集器可以多次收集
            for collector_type, collector in collectors.items():
                collector.collect_data()
                collector.collect_data()  # 收集两次，测试队列
        
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


def consume_queue_from_worker(port: int, collector_type: CollectorType):
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
        
        # 等待一下确保数据已收集
        time.sleep(0.5)
        
        # 从每个进程的服务器消费数据
        for rank in range(world_size):
            port = port_map[rank]
            print(f"\n[Main] Consuming data from rank {rank} (port {port})...")
            
            # 消费两次（每个进程收集了两次数据）
            data1 = consume_queue_from_worker(port, CollectorType.LOG_COLLECTOR)
            data2 = consume_queue_from_worker(port, CollectorType.LOG_COLLECTOR)
            
            print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
            print(f"[Main] Rank {rank} - Data 2 type: {type(data2)}")
            
            assert data1 is not None, f"Rank {rank} should have data (first consume)"
            assert data2 is not None, f"Rank {rank} should have data (second consume)"
            
            # 检查数据内容
            if hasattr(data1, 'data_content'):
                print(f"[Main] Rank {rank} - Data content: {data1.data_content}")
                assert f"rank-{rank}" in str(data1.data_content), f"Data should contain rank {rank}"
            else:
                print(f"[Main] Rank {rank} - Data (raw): {data1}")
                assert f"rank-{rank}" in str(data1), f"Data should contain rank {rank}"
            
            # 验证队列已空
            data3 = consume_queue_from_worker(port, CollectorType.LOG_COLLECTOR)
            assert data3 is None, f"Rank {rank} queue should be empty"
        
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
        
        # 等待一下确保数据已收集
        time.sleep(0.5)
        
        # 从每个进程的服务器消费数据
        for rank in range(world_size):
            port = port_map[rank]
            print(f"\n[Main] Consuming data from rank {rank} (port {port})...")
            
            # 消费两次
            data1 = consume_queue_from_worker(port, CollectorType.RESOURCE_COLLECTOR)
            data2 = consume_queue_from_worker(port, CollectorType.RESOURCE_COLLECTOR)
            
            print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
            
            assert data1 is not None, f"Rank {rank} should have data (first consume)"
            assert data2 is not None, f"Rank {rank} should have data (second consume)"
            
            # 验证资源数据内容
            if data1:
                if hasattr(data1, 'data_content'):
                    resource_dict = json.loads(data1.data_content)
                    assert "cpu" in str(resource_dict), "Resource should contain cpu info"
                    print(f"[Main] Rank {rank} resource: {resource_dict}")
                else:
                    assert "cpu" in str(data1), "Resource should contain cpu info"
            
            # 验证队列已空
            data3 = consume_queue_from_worker(port, CollectorType.RESOURCE_COLLECTOR)
            assert data3 is None, f"Rank {rank} queue should be empty"
        
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
        
        # 等待一下确保数据已收集
        time.sleep(0.5)
        
        # 从每个进程的服务器消费数据
        for rank in range(world_size):
            port = port_map[rank]
            print(f"\n[Main] Consuming data from rank {rank} (port {port})...")
            
            if rank == 0:
                # Rank 0: group_rank == 0，应该有两次数据（模拟了 15 秒间隔）
                # 注意：consume_data() 使用 LIFO（后进先出），最新的数据先被消费
                data1 = consume_queue_from_worker(port, CollectorType.METRIC_COLLECTOR)
                data2 = consume_queue_from_worker(port, CollectorType.METRIC_COLLECTOR)
                
                print(f"[Main] Rank {rank} - Data 1 type: {type(data1)}")
                print(f"[Main] Rank {rank} - Data 2 type: {type(data2)}")
                
                assert data1 is not None, f"Rank {rank} should have data (first consume)"
                assert data2 is not None, f"Rank {rank} should have data (second consume)"
                
                # 验证第一次消费的数据（LIFO：最新的数据，step=200）
                if hasattr(data1, 'data_content'):
                    metric_dict_1 = json.loads(data1.data_content)
                    assert "step" in metric_dict_1, "Metric should contain 'step'"
                    assert metric_dict_1["step"] == 200, "First consume should get latest data (step 200)"
                    print(f"[Main] Rank {rank} metric 1 (latest): step={metric_dict_1.get('step')}, "
                          f"loss={metric_dict_1.get('loss')}, accuracy={metric_dict_1.get('accuracy')}")
                
                # 验证第二次消费的数据（LIFO：较旧的数据，step=100）
                if hasattr(data2, 'data_content'):
                    metric_dict_2 = json.loads(data2.data_content)
                    assert "step" in metric_dict_2, "Metric should contain 'step'"
                    assert metric_dict_2["step"] == 100, "Second consume should get older data (step 100)"
                    print(f"[Main] Rank {rank} metric 2 (older): step={metric_dict_2.get('step')}, "
                          f"loss={metric_dict_2.get('loss')}, accuracy={metric_dict_2.get('accuracy')}")
                
                # 验证队列已空
                data3 = consume_queue_from_worker(port, CollectorType.METRIC_COLLECTOR)
                assert data3 is None, f"Rank {rank} queue should be empty after consuming all data"
                
            else:
                # Rank 1, 2: group_rank != 0，应该没有数据（因为 report_step 返回 None）
                data1 = consume_queue_from_worker(port, CollectorType.METRIC_COLLECTOR)
                
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

