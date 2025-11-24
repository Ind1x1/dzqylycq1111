"""
测试 StackTraceCollector 能否正确收集堆栈数据

测试两个场景：
1. 单进程场景：simple_my_logs_minimal
2. 多进程场景：multiprocess_my_logs
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent.data_collector.stack_collector import StackTraceCollector


def test_single_process_logs():
    """测试单进程日志收集"""
    print("\n" + "="*70)
    print("测试 1: 单进程场景 (simple_my_logs_minimal)")
    print("="*70)
    
    log_dir = os.path.join(project_root, "tests", "simple_my_logs_minimal")
    print(f"日志目录: {log_dir}")
    
    # 创建收集器
    collector = StackTraceCollector(log_dir=log_dir, n_line=100, rank=0)
    
    # 收集数据
    stack_data = collector.collect_data()
    
    # 验证结果
    print(f"\n收集结果:")
    print(f"  - 时间戳: {stack_data.timestamp}")
    print(f"  - PID: {stack_data.pid}")
    print(f"  - 事件名称: {stack_data.event_name}")
    print(f"  - 异常类型: {stack_data.exception_type}")
    print(f"  - 包含异常: {stack_data.has_exception()}")
    print(f"  - 堆栈行数: {len(stack_data.stack_traces)}")
    
    if stack_data.stack_traces:
        print(f"\n堆栈内容:")
        for i, line in enumerate(stack_data.stack_traces, 1):
            print(f"    [{i}] {line.strip()}")
    
    # 断言检查
    assert stack_data.has_exception(), "❌ 未检测到异常信息"
    assert stack_data.exception_type == "ValueError", f"❌ 异常类型错误: {stack_data.exception_type}"
    assert stack_data.pid == 3484, f"❌ PID 错误: {stack_data.pid}"
    assert len(stack_data.stack_traces) > 0, "❌ 堆栈为空"
    
    print(f"\n✅ 单进程测试通过!")
    return True


def test_multiprocess_logs():
    """测试多进程日志收集"""
    print("\n" + "="*70)
    print("测试 2: 多进程场景 (multiprocess_my_logs)")
    print("="*70)
    
    log_dir = os.path.join(project_root, "tests", "multiprocess_my_logs")
    print(f"日志目录: {log_dir}")
    
    # 创建收集器
    collector = StackTraceCollector(log_dir=log_dir, n_line=100)
    
    # 收集所有 rank 的数据
    all_stack_data = collector.collect_all_ranks()
    
    print(f"\n收集到 {len(all_stack_data)} 个进程的堆栈数据")
    
    # 验证每个 rank 的数据
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
    
    found_ranks = set()
    
    for stack_data in all_stack_data:
        print(f"\n--- Rank 信息 ---")
        print(f"  - 时间戳: {stack_data.timestamp}")
        print(f"  - PID: {stack_data.pid}")
        print(f"  - 事件名称: {stack_data.event_name}")
        print(f"  - 异常类型: {stack_data.exception_type}")
        print(f"  - 包含异常: {stack_data.has_exception()}")
        print(f"  - 堆栈行数: {len(stack_data.stack_traces)}")
        
        # 根据 PID 判断是哪个 rank
        rank = None
        for r, expected_pid in expected_pids.items():
            if stack_data.pid == expected_pid:
                rank = r
                break
        
        if rank is not None:
            found_ranks.add(rank)
            print(f"  - 识别为 Rank: {rank}")
            
            # 验证异常类型
            expected_exc = expected_exceptions[rank]
            assert stack_data.exception_type == expected_exc, \
                f"❌ Rank {rank} 异常类型错误: 期望 {expected_exc}, 实际 {stack_data.exception_type}"
        
        if stack_data.stack_traces:
            print(f"\n  堆栈摘要 (前2行):")
            for i, line in enumerate(stack_data.stack_traces[:2], 1):
                print(f"    [{i}] {line.strip()}")
    
    # 断言检查
    assert len(all_stack_data) == 4, f"❌ 应收集 4 个进程的数据，实际收集了 {len(all_stack_data)} 个"
    assert len(found_ranks) == 4, f"❌ 应识别 4 个 rank，实际识别了 {len(found_ranks)} 个: {found_ranks}"
    
    print(f"\n✅ 多进程测试通过! 成功收集了 {len(all_stack_data)} 个进程的堆栈数据")
    return True


def test_single_rank_collection():
    """测试收集指定 rank 的数据"""
    print("\n" + "="*70)
    print("测试 3: 收集指定 Rank (multiprocess_my_logs, rank=2)")
    print("="*70)
    
    log_dir = os.path.join(project_root, "tests", "multiprocess_my_logs")
    
    # 测试收集 rank 2 的数据
    collector = StackTraceCollector(log_dir=log_dir, n_line=100, rank=2)
    stack_data = collector.collect_data()
    
    print(f"\n收集结果 (Rank 2):")
    print(f"  - 时间戳: {stack_data.timestamp}")
    print(f"  - PID: {stack_data.pid}")
    print(f"  - 事件名称: {stack_data.event_name}")
    print(f"  - 异常类型: {stack_data.exception_type}")
    print(f"  - 包含异常: {stack_data.has_exception()}")
    print(f"  - 堆栈行数: {len(stack_data.stack_traces)}")
    
    if stack_data.stack_traces:
        print(f"\n堆栈内容:")
        for i, line in enumerate(stack_data.stack_traces, 1):
            print(f"    [{i}] {line.strip()}")
    
    # 验证
    assert stack_data.has_exception(), "❌ 未检测到异常信息"
    assert stack_data.exception_type == "ZeroDivisionError", \
        f"❌ 异常类型错误: 期望 ZeroDivisionError, 实际 {stack_data.exception_type}"
    assert stack_data.pid == 33740, f"❌ PID 错误: 期望 33740, 实际 {stack_data.pid}"
    
    print(f"\n✅ 指定 Rank 测试通过!")
    return True


def test_nonexistent_directory():
    """测试不存在的目录"""
    print("\n" + "="*70)
    print("测试 4: 不存在的日志目录")
    print("="*70)
    
    log_dir = os.path.join(project_root, "stack", "nonexistent_logs")
    print(f"日志目录: {log_dir}")
    
    # 创建收集器
    collector = StackTraceCollector(log_dir=log_dir, n_line=100, rank=0)
    
    # 收集数据
    stack_data = collector.collect_data()
    
    print(f"\n收集结果:")
    print(f"  - 包含异常: {stack_data.has_exception()}")
    print(f"  - 堆栈行数: {len(stack_data.stack_traces)}")
    
    # 验证：应该返回空数据
    assert not stack_data.has_exception(), "❌ 不应检测到异常信息"
    assert len(stack_data.stack_traces) == 0, "❌ 堆栈应为空"
    
    print(f"\n✅ 不存在目录测试通过!")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("开始测试 StackTraceCollector")
    print("="*70)
    
    tests = [
        ("单进程日志收集", test_single_process_logs),
        ("多进程日志收集", test_multiprocess_logs),
        ("指定 Rank 收集", test_single_rank_collection),
        ("不存在的目录", test_nonexistent_directory),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_name} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_name} 出错: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过!")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

