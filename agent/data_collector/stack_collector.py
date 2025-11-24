from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agent.data_collector.collected_data import StackTraceData
from agent.data_collector.data_collector import DataCollector
from common.constants import CollectorType
from util.file_util import read_last_n_lines


class StackTraceCollector(DataCollector):
    """
    StackTraceCollector 从异常堆栈日志文件中收集堆栈信息。
    
    支持单进程和多进程场景：
    - 单进程：读取 events_{rank}.log
    - 多进程：读取所有 events_*.log 文件
    
    日志格式示例：
    [2025-11-13T14:20:24.714461] [3484] [9d45d8da] [ErrorReporter] [exception] [INSTANT] 
    {"stack": ["Traceback...", ...], "pid": 3484}
    """

    def __init__(
        self,
        log_dir: str = "",
        n_line: int = 100,
        rank: Optional[int] = None,
    ) -> None:
        """
        初始化堆栈收集器。
        
        Args:
            log_dir: 堆栈日志目录，默认从环境变量 DLROVER_EVENT_FILE_DIR 读取
            n_line: 从每个日志文件读取的最大行数
            rank: 当前进程的 rank，如果为 None 则从环境变量 RANK 读取
        """
        super().__init__()
        
        # 设置日志目录
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            # 从环境变量获取
            log_dir_env = os.getenv("DLROVER_EVENT_FILE_DIR", "/tmp/dlrover")
            self._log_dir = Path(log_dir_env)
        
        self._n_line = max(0, n_line)
        
        # 获取 rank
        if rank is not None:
            self._rank = rank
        else:
            self._rank = int(os.getenv("RANK", "0") or "0")
        
        self._collector_type = CollectorType.STACK_TRACE_COLLECTOR
        
        # 日志行格式的正则表达式
        # [timestamp] [pid] [event_id] [target] [name] [type] {json_content}
        self._log_pattern = re.compile(
            r'\[([^\]]+)\]\s+\[(\d+)\]\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+'
            r'\[([^\]]+)\]\s+\[([^\]]+)\]\s+(.+)'
        )

    def collect_data(self) -> StackTraceData:
        """
        收集堆栈追踪数据。
        
        Returns:
            StackTraceData: 收集到的堆栈数据
        """
        if not self._log_dir or not self._log_dir.exists():
            stack_data = StackTraceData()
            return stack_data
        
        # 查找当前 rank 的日志文件
        log_file = self._log_dir / f"events_{self._rank}.log"
        
        if not log_file.exists():
            stack_data = StackTraceData()
            return stack_data
        
        # 读取最后 n 行日志
        raw_logs = read_last_n_lines(str(log_file), self._n_line)
        
        # 解析日志，查找异常堆栈
        stack_traces = []
        exception_type = ""
        event_name = ""
        pid = 0
        timestamp = 0
        
        for line in raw_logs:
            # 解码字节序列
            if isinstance(line, (bytes, bytearray)):
                line = line.decode("utf-8", errors="ignore")
            else:
                line = str(line)
            
            # 解析日志行
            parsed = self._parse_log_line(line)
            if parsed:
                ts, p, event_id, target, name, event_type, content = parsed
                
                # 只处理异常和信号事件
                if name in ["exception", "exit_sig"]:
                    event_name = name
                    pid = p
                    timestamp = ts
                    
                    # 解析堆栈信息
                    if "stack" in content:
                        stack_info = content["stack"]
                        if isinstance(stack_info, list):
                            stack_traces = stack_info
                        else:
                            stack_traces = [str(stack_info)]
                        
                        # 尝试提取异常类型
                        if stack_traces:
                            for stack_line in stack_traces:
                                # 查找异常类型（通常在最后一行）
                                if "Error" in stack_line or "Exception" in stack_line:
                                    exception_type = self._extract_exception_type(stack_line)
                                    if exception_type:
                                        break
                    
                    # 找到最新的堆栈信息后停止
                    if stack_traces:
                        break
        
        # 创建 StackTraceData 对象
        stack_data = StackTraceData(
            timestamp=timestamp,
            stack_traces=stack_traces,
            exception_type=exception_type,
            event_name=event_name,
            pid=pid,
        )
        
        return stack_data

    def _parse_log_line(self, line: str):
        """
        解析日志行。
        
        Args:
            line: 日志行字符串
            
        Returns:
            tuple: (timestamp, pid, event_id, target, name, event_type, content_dict)
                   如果解析失败返回 None
        """
        match = self._log_pattern.match(line.strip())
        if not match:
            return None
        
        try:
            timestamp_str, pid_str, event_id, target, name, event_type, content_str = match.groups()
            
            # 解析时间戳
            timestamp = self._parse_timestamp(timestamp_str)
            
            # 解析 PID
            pid = int(pid_str)
            
            # 解析 JSON 内容
            content = json.loads(content_str)
            
            return timestamp, pid, event_id, target, name, event_type, content
        except (ValueError, json.JSONDecodeError) as e:
            # 解析失败，忽略这一行
            return None

    def _parse_timestamp(self, timestamp_str: str) -> int:
        """
        解析时间戳字符串。
        
        Args:
            timestamp_str: ISO 格式的时间戳字符串
            
        Returns:
            int: Unix 时间戳（秒）
        """
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return int(dt.timestamp())
        except ValueError:
            return 0

    def _extract_exception_type(self, stack_line: str) -> str:
        """
        从堆栈行中提取异常类型。
        
        Args:
            stack_line: 堆栈行字符串
            
        Returns:
            str: 异常类型（如 "ValueError", "RuntimeError"）
        """
        # 匹配常见的异常类型
        exception_pattern = re.compile(r'(\w+Error|\w+Exception):')
        match = exception_pattern.search(stack_line)
        if match:
            return match.group(1)
        return ""

    def collect_all_ranks(self) -> List[StackTraceData]:
        """
        收集所有 rank 的堆栈数据（用于多进程场景）。
        
        Returns:
            List[StackTraceData]: 所有 rank 的堆栈数据列表
        """
        if not self._log_dir or not self._log_dir.exists():
            return []
        
        all_stack_data = []
        
        # 查找所有 events_*.log 文件
        for log_file in self._log_dir.glob("events_*.log"):
            # 提取 rank
            match = re.match(r'events_(\d+)\.log', log_file.name)
            if not match:
                continue
            
            rank = int(match.group(1))
            
            # 临时修改 rank 并收集数据
            old_rank = self._rank
            self._rank = rank
            stack_data = self.collect_data()
            self._rank = old_rank
            
            if stack_data.has_exception():
                all_stack_data.append(stack_data)
        
        return all_stack_data

    def is_enabled(self) -> bool:
        """检查收集器是否启用"""
        # 从环境变量检查
        enable_env = os.getenv("DLROVER_EVENT_ENABLE", "true").lower()
        return enable_env in ["true", "1", "yes", "y", "on", "enable", "enabled"]