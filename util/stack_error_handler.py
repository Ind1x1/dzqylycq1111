# Copyright 2025 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
最小核心版本 - 堆栈错误处理模块
只保留记录异常堆栈的核心功能，代码精简到约 400 行
"""

import atexit
import json
import logging
import os
import signal
import sys
import threading
import traceback
import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Optional


# ============================================================================
# Singleton 单例模式（简化版）
# ============================================================================

class Singleton:
    """简化的单例基类"""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def singleton_instance(cls):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = cls()
        return cls._instances[cls]


# ============================================================================
# Config 配置管理（简化版）
# ============================================================================

def get_env_bool(key: str, default: bool = False) -> bool:
    """从环境变量读取布尔值"""
    val = os.getenv(key, "").lower()
    if val in ["true", "1", "yes", "y", "on", "enable", "enabled"]:
        return True
    elif val in ["false", "0", "no", "n", "off", "disable", "disabled"]:
        return False
    return default


class Config(Singleton):
    """配置管理（简化版）"""
    
    def __init__(self):
        # 从环境变量读取配置
        self.enable = get_env_bool("DLROVER_EVENT_ENABLE", True)
        self.hook_error = get_env_bool("DLROVER_EVENT_HOOK_ERROR", False)
        self.file_dir = os.getenv("DLROVER_EVENT_FILE_DIR", "/tmp/dlrover")
        self.text_formatter = os.getenv("DLROVER_EVENT_TEXT_FORMATTER", "LOG")
        
        # 获取进程信息
        self.pid = str(os.getpid())
        self.rank = os.getenv("RANK", "0") or "0"
        
        # 初始化日志
        self._logger = self._init_logger()
    
    def _init_logger(self):
        """初始化系统日志"""
        logger = logging.getLogger("stack_error_handler")
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir, exist_ok=True)
        
        # 系统日志文件
        log_file = os.path.join(
            self.file_dir, f"events_sys_{self.rank}_{self.pid}.log"
        )
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    
    def get_logger(self):
        return self._logger


# 全局日志实例
def get_logger():
    return Config.singleton_instance().get_logger()


# ============================================================================
# Event 事件数据结构（简化版）
# ============================================================================

class EventType(Enum):
    """事件类型：只保留 INSTANT（即时事件）"""
    INSTANT = auto()


def _json_default(o):
    """JSON 序列化辅助函数"""
    if isinstance(o, datetime):
        return o.isoformat()
    return f"<<non-serializable: {type(o).__qualname__}>>"


class Event:
    """事件类（简化版，只支持即时事件）"""
    
    def __init__(self, event_id: str, target: str, name: str, content: dict):
        self.pid = Config.singleton_instance().pid
        self.event_id = event_id
        self.event_time = datetime.now()
        self.target = target
        self.name = name
        self.event_type = EventType.INSTANT
        self.content = content or {}
    
    def __str__(self):
        """格式化为日志字符串"""
        return "[%s] [%s] [%s] [%s] [%s] [%s] %s" % (
            self.event_time.isoformat(),
            self.pid,
            self.event_id,
            self.target,
            self.name,
            self.event_type.name,
            json.dumps(self.content, ensure_ascii=False, default=_json_default),
        )
    
    @classmethod
    def instant(cls, event_id: str, target: str, name: str, content: Optional[dict] = None):
        """创建即时事件"""
        return cls(event_id, target, name, content or {})


# ============================================================================
# Exporter 事件导出器（简化版）
# ============================================================================

class TextFileExporter:
    """文本文件导出器（同步版本）"""
    
    def __init__(self, file_dir: str):
        self._file_dir = file_dir
        
        # 创建目录
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        
        # 事件日志文件
        config = Config.singleton_instance()
        self._file_path = os.path.join(
            file_dir, f"events_{config.rank}.log"
        )
        
        # 使用 logging 确保线程安全
        self._logger = self._init_logger()
    
    def _init_logger(self):
        """初始化文件日志记录器"""
        logger_name = f"event_exporter_{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self._file_path)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    
    def export(self, event: Event):
        """导出事件到文件"""
        self._logger.info(str(event))
    
    def close(self):
        """关闭导出器"""
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()


# 全局导出器
_default_exporter: Optional[TextFileExporter] = None
_exporter_lock = threading.Lock()


def get_default_exporter():
    """获取默认导出器"""
    global _default_exporter
    if _default_exporter is None:
        with _exporter_lock:
            if _default_exporter is None:
                config = Config.singleton_instance()
                if config.enable:
                    _default_exporter = TextFileExporter(config.file_dir)
                    atexit.register(close_default_exporter)
                    get_logger().info(
                        f"EventExporter initialized: file_dir={config.file_dir}"
                    )
    return _default_exporter


def close_default_exporter():
    """关闭默认导出器"""
    global _default_exporter
    with _exporter_lock:
        if _default_exporter is not None:
            _default_exporter.close()
            _default_exporter = None
            get_logger().info("EventExporter closed")


# ============================================================================
# EventEmitter 事件发射器（简化版）
# ============================================================================

def generate_event_id():
    """生成短事件 ID"""
    return uuid.uuid4().hex[:8]


class EventEmitter:
    """事件发射器（简化版）"""
    
    def __init__(self, target: str, exporter: Optional[TextFileExporter] = None):
        self.target = target
        self.exporter = exporter or get_default_exporter()
    
    def instant(self, name: str, content: Optional[dict] = None):
        """发射即时事件"""
        if self.exporter is not None:
            event = Event.instant(generate_event_id(), self.target, name, content)
            self.exporter.export(event)


class Process:
    """进程事件记录器（简化版）"""
    
    def __init__(self, target: str):
        self._emitter = EventEmitter(target)
    
    def instant(self, name: str, content: Optional[dict] = None):
        """记录即时事件"""
        try:
            self._emitter.instant(name, content)
        except Exception as e:
            get_logger().error(f"Failed to emit event: {e}")


# ============================================================================
# ErrorHandler 错误处理器（完整保留）
# ============================================================================

class ErrorHandler(Singleton):
    """错误处理器 - 捕获异常和信号，记录堆栈信息"""
    
    def __init__(self):
        self._original_excepthook = None
        self._original_handlers = {}
        self._process = Process("ErrorReporter")
        self._registered = False
        self._lock = threading.Lock()
    
    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """处理异常，记录堆栈"""
        try:
            if exc_traceback is not None:
                # 提取异常堆栈
                stack_info = traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
                self._process.instant(
                    "exception",
                    {
                        "stack": stack_info,
                        "pid": os.getpid(),
                    },
                )
        except Exception as e:
            get_logger().error(f"Error in exception handler: {e}")
        
        # 调用原始异常处理器
        if self._original_excepthook:
            self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _handle_signal(self, signum, frame):
        """处理信号，记录堆栈"""
        try:
            content = {
                "sig": signum,
                "sig_name": signal.Signals(signum).name,
                "pid": os.getpid(),
            }
            
            # 提取信号时的堆栈
            try:
                if frame:
                    stack = traceback.extract_stack(frame)
                    content["stack"] = traceback.format_list(stack)
            except Exception as e:
                content["stack"] = f"get stack failed: {str(e)}"
            
            self._process.instant("exit_sig", content)
        except Exception as e:
            get_logger().error(f"Error in signal handler: {e}")
        finally:
            self._call_original_handler(signum, frame)
    
    def _call_original_handler(self, signum, frame):
        """调用原始信号处理器"""
        handler = self._original_handlers.get(signum)
        
        # 如果是可调用的处理器，调用它
        if callable(handler):
            handler(signum, frame)
        # 如果是 SIG_IGN 或 SIGCHLD，忽略
        elif handler == signal.SIG_IGN or signum == signal.SIGCHLD:
            return
        else:
            # 恢复默认处理
            if self._registered:
                self.unregister()
            os.kill(os.getpid(), signum)
    
    def register(self):
        """注册异常和信号处理器"""
        with self._lock:
            if self._registered:
                return
            
            # 注册异常钩子
            self._original_excepthook = sys.excepthook
            sys.excepthook = self._handle_exception
            
            # 注册信号处理器（只捕获退出信号）
            signals_to_catch = [
                signal.SIGINT,   # Ctrl+C
                signal.SIGTERM,  # 终止信号
                signal.SIGABRT,  # 异常终止
                signal.SIGFPE,   # 浮点异常
            ]
            
            for sig in signals_to_catch:
                try:
                    self._original_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, self._handle_signal)
                except (OSError, ValueError) as e:
                    get_logger().warning(f"Cannot register handler for signal {sig}: {e}")
            
            self._registered = True
            get_logger().info("ErrorHandler registered")
    
    def unregister(self):
        """注销异常和信号处理器"""
        with self._lock:
            if not self._registered:
                return
            
            # 恢复原始异常钩子
            if self._original_excepthook:
                sys.excepthook = self._original_excepthook
                self._original_excepthook = None
            
            # 恢复原始信号处理器
            for sig, handler in self._original_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (OSError, ValueError) as e:
                    get_logger().warning(f"Cannot restore handler for signal {sig}: {e}")
            self._original_handlers.clear()
            
            self._registered = False
            get_logger().info("ErrorHandler unregistered")


# ============================================================================
# 初始化函数
# ============================================================================

def init_error_handler():
    """初始化错误处理器"""
    config = Config.singleton_instance()
    if config.enable and config.hook_error:
        ErrorHandler.singleton_instance().register()


# ============================================================================
# 模块初始化
# ============================================================================

# 导入时自动初始化
get_default_exporter()  # 初始化导出器
init_error_handler()    # 初始化错误处理器

