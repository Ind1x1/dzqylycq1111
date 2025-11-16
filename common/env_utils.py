"""
提供与运行环境相关的辅助函数，当前实现主要用于测试。
"""

import os
import socket


def get_node_id() -> int:
    return int(os.getenv("AUTO_RL_NODE_ID", "-1"))


def get_node_type() -> str:
    return os.getenv("AUTO_RL_NODE_TYPE", "TRAIN_NODE")


def get_node_rank() -> int:
    return int(os.getenv("AUTO_RL_NODE_RANK", "-1"))


def get_node_ip() -> str:
    """
    获取当前机器的主机 IP，失败时回退到 127.0.0.1。
    """
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return "127.0.0.1"


def get_worker_local_process_id() -> int:
    """
    获取当前进程 ID。
    """
    return os.getpid()

