from __future__ import annotations

from pathlib import Path
from typing import List

from agent.data_collector.collected_data import TrainingLog
from agent.data_collector.data_collector import DataCollector
from util.file_util import read_last_n_lines


class TrainingLogCollector(DataCollector):
    """
    TrainingLogCollector collects the last n_line lines
    from the given logs.
    """
    def __init__(self, log_file: str = "", n_line: int = 0) -> None:
        super().__init__()
        self._log_file = Path(log_file) if log_file else None
        self._n_line = max(0, n_line)

    def collect_data(self) -> TrainingLog:
        if not self._log_file or not self._log_file.exists():
            return TrainingLog()

        raw_logs = read_last_n_lines(str(self._log_file), self._n_line)

        def _to_text(line: object) -> str:
            if isinstance(line, (bytes, bytearray)):
                return line.decode("utf-8", errors="ignore")
            return str(line)

        logs = [_to_text(line) for line in raw_logs]

        if logs:
            start_str = "DLRover agent started with:"
            for idx, text in enumerate(logs):
                if start_str in text:
                    logs = logs[idx:]
                    break

        return TrainingLog(logs=logs)

    def is_enabled(self) -> bool:
        return True