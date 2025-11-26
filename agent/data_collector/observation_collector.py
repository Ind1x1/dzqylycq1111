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
ObservationCollector - Collects diagnosis observations by matching logs with error codes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent.data_collector.collected_data import DiagnosisObservationData
from agent.data_collector.data_collector import DataCollector
from agent.diagnose.error_code_diagnostician import ErrorCodeDiagnostician
from agent.data_collector.log_collector import TrainingLogCollector
from common.constants import CollectorType
from common.log import default_logger as logger


class ObservationCollector(DataCollector):
    """
    ObservationCollector collects diagnosis observations by analyzing training logs
    and matching them against error codes from error_type.json.
    
    This collector reuses the TrainingLogCollector to read logs and uses
    ErrorCodeDiagnostician to perform the diagnosis.
    """

    def __init__(
        self, 
        log_file: str = "", 
        n_line: int = 5000,
        error_type_file: Optional[str] = None
    ) -> None:
        """
        Initialize ObservationCollector.
        
        Args:
            log_file: Path to the log file to analyze
            n_line: Number of lines to read from the end of the log file
            error_type_file: Path to error_type.json file. If None, will use
                           environment variable AUTO_RL_ERROR_TYPE_FILE or default path
        """
        super().__init__()
        self._log_file = Path(log_file) if log_file else None
        self._n_line = max(0, n_line)
        self._error_type_file = error_type_file
        self._collector_type = CollectorType.OBSERVATION_COLLECTOR

    def collect_data(self) -> DiagnosisObservationData:
        """
        Collect diagnosis observation data by analyzing training logs.
        
        This method:
        1. Reuses TrainingLogCollector to read logs (avoiding duplicate file reads)
        2. Uses ErrorCodeDiagnostician to match error codes
        3. Returns DiagnosisObservationData containing the diagnosis result
        
        Returns:
            DiagnosisObservationData with diagnosis results, or empty if no error detected
        """
        # Validate log file
        if not self._log_file or not self._log_file.exists():
            logger.warning(
                f"Log file does not exist or not specified: {self._log_file}"
            )
            return DiagnosisObservationData()

        log_file_str = str(self._log_file)

        try:
            # Step 1: Collect logs using TrainingLogCollector (reuse existing logic)
            log_collector = TrainingLogCollector(log_file_str, self._n_line)
            training_log = log_collector.collect_data()

            if not training_log or not training_log.logs:
                logger.warning(
                    f"No training logs collected from {log_file_str}"
                )
                return DiagnosisObservationData()

            # Step 2: Execute diagnosis using ErrorCodeDiagnostician
            diagnostician = ErrorCodeDiagnostician(self._error_type_file)
            observation = diagnostician.observe(log_file=log_file_str)

            # Step 3: Wrap observation result into DiagnosisObservationData
            if observation.observation:
                logger.info(
                    f"Diagnosis observation collected: {observation.observation}"
                )
                return DiagnosisObservationData(
                    observation=observation.observation,
                    extra_infos=observation.extra_infos,
                )
            else:
                # No error detected, return empty observation
                return DiagnosisObservationData()

        except Exception as e:
            logger.error(
                f"Failed to collect diagnosis observation from {log_file_str}: {e}"
            )
            return DiagnosisObservationData()

    def is_enabled(self) -> bool:
        """
        Check if the collector is enabled.
        
        Returns:
            True if enabled, False otherwise
        """
        return True

