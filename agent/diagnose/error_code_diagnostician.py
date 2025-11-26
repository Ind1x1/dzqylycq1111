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
ErrorCodeDiagnostician - Diagnose training failures by checking error codes
from error_type.json
"""

import json
import os
from typing import Optional

from common.log import default_logger as logger
from agent.data_collector.constants import DiagnosisErrorConstant
from agent.data_collector.log_collector import TrainingLogCollector


class DiagnosisObservation:
    """
    DiagnosisObservation is to describe the problem observed
    by ErrorCodeDiagnostician.observe
    """

    def __init__(self, observation: str = ""):
        # The simple description info for the problem.
        self._observation: str = observation
        self.extra_infos: dict = {}

    @property
    def observation(self):
        return self._observation


class ErrorCodeDiagnostician:
    """
    ErrorCodeDiagnostician is to observe and resolve the failure node problem
    by checking error codes from error_type.json
    """

    def __init__(self, error_type_file: Optional[str] = None):
        if error_type_file is None:
            # Try to read from environment variable
            error_type_file = os.getenv("AUTO_RL_ERROR_TYPE_FILE", "")
            if not error_type_file:
                # Default to error_type.json in the same directory as this file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                error_type_file = os.path.join(current_dir, "error_type.json")
        
        self.error_type_file = error_type_file
        self.error_items = self._load_error_codes()

    def _load_error_codes(self):
        """
        Load error items from error_type.json file.
        Each item contains error_type and sub_error pair for matching.
        """
        error_items = []
        try:
            if not os.path.exists(self.error_type_file):
                logger.warning(
                    f"Error type file not found: {self.error_type_file}"
                )
                return error_items

            with open(self.error_type_file, "r", encoding="utf-8") as f:
                error_list = json.load(f)

            if not isinstance(error_list, list):
                logger.error(
                    f"Invalid error_type.json format: expected list, "
                    f"got {type(error_list)}"
                )
                return error_items

            for error_item in error_list:
                if not isinstance(error_item, dict):
                    continue

                error_type = error_item.get("error_type", "")
                sub_error = error_item.get("sub_error", "")

                # Store complete error item with both error_type and sub_error
                if error_type and sub_error:
                    error_items.append({
                        "error_type": error_type,
                        "sub_error": sub_error,
                        "reason": error_item.get("reason", "")
                    })

            logger.info(
                f"Loaded {len(error_items)} error items from "
                f"{self.error_type_file}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse error_type.json: {e}")
        except Exception as e:
            logger.error(f"Failed to load error codes: {e}")

        return error_items

    def observe(self, **kwargs) -> DiagnosisObservation:
        """
        Observe the training logs to detect if any error pair
        (error_type + sub_error) from error_type.json is present.

        The log line must contain BOTH error_type and sub_error
        from the same error item to be considered a match.

        Args:
            log_file: Path to the log file to analyze
            **kwargs: Additional keyword arguments

        Returns:
            DiagnosisObservation with NODE_FAILED + error_type + sub_error
            if error detected, empty otherwise
        """
        # Parameter validation: log_file
        log_file_arg = kwargs.get("log_file")
        if log_file_arg is None or not isinstance(log_file_arg, str):
            logger.error(f"Invalid log_file: {log_file_arg}")
            return DiagnosisObservation()
        log_file = str(log_file_arg)

        # Check if log file exists
        if not os.path.exists(log_file):
            logger.error(f"Log file does not exist: {log_file}")
            return DiagnosisObservation()

        # Check if error items are loaded
        if not self.error_items or len(self.error_items) == 0:
            logger.warning(
                "No error items loaded from error_type.json, "
                "skipping diagnosis"
            )
            return DiagnosisObservation()

        # Log collection: create TrainingLogCollector instance,
        # read up to 5000 lines
        collector = TrainingLogCollector(log_file, 5000)
        training_log = collector.collect_data()
        logs = training_log.logs

        if not logs or len(logs) == 0:
            logger.warning(f"Failed to collect training logs from {log_file}")
            return DiagnosisObservation()

        # Failure node detection: iterate through all log lines
        # and check if they contain BOTH error_type and sub_error
        # from the same error item
        is_failure_node = False
        matched_error_type = None
        matched_sub_error = None
        matched_reason = None

        for log in logs:
            if is_failure_node:
                break
            for error_item in self.error_items:
                error_type = error_item["error_type"]
                sub_error = error_item["sub_error"]
                
                # Check if BOTH error_type and sub_error are in the same log line
                if error_type in log and sub_error in log:
                    logger.info(
                        f"Found matching error pair in log: "
                        f"error_type='{error_type}', sub_error='{sub_error}'"
                    )
                    logger.info(f"Log line: {log}")
                    matched_error_type = error_type
                    matched_sub_error = sub_error
                    matched_reason = error_item.get("reason", "")
                    is_failure_node = True
                    break

        # Return diagnosis result with error_type and sub_error
        if is_failure_node:
            # Build observation string: NODE_FAILED + error_type + sub_error
            observation_str = (
                f"{DiagnosisErrorConstant.NODE_FAILED} - "
                f"error_type: {matched_error_type}, "
                f"sub_error: {matched_sub_error}"
            )
            
            observation = DiagnosisObservation(
                observation=observation_str,
            )
            observation.extra_infos["error_type"] = matched_error_type
            observation.extra_infos["sub_error"] = matched_sub_error
            observation.extra_infos["reason"] = matched_reason
            observation.extra_infos["log_file"] = log_file
            return observation

        return DiagnosisObservation()

