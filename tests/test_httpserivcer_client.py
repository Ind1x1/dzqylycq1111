import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controllerclinet import HttpControllerClient
from agent.data_collector.constants import CollectedNodeType
from agent.data_collector.log_collector import TrainingLogCollector
from controller.controllerservicer import create_http_controller_handler
from util.comm_util import find_free_port


def test_http_servicer_client_report_and_get(capsys):
    # Use check_log.txt from the data_collector directory
    project_root = Path(__file__).resolve().parents[1]
    log_file = project_root / "agent" / "data_collector" / "check_log.txt"
    
    if not log_file.exists():
        pytest.skip(f"check_log.txt not found at {log_file}")
    
    collector = TrainingLogCollector(str(log_file), n_line=100)
    training_log = collector.collect_data()
    
    print(f"\n[Test] Collected {len(training_log.logs)} log lines from {log_file}")
    if training_log.logs:
        print(f"[Test] First few lines:")
        for i, line in enumerate(training_log.logs[:5], 1):
            print(f"  [{i}] {line}")
        if len(training_log.logs) > 5:
            print(f"  ... and {len(training_log.logs) - 5} more lines")

    port = find_free_port()
    server, servicer = create_http_controller_handler("127.0.0.1", port)
    server.start()

    try:
        client = HttpControllerClient(
            master_addr=f"127.0.0.1:{port}",
            node_id=1,
            node_type=CollectedNodeType.TRAIN_NODE,
            timeout=5,
        )

        response = client.report(training_log)
        assert response.success is True

        received_log = None
        for _ in range(10):
            received_log = servicer.peek_latest_report()
            if received_log is not None:
                break
            time.sleep(0.1)

        assert received_log is not None
        received_logs = getattr(received_log, "logs", [])
        assert len(received_logs) > 0, "Received log should not be empty"
        print(f"\n[Test] Verified received log has {len(received_logs)} lines")

        fetched_log = client.get(None)
        assert fetched_log is not None
        fetched_logs = getattr(fetched_log, "logs", [])
        assert len(fetched_logs) > 0, "Fetched log should not be empty"
        print(f"[Test] Verified fetched log has {len(fetched_logs)} lines")

        # Wait a bit for print statements to flush
        time.sleep(0.1)
        captured = capsys.readouterr()
        assert "Received report" in captured.out
        
        # Print the captured output to show what servicer received
        print("\n" + "="*60)
        print("Servicer Output (what servicer received):")
        print("="*60)
        print(captured.out)
        if captured.err:
            print("Stderr:")
            print(captured.err)
        print("="*60)
    finally:
        server.stop()
        time.sleep(0.2)

