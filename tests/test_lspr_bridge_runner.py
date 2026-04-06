import importlib.util
from pathlib import Path

import numpy as np


def _load_runner_module():
    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "lspr_bridge_runner.py"
    spec = importlib.util.spec_from_file_location("lspr_bridge_runner_test", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_payload_accepts_numpy_scalars():
    runner = _load_runner_module()
    payload = {
        "ok": True,
        "metrics": {
            "predicted_concentration_ng_ml": np.float32(12.34),
        },
        "wavelengths": np.asarray([500.0, 501.0], dtype=np.float32),
    }

    exit_code = runner._write_payload(payload)

    assert exit_code == 0

