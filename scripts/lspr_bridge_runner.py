#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _configure_windows_dll_paths() -> None:
    env_root = Path(sys.executable).resolve().parent
    candidate_dirs = [
        env_root / "bin",
        env_root / "Library" / "bin",
        env_root / "Scripts",
    ]

    prepend = []
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            prepend.append(candidate_str)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(candidate_str)
                except OSError:
                    pass

    if prepend:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ";".join(prepend + [current_path])


def _read_payload() -> Dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    return json.loads(raw)


def _write_payload(payload: Dict[str, Any], exit_code: int = 0) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return exit_code


def _health() -> Dict[str, Any]:
    try:
        from src.core.ai_engine import FullSpectrumAIEngine

        with contextlib.redirect_stdout(sys.stderr):
            engine = FullSpectrumAIEngine(models_dir=str(PROJECT_ROOT / "models"))
        return {
            "ok": bool(engine.is_loaded),
            "backend": "subprocess",
            "details": {
                "project_root": str(PROJECT_ROOT),
                "models_dir": str(PROJECT_ROOT / "models"),
                "v2_loaded": bool(getattr(engine, "v2_loaded", False)),
            },
            "error": None if engine.is_loaded else {
                "code": "models_not_loaded",
                "message": "模型已导入，但未成功加载所需权重",
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "backend": "subprocess",
            "details": {
                "project_root": str(PROJECT_ROOT),
            },
            "error": {
                "code": "health_check_failed",
                "message": str(exc),
            },
        }


def _predict_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.core.ai_engine import FullSpectrumAIEngine

        intensities = payload.get("intensities") or []
        with contextlib.redirect_stdout(sys.stderr):
            engine = FullSpectrumAIEngine(models_dir=str(PROJECT_ROOT / "models"))
        if not engine.is_loaded:
            return {
                "ok": False,
                "backend": "subprocess",
                "error": {
                    "code": "models_not_loaded",
                    "message": "模型未加载，无法执行 predict_single",
                },
            }

        with contextlib.redirect_stdout(sys.stderr):
            predicted = float(engine.predict_concentration(intensities))
            report = engine.interpret_concentration(predicted)
        return {
            "ok": True,
            "backend": "subprocess",
            "predicted_concentration_ng_ml": predicted,
            "report_mode": report.get("mode"),
            "reported_text": report.get("reported_text"),
            "uloq_ng_ml": report.get("uloq_ng_ml"),
            "super_quant_bin": report.get("super_quant_bin"),
            "metrics": {},
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "backend": "subprocess",
            "error": {
                "code": "predict_single_failed",
                "message": str(exc),
            },
        }


def main(argv: list[str]) -> int:
    _configure_windows_dll_paths()

    if len(argv) < 2:
        return _write_payload(
            {
                "ok": False,
                "backend": "subprocess",
                "error": {
                    "code": "missing_command",
                    "message": "必须提供命令，例如 health 或 predict_single",
                },
            },
            exit_code=1,
        )

    command = argv[1]
    payload = _read_payload()

    if command == "health":
        return _write_payload(_health())
    if command == "predict_single":
        return _write_payload(_predict_single(payload))

    return _write_payload(
        {
            "ok": False,
            "backend": "subprocess",
            "error": {
                "code": "unsupported_command",
                "message": f"不支持的命令: {command}",
            },
        },
        exit_code=1,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
