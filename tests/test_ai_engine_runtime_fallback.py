import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_fusion_mode_falls_back_to_v2_when_prediction_collapses_to_zero(monkeypatch):
    ai_engine = importlib.import_module("src.core.ai_engine")
    engine = ai_engine.FullSpectrumAIEngine.__new__(ai_engine.FullSpectrumAIEngine)
    engine._mode_registry = {
        "v2": {"predictor_type": "v2"},
        "v2_fusion": {"predictor_type": "fusion"},
    }

    class Runtime(dict):
        pass

    runtimes = {
        "v2": Runtime(mode="v2", predictor_type="v2"),
        "v2_fusion": Runtime(mode="v2_fusion", predictor_type="fusion"),
    }

    monkeypatch.setattr(engine, "_resolve_mode", lambda model_mode="auto": model_mode if model_mode != "auto" else "v2_fusion")
    monkeypatch.setattr(engine, "_load_runtime", lambda mode: runtimes[mode])

    def fake_predict(runtime, spectrum_ys):
        del spectrum_ys
        if runtime["mode"] == "v2_fusion":
            return 0.0
        return 12.34

    monkeypatch.setattr(engine, "_predict_concentration_with_runtime", fake_predict)

    result = ai_engine.FullSpectrumAIEngine.predict_concentration(engine, [0.1, 0.2, 0.3], model_mode="v2_fusion")

    assert result == 12.34
