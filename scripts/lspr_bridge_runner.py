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
    candidate_dirs = [env_root / 'bin', env_root / 'Library' / 'bin', env_root / 'Scripts']
    prepend = []
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            prepend.append(candidate_str)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(candidate_str)
                except OSError:
                    pass
    if prepend:
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = ';'.join(prepend + [current_path])


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
            engine = FullSpectrumAIEngine(models_dir=str(PROJECT_ROOT / 'models'))
        return {
            'ok': bool(engine.is_loaded),
            'backend': 'subprocess',
            'details': {
                'project_root': str(PROJECT_ROOT),
                'models_dir': str(PROJECT_ROOT / 'models'),
                'available_model_modes': engine.available_model_modes(),
            },
            'error': None if engine.is_loaded else {'code': 'models_not_loaded', 'message': 'models discovered but not loaded'},
        }
    except Exception as exc:
        return {'ok': False, 'backend': 'subprocess', 'details': {'project_root': str(PROJECT_ROOT)}, 'error': {'code': 'health_check_failed', 'message': str(exc)}}


def _predict_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.core.ai_engine import FullSpectrumAIEngine
        intensities = payload.get('intensities') or []
        model_mode = payload.get('model_mode', 'auto')
        with contextlib.redirect_stdout(sys.stderr):
            engine = FullSpectrumAIEngine(models_dir=str(PROJECT_ROOT / 'models'))
        if not engine.is_loaded:
            return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'models_not_loaded', 'message': 'models are not loaded'}}
        with contextlib.redirect_stdout(sys.stderr):
            predicted = float(engine.predict_concentration(intensities, model_mode=model_mode))
            report = engine.interpret_concentration(predicted)
            resolved_mode = engine._resolve_mode(model_mode)
        return {
            'ok': True,
            'backend': 'subprocess',
            'model_mode': resolved_mode,
            'predicted_concentration_ng_ml': predicted,
            'report_mode': report.get('mode'),
            'reported_text': report.get('reported_text'),
            'uloq_ng_ml': report.get('uloq_ng_ml'),
            'super_quant_bin': report.get('super_quant_bin'),
            'metrics': {},
            'error': None,
        }
    except Exception as exc:
        return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'predict_single_failed', 'message': str(exc)}}


def _build_comparison(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.core.ai_engine import FullSpectrumAIEngine
        import numpy as np
        intensities = payload.get('intensities') or []
        model_mode = payload.get('model_mode', 'auto')
        with contextlib.redirect_stdout(sys.stderr):
            engine = FullSpectrumAIEngine(models_dir=str(PROJECT_ROOT / 'models'))
        if not engine.is_loaded:
            return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'models_not_loaded', 'message': 'models are not loaded, cannot run build_comparison'}}

        def predict_for_mode(mode: str):
            return engine.predict_spectrum_from_spectrum(intensities, model_mode=mode)

        def is_flat_generated(result: Dict[str, Any]) -> bool:
            generated = np.asarray(result.get('pred_spectrum_raw', []), dtype=float).reshape(-1)
            if generated.size < 2:
                return True
            return float(np.ptp(generated)) <= 1e-8

        with contextlib.redirect_stdout(sys.stderr):
            result = predict_for_mode(model_mode)
            if is_flat_generated(result):
                for candidate in engine.available_model_modes():
                    if candidate == model_mode:
                        continue
                    try:
                        candidate_result = predict_for_mode(candidate)
                    except Exception:
                        continue
                    if not is_flat_generated(candidate_result):
                        result = candidate_result
                        break

        generated = list(result.get('pred_spectrum_raw', []))
        generator_supported = len(generated) > 1 and (max(generated) - min(generated)) > 1e-8
        return {
            'ok': True,
            'backend': 'subprocess',
            'model_mode': result.get('model_mode', model_mode),
            'wavelengths': list(result.get('wavelengths', [])),
            'input_spectrum': list(result.get('input_resampled', [])),
            'generated_spectrum': generated,
            'aligned_spectrum': list(result.get('pred_spectrum', [])),
            'physical_spectrum': None,
            'metrics': {
                'predicted_concentration_ng_ml': float(result.get('pred_concentration', 0.0)),
                'report_mode': result.get('report_mode'),
                'reported_text': result.get('reported_text'),
                'uloq_ng_ml': result.get('uloq_ng_ml'),
                'super_quant_bin': result.get('super_quant_bin'),
                'intensity_scale': float(result.get('intensity_scale', 1.0)),
                'intensity_offset': float(result.get('intensity_offset', 0.0)),
                'generator_supported': bool(generator_supported),
            },
            'error': None,
        }
    except Exception as exc:
        return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'build_comparison_failed', 'message': str(exc)}}


def _build_digital_twin(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.core.digital_twin_service import DigitalTwinService
        concentration = float(payload.get('concentration_ng_ml', 0.0))
        with contextlib.redirect_stdout(sys.stderr):
            service = DigitalTwinService(base_dir=str(PROJECT_ROOT))
            context = service.build_plot_context(concentration)
        prediction = context.prediction
        ai_spectrum = context.ai_spectrum_aligned if context.ai_spectrum_aligned is not None else context.ai_spectrum_raw
        return {
            'ok': True,
            'backend': 'subprocess',
            'concentration_ng_ml': concentration,
            'wavelengths': list(context.wavelengths),
            'baseline_spectrum': list(context.bsa_spectrum),
            'physical_spectrum': list(context.physical_spectrum),
            'ai_spectrum': list(ai_spectrum) if ai_spectrum is not None else None,
            'metrics': {
                'peak_wavelength_nm': float(prediction.peak_wavelength),
                'delta_lambda_nm': float(prediction.delta_lambda),
                'peak_intensity': float(prediction.peak_intensity),
            },
            'error': None,
        }
    except Exception as exc:
        return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'build_digital_twin_failed', 'message': str(exc)}}


def _predict_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.core.ai_engine import FullSpectrumAIEngine
        from src.utils.data_loader import read_spectrum_file
        items = payload.get('items') or []
        model_mode = payload.get('model_mode', 'auto')
        with contextlib.redirect_stdout(sys.stderr):
            engine = FullSpectrumAIEngine(models_dir=str(PROJECT_ROOT / 'models'))
        if not engine.is_loaded:
            return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'models_not_loaded', 'message': 'models are not loaded, cannot run predict_batch'}}
        rows = []
        for index, item in enumerate(items):
            label = str(item.get('label') or item.get('name') or f'sample_{index + 1}')
            if item.get('intensities') is not None:
                intensities = item.get('intensities') or []
            elif item.get('file_path'):
                intensities = read_spectrum_file(str(item['file_path'])).tolist()
            else:
                intensities = []
            with contextlib.redirect_stdout(sys.stderr):
                predicted = float(engine.predict_concentration(intensities, model_mode=model_mode))
                report = engine.interpret_concentration(predicted)
                resolved_mode = engine._resolve_mode(model_mode)
            rows.append({
                'label': label,
                'model_mode': resolved_mode,
                'predicted_concentration_ng_ml': predicted,
                'report_mode': report.get('mode'),
                'reported_text': report.get('reported_text'),
                'uloq_ng_ml': report.get('uloq_ng_ml'),
                'super_quant_bin': report.get('super_quant_bin'),
                'source_file': item.get('file_path'),
            })
        return {'ok': True, 'backend': 'subprocess', 'rows': rows, 'error': None}
    except Exception as exc:
        return {'ok': False, 'backend': 'subprocess', 'error': {'code': 'predict_batch_failed', 'message': str(exc)}}


def main(argv: list[str]) -> int:
    _configure_windows_dll_paths()
    if len(argv) < 2:
        return _write_payload({'ok': False, 'backend': 'subprocess', 'error': {'code': 'missing_command', 'message': 'missing command'}}, exit_code=1)
    command = argv[1]
    payload = _read_payload()
    if command == 'health':
        return _write_payload(_health())
    if command == 'predict_single':
        return _write_payload(_predict_single(payload))
    if command == 'build_comparison':
        return _write_payload(_build_comparison(payload))
    if command == 'build_digital_twin':
        return _write_payload(_build_digital_twin(payload))
    if command == 'predict_batch':
        return _write_payload(_predict_batch(payload))
    return _write_payload({'ok': False, 'backend': 'subprocess', 'error': {'code': 'unsupported_command', 'message': f'unsupported command: {command}'}}, exit_code=1)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
