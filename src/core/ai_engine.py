import os
import numpy as np
import torch

from src.core.full_spectrum_models import (
    SpectralPredictor,
    SpectralPredictorV2,
    SpectralPredictorV2_Fusion,
    SpectrumGenerator,
)
from src.core.physics_models import extract_spectrum_features


class FullSpectrumAIEngine:
    def __init__(self, models_dir=None):
        self.device = torch.device('cpu')

        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

        self.models_dir = os.path.abspath(models_dir)
        if os.path.basename(self.models_dir).lower() == 'pretrained':
            self.model_search_dirs = [self.models_dir, os.path.dirname(self.models_dir)]
        else:
            self.model_search_dirs = [os.path.join(self.models_dir, 'pretrained'), self.models_dir]

        self.linear_uloq_ng_ml = 18.0
        self._mode_registry = {}
        self._runtime_cache = {}
        self.is_loaded = False
        self.v2_loaded = False
        self.load_models()

    def _load_torch_file(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            try:
                return torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=self.device)

    def _resolve_artifact(self, file_name):
        for model_dir in self.model_search_dirs:
            p = os.path.join(model_dir, file_name)
            if os.path.exists(p):
                return p
        return None

    def _discover_mode_registry(self):
        registry = {}

        def add_mode(mode, predictor_type, predictor_file, norm_file, generator_file=None):
            predictor_path = self._resolve_artifact(predictor_file)
            norm_path = self._resolve_artifact(norm_file)
            if predictor_path is None or norm_path is None:
                return
            generator_path = self._resolve_artifact(generator_file) if generator_file else None
            registry[mode] = {
                'predictor_type': predictor_type,
                'predictor_path': predictor_path,
                'norm_path': norm_path,
                'generator_path': generator_path,
            }

        add_mode('v1', 'v1', 'spectral_predictor_v1_split.pth', 'predictor_v1_norm_params.pth', 'spectral_generator.pth')
        add_mode('v2', 'v2', 'spectral_predictor_v2.pth', 'predictor_v2_norm_params.pth')
        add_mode('v2_cycle', 'fusion', 'spectral_predictor_v2_cycle.pth', 'predictor_v2_cycle_norm_params.pth', 'spectral_generator_cycle.pth')
        add_mode('v2_fusion', 'fusion', 'spectral_predictor_v2_fusion.pth', 'predictor_v2_fusion_norm_params.pth')
        add_mode('stage3_3a_fixed_frozen', 'fusion', 'spectral_predictor_v2_stage3_3a_fixed_frozen.pth', 'predictor_v2_stage3_3a_fixed_frozen_norm_params.pth', 'spectral_generator_stage3_3a_fixed_frozen.pth')
        add_mode('stage3_3b_fixed_regressor', 'fusion', 'spectral_predictor_v2_stage3_3b_fixed_regressor.pth', 'predictor_v2_stage3_3b_fixed_regressor_norm_params.pth', 'spectral_generator_stage3_3b_fixed_regressor.pth')
        add_mode('stage3_3c_learnable_regressor', 'fusion', 'spectral_predictor_v2_stage3_3c_learnable_regressor.pth', 'predictor_v2_stage3_3c_learnable_regressor_norm_params.pth', 'spectral_generator_stage3_3c_learnable_regressor.pth')
        add_mode('stage3_ch_fixed_regressor', 'fusion', 'spectral_predictor_v2_stage3_ch_fixed_regressor.pth', 'predictor_v2_stage3_ch_fixed_regressor_norm_params.pth', 'spectral_generator_stage3_ch_fixed_regressor.pth')
        return registry

    def load_models(self):
        try:
            self._mode_registry = self._discover_mode_registry()
            self._runtime_cache = {}
            self.is_loaded = bool(self._mode_registry)
            self.v2_loaded = 'v2' in self._mode_registry
            if self.is_loaded:
                print(f'AI engine discovered model modes: {sorted(self._mode_registry.keys())}')
            else:
                print(f'AI engine could not find any supported predictor artifacts (searched: {self.model_search_dirs}).')
            return self.is_loaded
        except Exception as e:
            print(f'Failed to load models: {e}')
            self._mode_registry = {}
            self._runtime_cache = {}
            self.is_loaded = False
            self.v2_loaded = False
            return False

    def available_model_modes(self):
        return sorted(self._mode_registry.keys())

    def resolve_prediction_mode(self, model_mode='auto'):
        if model_mode and model_mode != 'auto' and model_mode in self._mode_registry:
            return model_mode
        for preferred in ('v2', 'stage3_ch_fixed_regressor', 'stage3_3c_learnable_regressor', 'v2_cycle', 'v1'):
            if preferred in self._mode_registry:
                return preferred
        if self._mode_registry:
            return sorted(self._mode_registry.keys())[0]
        return None

    def resolve_generator_mode(self, model_mode='auto'):
        if model_mode and model_mode != 'auto' and model_mode in self._mode_registry:
            return model_mode
        for preferred in (
            'stage3_3a_fixed_frozen',
            'stage3_3b_fixed_regressor',
            'stage3_3c_learnable_regressor',
            'stage3_ch_fixed_regressor',
            'v2_cycle',
            'v1',
        ):
            if preferred in self._mode_registry and self._mode_registry[preferred].get('generator_path'):
                return preferred
        for mode in sorted(self._mode_registry.keys()):
            if self._mode_registry[mode].get('generator_path'):
                return mode
        if self._mode_registry:
            return sorted(self._mode_registry.keys())[0]
        return None

    def _resolve_mode(self, model_mode='auto'):
        return self.resolve_prediction_mode(model_mode)

    def _load_runtime(self, mode):
        if mode in self._runtime_cache:
            return self._runtime_cache[mode]

        cfg = self._mode_registry[mode]
        norm_params = self._load_torch_file(cfg['norm_path'])
        seq_len = len(norm_params['wavelengths'])
        predictor_type = cfg['predictor_type']
        if predictor_type == 'v1':
            predictor = SpectralPredictor(seq_len=seq_len).to(self.device)
        elif predictor_type == 'fusion':
            predictor = SpectralPredictorV2_Fusion(seq_len=seq_len).to(self.device)
        else:
            predictor = SpectralPredictorV2(seq_len=seq_len).to(self.device)
        predictor.load_state_dict(self._load_torch_file(cfg['predictor_path']))
        predictor.eval()

        generator = None
        if cfg.get('generator_path'):
            generator = SpectrumGenerator(seq_len=seq_len).to(self.device)
            generator.load_state_dict(self._load_torch_file(cfg['generator_path']))
            generator.eval()

        runtime = {
            'mode': mode,
            'predictor_type': predictor_type,
            'predictor': predictor,
            'norm_params': norm_params,
            'generator': generator,
        }
        self._runtime_cache[mode] = runtime
        return runtime

    def get_wavelengths(self, model_mode='auto'):
        mode = self.resolve_generator_mode(model_mode)
        if mode is None:
            return np.linspace(400, 800, 745, dtype=np.float32)
        runtime = self._load_runtime(mode)
        return np.asarray(runtime['norm_params']['wavelengths'], dtype=np.float32)

    def _prepare_input_spectrum(self, spectrum_ys, target_wavelengths=None):
        if target_wavelengths is None:
            target_wavelengths = self.get_wavelengths()
        target_wl = np.asarray(target_wavelengths, dtype=np.float32).reshape(-1)
        target_len = len(target_wl)

        spec = np.asarray(spectrum_ys, dtype=np.float32).reshape(-1)
        if spec.size == 0:
            return np.zeros(target_len, dtype=np.float32)
        if spec.size == target_len:
            return spec

        src_x = np.linspace(float(target_wl[0]), float(target_wl[-1]), num=spec.size, dtype=np.float32)
        spec_rs = np.interp(target_wl, src_x, spec)
        return spec_rs.astype(np.float32)

    def _predict_concentration_with_runtime(self, runtime, spectrum_ys):
        predictor_type = runtime['predictor_type']
        norm_params = runtime['norm_params']
        wl = np.asarray(norm_params['wavelengths'], dtype=np.float32)
        spec = self._prepare_input_spectrum(spectrum_ys, target_wavelengths=wl)

        with torch.no_grad():
            if predictor_type == 'v1':
                spec_tensor = torch.FloatTensor(spec)
                local_min = spec_tensor.min()
                local_max = spec_tensor.max()
                spec_norm = (spec_tensor - local_min) / (local_max - local_min + 1e-8)
                spec_norm = spec_norm.unsqueeze(0).unsqueeze(0).to(self.device)
                log_conc = runtime['predictor'](spec_norm).item()
            else:
                diff = np.gradient(spec).astype(np.float32)
                raw_med = np.asarray(norm_params['raw_med'], dtype=np.float32)
                raw_iqr = np.asarray(norm_params['raw_iqr'], dtype=np.float32)
                diff_med = np.asarray(norm_params['diff_med'], dtype=np.float32)
                diff_iqr = np.asarray(norm_params['diff_iqr'], dtype=np.float32)
                raw_norm = (spec - raw_med) / (raw_iqr + 1e-8)
                diff_norm = (diff - diff_med) / (diff_iqr + 1e-8)
                x = np.stack([raw_norm, diff_norm], axis=0).astype(np.float32)
                x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
                if predictor_type == 'fusion':
                    center, amp, fwhm = extract_spectrum_features(wl, spec)
                    x_physics = torch.tensor([[center, amp, fwhm]], dtype=torch.float32).to(self.device)
                    log_conc = runtime['predictor'](x_t, x_physics).item()
                else:
                    log_conc = runtime['predictor'](x_t).item()
            conc = (10 ** log_conc) - 1e-3
            return max(0.0, float(conc))

    def predict_concentration_details(self, spectrum_ys, model_mode='auto'):
        requested_mode = model_mode or 'auto'
        resolved_mode = self.resolve_prediction_mode(requested_mode)
        if resolved_mode is None:
            return {
                'requested_prediction_model': requested_mode,
                'resolved_prediction_model': None,
                'predicted_concentration_ng_ml': 0.0,
                'fallback_applied': False,
                'fallback_reason': None,
            }
        runtime = self._load_runtime(resolved_mode)
        prediction = self._predict_concentration_with_runtime(runtime, spectrum_ys)
        fallback_applied = False
        fallback_reason = None
        actual_mode = resolved_mode
        if (
            runtime.get('predictor_type') == 'fusion'
            and prediction <= 0.0
            and resolved_mode != 'v2'
            and 'v2' in self._mode_registry
        ):
            stable_runtime = self._load_runtime('v2')
            stable_prediction = self._predict_concentration_with_runtime(stable_runtime, spectrum_ys)
            if stable_prediction > 0.0:
                prediction = stable_prediction
                actual_mode = 'v2'
                fallback_applied = True
                fallback_reason = 'prediction collapsed to zero; fell back to v2 predictor'
        return {
            'requested_prediction_model': requested_mode,
            'resolved_prediction_model': actual_mode,
            'predicted_concentration_ng_ml': float(prediction),
            'fallback_applied': fallback_applied,
            'fallback_reason': fallback_reason,
        }

    def predict_concentration(self, spectrum_ys, model_mode='auto'):
        details = self.predict_concentration_details(spectrum_ys, model_mode=model_mode)
        return float(details['predicted_concentration_ng_ml'])

    def interpret_concentration(self, pred_concentration):
        pred = max(0.0, float(pred_concentration))
        uloq = float(self.linear_uloq_ng_ml)
        if pred <= uloq:
            return {
                'mode': 'quantitative',
                'raw_pred_ng_ml': pred,
                'reported_ng_ml': pred,
                'reported_text': f'{pred:.4f} ng/ml',
                'uloq_ng_ml': uloq,
                'super_quant_bin': None,
            }
        if pred <= 40.0:
            bin_label = '18-40 ng/ml'
        elif pred <= 75.0:
            bin_label = '40-75 ng/ml'
        else:
            bin_label = '>75 ng/ml'
        return {
            'mode': 'super_quantitative',
            'raw_pred_ng_ml': pred,
            'reported_ng_ml': None,
            'reported_text': f'> {uloq:.1f} ng/ml (super-quantitative, bin: {bin_label})',
            'uloq_ng_ml': uloq,
            'super_quant_bin': bin_label,
        }

    def generate_spectrum(self, concentration, model_mode='auto'):
        mode = self.resolve_generator_mode(model_mode)
        if mode is None:
            return np.zeros(745, dtype=np.float32)
        runtime = self._load_runtime(mode)
        norm_params = runtime['norm_params']
        generator = runtime['generator']
        if generator is None:
            return np.zeros(len(norm_params['wavelengths']), dtype=np.float32)

        with torch.no_grad():
            log_conc = np.log10(float(concentration) + 1e-3)
            conc_tensor = torch.FloatTensor([[log_conc]]).to(self.device)
            gen_norm = generator(conc_tensor).squeeze().cpu().numpy()
            # Backward compatibility:
            # - Older full-spectrum checkpoints used global spec_min/spec_max.
            # - Stage2.5/Stage3 checkpoints use per-wavelength robust stats (raw_med/raw_iqr).
            if 'spec_min' in norm_params and 'spec_max' in norm_params:
                spec_min = float(norm_params['spec_min'])
                spec_max = float(norm_params['spec_max'])
                real_spectrum = gen_norm * (spec_max - spec_min + 1e-8) + spec_min
                return real_spectrum.astype(np.float32)
            if 'raw_med' in norm_params and 'raw_iqr' in norm_params:
                raw_med = np.asarray(norm_params['raw_med'], dtype=np.float32).reshape(-1)
                raw_iqr = np.asarray(norm_params['raw_iqr'], dtype=np.float32).reshape(-1)
                if raw_med.size == gen_norm.size and raw_iqr.size == gen_norm.size:
                    real_spectrum = gen_norm * (raw_iqr + 1e-8) + raw_med
                    return real_spectrum.astype(np.float32)
            return np.asarray(gen_norm, dtype=np.float32).reshape(-1)

    def _intensity_align(self, input_spectrum, pred_spectrum):
        x = np.asarray(input_spectrum, dtype=np.float32).reshape(-1)
        y = np.asarray(pred_spectrum, dtype=np.float32).reshape(-1)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            return y.astype(np.float32), 1.0, 0.0
        x_p10, x_p50, x_p90 = np.percentile(x, [10, 50, 90]).astype(np.float32)
        y_p10, y_p50, y_p90 = np.percentile(y, [10, 50, 90]).astype(np.float32)
        y_span = float(y_p90 - y_p10)
        if abs(y_span) < 1e-8:
            return y.astype(np.float32), 1.0, 0.0
        scale = float((x_p90 - x_p10) / (y_span + 1e-8))
        scale = float(np.clip(scale, 0.2, 5.0))
        offset = float(x_p50 - scale * y_p50)
        aligned = scale * y + offset
        return aligned.astype(np.float32), scale, offset

    def predict_spectrum_from_spectrum(self, spectrum_ys, model_mode='auto', prediction_model_mode=None, generator_model_mode=None):
        requested_prediction_model = prediction_model_mode or model_mode or 'auto'
        requested_generator_model = generator_model_mode or model_mode or 'auto'
        generator_mode = self.resolve_generator_mode(requested_generator_model)
        wavelengths = self.get_wavelengths(generator_mode)
        input_resampled = self._prepare_input_spectrum(spectrum_ys, target_wavelengths=wavelengths)
        prediction_details = self.predict_concentration_details(spectrum_ys, model_mode=requested_prediction_model)
        pred_concentration = float(prediction_details['predicted_concentration_ng_ml'])
        report = self.interpret_concentration(pred_concentration)
        pred_spectrum_raw = self.generate_spectrum(pred_concentration, model_mode=generator_mode)
        pred_spectrum, spec_scale, spec_offset = self._intensity_align(input_resampled, pred_spectrum_raw)
        return {
            'model_mode': generator_mode,
            'requested_prediction_model': requested_prediction_model,
            'resolved_prediction_model': prediction_details['resolved_prediction_model'],
            'requested_generator_model': requested_generator_model,
            'resolved_generator_model': generator_mode,
            'fallback_applied': prediction_details['fallback_applied'],
            'fallback_reason': prediction_details['fallback_reason'],
            'pred_concentration': float(pred_concentration),
            'report_mode': report['mode'],
            'reported_concentration': report['reported_ng_ml'],
            'reported_text': report['reported_text'],
            'uloq_ng_ml': report['uloq_ng_ml'],
            'super_quant_bin': report['super_quant_bin'],
            'input_resampled': input_resampled,
            'pred_spectrum_raw': np.asarray(pred_spectrum_raw, dtype=np.float32).reshape(-1),
            'pred_spectrum': np.asarray(pred_spectrum, dtype=np.float32).reshape(-1),
            'intensity_scale': float(spec_scale),
            'intensity_offset': float(spec_offset),
            'wavelengths': np.asarray(wavelengths, dtype=np.float32).reshape(-1),
        }


ai_engine_instance = None


def get_ai_engine():
    global ai_engine_instance
    if ai_engine_instance is None:
        ai_engine_instance = FullSpectrumAIEngine()
    return ai_engine_instance
