import os
import numpy as np
import torch

from src.core.full_spectrum_models import SpectralPredictor, SpectralPredictorV2, SpectrumGenerator


class FullSpectrumAIEngine:
    def __init__(self, models_dir=None):
        self.device = torch.device('cpu')

        if models_dir is None:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

        self.models_dir = models_dir
        self.predictor = None
        self.predictor_v2 = None
        self.generator = None
        self.norm_params = None
        self.v2_norm_params = None
        self.v2_calibration = None
        self.is_loaded = False
        self.v2_loaded = False
        self.linear_uloq_ng_ml = 18.0

        self.load_models()

    def _load_torch_file(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            return torch.load(path, map_location=self.device)

    def load_models(self):
        """Load trained model weights and normalization parameters."""
        try:
            norm_path = os.path.join(self.models_dir, 'norm_params.pth')
            pred_path = os.path.join(self.models_dir, 'spectral_predictor.pth')
            gen_path = os.path.join(self.models_dir, 'spectral_generator.pth')

            if not os.path.exists(norm_path) or not os.path.exists(pred_path) or not os.path.exists(gen_path):
                print('AI engine could not find pretrained weights under models/. Please run training first.')
                return False

            self.norm_params = self._load_torch_file(norm_path)
            seq_len = len(self.norm_params['wavelengths'])

            self.predictor = SpectralPredictor(seq_len=seq_len).to(self.device)
            self.predictor.load_state_dict(self._load_torch_file(pred_path))
            self.predictor.eval()

            self.generator = SpectrumGenerator(seq_len=seq_len).to(self.device)
            self.generator.load_state_dict(self._load_torch_file(gen_path))
            self.generator.eval()

            # Optional robust predictor v2.
            pred_v2_path = os.path.join(self.models_dir, 'spectral_predictor_v2.pth')
            v2_params_path = os.path.join(self.models_dir, 'predictor_v2_norm_params.pth')
            if os.path.exists(pred_v2_path) and os.path.exists(v2_params_path):
                self.v2_norm_params = self._load_torch_file(v2_params_path)
                v2_len = len(self.v2_norm_params['wavelengths'])
                self.predictor_v2 = SpectralPredictorV2(seq_len=v2_len).to(self.device)
                self.predictor_v2.load_state_dict(self._load_torch_file(pred_v2_path))
                self.predictor_v2.eval()
                self.v2_loaded = True
                print('AI engine loaded robust predictor v2.')

                cal_path = os.path.join(self.models_dir, 'predictor_v2_calibration.pth')
                if os.path.exists(cal_path):
                    self.v2_calibration = self._load_torch_file(cal_path)
                    print('AI engine loaded predictor v2 calibration layer.')

            self.is_loaded = True
            print('AI engine (predictor + generator) loaded successfully.')
            return True

        except Exception as e:
            print(f'Failed to load models: {e}')
            self.is_loaded = False
            self.v2_loaded = False
            self.v2_calibration = None
            return False

    def get_wavelengths(self):
        """Return wavelength axis used by the full-spectrum models."""
        if self.norm_params is not None:
            return np.asarray(self.norm_params['wavelengths'], dtype=np.float32)
        return np.linspace(400, 800, 745, dtype=np.float32)

    def _prepare_input_spectrum(self, spectrum_ys, target_wavelengths=None):
        """
        Ensure the input spectrum is 1D and aligned to target wavelength length.
        If length mismatches, linear-resample to target length.
        """
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

    def _predict_concentration_v1(self, spectrum_ys):
        with torch.no_grad():
            spec = self._prepare_input_spectrum(spectrum_ys)
            spec_tensor = torch.FloatTensor(spec)
            local_min = spec_tensor.min()
            local_max = spec_tensor.max()
            spec_norm = (spec_tensor - local_min) / (local_max - local_min + 1e-8)
            spec_norm = spec_norm.unsqueeze(0).unsqueeze(0).to(self.device)
            log_conc = self.predictor(spec_norm)
            val = log_conc.item()
            conc = (10 ** val) - 1e-3
            return max(0.0, float(conc))

    def _predict_concentration_v2(self, spectrum_ys):
        with torch.no_grad():
            wl = np.asarray(self.v2_norm_params['wavelengths'], dtype=np.float32)
            spec = self._prepare_input_spectrum(spectrum_ys, target_wavelengths=wl)
            diff = np.gradient(spec).astype(np.float32)

            raw_med = np.asarray(self.v2_norm_params['raw_med'], dtype=np.float32)
            raw_iqr = np.asarray(self.v2_norm_params['raw_iqr'], dtype=np.float32)
            diff_med = np.asarray(self.v2_norm_params['diff_med'], dtype=np.float32)
            diff_iqr = np.asarray(self.v2_norm_params['diff_iqr'], dtype=np.float32)

            raw_norm = (spec - raw_med) / (raw_iqr + 1e-8)
            diff_norm = (diff - diff_med) / (diff_iqr + 1e-8)
            x = np.stack([raw_norm, diff_norm], axis=0).astype(np.float32)  # [2, L]
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1,2,L]
            log_conc = self.predictor_v2(x_t).item()

            if self.v2_calibration is not None:
                x_thr = np.asarray(self.v2_calibration.get('x_thresholds', []), dtype=np.float32).reshape(-1)
                y_thr = np.asarray(self.v2_calibration.get('y_thresholds', []), dtype=np.float32).reshape(-1)
                if x_thr.size >= 2 and y_thr.size == x_thr.size:
                    log_conc = float(np.interp(log_conc, x_thr, y_thr, left=y_thr[0], right=y_thr[-1]))

            conc = (10 ** log_conc) - 1e-3
            return max(0.0, float(conc))

    def predict_concentration(self, spectrum_ys):
        """Input one spectrum and predict concentration (ng/ml)."""
        if not self.is_loaded:
            return 0.0

        if self.v2_loaded and self.predictor_v2 is not None and self.v2_norm_params is not None:
            try:
                return self._predict_concentration_v2(spectrum_ys)
            except Exception as e:
                print(f'Predictor v2 failed, fallback to v1: {e}')

        return self._predict_concentration_v1(spectrum_ys)

    def interpret_concentration(self, pred_concentration):
        """
        Convert raw predicted concentration into reporting mode:
        - <= ULOQ: quantitative report
        - > ULOQ: super-quantitative (over-range) report with coarse bins
        """
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

    def generate_spectrum(self, concentration):
        """Input concentration (ng/ml) and generate corresponding spectrum."""
        if not self.is_loaded:
            if self.norm_params is not None:
                return np.zeros(len(self.norm_params['wavelengths']), dtype=np.float32)
            return np.zeros(745, dtype=np.float32)

        with torch.no_grad():
            log_conc = np.log10(float(concentration) + 1e-3)
            conc_tensor = torch.FloatTensor([[log_conc]]).to(self.device)

            gen_norm = self.generator(conc_tensor)
            gen_norm = gen_norm.squeeze().cpu().numpy()

            spec_min = float(self.norm_params['spec_min'])
            spec_max = float(self.norm_params['spec_max'])
            real_spectrum = gen_norm * (spec_max - spec_min + 1e-8) + spec_min
            return real_spectrum.astype(np.float32)

    def _intensity_align(self, input_spectrum, pred_spectrum):
        """
        Align predicted spectrum intensity to input spectrum using robust
        scale + offset calibration on percentiles (to reduce peak-height bias).
        """
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

    def predict_spectrum_from_spectrum(self, spectrum_ys):
        """
        Spectrum-to-spectrum mapping via AI chain:
          input spectrum -> predicted concentration -> generated spectrum
        """
        wavelengths = self.get_wavelengths()
        input_resampled = self._prepare_input_spectrum(spectrum_ys)
        pred_concentration = self.predict_concentration(input_resampled)
        report = self.interpret_concentration(pred_concentration)
        pred_spectrum_raw = self.generate_spectrum(pred_concentration)
        pred_spectrum, spec_scale, spec_offset = self._intensity_align(input_resampled, pred_spectrum_raw)

        return {
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
