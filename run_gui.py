import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import optimize
from sklearn.preprocessing import StandardScaler

from src.core.ai_engine import get_ai_engine

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LSPRResidualNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def lorentzian_reconstruct(wavelengths, peak_pos, amplitude, fwhm):
    gamma = fwhm / 2.0
    return amplitude * (gamma**2) / ((wavelengths - peak_pos) ** 2 + gamma**2)


class LSPRDigitalTwinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LSPR Digital Twin Visualizer')
        self.resize(1200, 700)

        self.wavelengths = None
        self.baseline_features = None
        self.model = None
        self.scaler_x = None
        self.scaler_y = None

        self.init_ai_engine()
        self.full_spectrum_ai = get_ai_engine()
        self.init_ui()
        self.update_plot(5.0)

    def _estimate_fwhm(self, wl: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
        peak = float(y[peak_idx])
        base = float(np.min(y))
        half = base + (peak - base) / 2.0

        left = peak_idx
        while left > 0 and y[left] > half:
            left -= 1

        right = peak_idx
        while right < len(y) - 1 and y[right] > half:
            right += 1

        if right == left:
            return float(np.median(np.diff(wl)) * 8.0)
        return float(abs(wl[right] - wl[left]))

    @staticmethod
    def _gaussian_model(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    def _fit_gaussian_peak(self, wl: np.ndarray, y: np.ndarray):
        mask = (wl >= 500.0) & (wl <= 800.0) & np.isfinite(y)
        x = wl[mask]
        s = y[mask]
        if x.size < 5:
            return None

        idx = int(np.argmax(s))
        p0 = [max(float(s[idx]), 1e-8), float(x[idx]), max(float((x[-1] - x[0]) * 0.1), 1.0)]
        try:
            popt, _ = optimize.curve_fit(self._gaussian_model, x, s, p0=p0, maxfev=3000)
            amp, center, sigma = float(popt[0]), float(popt[1]), abs(float(popt[2]))
            if 500.0 <= center <= 800.0 and amp > 0 and sigma > 0:
                return amp, center, sigma
        except Exception:
            pass
        return None

    def _extract_spectrum_features(self, wl: np.ndarray, y: np.ndarray):
        fit = self._fit_gaussian_peak(wl, y)
        if fit is not None:
            amp, center, sigma = fit
            peak_idx = int(np.argmin(np.abs(wl - center)))
            fwhm = self._estimate_fwhm(wl, y, peak_idx)
            if not np.isfinite(fwhm) or fwhm <= 0:
                fwhm = float(2.355 * sigma)
            return center, amp, float(fwhm)

        # Fallback to argmax if gaussian fit fails.
        peak_idx = int(np.argmax(y))
        center = float(wl[peak_idx])
        amp = float(y[peak_idx])
        fwhm = self._estimate_fwhm(wl, y, peak_idx)
        return center, amp, float(fwhm)

    def _align_spectrum_intensity(self, ref_spec: np.ndarray, pred_spec: np.ndarray) -> np.ndarray:
        """
        Align predicted spectrum intensity to a reference spectrum using robust
        percentile-based scale/offset so displayed peak heights are comparable.
        """
        ref = np.asarray(ref_spec, dtype=np.float32).reshape(-1)
        pred = np.asarray(pred_spec, dtype=np.float32).reshape(-1)
        if ref.size == 0 or pred.size == 0 or ref.size != pred.size:
            return pred

        ref_p10, ref_p50, ref_p90 = np.percentile(ref, [10, 50, 90]).astype(np.float32)
        pred_p10, pred_p50, pred_p90 = np.percentile(pred, [10, 50, 90]).astype(np.float32)
        span = float(pred_p90 - pred_p10)
        if abs(span) < 1e-8:
            return pred

        scale = float((ref_p90 - ref_p10) / (span + 1e-8))
        scale = float(np.clip(scale, 0.2, 5.0))
        offset = float(ref_p50 - scale * pred_p50)
        return (scale * pred + offset).astype(np.float32)

    def _build_training_from_paired_file(self, data_path: str):
        ext = os.path.splitext(data_path)[1].lower()
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path, sheet_name=0)
        elif ext in ['.csv', '.txt']:
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f'Unsupported paired data format: {ext}')
        if 'Wavelength' not in df.columns:
            raise ValueError('The paired data must contain Wavelength column.')

        wl = pd.to_numeric(df['Wavelength'], errors='coerce').dropna().values.astype(np.float32)
        self.wavelengths = wl

        pattern = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)ng/ml-(Ag|BSA)-(.+)\s*$', re.IGNORECASE)
        pairs = {}
        for col in df.columns:
            if col == 'Wavelength':
                continue
            m = pattern.match(str(col))
            if not m:
                continue
            conc = float(m.group(1))
            phase = m.group(2).lower()
            rep = m.group(3).strip()
            pairs.setdefault((conc, rep), {})[phase] = col

        x_rows, y_rows = [], []
        bsa_feature_rows = []
        for (conc, _rep), cols in pairs.items():
            if 'bsa' not in cols or 'ag' not in cols:
                continue

            pre_raw = pd.to_numeric(df[cols['bsa']], errors='coerce').values
            post_raw = pd.to_numeric(df[cols['ag']], errors='coerce').values
            valid = np.isfinite(pre_raw) & np.isfinite(post_raw) & np.isfinite(df['Wavelength'].values)
            if np.count_nonzero(valid) < 10:
                continue

            pre = pre_raw[valid].astype(np.float32)
            post = post_raw[valid].astype(np.float32)
            wl_valid = df['Wavelength'].values[valid].astype(np.float32)

            lambda_pre, a_pre, fwhm_pre = self._extract_spectrum_features(wl_valid, pre)
            lambda_post, a_post, _fwhm_post = self._extract_spectrum_features(wl_valid, post)

            x_rows.append([np.log10(conc + 1e-3), lambda_pre, a_pre, fwhm_pre])
            y_rows.append([lambda_post - lambda_pre, a_post - a_pre])
            bsa_feature_rows.append((lambda_pre, a_pre, fwhm_pre))

        if not x_rows:
            raise ValueError('No complete Ag/BSA column pairs found in Excel.')

        if bsa_feature_rows:
            baseline = {
                'lambda': float(np.median([r[0] for r in bsa_feature_rows])),
                'A': float(np.median([r[1] for r in bsa_feature_rows])),
                'fwhm': float(np.median([r[2] for r in bsa_feature_rows])),
            }
        else:
            baseline = {'lambda': float(np.median(self.wavelengths)), 'A': 0.05, 'fwhm': 80.0}

        return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), baseline

    def init_ai_engine(self):
        """Train a lightweight residual model for concentration -> delta features."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        split_train_xlsx = os.path.join(
            base_dir, 'data', 'processed', 'splits_reconstructed', 'train_preprocessed_pairs.xlsx'
        )
        split_train_csv = os.path.join(
            base_dir, 'data', 'processed', 'splits_reconstructed', 'train_preprocessed_pairs.csv'
        )
        filtered_path = os.path.join(base_dir, 'data', 'filtered', 'cea_training_data_20pct.csv')
        spectra_path = os.path.join(base_dir, 'data', 'processed', 'Reconstructed_Preprocessed_Spectra.csv')

        # Priority 1: paired split training set generated from reconstructed spectra.
        if os.path.exists(split_train_xlsx) or os.path.exists(split_train_csv):
            paired_path = split_train_xlsx if os.path.exists(split_train_xlsx) else split_train_csv
            x, y, baseline = self._build_training_from_paired_file(paired_path)
            print(f'Using split train set for homepage Δλ model: {paired_path}')
        # Priority 2: legacy feature CSV + spectra (old path).
        elif os.path.exists(filtered_path) and os.path.exists(spectra_path):
            df_feat = pd.read_csv(filtered_path)
            df_spec = pd.read_csv(spectra_path)
            self.wavelengths = df_spec['Wavelength'].values.astype(np.float32)

            x = np.column_stack([
                np.log10(df_feat['c_ng_ml'].values.astype(np.float32) + 1e-3),
                df_feat['lambda_peak_nm_pre'].values.astype(np.float32),
                df_feat['Apeak_pre'].values.astype(np.float32),
                df_feat['fwhm_nm_pre'].values.astype(np.float32),
            ])
            y = np.column_stack([
                (df_feat['lambda_peak_nm_post'] - df_feat['lambda_peak_nm_pre']).values.astype(np.float32),
                (df_feat['Apeak_post'] - df_feat['Apeak_pre']).values.astype(np.float32),
            ])
            baseline = {
                'lambda': float(df_feat.iloc[0]['lambda_peak_nm_pre']),
                'A': float(df_feat.iloc[0]['Apeak_pre']),
                'fwhm': float(df_feat.iloc[0]['fwhm_nm_pre']),
            }
        else:
            paired_path = os.path.join(base_dir, 'data', 'processed', 'All_Absorbance_Spectra_Preprocessed.xlsx')
            if not os.path.exists(paired_path):
                raise FileNotFoundError(
                    'No usable training source found: missing split train file, legacy CEA CSV, and All_Absorbance_Spectra_Preprocessed.xlsx.'
                )
            x, y, baseline = self._build_training_from_paired_file(paired_path)

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        x_scaled = self.scaler_x.fit_transform(x)
        y_scaled = self.scaler_y.fit_transform(y)

        self.model = LSPRResidualNet()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x_t = torch.FloatTensor(x_scaled)
        y_t = torch.FloatTensor(y_scaled)
        for _ in range(300):
            optimizer.zero_grad()
            loss = criterion(self.model(x_t), y_t)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.baseline_features = baseline

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QFrame()
        control_panel.setFixedWidth(360)
        control_layout = QVBoxLayout(control_panel)

        slider_group = QGroupBox('Target Concentration Simulation')
        slider_layout = QVBoxLayout()

        self.conc_label = QLabel('Current concentration: 5.0 ng/ml')
        self.conc_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #E74C3C;')

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.slider_changed)

        slider_layout.addWidget(self.conc_label)
        slider_layout.addWidget(self.slider)
        slider_group.setLayout(slider_layout)

        data_group = QGroupBox('Realtime Physical Features')
        data_layout = QVBoxLayout()
        self.peak_label = QLabel('Predicted peak wavelength: -- nm')
        self.shift_label = QLabel('Predicted redshift Δλ: -- nm')
        self.amp_label = QLabel('Predicted amplitude A: -- a.u.')

        for lbl in [self.peak_label, self.shift_label, self.amp_label]:
            lbl.setStyleSheet('font-family: Consolas; font-size: 14px; padding: 5px;')
            data_layout.addWidget(lbl)
        data_group.setLayout(data_layout)

        ai_group = QGroupBox('Full-Spectrum AI Tools')
        ai_layout = QVBoxLayout()

        self.btn_inverse = QPushButton('Import spectrum and infer concentration')
        self.btn_inverse.setStyleSheet('background-color: #2ECC71; color: white; font-weight: bold; border-radius: 4px; padding: 8px;')
        self.btn_inverse.clicked.connect(self.inverse_concentration)

        self.btn_spec2spec = QPushButton('Import spectrum and predict spectrum')
        self.btn_spec2spec.setStyleSheet('background-color: #8E44AD; color: white; font-weight: bold; border-radius: 4px; padding: 8px;')
        self.btn_spec2spec.clicked.connect(self.predict_spectrum_from_input)

        self.ai_conc_label = QLabel('AI inferred concentration: -- ng/ml')
        self.ai_conc_label.setStyleSheet('font-size: 14px; color: #27AE60; font-weight: bold;')

        ai_layout.addWidget(self.btn_inverse)
        ai_layout.addWidget(self.btn_spec2spec)
        ai_layout.addWidget(self.ai_conc_label)
        ai_group.setLayout(ai_layout)

        control_layout.addWidget(slider_group)
        control_layout.addWidget(data_group)
        control_layout.addWidget(ai_group)
        control_layout.addStretch()

        canvas_group = QGroupBox('LSPR Spectrum View')
        canvas_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='#F5F7FA')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        canvas_layout.addWidget(self.canvas)
        canvas_group.setLayout(canvas_layout)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(canvas_group)

    def slider_changed(self, value):
        conc = value / 10.0
        self.conc_label.setText(f'Current concentration: {conc:.1f} ng/ml')
        self.update_plot(conc)

    def update_plot(self, conc):
        log_conc = np.log10(conc + 1e-3)
        x_input = np.array([[log_conc, self.baseline_features['lambda'], self.baseline_features['A'], self.baseline_features['fwhm']]], dtype=np.float32)
        x_scaled = self.scaler_x.transform(x_input)

        with torch.no_grad():
            pred_delta_scaled = self.model(torch.FloatTensor(x_scaled))
            pred_delta = self.scaler_y.inverse_transform(pred_delta_scaled.numpy())[0]

        delta_lambda, delta_A = float(pred_delta[0]), float(pred_delta[1])
        final_lambda = self.baseline_features['lambda'] + delta_lambda
        final_A = self.baseline_features['A'] + delta_A

        self.peak_label.setText(f'Predicted peak wavelength: {final_lambda:.2f} nm')
        self.shift_label.setText(f'Predicted redshift Δλ: +{delta_lambda:.2f} nm')
        self.amp_label.setText(f'Predicted amplitude A: {final_A:.4f} a.u.')

        bsa_spec = lorentzian_reconstruct(self.wavelengths, self.baseline_features['lambda'], self.baseline_features['A'], self.baseline_features['fwhm'])
        ag_spec = lorentzian_reconstruct(self.wavelengths, final_lambda, final_A, self.baseline_features['fwhm'])

        self.ax.clear()
        self.ax.plot(self.wavelengths, bsa_spec, color='#7F8C8D', linewidth=2, label='BSA baseline (pre)')
        self.ax.plot(self.wavelengths, ag_spec, color='#E74C3C', linestyle='--', linewidth=2.5, label='Physical formula (post)')

        gen_spec = self.full_spectrum_ai.generate_spectrum(conc)
        ai_wavelengths = self.full_spectrum_ai.get_wavelengths()
        if len(gen_spec) == len(ai_wavelengths):
            # Display-only intensity alignment to reduce peak-height mismatch.
            gen_spec_aligned = self._align_spectrum_intensity(ag_spec, gen_spec)
            self.ax.plot(
                ai_wavelengths,
                gen_spec_aligned,
                color='#9B59B6',
                linewidth=2.5,
                alpha=0.8,
                label='AI digital twin (generator, aligned)',
            )

        self.ax.set_title(f'LSPR Spectral Shift Simulation (CEA = {conc:.1f} ng/ml)', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Absorbance (a.u.)')
        self.ax.set_xlim(520, 700)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend(loc='upper right', fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def _read_spectrum_from_csv(self, file_name):
        ext = os.path.splitext(file_name)[1].lower()

        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_name, sheet_name=0)
        elif ext in ['.txt', '.tsv']:
            df = pd.read_csv(file_name, sep=None, engine='python')
        elif ext == '.csv':
            df = pd.read_csv(file_name)
        else:
            df = pd.read_csv(file_name, sep=None, engine='python')

        if df.empty:
            raise ValueError('File is empty.')

        preferred_cols = ['absorbance', 'intensity', 'signal', 'y']
        for col in df.columns:
            if str(col).strip().lower() in preferred_cols:
                vals = pd.to_numeric(df[col], errors='coerce').dropna().values
                if vals.size > 0:
                    return vals.astype(np.float32)

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        valid_cols = [c for c in numeric_df.columns if numeric_df[c].notna().sum() > 0]
        if not valid_cols:
            raise ValueError('No numeric spectrum column found.')

        vals = numeric_df[valid_cols[-1]].dropna().values
        if vals.size == 0:
            raise ValueError('No valid numeric values found in spectrum column.')
        return vals.astype(np.float32)

    def inverse_concentration(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            'Choose measured spectrum file',
            '',
            'Spectrum Files (*.csv *.txt *.tsv *.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt *.tsv);;Excel Files (*.xlsx *.xls);;All Files (*)',
            options=options,
        )

        if file_name:
            try:
                spectrum = self._read_spectrum_from_csv(file_name)
            except Exception as e:
                QMessageBox.warning(self, 'Import failed', f'Unable to parse spectrum file: {str(e)}')
                return
        else:
            conc = self.slider.value() / 10.0
            spectrum = self.full_spectrum_ai.generate_spectrum(conc)

        pred_conc = self.full_spectrum_ai.predict_concentration(spectrum)
        report = self.full_spectrum_ai.interpret_concentration(pred_conc)
        self.ai_conc_label.setText(f"AI inferred concentration: {report['reported_text']}")
        QMessageBox.information(
            self,
            'Inference done',
            (
                f"Prediction mode: {report['mode']}\n"
                f"Reported result: {report['reported_text']}\n"
                f"Raw model output: {pred_conc:.4f} ng/ml"
            ),
        )

    def predict_spectrum_from_input(self):
        """Input one spectrum and predict another spectrum through the AI chain."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            'Choose input spectrum file',
            '',
            'Spectrum Files (*.csv *.txt *.tsv *.xlsx *.xls);;CSV Files (*.csv);;Text Files (*.txt *.tsv);;Excel Files (*.xlsx *.xls);;All Files (*)',
            options=options,
        )
        if not file_name:
            return

        try:
            spectrum = self._read_spectrum_from_csv(file_name)
            result = self.full_spectrum_ai.predict_spectrum_from_spectrum(spectrum)

            pred_conc = result['pred_concentration']
            report_text = result.get('reported_text', f'{pred_conc:.4f} ng/ml')
            wl = result['wavelengths']
            input_spec = result['input_resampled']
            pred_spec = result['pred_spectrum']

            self.ax.clear()
            self.ax.plot(wl, input_spec, color='#1F77B4', linewidth=2.0, label='Input spectrum')
            self.ax.plot(wl, pred_spec, color='#D62728', linestyle='--', linewidth=2.2, label='Predicted spectrum')
            self.ax.set_title(
                f'Spectrum-to-Spectrum Prediction ({report_text})',
                fontsize=14,
                fontweight='bold',
            )
            self.ax.set_xlabel('Wavelength (nm)')
            self.ax.set_ylabel('Absorbance (a.u.)')
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.ax.legend(loc='upper right', fontsize=9)
            self.figure.tight_layout()
            self.canvas.draw()

            self.ai_conc_label.setText(f'AI inferred concentration: {report_text}')
            QMessageBox.information(
                self,
                'Prediction done',
                (
                    "Spectrum-to-spectrum prediction finished.\n\n"
                    f"Reported result: {report_text}\n"
                    f"Raw model output: {pred_conc:.4f} ng/ml"
                ),
            )
        except Exception as e:
            QMessageBox.warning(self, 'Prediction failed', f'Unable to run spectrum-to-spectrum prediction: {str(e)}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LSPRDigitalTwinApp()
    window.show()
    sys.exit(app.exec_())
