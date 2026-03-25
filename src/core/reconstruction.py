from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from .neural_network import LSPRResidualNet
from .physics_models import extract_spectrum_features, lorentzian_reconstruct
from ..utils.data_loader import (
    TrainingSamples,
    build_training_samples_from_legacy_files,
    build_training_samples_from_paired_file,
    find_training_source,
)


@dataclass
class ResidualPrediction:
    concentration: float
    delta_lambda: float
    delta_amplitude: float
    peak_wavelength: float
    peak_intensity: float
    fwhm: float


class ResidualPhysicsEngine:
    """Train/use a lightweight residual network for concentration -> spectral feature deltas."""

    def __init__(self, base_dir: str, epochs: int = 300, learning_rate: float = 0.01):
        self.base_dir = base_dir
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)

        self.wavelengths: np.ndarray | None = None
        self.baseline_features: Dict[str, float] | None = None
        self.model: LSPRResidualNet | None = None
        self.scaler_x: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        self.training_source: str | None = None

        self._train_model()

    def _load_training_samples(self) -> TrainingSamples:
        source_type, paths = find_training_source(self.base_dir)
        if source_type == "paired":
            path = paths[0]
            return build_training_samples_from_paired_file(path, feature_extractor=extract_spectrum_features)
        if source_type == "legacy":
            return build_training_samples_from_legacy_files(paths[0], paths[1])
        raise RuntimeError(f"Unsupported training source type: {source_type}")

    def _train_model(self) -> None:
        samples = self._load_training_samples()
        self.training_source = samples.source_path
        self.wavelengths = samples.wavelengths
        self.baseline_features = samples.baseline

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        x_scaled = self.scaler_x.fit_transform(samples.features)
        y_scaled = self.scaler_y.fit_transform(samples.targets)

        self.model = LSPRResidualNet()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        x_t = torch.FloatTensor(x_scaled)
        y_t = torch.FloatTensor(y_scaled)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(x_t), y_t)
            loss.backward()
            optimizer.step()

        self.model.eval()

    def get_wavelengths(self) -> np.ndarray:
        if self.wavelengths is None:
            raise RuntimeError("Residual model is not initialized.")
        return np.asarray(self.wavelengths, dtype=np.float32)

    def predict(self, concentration: float) -> ResidualPrediction:
        if self.model is None or self.scaler_x is None or self.scaler_y is None or self.baseline_features is None:
            raise RuntimeError("Residual model is not initialized.")

        conc = float(max(0.0, concentration))
        x_input = np.array(
            [
                [
                    np.log10(conc + 1e-3),
                    self.baseline_features["lambda"],
                    self.baseline_features["A"],
                    self.baseline_features["fwhm"],
                ]
            ],
            dtype=np.float32,
        )
        x_scaled = self.scaler_x.transform(x_input)

        with torch.no_grad():
            pred_delta_scaled = self.model(torch.FloatTensor(x_scaled))
            pred_delta = self.scaler_y.inverse_transform(pred_delta_scaled.numpy())[0]

        delta_lambda, delta_a = float(pred_delta[0]), float(pred_delta[1])
        peak_wavelength = float(self.baseline_features["lambda"] + delta_lambda)
        peak_intensity = float(self.baseline_features["A"] + delta_a)
        return ResidualPrediction(
            concentration=conc,
            delta_lambda=delta_lambda,
            delta_amplitude=delta_a,
            peak_wavelength=peak_wavelength,
            peak_intensity=peak_intensity,
            fwhm=float(self.baseline_features["fwhm"]),
        )

    def reconstruct(self, concentration: float) -> Tuple[np.ndarray, np.ndarray, ResidualPrediction]:
        prediction = self.predict(concentration)
        wl = self.get_wavelengths()
        baseline = self.baseline_features
        if baseline is None:
            raise RuntimeError("Residual model baseline is not initialized.")

        bsa_spec = lorentzian_reconstruct(wl, baseline["lambda"], baseline["A"], baseline["fwhm"])
        ag_spec = lorentzian_reconstruct(
            wl,
            prediction.peak_wavelength,
            prediction.peak_intensity,
            prediction.fwhm,
        )
        return (
            np.asarray(bsa_spec, dtype=np.float32),
            np.asarray(ag_spec, dtype=np.float32),
            prediction,
        )


def default_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
