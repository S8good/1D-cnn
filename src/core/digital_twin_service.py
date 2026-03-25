from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .ai_engine import get_ai_engine
from .physics_models import align_spectrum_intensity
from .reconstruction import ResidualPhysicsEngine, ResidualPrediction, default_project_root
from ..utils.data_loader import read_spectrum_file


@dataclass
class PlotContext:
    prediction: ResidualPrediction
    wavelengths: np.ndarray
    bsa_spectrum: np.ndarray
    physical_spectrum: np.ndarray
    ai_wavelengths: np.ndarray
    ai_spectrum_raw: np.ndarray
    ai_spectrum_aligned: Optional[np.ndarray]


class DigitalTwinService:
    """Application service: orchestrates residual physics + full-spectrum AI operations."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or default_project_root()
        self.residual_engine = ResidualPhysicsEngine(self.base_dir)
        self.full_spectrum_ai = get_ai_engine()

    def build_plot_context(self, concentration: float) -> PlotContext:
        bsa_spec, physical_spec, prediction = self.residual_engine.reconstruct(concentration)
        ai_spectrum_raw = np.asarray(
            self.full_spectrum_ai.generate_spectrum(concentration),
            dtype=np.float32,
        ).reshape(-1)
        ai_wavelengths = np.asarray(self.full_spectrum_ai.get_wavelengths(), dtype=np.float32).reshape(-1)

        ai_spectrum_aligned = None
        if ai_spectrum_raw.size == physical_spec.size and ai_wavelengths.size == self.residual_engine.get_wavelengths().size:
            ai_spectrum_aligned = align_spectrum_intensity(physical_spec, ai_spectrum_raw)

        return PlotContext(
            prediction=prediction,
            wavelengths=self.residual_engine.get_wavelengths(),
            bsa_spectrum=bsa_spec,
            physical_spectrum=physical_spec,
            ai_wavelengths=ai_wavelengths,
            ai_spectrum_raw=ai_spectrum_raw,
            ai_spectrum_aligned=ai_spectrum_aligned,
        )

    def infer_concentration_from_file(
        self,
        file_path: Optional[str] = None,
        fallback_concentration: float = 5.0,
    ) -> Dict[str, object]:
        if file_path:
            spectrum = read_spectrum_file(file_path)
        else:
            spectrum = self.full_spectrum_ai.generate_spectrum(float(fallback_concentration))

        pred_conc = float(self.full_spectrum_ai.predict_concentration(spectrum))
        report = self.full_spectrum_ai.interpret_concentration(pred_conc)
        return {
            "pred_concentration": pred_conc,
            "report": report,
            "spectrum": np.asarray(spectrum, dtype=np.float32),
        }

    def predict_spectrum_from_file(self, file_path: str) -> Dict[str, object]:
        spectrum = read_spectrum_file(file_path)
        result = self.full_spectrum_ai.predict_spectrum_from_spectrum(spectrum)
        result["input_file"] = file_path
        return result
