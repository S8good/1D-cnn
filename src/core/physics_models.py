from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import optimize


def lorentzian_reconstruct(
    wavelengths: np.ndarray,
    peak_pos: float,
    amplitude: float,
    fwhm: float,
) -> np.ndarray:
    gamma = float(fwhm) / 2.0
    wl = np.asarray(wavelengths, dtype=np.float32)
    return amplitude * (gamma**2) / ((wl - peak_pos) ** 2 + gamma**2)


def estimate_fwhm(wavelengths: np.ndarray, intensity: np.ndarray, peak_idx: int) -> float:
    wl = np.asarray(wavelengths, dtype=np.float32)
    y = np.asarray(intensity, dtype=np.float32)
    peak = float(y[peak_idx])
    base = float(np.min(y))
    half = base + (peak - base) / 2.0

    left = int(peak_idx)
    while left > 0 and y[left] > half:
        left -= 1

    right = int(peak_idx)
    while right < len(y) - 1 and y[right] > half:
        right += 1

    if right == left:
        return float(np.median(np.diff(wl)) * 8.0)
    return float(abs(wl[right] - wl[left]))


def gaussian_model(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def fit_gaussian_peak(wavelengths: np.ndarray, intensity: np.ndarray) -> Optional[Tuple[float, float, float]]:
    wl = np.asarray(wavelengths, dtype=np.float32)
    y = np.asarray(intensity, dtype=np.float32)
    mask = (wl >= 500.0) & (wl <= 800.0) & np.isfinite(y)
    x = wl[mask]
    s = y[mask]
    if x.size < 5:
        return None

    idx = int(np.argmax(s))
    p0 = [
        max(float(s[idx]), 1e-8),
        float(x[idx]),
        max(float((x[-1] - x[0]) * 0.1), 1.0),
    ]
    try:
        popt, _ = optimize.curve_fit(gaussian_model, x, s, p0=p0, maxfev=3000)
        amp, center, sigma = float(popt[0]), float(popt[1]), abs(float(popt[2]))
        if 500.0 <= center <= 800.0 and amp > 0 and sigma > 0:
            return amp, center, sigma
    except Exception:
        return None
    return None


def extract_spectrum_features(wavelengths: np.ndarray, intensity: np.ndarray) -> Tuple[float, float, float]:
    wl = np.asarray(wavelengths, dtype=np.float32)
    y = np.asarray(intensity, dtype=np.float32)
    fit = fit_gaussian_peak(wl, y)
    if fit is not None:
        amp, center, sigma = fit
        peak_idx = int(np.argmin(np.abs(wl - center)))
        fwhm = estimate_fwhm(wl, y, peak_idx)
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = float(2.355 * sigma)
        return float(center), float(amp), float(fwhm)

    peak_idx = int(np.argmax(y))
    center = float(wl[peak_idx])
    amp = float(y[peak_idx])
    fwhm = estimate_fwhm(wl, y, peak_idx)
    return center, amp, float(fwhm)


def align_spectrum_intensity(ref_spec: np.ndarray, pred_spec: np.ndarray) -> np.ndarray:
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
