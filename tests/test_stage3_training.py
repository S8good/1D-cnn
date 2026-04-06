import torch
import torch.nn as nn

from src.core.stage3_hill import FixedHillCurve
from src.core.stage3_training import generator_step_with_hill, run_stage3_alternating_epoch


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 1)

    def forward(self, x_spectrum, x_physics):
        flat = x_spectrum.view(x_spectrum.size(0), -1)
        return self.linear(torch.cat([flat, x_physics], dim=1))


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, y_log):
        return self.linear(y_log).unsqueeze(1)


class PeakGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, y_log):
        low_peak = torch.tensor([0.0, 6.0, 6.0, 0.0, 0.0], dtype=y_log.dtype, device=y_log.device) + self.dummy
        rows = [low_peak for _ in range(y_log.size(0))]
        return torch.stack(rows, dim=0).unsqueeze(1)


def _build_batch():
    xb = torch.tensor([[[0.1, 0.2], [0.0, 0.1]], [[0.3, 0.4], [0.1, 0.0]]], dtype=torch.float32)
    pb = torch.tensor([[603.0, 0.1, 0.2], [603.5, 0.2, 0.1]], dtype=torch.float32)
    yb = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    rb = xb[:, 0, :]
    wavelengths = torch.tensor([602.0, 603.0], dtype=torch.float32)
    return xb, pb, yb, rb, wavelengths


def test_generator_step_with_hill_reports_loss_hill():
    predictor = TinyPredictor()
    generator = TinyGenerator()
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    xb, pb, yb, rb, wavelengths = _build_batch()
    losses = generator_step_with_hill(
        predictor=predictor,
        generator=generator,
        batch=(xb, pb, yb, rb),
        wavelengths_nm=wavelengths,
        generator_optimizer=optimizer,
        hill_curve=FixedHillCurve(delta_lambda_max=5.0, k_half=10.0, hill_n=1.2),
        hill_weight=0.1,
        cycle_weight=0.01,
        recon_weight=0.05,
        hill_window_center_nm=603.0,
        hill_window_half_width_nm=1.0,
        hill_temperature=0.25,
        hill_reg_weight=0.0,
    )
    assert losses["loss_hill"] >= 0


def test_run_stage3_alternating_epoch_counts_hill_generator_steps():
    predictor = TinyPredictor()
    generator = TinyGenerator()
    predictor_optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    xb, pb, yb, rb, wavelengths = _build_batch()
    stats = run_stage3_alternating_epoch(
        predictor=predictor,
        generator=generator,
        train_batches=[(xb, pb, yb, rb)],
        wavelengths_nm=wavelengths,
        predictor_optimizer=predictor_optimizer,
        generator_optimizer=generator_optimizer,
        hill_curve=FixedHillCurve(delta_lambda_max=5.0, k_half=10.0, hill_n=1.2),
        p_steps=0,
        g_steps=1,
        mono_weight=0.05,
        cycle_weight=0.01,
        recon_weight=0.05,
        hill_weight=0.1,
        hill_window_center_nm=603.0,
        hill_window_half_width_nm=1.0,
        hill_temperature=0.25,
        hill_reg_weight=0.0,
    )
    assert stats["generator_steps"] == 1
    assert stats["generator_loss_hill"] >= 0


def test_generator_step_with_hill_uses_raw_lambda_bsa_when_batch_provides_it():
    class WidePredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(13, 1)

        def forward(self, x_spectrum, x_physics):
            flat = x_spectrum.view(x_spectrum.size(0), -1)
            return self.linear(torch.cat([flat, x_physics], dim=1))

    predictor = WidePredictor()
    generator = PeakGenerator()
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    xb = torch.tensor([[[0.1, 0.2, 0.2, 0.1, 0.0], [0.1, 0.0, -0.1, -0.2, -0.1]]], dtype=torch.float32)
    pb = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
    yb = torch.tensor([[0.0]], dtype=torch.float32)
    rb = xb[:, 0, :]
    lambda_bsa_nm = torch.tensor([[600.0]], dtype=torch.float32)
    wavelengths = torch.tensor([601.0, 602.0, 603.0, 604.0, 605.0], dtype=torch.float32)
    losses = generator_step_with_hill(
        predictor=predictor,
        generator=generator,
        batch=(xb, pb, yb, rb, lambda_bsa_nm),
        wavelengths_nm=wavelengths,
        generator_optimizer=optimizer,
        hill_curve=FixedHillCurve(delta_lambda_max=2.5, k_half=0.01, hill_n=10.0),
        hill_weight=1.0,
        cycle_weight=0.0,
        recon_weight=0.0,
        hill_window_center_nm=603.0,
        hill_window_half_width_nm=5.0,
        hill_temperature=0.01,
        hill_reg_weight=0.0,
    )
    assert losses["loss_hill"] < 0.1
