import torch
import torch.nn as nn

from src.core.stage25_training import generator_step, predictor_step, run_alternating_epoch


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


def _clone_params(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, after):
    return any(not torch.allclose(b, a) for b, a in zip(before, after))


def _build_batch():
    xb = torch.tensor(
        [
            [[0.1, 0.2], [0.0, 0.1]],
            [[0.3, 0.4], [0.1, 0.0]],
        ],
        dtype=torch.float32,
    )
    pb = torch.tensor([[0.5, 0.1, 0.2], [0.6, 0.2, 0.1]], dtype=torch.float32)
    yb = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    rb = xb[:, 0, :]
    return xb, pb, yb, rb


def test_predictor_step_updates_predictor_only():
    predictor = TinyPredictor()
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    before = _clone_params(predictor)

    losses = predictor_step(predictor, _build_batch(), optimizer, mono_weight=0.05)
    after = _clone_params(predictor)

    assert losses["loss_conc"] >= 0
    assert losses["loss_mono"] >= 0
    assert _changed(before, after)


def test_generator_step_updates_generator_without_moving_predictor():
    predictor = TinyPredictor()
    generator = TinyGenerator()
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    predictor_before = _clone_params(predictor)
    generator_before = _clone_params(generator)

    losses = generator_step(predictor, generator, _build_batch(), optimizer, cycle_weight=0.01, recon_weight=0.05)
    predictor_after = _clone_params(predictor)
    generator_after = _clone_params(generator)

    assert losses["loss_cycle"] >= 0
    assert losses["loss_recon"] >= 0
    assert not _changed(predictor_before, predictor_after)
    assert _changed(generator_before, generator_after)


def test_run_alternating_epoch_reports_expected_step_counts():
    predictor = TinyPredictor()
    generator = TinyGenerator()
    predictor_opt = torch.optim.SGD(predictor.parameters(), lr=0.05)
    generator_opt = torch.optim.SGD(generator.parameters(), lr=0.05)
    batch = _build_batch()

    stats = run_alternating_epoch(
        predictor=predictor,
        generator=generator,
        train_batches=[batch, batch],
        predictor_optimizer=predictor_opt,
        generator_optimizer=generator_opt,
        p_steps=2,
        g_steps=1,
        mono_weight=0.05,
        cycle_weight=0.01,
        recon_weight=0.05,
    )

    assert stats["predictor_steps"] == 4
    assert stats["generator_steps"] == 2
