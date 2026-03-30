from typing import Iterable

import torch
import torch.nn.functional as F


def _tensor_gradient_1d(x: torch.Tensor) -> torch.Tensor:
    left = x[:, 1:2] - x[:, 0:1]
    mid = (x[:, 2:] - x[:, :-2]) * 0.5
    right = x[:, -1:] - x[:, -2:-1]
    return torch.cat([left, mid, right], dim=1)


def monotonic_penalty(pred_logc: torch.Tensor, true_logc: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(true_logc.squeeze(1))
    sorted_pred = pred_logc[order].squeeze(1)
    if sorted_pred.numel() <= 1:
        return torch.zeros((), dtype=sorted_pred.dtype, device=sorted_pred.device)
    return torch.relu(sorted_pred[:-1] - sorted_pred[1:]).mean()


def predictor_step(predictor, batch, predictor_optimizer, mono_weight: float):
    xb, pb, yb, _rb = batch
    predictor_optimizer.zero_grad()
    pred_real = predictor(xb, pb)
    loss_conc = F.huber_loss(pred_real, yb, delta=0.2)
    loss_mono = monotonic_penalty(pred_real, yb)
    loss = loss_conc + mono_weight * loss_mono
    loss.backward()
    predictor_optimizer.step()
    return {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_conc": float(loss_conc.detach().cpu().item()),
        "loss_mono": float(loss_mono.detach().cpu().item()),
    }


def generator_step(predictor, generator, batch, generator_optimizer, cycle_weight: float, recon_weight: float):
    _xb, pb, yb, rb = batch
    old_flags = [p.requires_grad for p in predictor.parameters()]
    for param in predictor.parameters():
        param.requires_grad = False

    generator_optimizer.zero_grad()
    gen_raw = generator(yb).squeeze(1)
    gen_diff = _tensor_gradient_1d(gen_raw)
    x_gen = torch.stack([gen_raw, gen_diff], dim=1)
    pred_cycle = predictor(x_gen, pb)
    loss_cycle = F.mse_loss(pred_cycle, yb)
    loss_recon = F.mse_loss(gen_raw, rb)
    loss = cycle_weight * loss_cycle + recon_weight * loss_recon
    loss.backward()
    generator_optimizer.step()

    for param, flag in zip(predictor.parameters(), old_flags):
        param.requires_grad = flag

    return {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_cycle": float(loss_cycle.detach().cpu().item()),
        "loss_recon": float(loss_recon.detach().cpu().item()),
    }


def run_alternating_epoch(
    predictor,
    generator,
    train_batches: Iterable,
    predictor_optimizer,
    generator_optimizer,
    p_steps: int,
    g_steps: int,
    mono_weight: float,
    cycle_weight: float,
    recon_weight: float,
):
    predictor_losses = []
    generator_losses = []
    predictor_steps = 0
    generator_steps = 0

    for batch in train_batches:
        for _ in range(max(p_steps, 0)):
            predictor_losses.append(predictor_step(predictor, batch, predictor_optimizer, mono_weight))
            predictor_steps += 1
        for _ in range(max(g_steps, 0)):
            generator_losses.append(
                generator_step(predictor, generator, batch, generator_optimizer, cycle_weight, recon_weight)
            )
            generator_steps += 1

    mean_predictor_loss = (
        sum(item["loss_total"] for item in predictor_losses) / len(predictor_losses) if predictor_losses else 0.0
    )
    mean_generator_loss = (
        sum(item["loss_total"] for item in generator_losses) / len(generator_losses) if generator_losses else 0.0
    )

    return {
        "predictor_steps": predictor_steps,
        "generator_steps": generator_steps,
        "predictor_loss": mean_predictor_loss,
        "generator_loss": mean_generator_loss,
    }
