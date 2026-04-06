from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from src.core.stage3_hill import soft_argmax_peak_nm


def _tensor_gradient_1d(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) < 2:
        return torch.zeros_like(x)
    if x.size(1) == 2:
        grad = x[:, 1:2] - x[:, 0:1]
        return torch.cat([grad, grad], dim=1)
    left = x[:, 1:2] - x[:, 0:1]
    mid = 0.5 * (x[:, 2:] - x[:, :-2])
    right = x[:, -1:] - x[:, -2:-1]
    return torch.cat([left, mid, right], dim=1)


def monotonic_penalty(pred_logc: torch.Tensor, true_logc: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(true_logc.squeeze(1))
    sorted_pred = pred_logc[order].squeeze(1)
    if sorted_pred.numel() < 2:
        return torch.zeros((), dtype=pred_logc.dtype, device=pred_logc.device)
    return torch.relu(sorted_pred[:-1] - sorted_pred[1:]).mean()


def predictor_step(predictor, batch, predictor_optimizer, mono_weight):
    xb, pb, yb, _rb = batch[:4]
    predictor_optimizer.zero_grad()
    pred = predictor(xb, pb)
    loss_conc = F.mse_loss(pred, yb)
    loss_mono = monotonic_penalty(pred, yb)
    loss = loss_conc + mono_weight * loss_mono
    loss.backward()
    predictor_optimizer.step()
    return {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_conc": float(loss_conc.detach().cpu().item()),
        "loss_mono": float(loss_mono.detach().cpu().item()),
    }


def generator_step_with_hill(
    predictor,
    generator,
    batch,
    wavelengths_nm,
    generator_optimizer,
    hill_curve,
    hill_weight,
    cycle_weight,
    recon_weight,
    hill_window_center_nm,
    hill_window_half_width_nm,
    hill_temperature,
    hill_reg_weight,
):
    _xb, pb, yb, rb = batch[:4]
    lambda_bsa_nm = batch[4] if len(batch) > 4 else pb[:, 0:1]
    old_flags = [param.requires_grad for param in predictor.parameters()]
    for param in predictor.parameters():
        param.requires_grad = False

    generator_optimizer.zero_grad()
    gen_raw = generator(yb).squeeze(1)
    gen_diff = _tensor_gradient_1d(gen_raw)
    x_gen = torch.stack([gen_raw, gen_diff], dim=1)
    pred_cycle = predictor(x_gen, pb)
    loss_cycle = F.mse_loss(pred_cycle, yb)
    loss_recon = F.mse_loss(gen_raw, rb)
    window_mask = torch.abs(wavelengths_nm - hill_window_center_nm) <= hill_window_half_width_nm
    lambda_ag_hat = soft_argmax_peak_nm(
        gen_raw,
        wavelengths_nm=wavelengths_nm.to(device=gen_raw.device, dtype=gen_raw.dtype),
        window_mask=window_mask.to(device=gen_raw.device),
        temperature=hill_temperature,
    )
    lambda_bsa = lambda_bsa_nm.to(device=gen_raw.device, dtype=gen_raw.dtype)
    conc_ng_ml = torch.clamp(torch.pow(torch.tensor(10.0, dtype=yb.dtype, device=yb.device), yb) - 1e-3, min=0.0)
    delta_lambda_hat = lambda_ag_hat - lambda_bsa
    delta_lambda_target = hill_curve(conc_ng_ml)
    loss_hill = F.mse_loss(delta_lambda_hat, delta_lambda_target)
    loss_reg = hill_curve.regularization_loss()
    loss = cycle_weight * loss_cycle + recon_weight * loss_recon + hill_weight * loss_hill + hill_reg_weight * loss_reg
    loss.backward()
    generator_optimizer.step()

    for param, old_flag in zip(predictor.parameters(), old_flags):
        param.requires_grad = old_flag

    return {
        "loss_total": float(loss.detach().cpu().item()),
        "loss_cycle": float(loss_cycle.detach().cpu().item()),
        "loss_recon": float(loss_recon.detach().cpu().item()),
        "loss_hill": float(loss_hill.detach().cpu().item()),
        "loss_hill_reg": float(loss_reg.detach().cpu().item()),
    }


def run_stage3_alternating_epoch(
    predictor,
    generator,
    train_batches: Iterable,
    wavelengths_nm,
    predictor_optimizer,
    generator_optimizer,
    hill_curve,
    p_steps,
    g_steps,
    mono_weight,
    cycle_weight,
    recon_weight,
    hill_weight,
    hill_window_center_nm,
    hill_window_half_width_nm,
    hill_temperature,
    hill_reg_weight,
):
    predictor_losses = []
    generator_losses = []
    predictor_steps = 0
    generator_steps = 0

    for batch in train_batches:
        for _ in range(max(int(p_steps), 0)):
            predictor_losses.append(predictor_step(predictor, batch, predictor_optimizer, mono_weight))
            predictor_steps += 1
        for _ in range(max(int(g_steps), 0)):
            generator_losses.append(
                generator_step_with_hill(
                    predictor=predictor,
                    generator=generator,
                    batch=batch,
                    wavelengths_nm=wavelengths_nm,
                    generator_optimizer=generator_optimizer,
                    hill_curve=hill_curve,
                    hill_weight=hill_weight,
                    cycle_weight=cycle_weight,
                    recon_weight=recon_weight,
                    hill_window_center_nm=hill_window_center_nm,
                    hill_window_half_width_nm=hill_window_half_width_nm,
                    hill_temperature=hill_temperature,
                    hill_reg_weight=hill_reg_weight,
                )
            )
            generator_steps += 1

    predictor_loss = sum(item["loss_total"] for item in predictor_losses) / len(predictor_losses) if predictor_losses else 0.0
    generator_loss = sum(item["loss_total"] for item in generator_losses) / len(generator_losses) if generator_losses else 0.0
    generator_loss_hill = sum(item["loss_hill"] for item in generator_losses) / len(generator_losses) if generator_losses else 0.0
    return {
        "predictor_steps": predictor_steps,
        "generator_steps": generator_steps,
        "predictor_loss": predictor_loss,
        "generator_loss": generator_loss,
        "generator_loss_hill": generator_loss_hill,
    }
