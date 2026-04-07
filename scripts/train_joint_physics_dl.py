import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.full_spectrum_models import SpectralPredictorV2_Fusion, SpectrumGenerator
from src.core.stage25_config import apply_profile_overrides, build_stage25_profile
from src.core.stage25_training import run_alternating_epoch


def _prepare_model_dirs(project_root: str):
    models_root = os.path.join(project_root, "models")
    checkpoints_dir = os.path.join(models_root, "checkpoints")
    pretrained_dir = os.path.join(models_root, "pretrained")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)
    return models_root, checkpoints_dir, pretrained_dir


def _load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def load_bsa_feature_lookup(features_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_excel(features_path, sheet_name="sheet1")
    required = ["spectrum_col", "stage", "peak_wavelength_nm", "peak_intensity_au", "fwhm_nm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in sheet1: {missing}")

    df = df.copy()
    df["stage"] = df["stage"].astype(str).str.upper()
    bsa_df = df[df["stage"] == "BSA"].copy()
    lookup: Dict[str, np.ndarray] = {}
    for _, row in bsa_df.iterrows():
        key = str(row["spectrum_col"])
        vals = np.array(
            [
                float(row["peak_wavelength_nm"]),
                float(row["peak_intensity_au"]),
                float(row["fwhm_nm"]),
            ],
            dtype=np.float32,
        )
        if np.all(np.isfinite(vals)):
            lookup[key] = vals
    if not lookup:
        raise RuntimeError("No valid BSA features found in sheet1.")
    return lookup


def derive_bsa_col(ag_col: str) -> str:
    return re.sub(r"-Ag-", "-BSA-", ag_col, flags=re.IGNORECASE)


def parse_training_data_fusion(
    data_path: str,
    bsa_lookup: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _load_table(data_path)
    if "Wavelength" not in df.columns:
        raise ValueError("Input data must contain 'Wavelength' column.")

    wl = pd.to_numeric(df["Wavelength"], errors="coerce").dropna().values.astype(np.float32)
    wl_raw = df["Wavelength"].values.astype(np.float32)
    spectra, physics, concs = [], [], []

    for col in df.columns:
        if col == "Wavelength" or "Ag" not in str(col):
            continue
        m = re.search(r"([0-9.]+)\s*ng/ml", str(col), flags=re.IGNORECASE)
        if not m:
            continue
        conc = float(m.group(1))
        bsa_col = derive_bsa_col(str(col))
        if bsa_col not in bsa_lookup:
            continue

        spec = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        valid = np.isfinite(spec) & np.isfinite(wl_raw)
        if np.count_nonzero(valid) < int(0.9 * len(wl)):
            continue
        spec_valid = spec[valid]
        wl_valid = wl_raw[valid]
        if spec_valid.size != wl.size:
            spec_valid = np.interp(wl, wl_valid, spec_valid).astype(np.float32)

        spectra.append(spec_valid)
        physics.append(bsa_lookup[bsa_col])
        concs.append(np.log10(conc + 1e-3))

    if not spectra:
        raise RuntimeError(f"No valid Ag spectra with BSA physics were parsed from: {data_path}")

    return (
        np.stack(spectra).astype(np.float32),
        np.stack(physics).astype(np.float32),
        np.asarray(concs, dtype=np.float32),
        wl,
    )


def build_input_channels(specs: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    diff_specs = np.gradient(specs, axis=1)
    raw_norm = (specs - stats["raw_med"][None, :]) / (stats["raw_iqr"][None, :] + 1e-8)
    diff_norm = (diff_specs - stats["diff_med"][None, :]) / (stats["diff_iqr"][None, :] + 1e-8)
    return np.stack([raw_norm, diff_norm], axis=1).astype(np.float32)


def build_physics_inputs(physics: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return ((physics - stats["phy_med"][None, :]) / (stats["phy_iqr"][None, :] + 1e-8)).astype(np.float32)


def monotonic_penalty(pred_logc: torch.Tensor, true_logc: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(true_logc.squeeze(1))
    sorted_pred = pred_logc[order].squeeze(1)
    return torch.relu(sorted_pred[:-1] - sorted_pred[1:]).mean()


def evaluate_regression(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true = np.power(10.0, y_true_log) - 1e-3
    y_pred = np.power(10.0, y_pred_log) - 1e-3
    y_true = np.clip(y_true, a_min=0.0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(abs_err / np.maximum(y_true, 1e-8)) * 100.0)
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum((y_pred - y_true) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"mae_ng_ml": mae, "rmse_ng_ml": rmse, "mape_pct": mape, "r2": r2}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_predictor_trainability(
    predictor: SpectralPredictorV2_Fusion,
    train_mode: str,
) -> List[nn.Parameter]:
    for p in predictor.parameters():
        p.requires_grad = False

    if train_mode == "all":
        for p in predictor.parameters():
            p.requires_grad = True
    elif train_mode == "tail":
        # Keep low-level spectral feature extractor fixed and tune fusion tail.
        for module in (predictor.spectral_head, predictor.physics_encoder, predictor.regressor):
            for p in module.parameters():
                p.requires_grad = True
    elif train_mode == "regressor":
        for p in predictor.regressor.parameters():
            p.requires_grad = True
    elif train_mode == "frozen":
        pass
    else:
        raise ValueError(f"Unsupported predictor train mode: {train_mode}")

    return [p for p in predictor.parameters() if p.requires_grad]


def set_frozen_submodules_eval(module: nn.Module) -> None:
    for sub in module.modules():
        own_params = list(sub.parameters(recurse=False))
        if own_params and not any(p.requires_grad for p in own_params):
            sub.eval()


def parse_args() -> argparse.Namespace:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    split_dir = os.path.join(root, "data", "processed", "splits_reconstructed")
    parser = argparse.ArgumentParser(description="Stage2 joint training: Fusion predictor + cycle loss.")
    parser.add_argument("--train", type=str, default=os.path.join(split_dir, "train_preprocessed_pairs.xlsx"))
    parser.add_argument("--val", type=str, default=os.path.join(split_dir, "val_preprocessed_pairs.xlsx"))
    parser.add_argument("--test", type=str, default=os.path.join(split_dir, "test_preprocessed_pairs.xlsx"))
    parser.add_argument(
        "--features-xlsx",
        type=str,
        default=os.path.join(root, "data", "processed", "Reconstructed_Preprocessed_Features_and_Delta.xlsx"),
    )
    parser.add_argument(
        "--fusion-weights",
        type=str,
        default=os.path.join(root, "models", "pretrained", "spectral_predictor_v2_fusion.pth"),
    )
    parser.add_argument(
        "--fusion-norm",
        type=str,
        default=os.path.join(root, "models", "pretrained", "predictor_v2_fusion_norm_params.pth"),
    )
    parser.add_argument(
        "--generator-weights",
        type=str,
        default=os.path.join(root, "models", "pretrained", "spectral_generator_cycle.pth"),
    )
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-gen-epochs", type=int, default=40)
    parser.add_argument("--joint-epochs", type=int, default=180)
    parser.add_argument("--predictor-lr", type=float, default=2e-4)
    parser.add_argument("--generator-lr", type=float, default=1e-3)
    parser.add_argument("--update-strategy", type=str, default="joint", choices=["joint", "alternating"])
    parser.add_argument("--stage25-profile", type=str, choices=["2.5A", "2.5B", "2.5C"])
    parser.add_argument("--freeze-predictor", action="store_true")
    parser.add_argument(
        "--predictor-train-mode",
        type=str,
        default="all",
        choices=["all", "tail", "regressor", "frozen"],
        help="Predictor finetune scope during joint phase. Ignored when --freeze-predictor is set.",
    )
    parser.add_argument("--w-conc", type=float, default=1.0)
    parser.add_argument("--w-cycle", type=float, default=0.1)
    parser.add_argument("--w-mono", type=float, default=0.05)
    parser.add_argument("--w-recon", type=float, default=0.05)
    parser.add_argument("--p-steps", type=int, default=1)
    parser.add_argument("--g-steps", type=int, default=1)
    args = parser.parse_args()
    if args.stage25_profile:
        apply_stage25_profile(args, build_stage25_profile(args.stage25_profile))
    return args


def _tensor_gradient_1d(x: torch.Tensor) -> torch.Tensor:
    left = x[:, 1:2] - x[:, 0:1]
    mid = (x[:, 2:] - x[:, :-2]) * 0.5
    right = x[:, -1:] - x[:, -2:-1]
    return torch.cat([left, mid, right], dim=1)


def evaluate_predictor(
    model: SpectralPredictorV2_Fusion,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    ys, ps = [], []
    with torch.no_grad():
        for xb, pb, yb, _rb in loader:
            xb = xb.to(device)
            pb = pb.to(device)
            yb = yb.to(device)
            pred = model(xb, pb)
            losses.append(nn.functional.mse_loss(pred, yb).item())
            ys.append(yb.cpu().numpy().reshape(-1))
            ps.append(pred.cpu().numpy().reshape(-1))
    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(ps)


def apply_stage25_profile(args: argparse.Namespace, profile) -> argparse.Namespace:
    return apply_profile_overrides(args, profile)


def build_output_tag(stage25_profile: str) -> str:
    if not stage25_profile:
        return "cycle"
    return f"stage25_{stage25_profile.lower().replace('.', 'p')}"


def run_joint_training_epoch(
    predictor,
    generator,
    train_loader,
    predictor_optimizer,
    generator_optimizer,
    update_strategy: str,
    mono_weight: float,
    cycle_weight: float,
    recon_weight: float,
    p_steps: int,
    g_steps: int,
    *,
    alternating_runner=run_alternating_epoch,
    joint_optimizer=None,
    device=None,
    huber_loss=None,
    mse_loss=None,
    conc_weight: float = 1.0,
):
    if update_strategy == "alternating":
        if predictor_optimizer is None and p_steps > 0:
            raise ValueError("predictor_optimizer is required when alternating predictor updates are enabled.")
        if generator_optimizer is None and g_steps > 0:
            raise ValueError("generator_optimizer is required when alternating generator updates are enabled.")
        stats = alternating_runner(
            predictor=predictor,
            generator=generator,
            train_batches=train_loader,
            predictor_optimizer=predictor_optimizer,
            generator_optimizer=generator_optimizer,
            p_steps=p_steps,
            g_steps=g_steps,
            mono_weight=mono_weight,
            cycle_weight=cycle_weight,
            recon_weight=recon_weight,
        )
        return {
            "loss": float((stats["predictor_loss"] + stats["generator_loss"]) * 0.5),
            "predictor_steps": int(stats["predictor_steps"]),
            "generator_steps": int(stats["generator_steps"]),
            "predictor_loss": float(stats["predictor_loss"]),
            "generator_loss": float(stats["generator_loss"]),
        }

    if joint_optimizer is None or device is None or huber_loss is None or mse_loss is None:
        raise ValueError("joint strategy requires joint_optimizer, device, huber_loss, and mse_loss.")

    losses = []
    for xb, pb, yb, rb in train_loader:
        xb = xb.to(device)
        pb = pb.to(device)
        yb = yb.to(device)
        rb = rb.to(device)

        joint_optimizer.zero_grad()
        pred_real = predictor(xb, pb)
        loss_conc = huber_loss(pred_real, yb)

        gen_raw = generator(yb).squeeze(1)
        gen_diff = _tensor_gradient_1d(gen_raw)
        x_gen = torch.stack([gen_raw, gen_diff], dim=1)
        pred_cycle = predictor(x_gen, pb)
        loss_cycle = mse_loss(pred_cycle, yb)
        loss_mono = monotonic_penalty(pred_real, yb)
        loss_recon = mse_loss(gen_raw, rb)

        loss = conc_weight * loss_conc + cycle_weight * loss_cycle + mono_weight * loss_mono + recon_weight * loss_recon
        loss.backward()
        joint_optimizer.step()
        losses.append(loss.item())

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "predictor_steps": len(train_loader),
        "generator_steps": len(train_loader),
        "predictor_loss": 0.0,
        "generator_loss": 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Update strategy: {args.update_strategy}")
    if args.stage25_profile:
        print(f"Stage 2.5 profile: {args.stage25_profile}")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _models_root, checkpoints_dir, pretrained_dir = _prepare_model_dirs(root)
    train_path = os.path.abspath(args.train)
    val_path = os.path.abspath(args.val)
    test_path = os.path.abspath(args.test)
    features_path = os.path.abspath(args.features_xlsx)

    norm_params = torch.load(os.path.abspath(args.fusion_norm), map_location="cpu")
    stats = {
        "raw_med": np.asarray(norm_params["raw_med"], dtype=np.float32),
        "raw_iqr": np.asarray(norm_params["raw_iqr"], dtype=np.float32),
        "diff_med": np.asarray(norm_params["diff_med"], dtype=np.float32),
        "diff_iqr": np.asarray(norm_params["diff_iqr"], dtype=np.float32),
        "phy_med": np.asarray(norm_params["phy_med"], dtype=np.float32),
        "phy_iqr": np.asarray(norm_params["phy_iqr"], dtype=np.float32),
    }

    bsa_lookup = load_bsa_feature_lookup(features_path)
    specs_train, phy_train, y_train, wl_train = parse_training_data_fusion(train_path, bsa_lookup)
    specs_val, phy_val, y_val, wl_val = parse_training_data_fusion(val_path, bsa_lookup)
    specs_test, phy_test, y_test, wl_test = parse_training_data_fusion(test_path, bsa_lookup)
    if not (np.allclose(wl_train, wl_val) and np.allclose(wl_train, wl_test)):
        raise RuntimeError("Train/val/test wavelength grids are not identical.")

    x_train = build_input_channels(specs_train, stats)
    x_val = build_input_channels(specs_val, stats)
    x_test = build_input_channels(specs_test, stats)
    p_train = build_physics_inputs(phy_train, stats)
    p_val = build_physics_inputs(phy_val, stats)
    p_test = build_physics_inputs(phy_test, stats)

    raw_train = x_train[:, 0, :]
    raw_val = x_val[:, 0, :]
    raw_test = x_test[:, 0, :]

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(p_train),
        torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32)),
        torch.from_numpy(raw_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(p_val),
        torch.from_numpy(y_val.reshape(-1, 1).astype(np.float32)),
        torch.from_numpy(raw_val.astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test),
        torch.from_numpy(p_test),
        torch.from_numpy(y_test.reshape(-1, 1).astype(np.float32)),
        torch.from_numpy(raw_test.astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    seq_len = x_train.shape[-1]
    predictor = SpectralPredictorV2_Fusion(seq_len=seq_len).to(device)
    predictor.load_state_dict(torch.load(os.path.abspath(args.fusion_weights), map_location="cpu"))
    generator = SpectrumGenerator(seq_len=seq_len).to(device)

    gen_path = os.path.abspath(args.generator_weights)
    if os.path.exists(gen_path):
        generator.load_state_dict(torch.load(gen_path, map_location="cpu"))
        print(f"Loaded generator: {gen_path}")
    else:
        print("Generator weights not found, using random init.")

    # Optional generator warm-up on concentration->spectrum mapping.
    if args.pretrain_gen_epochs > 0:
        gen_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
        gen_mse = nn.MSELoss()
        for ep in range(args.pretrain_gen_epochs):
            generator.train()
            loss_ep = []
            for _xb, _pb, yb, rb in train_loader:
                yb = yb.to(device)
                rb = rb.to(device)
                gen_optimizer.zero_grad()
                g = generator(yb).squeeze(1)
                loss = gen_mse(g, rb)
                loss.backward()
                gen_optimizer.step()
                loss_ep.append(loss.item())
            if (ep + 1) % 10 == 0:
                print(f"Pretrain-Gen {ep + 1:3d}/{args.pretrain_gen_epochs} | loss={float(np.mean(loss_ep)):.5f}")

    predictor_train_mode = "frozen" if args.freeze_predictor else args.predictor_train_mode
    trainable_predictor_params = configure_predictor_trainability(predictor, predictor_train_mode)
    predictor_trainable_count = int(sum(p.numel() for p in trainable_predictor_params))
    predictor_total_count = int(sum(p.numel() for p in predictor.parameters()))
    print(
        f"Predictor train mode: {predictor_train_mode} | "
        f"trainable_params={predictor_trainable_count}/{predictor_total_count}"
    )

    optim_groups = []
    if trainable_predictor_params:
        optim_groups.append({"params": trainable_predictor_params, "lr": float(args.predictor_lr)})
    optim_groups.append({"params": generator.parameters(), "lr": float(args.generator_lr)})
    optimizer = optim.AdamW(optim_groups, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    predictor_optimizer = None
    predictor_scheduler = None
    if trainable_predictor_params:
        predictor_optimizer = optim.AdamW(trainable_predictor_params, lr=float(args.predictor_lr), weight_decay=3e-4)
        predictor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            predictor_optimizer, mode="min", factor=0.5, patience=10
        )
    generator_optimizer = optim.AdamW(generator.parameters(), lr=float(args.generator_lr), weight_decay=3e-4)
    generator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        generator_optimizer, mode="min", factor=0.5, patience=10
    )
    huber = nn.HuberLoss(delta=0.2)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(args.joint_epochs):
        if not trainable_predictor_params:
            predictor.eval()
        else:
            predictor.train()
            if predictor_train_mode != "all":
                # Keep frozen branches in eval mode to avoid BN/Dropout drift.
                set_frozen_submodules_eval(predictor)
        generator.train()
        train_stats = run_joint_training_epoch(
            predictor=predictor,
            generator=generator,
            train_loader=train_loader,
            predictor_optimizer=predictor_optimizer,
            generator_optimizer=generator_optimizer,
            update_strategy=args.update_strategy,
            mono_weight=args.w_mono,
            cycle_weight=args.w_cycle,
            recon_weight=args.w_recon,
            p_steps=args.p_steps,
            g_steps=args.g_steps,
            joint_optimizer=optimizer,
            device=device,
            huber_loss=huber,
            mse_loss=mse,
            conc_weight=args.w_conc,
        )

        val_loss, _, _ = evaluate_predictor(predictor, val_loader, device)
        if args.update_strategy == "alternating":
            if predictor_scheduler is not None:
                predictor_scheduler.step(val_loss)
            generator_scheduler.step(val_loss)
        else:
            scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "predictor": {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
                "generator": {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()},
                "epoch": epoch + 1,
            }
        if (epoch + 1) % 20 == 0:
            if args.update_strategy == "alternating":
                if predictor_optimizer is not None:
                    lr_display = predictor_optimizer.param_groups[0]["lr"]
                else:
                    lr_display = generator_optimizer.param_groups[0]["lr"]
            else:
                lr_display = optimizer.param_groups[0]["lr"]
            print(
                f"Joint {epoch + 1:3d}/{args.joint_epochs} | train={train_stats['loss']:.5f} | "
                f"val={val_loss:.5f} | lr={lr_display:.2e}"
            )

    if best_state is not None:
        predictor.load_state_dict(best_state["predictor"])
        generator.load_state_dict(best_state["generator"])

    _, y_true, y_pred = evaluate_predictor(predictor, test_loader, device)
    test_metrics = evaluate_regression(y_true, y_pred)

    output_tag = build_output_tag(args.stage25_profile)
    pred_out = os.path.join(pretrained_dir, f"spectral_predictor_v2_{output_tag}.pth")
    gen_out = os.path.join(pretrained_dir, f"spectral_generator_{output_tag}.pth")
    params_out = os.path.join(pretrained_dir, f"predictor_v2_{output_tag}_norm_params.pth")
    torch.save(predictor.state_dict(), pred_out)
    torch.save(generator.state_dict(), gen_out)
    torch.save(norm_params, params_out)

    ckpt_out = os.path.join(checkpoints_dir, f"joint_{output_tag}_best.pth")
    torch.save(
        {
            "best_epoch": int(best_state["epoch"] if best_state else args.joint_epochs),
            "best_val_mse": float(best_val),
            "predictor_state_dict": predictor.state_dict(),
            "generator_state_dict": generator.state_dict(),
            "weights": {
                "w_conc": float(args.w_conc),
                "w_cycle": float(args.w_cycle),
                "w_mono": float(args.w_mono),
                "w_recon": float(args.w_recon),
            },
            "stage25_profile": args.stage25_profile,
            "update_strategy": str(args.update_strategy),
            "predictor_train_mode": str(predictor_train_mode),
            "predictor_trainable_params": int(predictor_trainable_count),
            "p_steps": int(args.p_steps),
            "g_steps": int(args.g_steps),
        },
        ckpt_out,
    )

    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"split_test_metrics_predictor_v2_{output_tag}.csv")
    details_path = os.path.join(out_dir, f"split_test_predictions_predictor_v2_{output_tag}.csv")
    pd.DataFrame(
        [
            {
                "train_file": train_path,
                "val_file": val_path,
                "test_file": test_path,
                "features_file": features_path,
                "samples_train_ag": int(len(y_train)),
                "samples_val_ag": int(len(y_val)),
                "samples_test_ag": int(len(y_test)),
                "seed": int(args.seed),
                "pretrain_gen_epochs": int(args.pretrain_gen_epochs),
                "joint_epochs": int(args.joint_epochs),
                "stage25_profile": args.stage25_profile or "",
                "update_strategy": str(args.update_strategy),
                "predictor_lr": float(args.predictor_lr),
                "generator_lr": float(args.generator_lr),
                "freeze_predictor": bool(args.freeze_predictor),
                "predictor_train_mode": str(predictor_train_mode),
                "predictor_trainable_params": int(predictor_trainable_count),
                "p_steps": int(args.p_steps),
                "g_steps": int(args.g_steps),
                "w_conc": float(args.w_conc),
                "w_cycle": float(args.w_cycle),
                "w_mono": float(args.w_mono),
                "w_recon": float(args.w_recon),
                **test_metrics,
            }
        ]
    ).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {
            "true_log10_conc": y_true,
            "pred_log10_conc": y_pred,
            "true_ng_ml": np.clip(np.power(10.0, y_true) - 1e-3, a_min=0.0, a_max=None),
            "pred_ng_ml": np.clip(np.power(10.0, y_pred) - 1e-3, a_min=0.0, a_max=None),
        }
    ).to_csv(details_path, index=False)

    print("Test metrics (C+Cycle):")
    print(
        f"  MAE={test_metrics['mae_ng_ml']:.6f} ng/ml | RMSE={test_metrics['rmse_ng_ml']:.6f} ng/ml | "
        f"MAPE={test_metrics['mape_pct']:.3f}% | R2={test_metrics['r2']:.6f}"
    )
    print(f"Saved: {pred_out}")
    print(f"Saved: {gen_out}")
    print(f"Saved: {params_out}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()
