"""Train the physical-time latent dynamics model with its flow-matching head."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import load_monthly_archive
from .dynamics import (
    DynamicsLossConfig,
    DynamicsModelConfig,
    LatentDynamicsFlow,
    TrajectoryWindowDataset,
)
from .train import _sha256, build_purged_temporal_split, parameter_groups


def _field_grid(schema: dict) -> tuple[int, int, int]:
    """Derive the (channels, lat, lon) grid the conv autoencoder needs."""
    shapes = {tuple(variable["shape"]) for variable in schema["variables"]}
    if len(shapes) != 1:
        raise ValueError("Conv autoencoder needs every variable on one grid")
    height, width = shapes.pop()
    return len(schema["variables"]), int(height), int(width)


def _epoch(
    model: LatentDynamicsFlow,
    loader: DataLoader,
    loss_config: DynamicsLossConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            history = batch["history"].to(device)
            origin = batch["origin"].to(device)
            targets = batch["targets"].to(device)
            losses = model.loss(history, origin, targets, loss_config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("Non-finite training loss")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * history.shape[0]
            examples += history.shape[0]
    if examples == 0:
        raise ValueError("Split produced no batches")
    return {name: value / examples for name, value in totals.items()}


@torch.no_grad()
def _validation_trajectory_rmse(
    model: LatentDynamicsFlow, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Deterministic trajectory RMSE, overall and at the final lead time."""
    model.eval()
    squared, elements = 0.0, 0
    final_squared, final_elements = 0.0, 0
    for batch in loader:
        history = batch["history"].to(device)
        origin = batch["origin"].to(device)
        targets = batch["targets"].to(device)
        prediction = model.deterministic_forecast(history, origin)
        error = prediction - targets
        squared += float(error.square().sum())
        elements += targets.numel()
        final_squared += float(error[:, -1].square().sum())
        final_elements += targets[:, -1].numel()
    return {
        "trajectory_rmse": float(np.sqrt(squared / elements)),
        "horizon_rmse": float(np.sqrt(final_squared / final_elements)),
    }


def train_dynamics_model(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    history_steps: int = 6,
    history_stride: int = 120,
    horizon_steps: int = 120,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    autoencoder_hidden_dim: int | None = 512,
    autoencoder_blocks: int = 3,
    autoencoder_dropout: float = 0.0,
    autoencoder_kind: str = "mlp",
    autoencoder_weight_decay: float = 0.0,
    dynamics_solver: str = "rk4",
    dynamics_adjoint: bool = True,
    flow_solver: str = "midpoint",
    latent_normalization: bool = True,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    window_stride: int = 24,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.1,
    purge_windows: int = 1,
    recency_halflife: float | None = None,
    normalization_states: int | None = None,
    ensemble_size: int = 0,
    ensemble_weight: float = 0.0,
    ensemble_steps: int = 8,
    flow_steps_per_batch: int = 8,
    seed: int = 7,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    states, times, schema = load_monthly_archive(archive_path)
    step_hours = int(schema.get("forecast_step_hours", 0))
    if step_hours <= 0:
        raise ValueError(
            "Dynamics training needs a fixed-step archive; build one with "
            "prepare-climate-fixed-step-data"
        )
    actual = np.diff(times)
    if not np.all(actual == np.timedelta64(step_hours, "h")):
        raise ValueError(f"Archive timestamps violate the {step_hours}h step contract")
    if min(epochs, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if purge_windows < 0:
        raise ValueError("purge_windows cannot be negative")
    if window_stride < 1:
        raise ValueError("window_stride must be positive")

    model_config = DynamicsModelConfig(
        state_dim=states.shape[1],
        history_steps=history_steps,
        history_stride=history_stride,
        horizon_steps=horizon_steps,
        step_hours=step_hours,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        autoencoder_hidden_dim=autoencoder_hidden_dim,
        autoencoder_blocks=autoencoder_blocks,
        autoencoder_dropout=autoencoder_dropout,
        autoencoder_kind=autoencoder_kind,
        autoencoder_grid=_field_grid(schema) if autoencoder_kind == "conv" else None,
        latent_normalization=latent_normalization,
        dynamics_solver=dynamics_solver,
        dynamics_adjoint=dynamics_adjoint,
        flow_solver=flow_solver,
    )
    span = model_config.history_span_steps
    sample_count = len(states) - span - horizon_steps + 1
    if sample_count < 3:
        raise ValueError("Archive is too short for this history/horizon window")
    # Windows start one archive step apart before subsampling. Prevent future
    # target intervals from overlapping across splits; past context may recur.
    effective_purge = max(purge_windows, horizon_steps - 1)
    split = build_purged_temporal_split(
        sample_count,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        purge_windows=effective_purge,
    )

    # Normalization statistics stay inside train, and only ever move their start
    # forward, so nothing from validation or test leaks in.
    normalization_end = split.train[-1] + span + horizon_steps
    normalization_start = 0
    if normalization_states is not None:
        if normalization_states < 2:
            raise ValueError("normalization_states must be at least 2")
        normalization_start = max(0, normalization_end - normalization_states)
    reference = states[normalization_start:normalization_end]
    state_mean = reference.mean(axis=0).astype(np.float32)
    state_scale = reference.std(axis=0).astype(np.float32)
    state_scale = np.where(state_scale > 1e-6, state_scale, 1.0).astype(np.float32)
    normalized = ((states - state_mean) / state_scale).astype(np.float32)

    # A 6-hourly archive yields far more windows than an epoch needs, and
    # neighbouring ones are near-duplicates; striding keeps epochs affordable.
    train_indices = split.train[::window_stride]
    validation_indices = split.validation[::window_stride]
    train_dataset = TrajectoryWindowDataset(normalized, model_config, train_indices)
    validation_dataset = TrajectoryWindowDataset(
        normalized, model_config, validation_indices
    )
    generator = torch.Generator().manual_seed(seed)
    if recency_halflife is None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
    else:
        if recency_halflife <= 0:
            raise ValueError("recency_halflife must be positive")
        age = np.arange(len(train_indices) - 1, -1, -1, dtype=np.float64)
        weights = torch.from_numpy(0.5 ** (age / recency_halflife))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                weights, num_samples=len(train_dataset), replacement=True,
                generator=generator,
            ),
        )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    loss_config = DynamicsLossConfig(
        ensemble_weight=ensemble_weight,
        ensemble_size=ensemble_size,
        ensemble_steps=ensemble_steps,
        flow_steps_per_batch=flow_steps_per_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentDynamicsFlow(model_config).to(device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, 1e-4, autoencoder_weight_decay), lr=learning_rate
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_format = "climate_diffusion.latent_dynamics_flow.v1"
    best = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_config, device, optimizer)
        validation_metrics = _epoch(
            model, validation_loader, loss_config, device, optimizer=None
        )
        validation_metrics.update(
            _validation_trajectory_rmse(model, validation_loader, device)
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        criterion = validation_metrics["trajectory_rmse"]
        print(
            f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} "
            f"traj_rmse={criterion:.6f} horizon_rmse={validation_metrics['horizon_rmse']:.6f}",
            flush=True,
        )
        if criterion < best:
            best = criterion
            best_epoch = epoch
            torch.save(
                {
                    "format": checkpoint_format,
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "loss_config": asdict(loss_config),
                    "state_mean": torch.from_numpy(state_mean),
                    "state_scale": torch.from_numpy(state_scale),
                    "schema": schema,
                    "training": {
                        "step_hours": step_hours,
                        "forecast_step_hours": step_hours,
                        "requested_purge_windows": purge_windows,
                        "split_contract": "disjoint_future_targets.v1",
                        "archive_state_count": len(states),
                        "horizon_hours": model_config.horizon_hours,
                        "history_span_steps": span,
                        "window_stride": window_stride,
                        "seed": seed,
                        "recency_halflife": recency_halflife,
                        "normalization_states": normalization_states,
                        "normalization_span": [normalization_start, normalization_end],
                        "archive": str(archive_path),
                        "first_time": str(times[0]),
                        "last_time": str(times[-1]),
                        "best_epoch": best_epoch,
                        "best_validation_trajectory_rmse": best,
                        "best_validation_horizon_rmse": validation_metrics["horizon_rmse"],
                        "split": asdict(split),
                    },
                },
                output,
            )

    output.with_suffix(".metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checkpoint_sha256 = _sha256(output)
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_kind": "latent_dynamics_flow_matching",
                "checkpoint_format": checkpoint_format,
                "checkpoint_sha256": checkpoint_sha256,
                "forecast_step_hours": step_hours,
                "horizon_hours": model_config.horizon_hours,
                "horizon_steps": horizon_steps,
                "history_steps": history_steps,
                "history_stride": history_stride,
                "state_dim": int(states.shape[1]),
                "model_config": asdict(model_config),
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "autoencoder_parameter_count": int(
                    sum(p.numel() for p in model.autoencoder.parameters())
                ),
                "dynamics_parameter_count": int(
                    sum(p.numel() for p in model.dynamics.parameters())
                ),
                "schema_format": schema.get("format"),
                "split": asdict(split),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "format": "climate_diffusion.artifact.v2",
                "checkpoint": output.name,
                "checkpoint_sha256": checkpoint_sha256,
                "metrics": output.with_suffix(".metrics.json").name,
                "metadata": output.with_suffix(".metadata.json").name,
                "archive": str(Path(archive_path)),
                "schema_format": schema.get("format"),
                "forecast_step_hours": step_hours,
                "variables": [item["name"] for item in schema["variables"]],
                "split": asdict(split),
                "seed": seed,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a physical-time latent ODE with a flow-matching ensemble head"
    )
    parser.add_argument("--archive", required=True, help="Fixed-step archive")
    parser.add_argument("--history-steps", type=int, default=6)
    parser.add_argument(
        "--history-stride",
        type=int,
        default=120,
        help="Archive steps between history samples; 120 x 6h spans a month per step",
    )
    parser.add_argument("--horizon-steps", type=int, default=120)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--autoencoder-hidden-dim", type=int, default=512)
    parser.add_argument("--autoencoder-blocks", type=int, default=3)
    parser.add_argument("--autoencoder-dropout", type=float, default=0.0)
    parser.add_argument("--autoencoder-kind", choices=("mlp", "conv"), default="mlp")
    parser.add_argument(
        "--autoencoder-weight-decay",
        type=float,
        default=0.0,
        help="Decay applied to autoencoder parameters only; 0 keeps the latent "
             "from being shrunk into the scale floor",
    )
    parser.add_argument("--dynamics-solver", default="rk4")
    parser.add_argument("--no-dynamics-adjoint", dest="dynamics_adjoint",
                        action="store_false")
    parser.add_argument("--flow-solver", default="midpoint")
    parser.add_argument("--no-latent-normalization", dest="latent_normalization",
                        action="store_false")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--window-stride", type=int, default=24)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--purge-windows", type=int, default=1)
    parser.add_argument("--recency-halflife", type=float)
    parser.add_argument("--normalization-states", type=int)
    parser.add_argument("--ensemble-size", type=int, default=0)
    parser.add_argument("--ensemble-weight", type=float, default=0.0)
    parser.add_argument("--ensemble-steps", type=int, default=8)
    parser.add_argument("--flow-steps-per-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", default="download/flow-matching/dynamics/dynamics-v1.pt"
    )
    args = parser.parse_args(argv)
    path = train_dynamics_model(
        args.archive,
        args.output,
        history_steps=args.history_steps,
        history_stride=args.history_stride,
        horizon_steps=args.horizon_steps,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
        autoencoder_blocks=args.autoencoder_blocks,
        autoencoder_dropout=args.autoencoder_dropout,
        autoencoder_kind=args.autoencoder_kind,
        autoencoder_weight_decay=args.autoencoder_weight_decay,
        dynamics_solver=args.dynamics_solver,
        dynamics_adjoint=args.dynamics_adjoint,
        flow_solver=args.flow_solver,
        latent_normalization=args.latent_normalization,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        window_stride=args.window_stride,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        purge_windows=args.purge_windows,
        recency_halflife=args.recency_halflife,
        normalization_states=args.normalization_states,
        ensemble_size=args.ensemble_size,
        ensemble_weight=args.ensemble_weight,
        ensemble_steps=args.ensemble_steps,
        flow_steps_per_batch=args.flow_steps_per_batch,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
