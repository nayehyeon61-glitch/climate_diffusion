"""Train a conditional latent flow matcher over ordered state snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import FlowLossConfig, FlowModelConfig
from .data import MonthlyWindowDataset, load_monthly_archive
from .model import MonthlyLatentFlow


@dataclass(frozen=True)
class TemporalSplit:
    train: list[int]
    validation: list[int]
    test: list[int]
    purge_windows: int


def build_purged_temporal_split(
    sample_count: int,
    *,
    validation_fraction: float,
    test_fraction: float,
    purge_windows: int,
) -> TemporalSplit:
    """Create ordered train/validation/test windows with embargoed boundaries."""
    if sample_count < 3:
        raise ValueError("At least three forecast windows are required")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation_fraction + test_fraction must be below 0.8")
    if purge_windows < 0:
        raise ValueError("purge_windows cannot be negative")

    validation_count = max(1, int(sample_count * validation_fraction))
    test_count = max(1, int(sample_count * test_fraction))
    train_count = sample_count - validation_count - test_count - 2 * purge_windows
    if train_count < 1:
        raise ValueError(
            "Not enough windows for purged train/validation/test splits; "
            "reduce fractions or purge_windows"
        )
    validation_start = train_count + purge_windows
    test_start = validation_start + validation_count + purge_windows
    return TemporalSplit(
        train=list(range(train_count)),
        validation=list(range(validation_start, validation_start + validation_count)),
        test=list(range(test_start, sample_count)),
        purge_windows=purge_windows,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
    autoencoder_weight_decay: float | None,
) -> list[dict]:
    """Split off the autoencoder so it can opt out of weight decay.

    With a normalized latent the decoder undoes any rescaling, so decay on the
    encoder has no reconstruction cost to trade against: it just shrinks the
    latent until the scale estimate hits its floor. Excluding those parameters
    removes that degenerate direction.
    """
    if autoencoder_weight_decay is None or not hasattr(model, "autoencoder"):
        return [{"params": list(model.parameters()), "weight_decay": weight_decay}]
    autoencoder = set(id(p) for p in model.autoencoder.parameters())
    rest = [p for p in model.parameters() if id(p) not in autoencoder]
    return [
        {"params": list(model.autoencoder.parameters()),
         "weight_decay": autoencoder_weight_decay},
        {"params": rest, "weight_decay": weight_decay},
    ]


def _epoch(
    model: MonthlyLatentFlow,
    loader: DataLoader,
    loss_config: FlowLossConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            losses = model.loss(history, target, loss_config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
    if batches == 0:
        raise ValueError("Flow split produced no training/validation batches")
    return {name: value / batches for name, value in totals.items()}


def _validation_forecast_rmse(
    model: MonthlyLatentFlow,
    loader: DataLoader,
    device: torch.device,
    *,
    integration_steps: int,
    seed: int,
) -> float:
    """RMSE of an actual generated forecast on the validation windows.

    Total loss is dominated by the reconstruction term, so selecting on it
    tracks compression rather than forecast skill. This samples the model the
    way inference does, with a fixed seed so epochs stay comparable.
    """
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    squared_error = 0.0
    elements = 0
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        prediction = model.sample(
            history, integration_steps=integration_steps, generator=generator
        )
        squared_error += float(torch.square(prediction - target).sum())
        elements += target.numel()
    if elements == 0:
        raise ValueError("Validation split produced no windows")
    return math.sqrt(squared_error / elements)


def train_flow_model(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    history_months: int = 6,
    lead_months: int = 1,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    autoencoder_hidden_dim: int | None = None,
    autoencoder_blocks: int = 0,
    autoencoder_dropout: float = 0.0,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    purge_windows: int = 1,
    recency_halflife: float | None = None,
    normalization_states: int | None = None,
    latent_normalization: bool = False,
    flow_solver: str = "midpoint",
    flow_adjoint: bool = False,
    ensemble_size: int = 0,
    ensemble_weight: float = 0.0,
    ensemble_steps: int = 4,
    select_by: str = "forecast_rmse",
    forecast_eval_steps: int = 16,
    seed: int = 7,
) -> Path:
    if select_by not in {"loss", "forecast_rmse"}:
        raise ValueError("select_by must be 'loss' or 'forecast_rmse'")
    torch.manual_seed(seed)
    np.random.seed(seed)
    states, times, schema = load_monthly_archive(archive_path)
    forecast_step_hours = int(schema.get("forecast_step_hours", 30 * 24))
    if forecast_step_hours <= 0:
        raise ValueError("Archive forecast_step_hours must be positive")
    if len(times) > 1 and "forecast_step_hours" in schema:
        actual = np.diff(times).astype("timedelta64[h]").astype(np.int64)
        if not np.all(actual == forecast_step_hours):
            raise ValueError(
                f"Archive timestamps violate {forecast_step_hours}h forecast-step contract"
            )

    sample_count = len(states) - history_months - lead_months + 1
    split = build_purged_temporal_split(
        sample_count,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        purge_windows=purge_windows,
    )

    last_train_start = split.train[-1]
    normalization_end = last_train_start + history_months + lead_months
    # A long record is non-stationary, so the oldest train states can bias the
    # statistics away from the forecast period. Restricting the window keeps the
    # estimate causal because it only ever moves the start forward inside train.
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

    train_dataset = MonthlyWindowDataset(
        normalized, history_months, lead_months, indices=split.train
    )
    validation_dataset = MonthlyWindowDataset(
        normalized, history_months, lead_months, indices=split.validation
    )
    generator = torch.Generator().manual_seed(seed)
    if recency_halflife is None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
    else:
        if recency_halflife <= 0:
            raise ValueError("recency_halflife must be positive")
        age = np.arange(len(split.train) - 1, -1, -1, dtype=np.float64)
        weights = torch.from_numpy(0.5 ** (age / recency_halflife))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=generator,
            ),
        )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    model_config = FlowModelConfig(
        state_dim=states.shape[1],
        history_months=history_months,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        autoencoder_hidden_dim=autoencoder_hidden_dim,
        autoencoder_blocks=autoencoder_blocks,
        autoencoder_dropout=autoencoder_dropout,
        latent_normalization=latent_normalization,
        flow_solver=flow_solver,
        flow_adjoint=flow_adjoint,
    )
    loss_config = FlowLossConfig(
        ensemble_weight=ensemble_weight,
        ensemble_size=ensemble_size,
        ensemble_steps=ensemble_steps,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonthlyLatentFlow(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_epoch = 0
    history = []
    checkpoint_format = "climate_diffusion.latent_flow.v3"
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_config, device, optimizer)
        validation_metrics = _epoch(model, validation_loader, loss_config, device, optimizer=None)
        forecast_rmse = _validation_forecast_rmse(
            model,
            validation_loader,
            device,
            integration_steps=forecast_eval_steps,
            seed=seed,
        )
        validation_metrics["forecast_rmse"] = forecast_rmse
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        criterion = (
            forecast_rmse if select_by == "forecast_rmse" else validation_metrics["loss"]
        )
        print(
            f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f} "
            f"forecast_rmse={forecast_rmse:.6f}"
        )
        if criterion < best_validation:
            best_validation = criterion
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
                        "lead_steps": lead_months,
                        "lead_months": lead_months,
                        "forecast_step_hours": forecast_step_hours,
                        "seed": seed,
                        "recency_halflife": recency_halflife,
                        "normalization_states": normalization_states,
                        "latent_normalization": latent_normalization,
                        "flow_solver": flow_solver,
                        "flow_adjoint": flow_adjoint,
                        "select_by": select_by,
                        "forecast_eval_steps": forecast_eval_steps,
                        "best_epoch": best_epoch,
                        "best_validation_forecast_rmse": forecast_rmse,
                        "normalization_span": [normalization_start, normalization_end],
                        "archive": str(archive_path),
                        "first_time": str(times[0]),
                        "last_time": str(times[-1]),
                        "best_validation_criterion": best_validation,
                        "best_validation_loss": validation_metrics["loss"],
                        "split": asdict(split),
                    },
                },
                output,
            )

    output.with_suffix(".metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checkpoint_sha256 = _sha256(output)
    checkpoint_kind = (
        "fixed_step_latent_flow_matching"
        if "forecast_step_hours" in schema
        else "monthly_latent_flow_matching"
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_format": checkpoint_format,
                "checkpoint_sha256": checkpoint_sha256,
                "forecast_step_hours": forecast_step_hours,
                "history_steps": history_months,
                "state_dim": states.shape[1],
                "model_config": asdict(model_config),
                "parameter_count": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "autoencoder_parameter_count": int(
                    sum(parameter.numel() for parameter in model.autoencoder.parameters())
                ),
                "weather_next_compatible_runner": True,
                "inference_ready": True,
                "frozen_inference_required": True,
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
                "forecast_step_hours": forecast_step_hours,
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
    parser = argparse.ArgumentParser(description="Train latent Flow Matching over ordered state snapshots")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--history-months", type=int, default=6, help="Number of prior archive states used as history")
    parser.add_argument("--lead-months", type=int, default=1, help="Number of archive steps to the training target")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--autoencoder-hidden-dim",
        type=int,
        help="Autoencoder width; defaults to --hidden-dim",
    )
    parser.add_argument(
        "--autoencoder-blocks",
        type=int,
        default=0,
        help="Residual blocks per autoencoder half; 0 keeps the original MLP",
    )
    parser.add_argument("--autoencoder-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--purge-windows", type=int, default=1)
    parser.add_argument(
        "--recency-halflife",
        type=float,
        help="Exponential sampling half-life in windows; recent windows are drawn more often",
    )
    parser.add_argument(
        "--normalization-states",
        type=int,
        help="Use only the last N train states for normalization statistics",
    )
    parser.add_argument(
        "--latent-normalization",
        action="store_true",
        help="Rescale the latent to unit scale so it matches the N(0, I) flow prior",
    )
    parser.add_argument(
        "--flow-solver",
        default="midpoint",
        help="'midpoint' keeps the built-in loop; any other value is a torchdiffeq "
             "method over flow time tau, e.g. rk4 or dopri5",
    )
    parser.add_argument(
        "--flow-adjoint",
        action="store_true",
        help="Solve the adjoint backward for O(1) memory in the rollout length",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=0,
        help="Members generated per window for the training CRPS term; 0 disables it",
    )
    parser.add_argument("--ensemble-weight", type=float, default=0.0)
    parser.add_argument(
        "--ensemble-steps",
        type=int,
        default=4,
        help="ODE steps in the differentiable rollout used by the CRPS term",
    )
    parser.add_argument(
        "--select-by",
        choices=("forecast_rmse", "loss"),
        default="forecast_rmse",
        help="Checkpoint selection criterion on the validation split",
    )
    parser.add_argument("--forecast-eval-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="download/flow-matching/flow-v3/climate-flow-v3.pt",
    )
    args = parser.parse_args(argv)
    path = train_flow_model(
        args.archive,
        args.output,
        history_months=args.history_months,
        lead_months=args.lead_months,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
        autoencoder_blocks=args.autoencoder_blocks,
        autoencoder_dropout=args.autoencoder_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        purge_windows=args.purge_windows,
        recency_halflife=args.recency_halflife,
        normalization_states=args.normalization_states,
        latent_normalization=args.latent_normalization,
        flow_solver=args.flow_solver,
        flow_adjoint=args.flow_adjoint,
        ensemble_size=args.ensemble_size,
        ensemble_weight=args.ensemble_weight,
        ensemble_steps=args.ensemble_steps,
        select_by=args.select_by,
        forecast_eval_steps=args.forecast_eval_steps,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
