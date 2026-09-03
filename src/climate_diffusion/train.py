"""Train a conditional latent flow matcher over ordered state snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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


def train_flow_model(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    history_months: int = 6,
    lead_months: int = 1,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    purge_windows: int = 1,
    seed: int = 7,
) -> Path:
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
    state_mean = states[:normalization_end].mean(axis=0).astype(np.float32)
    state_scale = states[:normalization_end].std(axis=0).astype(np.float32)
    state_scale = np.where(state_scale > 1e-6, state_scale, 1.0).astype(np.float32)
    normalized = ((states - state_mean) / state_scale).astype(np.float32)

    train_dataset = MonthlyWindowDataset(
        normalized, history_months, lead_months, indices=split.train
    )
    validation_dataset = MonthlyWindowDataset(
        normalized, history_months, lead_months, indices=split.validation
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    model_config = FlowModelConfig(
        state_dim=states.shape[1],
        history_months=history_months,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    loss_config = FlowLossConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonthlyLatentFlow(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    history = []
    checkpoint_format = "climate_diffusion.latent_flow.v3"
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_config, device, optimizer)
        validation_metrics = _epoch(model, validation_loader, loss_config, device, optimizer=None)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f}"
        )
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
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
                        "archive": str(archive_path),
                        "first_time": str(times[0]),
                        "last_time": str(times[-1]),
                        "best_validation_loss": best_validation,
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--purge-windows", type=int, default=1)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        purge_windows=args.purge_windows,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
