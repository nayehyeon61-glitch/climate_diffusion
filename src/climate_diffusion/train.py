"""Train a monthly conditional flow matcher in an autoencoded latent space."""

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
from .data import MonthlyWindowDataset, load_auxiliary_states, load_monthly_archive
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
    *,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with context:
        for batch_index, batch in enumerate(loader):
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            history_auxiliary = batch.get("history_auxiliary")
            if history_auxiliary is not None:
                history_auxiliary = history_auxiliary.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                losses = model.loss(
                    history,
                    target,
                    loss_config,
                    history_auxiliary=history_auxiliary,
                )
            if training:
                (losses["loss"] / gradient_accumulation_steps).backward()
                if (
                    (batch_index + 1) % gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
    if batches == 0:
        raise ValueError("Monthly split produced no training/validation batches")
    return {name: value / batches for name, value in totals.items()}


def _spatial_channel_statistics(
    states: np.ndarray, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only per-channel statistics without loading a global archive."""
    channels = states.shape[1]
    total = np.zeros(channels, dtype=np.float64)
    total_square = np.zeros(channels, dtype=np.float64)
    count = np.zeros(channels, dtype=np.int64)
    for index in range(end):
        values = np.asarray(states[index], dtype=np.float32)
        valid = np.isfinite(values)
        total += np.where(valid, values, 0.0).sum(axis=(1, 2), dtype=np.float64)
        total_square += np.where(valid, values * values, 0.0).sum(
            axis=(1, 2), dtype=np.float64
        )
        count += valid.sum(axis=(1, 2))
    count = np.maximum(count, 1)
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32)[:, None, None], scale.astype(np.float32)[:, None, None]


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
    model_backend: str | None = None,
    spatial_latent_channels: int = 16,
    spatial_base_channels: int = 32,
    spatial_downsample_levels: int = 3,
    operator_modes_lat: int = 12,
    operator_modes_lon: int = 24,
    patch_height: int | None = None,
    patch_width: int | None = None,
    tile_overlap: int = 32,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 0,
    gradient_checkpointing: bool = False,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    states, times, schema = load_monthly_archive(archive_path)
    sample_count = len(states) - history_months - lead_months + 1
    split = build_purged_temporal_split(
        sample_count,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        purge_windows=purge_windows,
    )

    last_train_start = split.train[-1]
    normalization_end = last_train_start + history_months + lead_months
    layout = schema.get("layout", "vector")
    backend = model_backend or ("spatial_conv" if layout == "spatial" else "vector_mlp")
    if (layout == "spatial") != (backend != "vector_mlp"):
        raise ValueError("Archive layout and model backend are incompatible")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if layout == "spatial":
        state_mean, state_scale = _spatial_channel_statistics(states, normalization_end)
        normalized = states
        auxiliary = load_auxiliary_states(archive_path, schema)
        auxiliary_dim = int(schema.get("auxiliary_dim", 0))
        if auxiliary_dim:
            auxiliary_mean = np.asarray(auxiliary[:normalization_end]).mean(axis=0).astype(np.float32)
            auxiliary_scale = np.asarray(auxiliary[:normalization_end]).std(axis=0).astype(np.float32)
            auxiliary_scale = np.where(auxiliary_scale > 1e-6, auxiliary_scale, 1.0)
        else:
            auxiliary_mean = np.empty(0, dtype=np.float32)
            auxiliary_scale = np.empty(0, dtype=np.float32)
    else:
        state_mean = states[:normalization_end].mean(axis=0).astype(np.float32)
        state_scale = states[:normalization_end].std(axis=0).astype(np.float32)
        state_scale = np.where(state_scale > 1e-6, state_scale, 1.0).astype(np.float32)
        normalized = ((states - state_mean) / state_scale).astype(np.float32)
        auxiliary = None
        auxiliary_dim = 0
        auxiliary_mean = np.empty(0, dtype=np.float32)
        auxiliary_scale = np.empty(0, dtype=np.float32)

    patch_size = None
    if layout == "spatial":
        grid_height, grid_width = schema["grid_shape"]
        patch_size = (
            patch_height or min(256, grid_height),
            patch_width or min(256, grid_width),
        )
        if tile_overlap < 0 or tile_overlap >= min(patch_size):
            raise ValueError("tile_overlap must be non-negative and smaller than each patch axis")

    train_dataset = MonthlyWindowDataset(
        normalized,
        history_months,
        lead_months,
        indices=split.train,
        mean=state_mean if layout == "spatial" else None,
        scale=state_scale if layout == "spatial" else None,
        auxiliary_states=auxiliary,
        auxiliary_mean=auxiliary_mean,
        auxiliary_scale=auxiliary_scale,
        patch_size=patch_size,
        random_crop=layout == "spatial",
    )
    validation_dataset = MonthlyWindowDataset(
        normalized,
        history_months,
        lead_months,
        indices=split.validation,
        mean=state_mean if layout == "spatial" else None,
        scale=state_scale if layout == "spatial" else None,
        auxiliary_states=auxiliary,
        auxiliary_mean=auxiliary_mean,
        auxiliary_scale=auxiliary_scale,
        patch_size=patch_size,
        random_crop=False,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_config = FlowModelConfig(
        state_dim=int(schema.get("state_dim", states.shape[1])),
        history_months=history_months,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        backend=backend,
        spatial_channels=int(schema.get("spatial_channels", 0)),
        grid_height=int(schema.get("grid_shape", [0, 0])[0]),
        grid_width=int(schema.get("grid_shape", [0, 0])[1]),
        spatial_latent_channels=spatial_latent_channels,
        spatial_base_channels=spatial_base_channels,
        spatial_downsample_levels=spatial_downsample_levels,
        operator_modes_lat=operator_modes_lat,
        operator_modes_lon=operator_modes_lon,
        auxiliary_dim=auxiliary_dim,
        gradient_checkpointing=gradient_checkpointing,
    )
    loss_config = FlowLossConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonthlyLatentFlow(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(
            model,
            train_loader,
            loss_config,
            device,
            optimizer,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        validation_metrics = _epoch(
            model,
            validation_loader,
            loss_config,
            device,
            optimizer=None,
            mixed_precision=mixed_precision,
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        print(
            f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f}"
        )
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(
                {
                    "format": "climate_diffusion.monthly_latent_flow.v3",
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "loss_config": asdict(loss_config),
                    "state_mean": torch.from_numpy(state_mean),
                    "state_scale": torch.from_numpy(state_scale),
                    "auxiliary_mean": torch.from_numpy(auxiliary_mean),
                    "auxiliary_scale": torch.from_numpy(auxiliary_scale),
                    "schema": schema,
                    "training": {
                        "lead_months": lead_months,
                        "seed": seed,
                        "archive": str(archive_path),
                        "first_time": str(times[0]),
                        "last_time": str(times[-1]),
                        "best_validation_loss": best_validation,
                        "split": asdict(split),
                        "patch_size": patch_size,
                        "tile_overlap": tile_overlap,
                        "mixed_precision": mixed_precision,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "gradient_checkpointing": gradient_checkpointing,
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
                "checkpoint_kind": "monthly_latent_flow_matching",
                "checkpoint_format": "climate_diffusion.monthly_latent_flow.v3",
                "checkpoint_sha256": checkpoint_sha256,
                "forecast_step": "720 hours over monthly aggregate targets",
                "history_months": history_months,
                "state_dim": int(schema.get("state_dim", states.shape[1])),
                "state_layout": layout,
                "model_backend": backend,
                "grid_shape": schema.get("grid_shape"),
                "spatial_channels": schema.get("spatial_channels"),
                "weather_next_compatible_runner": True,
                "inference_ready": True,
                "frozen_inference_required": True,
                "split": asdict(split),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "format": "climate_diffusion.artifact.v1",
                "checkpoint": output.name,
                "checkpoint_sha256": checkpoint_sha256,
                "metrics": output.with_suffix(".metrics.json").name,
                "metadata": output.with_suffix(".metadata.json").name,
                "archive": str(Path(archive_path)),
                "schema_format": schema.get("format"),
                "variables": [item["name"] for item in schema["variables"]],
                "split": asdict(split),
                "seed": seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train monthly latent flow matching")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--history-months", type=int, default=6)
    parser.add_argument("--lead-months", type=int, default=1)
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
        "--model-backend",
        choices=("vector_mlp", "spatial_conv", "spatial_operator"),
        help="Defaults to vector_mlp for vector archives and spatial_conv otherwise",
    )
    parser.add_argument("--spatial-latent-channels", type=int, default=16)
    parser.add_argument("--spatial-base-channels", type=int, default=32)
    parser.add_argument("--spatial-downsample-levels", type=int, default=3)
    parser.add_argument("--operator-modes-lat", type=int, default=12)
    parser.add_argument("--operator-modes-lon", type=int, default=24)
    parser.add_argument("--patch-height", type=int)
    parser.add_argument("--patch-width", type=int)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--output",
        default="download/flow-matching/monthly-v1/climate-flow-monthly-v1.pt",
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
        model_backend=args.model_backend,
        spatial_latent_channels=args.spatial_latent_channels,
        spatial_base_channels=args.spatial_base_channels,
        spatial_downsample_levels=args.spatial_downsample_levels,
        operator_modes_lat=args.operator_modes_lat,
        operator_modes_lon=args.operator_modes_lon,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        tile_overlap=args.tile_overlap,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
