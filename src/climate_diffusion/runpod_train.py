"""Resumable RunPod training profile for 0.25-degree monthly spatial Flow Matching."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FlowLossConfig, FlowModelConfig
from .data import (
    MonthlyWindowDataset,
    load_auxiliary_states,
    load_monthly_archive,
    load_observation_mask,
)
from .model import MonthlyLatentFlow
from .train import (
    TemporalSplit,
    _epoch,
    _train_only_statistics,
    build_raw_month_temporal_split,
    forecast_skill,
    patch_latitude_weights,
)
from .validation import archive_contract_fingerprint, require_finite_numpy


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(temporary)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _rng_state(generator: torch.Generator) -> dict:
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["loader_generator"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _checkpoint_payload(
    *,
    model: MonthlyLatentFlow,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model_config: FlowModelConfig,
    loss_config: FlowLossConfig,
    state_mean: np.ndarray,
    state_scale: np.ndarray,
    auxiliary_mean: np.ndarray,
    auxiliary_scale: np.ndarray,
    schema: dict,
    split: TemporalSplit,
    archive: Path,
    contract_fingerprint: str,
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    training_config: dict,
    loader_generator: torch.Generator,
) -> dict:
    return {
        "format": "climate_diffusion.monthly_latent_flow.v3",
        "model": model.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "state_mean": torch.from_numpy(state_mean),
        "state_scale": torch.from_numpy(state_scale),
        "auxiliary_mean": torch.from_numpy(auxiliary_mean),
        "auxiliary_scale": torch.from_numpy(auxiliary_scale),
        "schema": schema,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": _rng_state(loader_generator),
        "training": {
            **training_config,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "archive": str(archive.resolve()),
            "archive_contract_fingerprint": contract_fingerprint,
            "split": asdict(split),
            "checkpoint_role": "resumable_latest",
        },
    }


def _best_payload(latest: dict) -> dict:
    # Keep the best artifact inference-friendly. Optimizer/RNG state belongs only
    # in latest/snapshots and is deliberately omitted from the frozen artifact.
    result = dict(latest)
    result.pop("optimizer", None)
    result.pop("scaler", None)
    result.pop("rng_state", None)
    result["training"] = dict(result["training"])
    result["training"]["checkpoint_role"] = "best_frozen_candidate"
    return result


def train_runpod_spatial_flow(
    archive_path: str | Path,
    checkpoint_dir: str | Path,
    *,
    backend: str = "spatial_conv",
    history_months: int = 6,
    lead_months: int = 1,
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    purge_months: int = 1,
    spatial_base_channels: int = 32,
    spatial_latent_channels: int = 16,
    spatial_downsample_levels: int = 3,
    operator_modes_lat: int = 12,
    operator_modes_lon: int = 24,
    patch_height: int = 256,
    patch_width: int = 256,
    tile_overlap: int = 64,
    gradient_accumulation_steps: int = 8,
    gradient_checkpointing: bool = True,
    mixed_precision: bool = True,
    min_observed_fraction: float = 0.95,
    num_workers: int = 2,
    seed: int = 7,
    resume: bool = True,
    save_every_epochs: int = 5,
    keep_epoch_snapshots: int = 3,
    min_free_disk_gb: float = 25.0,
    skill_every_epochs: int = 1,
    skill_windows: int = 2,
    skill_ensemble_size: int = 2,
    skill_integration_steps: int = 8,
) -> Path:
    if backend not in {"spatial_conv", "spatial_operator"}:
        raise ValueError("RunPod spatial trainer requires spatial_conv or spatial_operator")
    if min(epochs, batch_size, gradient_accumulation_steps, save_every_epochs) < 1:
        raise ValueError("epochs/batch/accumulation/save cadence must be positive")
    archive = Path(archive_path).expanduser().resolve()
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if _free_gib(checkpoint_root) < min_free_disk_gb:
        raise RuntimeError(
            f"Only {_free_gib(checkpoint_root):.1f} GiB free under {checkpoint_root}; "
            f"require at least {min_free_disk_gb:.1f} GiB before training"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    states, times, schema = load_monthly_archive(archive)
    if schema.get("layout") != "spatial":
        raise ValueError("500GB RunPod profile requires a spatial monthly archive")
    split0 = build_raw_month_temporal_split(
        len(states),
        history_months=history_months,
        lead_months=lead_months,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        purge_months=purge_months,
    )
    raw_time_ranges = {
        name: [str(times[start]), str(times[end - 1])]
        for name, (start, end) in split0.raw_month_ranges.items()
    }
    split = TemporalSplit(
        train=split0.train,
        validation=split0.validation,
        test=split0.test,
        purge_windows=split0.purge_windows,
        raw_month_ranges=split0.raw_month_ranges,
        raw_time_ranges=raw_time_ranges,
        window_span_months=split0.window_span_months,
    )
    train_start, train_end = split.raw_month_ranges["train"]
    train_raw_indices = list(range(train_start, train_end))
    observation_mask = load_observation_mask(archive, states, schema)
    state_mean, state_scale = _train_only_statistics(
        states,
        train_raw_indices,
        observation_mask=observation_mask,
        spatial=True,
        name="spatial state",
    )
    auxiliary = load_auxiliary_states(archive, schema)
    auxiliary_dim = int(schema.get("auxiliary_dim", 0))
    if auxiliary_dim:
        auxiliary_mean, auxiliary_scale = _train_only_statistics(
            auxiliary,
            train_raw_indices,
            spatial=False,
            name="spatial auxiliary",
        )
    else:
        auxiliary_mean = np.empty(0, dtype=np.float32)
        auxiliary_scale = np.empty(0, dtype=np.float32)
    for name, value in {
        "state_mean": state_mean,
        "state_scale": state_scale,
        "auxiliary_mean": auxiliary_mean,
        "auxiliary_scale": auxiliary_scale,
    }.items():
        require_finite_numpy(value, name)

    grid_height, grid_width = map(int, schema["grid_shape"])
    patch_size = (min(patch_height, grid_height), min(patch_width, grid_width))
    if tile_overlap < 0 or tile_overlap >= min(patch_size):
        raise ValueError("tile_overlap must be smaller than each patch dimension")

    train_dataset = MonthlyWindowDataset(
        states,
        history_months,
        lead_months,
        indices=split.train,
        mean=state_mean,
        scale=state_scale,
        observation_mask=observation_mask,
        times=times,
        min_observed_fraction=min_observed_fraction,
        auxiliary_states=auxiliary,
        auxiliary_mean=auxiliary_mean,
        auxiliary_scale=auxiliary_scale,
        patch_size=patch_size,
        random_crop=True,
    )
    validation_dataset = MonthlyWindowDataset(
        states,
        history_months,
        lead_months,
        indices=split.validation,
        mean=state_mean,
        scale=state_scale,
        observation_mask=observation_mask,
        times=times,
        min_observed_fraction=min_observed_fraction,
        auxiliary_states=auxiliary,
        auxiliary_mean=auxiliary_mean,
        auxiliary_scale=auxiliary_scale,
        patch_size=patch_size,
        random_crop=False,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    model_config = FlowModelConfig(
        state_dim=int(schema.get("state_dim", np.prod(states.shape[1:]))),
        history_months=history_months,
        backend=backend,
        spatial_channels=int(schema["spatial_channels"]),
        grid_height=grid_height,
        grid_width=grid_width,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision and device.type == "cuda")
    fingerprint = archive_contract_fingerprint(
        schema, times, tuple(int(value) for value in states.shape)
    )
    training_config = {
        "history_months": history_months,
        "lead_months": lead_months,
        "seed": seed,
        "patch_size": list(patch_size),
        "tile_overlap": tile_overlap,
        "mixed_precision": mixed_precision,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "min_observed_fraction": min_observed_fraction,
        "model_backend": backend,
        "raw_time_ranges": raw_time_ranges,
        "missing_value_policy": "train_only_mean_with_observation_mask",
        "selection_metric": "forecast_rmse" if skill_every_epochs > 0 else "loss",
        "skill_every_epochs": skill_every_epochs,
        "skill_windows": skill_windows,
        "skill_ensemble_size": skill_ensemble_size,
        "skill_integration_steps": skill_integration_steps,
    }
    skill_weights = patch_latitude_weights(schema, patch_size)

    latest_path = checkpoint_root / "latest.pt"
    best_path = checkpoint_root / "best.pt"
    metrics_path = checkpoint_root / "metrics.jsonl"
    start_epoch = 1
    best_validation = float("inf")
    best_epoch = 0
    if resume and latest_path.is_file():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        if payload.get("format") != "climate_diffusion.monthly_latent_flow.v3":
            raise ValueError("Unsupported resume checkpoint format")
        saved_training = payload.get("training", {})
        if saved_training.get("archive_contract_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint belongs to a different archive contract")
        if payload.get("model_config") != asdict(model_config):
            raise ValueError("Resume model configuration differs from current RunPod arguments")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        if payload.get("rng_state"):
            _restore_rng(payload["rng_state"], loader_generator)
        start_epoch = int(saved_training["epoch"]) + 1
        best_validation = float(saved_training["best_validation_loss"])
        best_epoch = int(saved_training["best_epoch"])
        print({"resume": str(latest_path), "start_epoch": start_epoch, "best_epoch": best_epoch})

    for epoch in range(start_epoch, epochs + 1):
        if _free_gib(checkpoint_root) < min_free_disk_gb:
            raise RuntimeError("Free disk fell below safety threshold before epoch start")
        train_metrics = _epoch(
            model,
            train_loader,
            loss_config,
            device,
            optimizer,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scaler=scaler,
        )
        validation_metrics = _epoch(
            model,
            validation_loader,
            loss_config,
            device,
            optimizer=None,
            mixed_precision=mixed_precision,
            eval_seed=seed,
        )
        skill_metrics = None
        if skill_every_epochs > 0 and epoch % skill_every_epochs == 0:
            skill_metrics = forecast_skill(
                model,
                validation_dataset,
                device,
                windows=skill_windows,
                ensemble_size=skill_ensemble_size,
                integration_steps=skill_integration_steps,
                seed=seed,
                weights=skill_weights,
            )
        if skill_every_epochs > 0:
            # Only an epoch that sampled forecasts can be compared on them.
            candidate = None if skill_metrics is None else skill_metrics["forecast_rmse"]
        else:
            candidate = float(validation_metrics["loss"])
        improved = candidate is not None and candidate < best_validation
        if improved:
            best_validation = float(candidate)
            best_epoch = epoch
        latest = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            loss_config=loss_config,
            state_mean=state_mean,
            state_scale=state_scale,
            auxiliary_mean=auxiliary_mean,
            auxiliary_scale=auxiliary_scale,
            schema=schema,
            split=split,
            archive=archive,
            contract_fingerprint=fingerprint,
            epoch=epoch,
            best_validation_loss=best_validation,
            best_epoch=best_epoch,
            training_config=training_config,
            loader_generator=loader_generator,
        )
        _atomic_torch_save(latest, latest_path)
        if improved:
            _atomic_torch_save(_best_payload(latest), best_path)
        if epoch % save_every_epochs == 0:
            snapshot = checkpoint_root / f"epoch-{epoch:04d}.pt"
            _atomic_torch_save(latest, snapshot)
            snapshots = sorted(checkpoint_root.glob("epoch-*.pt"))
            for old in snapshots[:-keep_epoch_snapshots] if keep_epoch_snapshots > 0 else snapshots:
                old.unlink(missing_ok=True)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "forecast_skill": skill_metrics,
            "best_validation_loss": best_validation,
            "best_epoch": best_epoch,
            "free_disk_gib": round(_free_gib(checkpoint_root), 3),
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(record)

    summary = {
        "format": "climate_diffusion.runpod_500gb_summary.v1",
        "archive": str(archive),
        "archive_contract_fingerprint": fingerprint,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "model_backend": backend,
        "grid_shape": schema["grid_shape"],
        "spatial_channels": schema["spatial_channels"],
        "pressure_levels_hpa": (schema.get("source_metadata") or {}).get("pressure_levels_hpa"),
        "training_config": training_config,
    }
    (checkpoint_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return best_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume-safe 500GB RunPod spatial Flow training")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-backend", choices=("spatial_conv", "spatial_operator"), default="spatial_conv")
    parser.add_argument("--history-months", type=int, default=6)
    parser.add_argument("--lead-months", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--purge-months", type=int, default=1)
    parser.add_argument("--spatial-base-channels", type=int, default=32)
    parser.add_argument("--spatial-latent-channels", type=int, default=16)
    parser.add_argument("--spatial-downsample-levels", type=int, default=3)
    parser.add_argument("--operator-modes-lat", type=int, default=12)
    parser.add_argument("--operator-modes-lon", type=int, default=24)
    parser.add_argument("--patch-height", type=int, default=256)
    parser.add_argument("--patch-width", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-observed-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--keep-epoch-snapshots", type=int, default=3)
    parser.add_argument("--min-free-disk-gb", type=float, default=25.0)
    parser.add_argument(
        "--skill-every-epochs",
        type=int,
        default=1,
        help="Sample forecasts every N epochs and select on them; 0 selects on validation loss",
    )
    parser.add_argument("--skill-windows", type=int, default=2)
    parser.add_argument("--skill-ensemble-size", type=int, default=2)
    parser.add_argument("--skill-integration-steps", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    path = train_runpod_spatial_flow(
        args.archive,
        args.checkpoint_dir,
        backend=args.model_backend,
        history_months=args.history_months,
        lead_months=args.lead_months,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        purge_months=args.purge_months,
        spatial_base_channels=args.spatial_base_channels,
        spatial_latent_channels=args.spatial_latent_channels,
        spatial_downsample_levels=args.spatial_downsample_levels,
        operator_modes_lat=args.operator_modes_lat,
        operator_modes_lon=args.operator_modes_lon,
        patch_height=args.patch_height,
        patch_width=args.patch_width,
        tile_overlap=args.tile_overlap,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        mixed_precision=not args.no_mixed_precision,
        min_observed_fraction=args.min_observed_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=not args.no_resume,
        save_every_epochs=args.save_every_epochs,
        keep_epoch_snapshots=args.keep_epoch_snapshots,
        min_free_disk_gb=args.min_free_disk_gb,
        skill_every_epochs=args.skill_every_epochs,
        skill_windows=args.skill_windows,
        skill_ensemble_size=args.skill_ensemble_size,
        skill_integration_steps=args.skill_integration_steps,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
