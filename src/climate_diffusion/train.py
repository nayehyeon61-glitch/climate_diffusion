"""Train a monthly conditional flow matcher in an autoencoded latent space."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FlowLossConfig, FlowModelConfig
from .data import MonthlyWindowDataset, load_monthly_archive
from .model import MonthlyLatentFlow


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
        raise ValueError("Monthly split produced no training/validation batches")
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
    seed: int = 7,
) -> Path:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    torch.manual_seed(seed)
    np.random.seed(seed)
    states, times, schema = load_monthly_archive(archive_path)
    sample_count = len(states) - history_months - lead_months + 1
    if sample_count < 2:
        raise ValueError("Not enough monthly states for temporal train/validation splits")
    train_count = max(1, int(sample_count * (1.0 - validation_fraction)))
    if train_count >= sample_count:
        train_count = sample_count - 1

    normalization_end = train_count + history_months + lead_months - 1
    state_mean = states[:normalization_end].mean(axis=0).astype(np.float32)
    state_scale = states[:normalization_end].std(axis=0).astype(np.float32)
    state_scale = np.where(state_scale > 1e-6, state_scale, 1.0).astype(np.float32)
    normalized = ((states - state_mean) / state_scale).astype(np.float32)

    train_dataset = MonthlyWindowDataset(
        normalized,
        history_months,
        lead_months,
        indices=list(range(train_count)),
    )
    validation_dataset = MonthlyWindowDataset(
        normalized,
        history_months,
        lead_months,
        indices=list(range(train_count, sample_count)),
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(model, train_loader, loss_config, device, optimizer)
        validation_metrics = _epoch(
            model, validation_loader, loss_config, device, optimizer=None
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
                    "format": "climate_diffusion.monthly_latent_flow.v1",
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "loss_config": asdict(loss_config),
                    "state_mean": torch.from_numpy(state_mean),
                    "state_scale": torch.from_numpy(state_scale),
                    "schema": schema,
                    "training": {
                        "lead_months": lead_months,
                        "seed": seed,
                        "archive": str(archive_path),
                        "first_time": str(times[0]),
                        "last_time": str(times[-1]),
                        "best_validation_loss": best_validation,
                    },
                },
                output,
            )

    output.with_suffix(".metrics.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_kind": "monthly_latent_flow_matching",
                "forecast_step": "1 calendar month",
                "history_months": history_months,
                "state_dim": states.shape[1],
                "weather_next_compatible_runner": True,
                "inference_ready": True,
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="checkpoints/climate-flow-matching.pt")
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
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
