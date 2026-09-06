"""Train the autoencoder on its own and hold it to the linear-PCA floor.

Inside the full model the reconstruction term competes with the trajectory and
flow-matching terms, so a bad reconstruction has two possible causes: the
autoencoder cannot do the job, or it is being outvoted. Training it alone
separates those. Linear PCA at the same latent width is the bar -- a learned
encoder that lands above it is broken, not merely small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.data import load_monthly_archive
from climate_diffusion.model import build_autoencoder

OUTPUT_ROOT = Path("outputs")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df",
)


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.dpi": 150,
    })


def field_grid(schema: dict) -> tuple[int, int, int]:
    shapes = {tuple(variable["shape"]) for variable in schema["variables"]}
    if len(shapes) != 1:
        raise ValueError("Conv autoencoder needs every variable on one grid")
    height, width = shapes.pop()
    return len(schema["variables"]), int(height), int(width)


def pca_floor(train: np.ndarray, latents: list[int], sample: int = 6000) -> dict[int, float]:
    rng = np.random.default_rng(0)
    rows = train[rng.choice(len(train), size=min(sample, len(train)), replace=False)]
    spectrum = np.linalg.svd(rows, compute_uv=False) ** 2
    total = spectrum.sum()
    return {
        k: float(np.sqrt(max(0.0, 1.0 - spectrum[:k].sum() / total)))
        for k in latents if k < min(rows.shape)
    }


def train_autoencoder(
    train: np.ndarray,
    holdout: np.ndarray,
    *,
    kind: str,
    grid: tuple[int, int, int],
    latent_dim: int,
    hidden_dim: int,
    blocks: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int = 7,
) -> dict:
    torch.manual_seed(seed)
    config = FlowModelConfig(
        state_dim=train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        autoencoder_hidden_dim=hidden_dim,
        autoencoder_blocks=blocks,
        autoencoder_kind=kind,
        autoencoder_grid=grid if kind == "conv" else None,
    )
    model = build_autoencoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train)), batch_size=batch_size, shuffle=True
    )
    evaluation = torch.from_numpy(holdout).to(device)
    curve = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            loss = torch.nn.functional.mse_loss(model(batch), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            errors = [
                float((model(chunk) - chunk).square().mean()) * len(chunk)
                for chunk in evaluation.split(512)
            ]
        rmse = float(np.sqrt(sum(errors) / len(evaluation)))
        curve.append(rmse)
        best = min(best, rmse)
        print(f"  {kind} latent={latent_dim} epoch={epoch:03d} holdout_rmse={rmse:.4f}",
              flush=True)
    with torch.no_grad():
        latent = torch.cat([model.encode(c) for c in evaluation.split(512)]).cpu().numpy()
    variance = latent.var(axis=0)
    return {
        "kind": kind,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "blocks": blocks,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "best_holdout_rmse": best,
        "final_holdout_rmse": curve[-1],
        "curve": curve,
        "effective_rank": float(variance.sum() ** 2 / (variance ** 2).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/era5_6h_states.npz")
    parser.add_argument("--latents", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--kinds", nargs="+", default=["mlp", "conv"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--holdout-samples", type=int, default=2000)
    parser.add_argument("--output", default="autoencoder-probe")
    args = parser.parse_args(argv)
    _style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    states, times, schema = load_monthly_archive(args.archive)
    grid = field_grid(schema)
    end = int(len(states) * 0.75)
    mean = states[:end].mean(axis=0)
    scale = states[:end].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = ((states - mean) / scale).astype(np.float32)

    rng = np.random.default_rng(0)
    train_rows = rng.choice(end, size=min(args.train_samples, end), replace=False)
    holdout_rows = rng.choice(
        np.arange(end, len(states)), size=args.holdout_samples, replace=False
    )
    train = normalized[np.sort(train_rows)]
    holdout = normalized[np.sort(holdout_rows)]

    floor = pca_floor(normalized[:end], args.latents)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for kind in args.kinds:
        for latent_dim in args.latents:
            results.append(train_autoencoder(
                train, holdout, kind=kind, grid=grid, latent_dim=latent_dim,
                hidden_dim=args.hidden_dim, blocks=args.blocks, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                weight_decay=args.weight_decay, device=device,
            ))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    latents = sorted(floor)
    axes[0].plot(latents, [floor[k] for k in latents], color=INK_MUTED, marker="o",
                 markersize=5, label="linear PCA floor")
    for index, kind in enumerate(args.kinds):
        rows = [r for r in results if r["kind"] == kind]
        axes[0].plot([r["latent_dim"] for r in rows],
                     [r["best_holdout_rmse"] for r in rows],
                     color=SERIES[index], marker="D", markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=1.2,
                     label=f"{kind} autoencoder")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(latents, [str(k) for k in latents])
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("holdout reconstruction RMSE")
    axes[0].set_title("Autoencoder trained alone, versus its linear floor",
                      color=INK, loc="left")
    axes[0].legend(fontsize=8, labelcolor=INK_SECONDARY)

    for index, result in enumerate(results):
        axes[1].plot(range(1, len(result["curve"]) + 1), result["curve"],
                     color=SERIES[index % len(SERIES)],
                     label=f"{result['kind']} latent {result['latent_dim']}")
    for latent_dim in args.latents:
        if latent_dim in floor:
            axes[1].axhline(floor[latent_dim], color=INK_MUTED, linewidth=1.0,
                            linestyle=(0, (1, 2)))
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("holdout reconstruction RMSE")
    axes[1].set_title("Dotted lines are the PCA floors", color=INK, loc="left")
    axes[1].legend(fontsize=8, labelcolor=INK_SECONDARY)

    figure.suptitle("Reconstruction probe on 6-hourly ERA5 snapshots",
                    color=INK, x=0.008, ha="left", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(OUTPUT_ROOT / f"{args.output}.png", bbox_inches="tight")
    plt.close(figure)

    summary = {
        "archive": args.archive,
        "grid": list(grid),
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "epochs": args.epochs,
        "pca_floor": floor,
        "runs": results,
    }
    path = OUTPUT_ROOT / f"{args.output}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
