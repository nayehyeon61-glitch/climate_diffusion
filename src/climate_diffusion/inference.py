"""Load and sample trained monthly latent flow checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import FlowModelConfig
from .data import load_monthly_archive
from .model import MonthlyLatentFlow


class LatentFlowForecaster:
    inference_only = True

    def __init__(self, checkpoint: str | Path, *, device: str | None = None):
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if payload.get("format") != "climate_diffusion.monthly_latent_flow.v1":
            raise ValueError("Unsupported climate flow checkpoint format")
        self.config = FlowModelConfig(**payload["model_config"])
        self.model = MonthlyLatentFlow(self.config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.state_mean = payload["state_mean"].to(self.device)
        self.state_scale = payload["state_scale"].to(self.device)
        self.schema = payload["schema"]
        self.training_metadata = payload.get("training", {})

    def _normalise(self, values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return (tensor - self.state_mean) / self.state_scale

    def _denormalise(self, values: torch.Tensor) -> np.ndarray:
        result = values * self.state_scale + self.state_mean
        return result.detach().cpu().numpy().astype(np.float32)

    def forecast(
        self,
        history_states: np.ndarray,
        *,
        months: int = 1,
        ensemble_size: int = 1,
        integration_steps: int = 32,
        seed: int = 0,
    ) -> np.ndarray:
        """Return [ensemble, month, state] autoregressive samples."""
        history = np.asarray(history_states, dtype=np.float32)
        expected = (self.config.history_months, self.config.state_dim)
        if history.shape != expected:
            raise ValueError(f"Expected history shape {expected}, received {history.shape}")
        if min(months, ensemble_size, integration_steps) < 1:
            raise ValueError("months, ensemble_size and integration_steps must be positive")

        normalized = self._normalise(history)
        outputs = []
        for member in range(ensemble_size):
            member_history = normalized.clone()
            generator = torch.Generator(device=self.device).manual_seed(seed + member)
            member_outputs = []
            for _ in range(months):
                prediction = self.model.sample(
                    member_history.unsqueeze(0),
                    integration_steps=integration_steps,
                    generator=generator,
                )[0]
                member_outputs.append(prediction)
                member_history = torch.cat(
                    (member_history[1:], prediction.unsqueeze(0)), dim=0
                )
            outputs.append(torch.stack(member_outputs))
        return self._denormalise(torch.stack(outputs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample a monthly climate flow model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--archive", required=True, help="Archive providing latest history")
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/climate-flow-forecast.npz")
    args = parser.parse_args(argv)

    states, times, _ = load_monthly_archive(args.archive)
    forecaster = LatentFlowForecaster(args.checkpoint)
    history = states[-forecaster.config.history_months :]
    predictions = forecaster.forecast(
        history,
        months=args.months,
        ensemble_size=args.ensemble_size,
        integration_steps=args.integration_steps,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        predictions=predictions,
        last_history_time=times[-1],
        checkpoint=str(forecaster.checkpoint_path),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
