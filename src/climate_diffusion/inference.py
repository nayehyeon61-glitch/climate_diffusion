"""Load and sample trained latent Flow Matching checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import FlowModelConfig
from .data import load_monthly_archive
from .model import MonthlyLatentFlow
from .dynamics import DynamicsModelConfig, LatentDynamicsFlow
from .moe import MOE_FORMAT, FlowMatchingMoE, MoEConfig
from .moe_data import load_moe_archive
from .manifold_moe import MANIFOLD_FORMAT, ManifoldMoE, ManifoldMoEConfig


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
        if payload.get("format") not in {
            "climate_diffusion.monthly_latent_flow.v1",
            "climate_diffusion.monthly_latent_flow.v2",
            "climate_diffusion.latent_flow.v3",
            "climate_diffusion.latent_dynamics_flow.v1",
            MOE_FORMAT,
            MANIFOLD_FORMAT,
        }:
            raise ValueError("Unsupported climate flow checkpoint format")
        self.is_manifold = payload["format"] == MANIFOLD_FORMAT
        self.is_moe = self.is_manifold or payload["format"] == MOE_FORMAT
        # is_dynamics denotes the bounded, strided, multi-lead temporal contract.
        self.is_dynamics = self.is_moe or payload["format"] == "climate_diffusion.latent_dynamics_flow.v1"
        config_class = (ManifoldMoEConfig if self.is_manifold else MoEConfig if self.is_moe
                        else DynamicsModelConfig if self.is_dynamics else FlowModelConfig)
        model_class = FlowMatchingMoE if self.is_moe else LatentDynamicsFlow if self.is_dynamics else MonthlyLatentFlow
        self.config = config_class(**payload["model_config"])
        self.model = (ManifoldMoE(self.config, payload["schema"], payload["state_mean"], payload["state_scale"])
                      if self.is_manifold else model_class(self.config)).to(self.device)
        self.history_steps = self.config.history_steps if self.is_dynamics else self.config.history_months
        self.history_stride = self.config.history_stride if self.is_dynamics else 1
        self.history_span_steps = (self.history_steps - 1) * self.history_stride + 1
        self.model.load_state_dict(payload["model"])
        if self.is_moe:
            self.model.set_stage(payload["training"]["stage"])
        self.model.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Flow checkpoint could not be frozen")
        self.state_mean = payload["state_mean"].to(self.device)
        self.state_scale = payload["state_scale"].to(self.device)
        self.schema = payload["schema"]
        self.training_metadata = payload.get("training", {})
        self.checkpoint_format = str(payload["format"])
        self.forecast_step_hours = int(
            self.training_metadata.get(
                "forecast_step_hours", self.training_metadata.get(
                    "step_hours", self.schema.get("forecast_step_hours", 30 * 24)
                )
            )
        )
        if self.forecast_step_hours <= 0:
            raise ValueError("Flow checkpoint forecast_step_hours must be positive")
        if self.is_dynamics and self.forecast_step_hours != self.config.step_hours:
            raise ValueError("Dynamics model/checkpoint forecast-step mismatch")
        self.checkpoint_sha256 = self._verify_manifest()

    def validate_archive(self, schema: dict, times: np.ndarray) -> None:
        """Reject silently reordered fields/grids or an incompatible time step."""
        for key in ("state_dim", "variables", "integrated_feature_names"):
            if schema.get(key) != self.schema.get(key):
                raise ValueError(f"Archive/checkpoint schema mismatch: {key}")
        if self.is_dynamics:
            if int(schema.get("forecast_step_hours", 0)) != self.forecast_step_hours:
                raise ValueError("Archive/checkpoint forecast-step mismatch")
            if not np.all(np.diff(times) == np.timedelta64(self.forecast_step_hours, "h")):
                raise ValueError("Archive timestamps violate the fixed-step contract")

    def select_history(self, states: np.ndarray) -> np.ndarray:
        """Select the trained cadence from a dense archive ending at the origin."""
        if len(states) < self.history_span_steps:
            raise ValueError(
                f"Flow checkpoint requires {self.history_span_steps} consecutive history states; "
                f"received {len(states)}"
            )
        return states[-self.history_span_steps::self.history_stride]

    def _verify_manifest(self) -> str:
        hasher = hashlib.sha256()
        with self.checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        manifest_path = self.checkpoint_path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("checkpoint_sha256")
            if expected and expected != digest:
                raise ValueError(
                    f"Checkpoint checksum mismatch for {self.checkpoint_path}"
                )
            manifest_step = manifest.get("forecast_step_hours")
            if manifest_step is not None and int(manifest_step) != self.forecast_step_hours:
                raise ValueError("Flow manifest/checkpoint forecast-step mismatch")
        return digest

    def _normalise(self, values: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return (tensor - self.state_mean) / self.state_scale

    def _denormalise(self, values: torch.Tensor) -> np.ndarray:
        result = values * self.state_scale + self.state_mean
        return result.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()  # Manifold decoder Jacobians require forward-mode AD during sampling.
    def forecast(
        self,
        history_states: np.ndarray,
        *,
        months: int = 1,
        ensemble_size: int = 1,
        integration_steps: int = 32,
        seed: int = 0,
        moe_mode: str | None = None,
    ) -> np.ndarray:
        """Return [ensemble, forecast_step, state]; dynamics uses per-lead marginals."""
        history = np.asarray(history_states, dtype=np.float32)
        expected = (self.history_steps, self.config.state_dim)
        if history.shape != expected:
            raise ValueError(f"Expected history shape {expected}, received {history.shape}")
        if min(months, ensemble_size, integration_steps) < 1:
            raise ValueError("steps, ensemble_size and integration_steps must be positive")

        if not np.isfinite(history).all():
            raise ValueError("History states must be finite")
        if moe_mode is not None and not self.is_moe:
            raise ValueError("moe_mode is only supported by MoE checkpoints")
        if self.is_moe and not self.is_manifold and moe_mode == "meta" and self.model.stage != "meta":
            raise ValueError("Meta inference requires a completed meta-stage checkpoint")
        normalized = self._normalise(history)
        if self.is_dynamics:
            if months > self.config.horizon_steps:
                raise ValueError("Requested steps exceed the trained dynamics horizon")
            generator = torch.Generator(device=self.device).manual_seed(seed)
            samples = self.model.forecast(
                normalized.unsqueeze(0), normalized[-1:].clone(),
                ensemble_size=ensemble_size, integration_steps=integration_steps,
                lead_indices=list(range(months)), generator=generator,
                **({"mode": moe_mode} if self.is_moe else {}),
            )[0]
            return self._denormalise(samples)
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
    parser = argparse.ArgumentParser(description="Sample a trained latent climate Flow model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--archive", required=True, help="Archive providing latest history")
    parser.add_argument("--forecast-steps", "--months", dest="months", type=int, default=None,
                        help="Number of steps; default: full dynamics horizon, otherwise 1")
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--integration-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="Inference device, e.g. cpu or cuda; default selects automatically")
    parser.add_argument("--moe-mode", help="MoE: experts/meta/uniform; manifold: local/uniform/expert:<index>")
    parser.add_argument("--output", default="outputs/climate-flow-forecast.npz")
    args = parser.parse_args(argv)

    forecaster = LatentFlowForecaster(args.checkpoint, device=args.device)
    loader = load_moe_archive if forecaster.is_moe else load_monthly_archive
    states, times, schema = loader(args.archive)
    forecaster.validate_archive(schema, times)
    history = forecaster.select_history(states)
    predictions = forecaster.forecast(
        history,
        months=(args.months if args.months is not None else
                forecaster.config.horizon_steps if forecaster.is_dynamics else 1),
        ensemble_size=args.ensemble_size,
        integration_steps=args.integration_steps,
        seed=args.seed,
        moe_mode=args.moe_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    origin_time = times[-1].astype("datetime64[ns]")
    lead_hours = np.arange(1, predictions.shape[1] + 1) * forecaster.forecast_step_hours
    np.savez_compressed(
        output,
        predictions=predictions,
        origin_time=origin_time,
        last_history_time=origin_time,  # backward-compatible alias
        lead_hours=lead_hours,
        valid_times=origin_time + lead_hours.astype("timedelta64[h]"),
        checkpoint=str(forecaster.checkpoint_path),
        forecast_step_hours=np.asarray(forecaster.forecast_step_hours, dtype=np.int64),
        time_contract=np.asarray("valid_time = origin_time + lead_hours; exact UTC snapshots"),
        sampling_contract=np.asarray(forecaster.training_metadata.get(
            "sampling_contract", "unspecified")),
        moe_mode=str(args.moe_mode or ("local" if forecaster.is_manifold else
                                      forecaster.model.stage if forecaster.is_moe else "")),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
