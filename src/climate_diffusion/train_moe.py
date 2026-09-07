"""Two-stage full-state FM-MoE training with separate calibration data."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dynamics import TrajectoryWindowDataset
from .moe import MOE_FORMAT, FlowMatchingMoE, MoEConfig, ensemble_scores
from .moe_data import build_moe_split, field_grid, load_moe_archive, validate_moe_split
from .train import _sha256


def _pairs(model, batch, generator, lead_count):
    history, targets = batch["history"], batch["targets"]
    context = model.encode_history(history)
    batch_size, horizon, dimension = targets.shape
    count = min(lead_count, horizon)
    picks = torch.randint(horizon, (batch_size, count), device=targets.device, generator=generator)
    rows = torch.arange(batch_size, device=targets.device)[:, None]
    target = targets[rows, picks].reshape(-1, dimension)
    context = context[:, None].expand(-1, count, -1).reshape(len(target), -1)
    lead = (picks.flatten().to(target.dtype) + 1) / horizon
    source = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    tau = torch.rand(len(target), device=target.device, generator=generator)
    state = (1 - tau[:, None]) * source + tau[:, None] * target
    return state, target - source, tau, context, lead, target


def _sample_pairs(model, context, lead, members, steps, generator):
    initial = torch.randn(len(context) * members, model.config.state_dim,
                          device=context.device, dtype=context.dtype, generator=generator)
    prediction = model.integrate(initial, context.repeat_interleave(members, 0),
                                 lead.repeat_interleave(members, 0), integration_steps=steps)
    return prediction.reshape(len(context), members, -1)


def _epoch(model, loader, device, generator, *, optimizer, members, steps, lead_count, weights):
    training = optimizer is not None
    model.train(training)
    totals, examples = {}, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            state, velocity, tau, context, lead, target = _pairs(model, batch, generator, lead_count)
            if model.stage == "experts":
                metrics = model.warmup_loss(state, velocity, tau, context, lead,
                                            balance_weight=weights["balance"],
                                            diversity_weight=weights["diversity"])
                if not training:
                    samples = _sample_pairs(model, context, lead, members, steps, generator)
                    metrics.update(ensemble_scores(samples, target))
                    metrics["ensemble_spread"] = samples.std(1, unbiased=False).mean()
                    metrics["forecast_rmse"] = (samples.mean(1) - target).square().mean().sqrt()
            else:
                samples = _sample_pairs(model, context, lead, members, steps, generator)
                metrics = model.meta_loss(state, velocity, tau, context, lead,
                                          samples=samples, target=target,
                                          fm_weight=weights["fm"], energy_weight=weights["energy"],
                                          crps_weight=weights["crps"], diversity_weight=weights["diversity"])
                metrics["forecast_rmse"] = (samples.mean(1) - target).square().mean().sqrt()
            if not all(bool(torch.isfinite(value)) for value in metrics.values()):
                raise FloatingPointError(f"Non-finite {model.stage} metrics")
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
            size = len(target)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
            examples += size
    if not examples:
        raise ValueError("Empty training/validation loader")
    return {key: value / examples for key, value in totals.items()}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_artifacts(path, payload, metrics):
    torch.save(payload, path)
    _write_json(path.with_suffix(".metrics.json"), metrics)
    metadata = {"checkpoint_format": MOE_FORMAT, "model_config": payload["model_config"],
                "training": payload["training"], "forecast_step_hours": payload["training"]["step_hours"]}
    _write_json(path.with_suffix(".metadata.json"), metadata)
    _write_json(path.with_suffix(".manifest.json"), {
        "format": "climate_diffusion.artifact.v2", "checkpoint": path.name,
        "checkpoint_sha256": _sha256(path), "forecast_step_hours": metadata["forecast_step_hours"],
        "metrics": path.with_suffix(".metrics.json").name,
        "metadata": path.with_suffix(".metadata.json").name})


def train_moe(archive_path, output_path, *, stage="all", init_checkpoint=None,
              model_options=None, expert_epochs=20, meta_epochs=10, batch_size=8,
              learning_rate=1e-4, window_stride=48, purge_windows=0,
              ensemble_size=4, integration_steps=8, expert_leads=8, meta_leads=2,
              max_validation_windows=64, seed=7, device=None,
              fm_weight=1.0, energy_weight=0.5, crps_weight=0.5,
              balance_weight=0.05, diversity_weight=0.01):
    if stage not in {"all", "experts", "meta"}:
        raise ValueError("stage must be all, experts or meta")
    if min(expert_epochs, meta_epochs, batch_size, window_stride, integration_steps,
           expert_leads, meta_leads, max_validation_windows) < 1 or ensemble_size < 2:
        raise ValueError("Counts must be positive; ensemble_size must be at least 2")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    weights = {"fm": fm_weight, "energy": energy_weight, "crps": crps_weight,
               "balance": balance_weight, "diversity": diversity_weight}
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    if stage in {"all", "meta"} and energy_weight + crps_weight <= 0:
        raise ValueError("Meta training requires a probabilistic ensemble objective")
    if (stage == "meta") != (init_checkpoint is not None):
        raise ValueError("Use --init-checkpoint only with --stage meta, after experts warm-up")

    output = Path(output_path)
    if output.suffix != ".pt":
        raise ValueError("output must end in .pt")
    if init_checkpoint and output.resolve() == Path(init_checkpoint).resolve():
        raise ValueError("Do not overwrite the warm-up checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    states, times, schema = load_moe_archive(archive_path)
    archive_hash = _sha256(Path(archive_path))
    options = dict(model_options or {})
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    initial = None
    if init_checkpoint:
        from .inference import LatentFlowForecaster
        frozen = LatentFlowForecaster(init_checkpoint, device="cpu")  # verifies manifest
        if not frozen.is_moe or frozen.training_metadata["stage"] != "experts":
            raise ValueError("Meta stage requires an experts-stage MoE checkpoint")
        initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        config = MoEConfig(**initial["model_config"])
        for key, value in options.items():
            if getattr(config, key) != value:
                raise ValueError(f"Model option conflicts with warm-up checkpoint: {key}")
        if initial["schema"] != schema or initial["training"]["archive_sha256"] != archive_hash:
            raise ValueError("Meta stage must use the identical warm-up archive and schema")
        split = initial["training"]["split"]
        mean = initial["state_mean"].numpy()
        scale = initial["state_scale"].numpy()
        normalization_end = initial["training"]["normalization_span"][1]
    else:
        config = MoEConfig(state_dim=states.shape[1], grid=field_grid(schema),
                           step_hours=int(schema["forecast_step_hours"]), **options)
        count = len(states) - config.history_span_steps - config.horizon_steps + 1
        split = build_moe_split(count, config.horizon_steps, purge_windows=purge_windows)
        normalization_end = split["train"][-1] + config.history_span_steps + config.horizon_steps
        reference = states[:normalization_end]
        mean = reference.mean(0).astype(np.float32)
        scale = reference.std(0).astype(np.float32)
        scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    count = len(states) - config.history_span_steps - config.horizon_steps + 1
    validate_moe_split(split, config.horizon_steps, count)
    normalized = ((states - mean) / scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Non-finite normalized archive")
    model = FlowMatchingMoE(config).to(device)
    if initial:
        model.load_state_dict(initial["model"])
    base_training = {
        "archive": str(archive_path), "archive_sha256": archive_hash,
        "archive_state_count": len(states), "first_time": str(times[0]), "last_time": str(times[-1]),
        "step_hours": config.step_hours, "forecast_step_hours": config.step_hours,
        "horizon_hours": config.horizon_hours, "history_span_steps": config.history_span_steps,
        "normalization_span": [0, normalization_end], "split": split,
        "split_contract": "moe_five_way_disjoint_future_targets.v1",
        "missing_value_policy": "fully_observed_or_fail", "seed": seed,
        "sampling_contract": "shared_member_noise_across_leads_not_joint_trajectory_training",
        "batch_size": batch_size, "window_stride": window_stride, "learning_rate": learning_rate,
        "ensemble_size": ensemble_size, "integration_steps": integration_steps,
        "expert_leads": expert_leads, "meta_leads": meta_leads,
        "max_validation_windows": max_validation_windows, "loss_weights": weights,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    all_metrics = []
    phases = ["experts", "meta"] if stage == "all" else [stage]
    warmup_path = output.with_name(output.stem + ".experts.pt") if stage == "all" else output
    for phase in phases:
        model.set_stage(phase)
        train_name, val_name = (("train", "expert_validation") if phase == "experts"
                                else ("calibration", "validation"))
        train_indices = split[train_name][::window_stride]
        val_indices = split[val_name][::window_stride]
        if len(val_indices) > max_validation_windows:
            positions = np.linspace(0, len(val_indices) - 1, max_validation_windows, dtype=int)
            val_indices = [val_indices[i] for i in positions]
        loader_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(TrajectoryWindowDataset(normalized, config, train_indices),
                                  batch_size=batch_size, shuffle=True, generator=loader_generator)
        val_loader = DataLoader(TrajectoryWindowDataset(normalized, config, val_indices), batch_size=batch_size)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=learning_rate, weight_decay=0.0)
        generator = torch.Generator(device=device).manual_seed(seed)
        phase_output = warmup_path if phase == "experts" else output
        best = float("inf")
        epochs = expert_epochs if phase == "experts" else meta_epochs
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(model, train_loader, device, generator, optimizer=optimizer,
                                   members=ensemble_size, steps=integration_steps,
                                   lead_count=expert_leads if phase == "experts" else meta_leads, weights=weights)
            validation = _epoch(model, val_loader, device,
                                 torch.Generator(device=device).manual_seed(seed + 10000),
                                 optimizer=None, members=ensemble_size, steps=integration_steps,
                                 lead_count=meta_leads, weights=weights)
            # Actual generated ensemble score, not compression or FM loss alone.
            criterion = validation["energy"] + validation["crps"]
            row = {"stage": phase, "epoch": epoch, "train": train_metrics,
                   "validation": validation, "selection_score": criterion}
            all_metrics.append(row)
            print(f"stage={phase} epoch={epoch:04d} loss={train_metrics['loss']:.5f} "
                  f"val_energy+crps={criterion:.5f} spread={validation['ensemble_spread']:.5f}", flush=True)
            if criterion < best:
                best = criterion
                payload = {"format": MOE_FORMAT, "model_config": asdict(config),
                           "model": model.state_dict(), "schema": schema,
                           "state_mean": torch.from_numpy(mean), "state_scale": torch.from_numpy(scale),
                           "training": {**base_training, "stage": phase, "best_epoch": epoch,
                                        "best_selection_score": best, "selection_metric": "energy_plus_crps",
                                        "train_split": train_name, "validation_split": val_name,
                                        "train_window_count": len(train_indices),
                                        "validation_windows": val_indices,
                                        "warmup_checkpoint_sha256": (_sha256(Path(init_checkpoint or warmup_path))
                                                                    if phase == "meta" else None)}}
                _save_artifacts(phase_output, payload, all_metrics)
            _write_json(phase_output.with_suffix(".metrics.json"), all_metrics)
        # Stage 2 must start from the SELECTED stage-1 parameters, not the last epoch.
        chosen = torch.load(phase_output, map_location=device, weights_only=False)
        model.load_state_dict(chosen["model"])
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="download/flow-matching/moe/moe.pt")
    parser.add_argument("--stage", choices=("all", "experts", "meta"), default="all")
    parser.add_argument("--init-checkpoint")
    model_keys = ("history_steps", "history_stride", "horizon_steps", "num_experts",
                  "expert_latent_dim", "meta_latent_dim", "hidden_dim", "context_dim")
    for name in model_keys:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    for name, default in (("expert_epochs", 20), ("meta_epochs", 10), ("batch_size", 8),
                          ("window_stride", 48), ("purge_windows", 0), ("ensemble_size", 4),
                          ("integration_steps", 8), ("expert_leads", 8), ("meta_leads", 2),
                          ("max_validation_windows", 64), ("seed", 7)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (("learning_rate", 1e-4), ("fm_weight", 1.0), ("energy_weight", 0.5),
                          ("crps_weight", 0.5), ("balance_weight", 0.05), ("diversity_weight", 0.01)):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--device")
    args = vars(parser.parse_args(argv))
    options = {key: args.pop(key) for key in model_keys}
    print(train_moe(args.pop("archive"), args.pop("output"),
                    model_options={k: v for k, v in options.items() if v is not None}, **args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
