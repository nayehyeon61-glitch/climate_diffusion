"""A: PI-AE, B: local expert specialization, C: anchored joint calibration."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dynamics import TrajectoryWindowDataset
from .manifold_moe import MANIFOLD_FORMAT, ManifoldMoE, ManifoldMoEConfig
from .moe import ensemble_scores
from .moe_data import build_moe_split, field_grid, load_moe_archive, validate_moe_split
from .train import _sha256


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save(path, model, schema, mean, scale, training, rows):
    payload = {"format": MANIFOLD_FORMAT, "model_config": asdict(model.config),
               "model": model.state_dict(), "schema": schema,
               "state_mean": torch.as_tensor(mean), "state_scale": torch.as_tensor(scale),
               "training": training}
    torch.save(payload, path)
    _json(path.with_suffix(".metadata.json"), {k: payload[k] for k in ("format", "model_config", "schema", "training")})
    _json(path.with_suffix(".metrics.json"), rows)
    _json(path.with_suffix(".manifest.json"), {"format": "climate_diffusion.artifact.v2",
          "checkpoint": path.name, "checkpoint_sha256": _sha256(path),
          "forecast_step_hours": model.config.step_hours,
          "metadata": path.with_suffix(".metadata.json").name,
          "metrics": path.with_suffix(".metrics.json").name})


def _pairs(model, batch, generator, lead_count):
    targets = batch["targets"]
    count = min(lead_count, targets.shape[1])
    horizon = targets.shape[1]
    if count == 1:
        picks = torch.randint(horizon, (len(targets), 1), device=targets.device, generator=generator)
    else:
        starts = torch.randint(horizon - count + 1, (len(targets), 1),
                               device=targets.device, generator=generator)
        picks = starts + torch.arange(count, device=targets.device)[None]
    rows = torch.arange(len(targets), device=targets.device)[:, None]
    target = targets[rows, picks].reshape(-1, targets.shape[-1])
    context = model.encode_history(batch["history"])[:, None].expand(-1, count, -1).reshape(len(target), -1)
    # Do not let the encoder shrink/move FM targets to make transport loss easy.
    code = model.encode(target).detach()
    # Reuse the same intrinsic source across adjacent leads, consistent with
    # member coupling at inference. This is conditioning repair, not joint-law training.
    source = torch.randn((len(targets), 1, code.shape[-1]), device=code.device,
                         dtype=code.dtype, generator=generator).expand(-1, count, -1).reshape_as(code)
    tau = torch.rand(len(code), device=code.device, dtype=code.dtype, generator=generator)
    lead = (picks.flatten().to(code) + 1) / targets.shape[1]
    q = (1 - tau[:, None]) * source + tau[:, None] * code
    return q, code - source, tau, context, lead, target


def _sample(model, context, lead, members, steps, generator):
    q0 = torch.randn(len(context) * members, model.config.manifold_dim,
                     device=context.device, dtype=context.dtype, generator=generator)
    q = model.integrate(q0, context.repeat_interleave(members, 0), lead.repeat_interleave(members, 0),
                        integration_steps=steps)
    return model.decode(q).reshape(len(context), members, -1)


def _epoch(model, loader, device, generator, optimizer, options):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    pi_options = {k: options[k] for k in ("physics_weight", "invariant_weight", "metric_weight", "dynamics_weight")}
    specialist_options = {k: options[k] for k in ("gate_weight", "balance_weight", "diversity_weight",
                                                  "projection_weight", "entropy_weight")}
    with torch.set_grad_enabled(training):
        for batch in loader:
            if model.stage == "manifold":
                state, next_state = [v.to(device) for v in batch]
                metrics = model.manifold_loss(state, next_state, **pi_options)
                size = len(state)
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
                q, velocity, tau, context, lead, target = _pairs(model, batch, generator, options["sampled_leads"])
                metrics = model.specialization_loss(q, velocity, tau, context, lead, **specialist_options)
                if model.stage == "joint" or not training:
                    samples = _sample(model, context, lead, options["ensemble_size"],
                                      options["integration_steps"], generator)
                    metrics.update(ensemble_scores(samples, target))
                    spread = samples.std(1, unbiased=False)
                    metrics["ensemble_spread"] = spread.mean()
                    metrics["forecast_rmse"] = (samples.mean(1) - target).square().mean().sqrt()
                    metrics["spread_guard"] = (torch.relu(0.02 - spread).square()
                                                + torch.relu(spread - 3).square()).mean()
                if model.stage == "joint":
                    auxiliary = model.manifold_loss(batch["origin"], batch["targets"][:, 0], **pi_options)
                    anchor_states = torch.cat((batch["origin"], target), 0)
                    anchor = (model.encode(anchor_states) - model.reference_encode(anchor_states).detach()).square().mean()
                    for key, value in auxiliary.items():
                        metrics["pi_" + key] = value
                    metrics["anchor"] = anchor
                    metrics["loss"] = (metrics["loss"] + options["energy_weight"] * metrics["energy"]
                                       + options["crps_weight"] * metrics["crps"]
                                       + options["joint_pi_weight"] * auxiliary["loss"]
                                       + options["anchor_weight"] * anchor + 0.01 * metrics["spread_guard"])
                size = len(target)
            if not all(bool(torch.isfinite(v)) for v in metrics.values()):
                raise FloatingPointError(f"Non-finite {model.stage} metrics")
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              1.0, error_if_nonfinite=True)
                optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
            count += size
    if not count:
        raise ValueError("Empty stage dataset")
    return {k: v / count for k, v in totals.items()}


def train_manifold_moe(archive_path, output_path, *, stage="all", init_checkpoint=None,
                       model_options=None, manifold_epochs=30, expert_epochs=30, joint_epochs=10,
                       batch_size=8, window_stride=4, purge_windows=0, learning_rate=1e-3,
                       joint_lr_factor=0.1, encoder_lr_factor=0.1, ensemble_size=4,
                       integration_steps=4, sampled_leads=1, max_validation_windows=32,
                       seed=7, device=None, physics_weight=0.1, invariant_weight=0.05,
                       metric_weight=0.1, dynamics_weight=0.1, gate_weight=0.2,
                       balance_weight=0.05, diversity_weight=0.001, projection_weight=0.05,
                       entropy_weight=0.01, energy_weight=0.5, crps_weight=0.5,
                       joint_pi_weight=0.5, anchor_weight=1.0):
    options = {k: v for k, v in locals().items() if k.endswith("_weight")}
    options.update(ensemble_size=ensemble_size, integration_steps=integration_steps, sampled_leads=sampled_leads)
    if stage not in {"all", "manifold", "specialize", "joint"}:
        raise ValueError("Invalid stage")
    if (stage in {"specialize", "joint"}) != (init_checkpoint is not None):
        raise ValueError("Only Stage B/C require --init-checkpoint from the preceding stage")
    if min(manifold_epochs, expert_epochs, joint_epochs, batch_size, window_stride,
           integration_steps, sampled_leads, max_validation_windows) < 1 or ensemble_size < 2:
        raise ValueError("Positive counts and ensemble_size >= 2 are required")
    for value in (learning_rate, joint_lr_factor, encoder_lr_factor):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rates/factors must be finite and positive")
    if joint_lr_factor > 1 or encoder_lr_factor >= 1:
        raise ValueError("Stage C must not raise LR; encoder_lr_factor must be below 1")
    if any(not math.isfinite(v) or v < 0 for k, v in options.items() if k.endswith("_weight")):
        raise ValueError("Loss weights must be finite and nonnegative")
    if min(physics_weight, metric_weight, gate_weight, projection_weight, anchor_weight) <= 0:
        raise ValueError("Final manifold model requires physics/metric/gate/projection/anchor losses")
    if energy_weight + crps_weight <= 0:
        raise ValueError("At least one probabilistic ensemble objective is required")
    output = Path(output_path)
    if output.suffix != ".pt":
        raise ValueError("output must end with .pt")
    if init_checkpoint and output.resolve() == Path(init_checkpoint).resolve():
        raise ValueError("Do not overwrite the previous-stage checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    states, times, schema = load_moe_archive(archive_path)
    archive_hash = _sha256(Path(archive_path))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    initial = None
    if init_checkpoint:
        from .inference import LatentFlowForecaster
        loaded = LatentFlowForecaster(init_checkpoint, device="cpu")
        required = "manifold" if stage == "specialize" else "specialize"
        if not loaded.is_manifold or loaded.model.stage != required:
            raise ValueError(f"{stage} requires a {required}-stage manifold checkpoint")
        initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        config = ManifoldMoEConfig(**initial["model_config"])
        if any(getattr(config, k) != v for k, v in (model_options or {}).items()):
            raise ValueError("Model options conflict with the previous checkpoint")
        if initial["schema"] != schema or initial["training"]["archive_sha256"] != archive_hash:
            raise ValueError("All stages must use the identical archive and schema")
        split = initial["training"]["split"]
        end = initial["training"]["normalization_span"][1]
        mean, scale = initial["state_mean"].numpy(), initial["state_scale"].numpy()
    else:
        config = ManifoldMoEConfig(state_dim=states.shape[1], grid=field_grid(schema),
                                   step_hours=int(schema["forecast_step_hours"]), **(model_options or {}))
        count = len(states) - config.history_span_steps - config.horizon_steps + 1
        split = build_moe_split(count, config.horizon_steps, purge_windows=purge_windows)
        end = split["train"][-1] + config.history_span_steps + config.horizon_steps
        mean, scale = states[:end].mean(0), states[:end].std(0)
        scale = np.where(scale > 1e-6, scale, 1).astype(np.float32)
    count = len(states) - config.history_span_steps - config.horizon_steps + 1
    validate_moe_split(split, config.horizon_steps, count)
    normalized = torch.from_numpy(((states - mean) / scale).astype(np.float32))
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("Non-finite normalized archive")
    model = ManifoldMoE(config, schema, mean, scale).to(device)
    if initial:
        model.load_state_dict(initial["model"])
        if not bool(model.manifold_ready):
            raise ValueError("Preceding checkpoint has no sealed manifold coordinates")
    else:
        model.physics.fit(normalized[:end].to(device))
    base = {"archive": str(archive_path), "archive_sha256": archive_hash,
            "archive_state_count": len(states), "first_time": str(times[0]), "last_time": str(times[-1]),
            "step_hours": config.step_hours, "forecast_step_hours": config.step_hours,
            "history_span_steps": config.history_span_steps, "horizon_hours": config.horizon_hours,
            "split": split, "split_contract": "moe_five_way_disjoint_future_targets.v1",
            "normalization_span": [0, end], "manifold_fit_span": [0, end],
            "missing_value_policy": "fully_observed_or_fail",
            "sampling_contract": "intrinsic_ODE_shared_member_noise_across_leads_not_joint_trajectory_training",
            "training_lead_sampling": "adjacent_leads_with_shared_source_noise.v1",
            "lead_condition_contract": "s=(lead_index+1)/horizon_steps; physical_hours=s*horizon_hours",
            "target_time_semantics": "target[j]=origin+(j+1)*forecast_step_hours",
            "physics_contract": "surface_diagnostic_reconstruction_and_metric_not_primitive_PDE",
            "projection_contract": "decoder_Jacobian_damped_weighted_tangent_lift.v1",
            "seed": seed, "learning_rate": learning_rate, "joint_lr_factor": joint_lr_factor,
            "encoder_lr_factor": encoder_lr_factor, "batch_size": batch_size,
            "window_stride": window_stride, "loss_options": options,
            "parameter_count": sum(p.numel() for p in model.parameters())}
    phases = ("manifold", "specialize", "joint") if stage == "all" else (stage,)
    rows = []
    previous = Path(init_checkpoint) if init_checkpoint else None
    for phase in phases:
        model.set_stage(phase)
        val_name = "validation" if phase == "joint" else "expert_validation"
        train_name = "calibration" if phase == "joint" else "train"
        if phase == "manifold":
            train_indices = list(range(0, end - 1, window_stride))
            # Both observations in each validation pair belong to the validation target interval.
            left = split[val_name][0] + config.history_span_steps
            right = split[val_name][-1] + config.history_span_steps + config.horizon_steps - 1
            val_indices = list(range(left, right, window_stride))
            if not val_indices:
                raise ValueError("Stage A requires two validation observations")
            val_indices = val_indices[::max(1, math.ceil(len(val_indices) / max_validation_windows))]
            train_data = TensorDataset(normalized[train_indices], normalized[np.asarray(train_indices) + 1])
            val_data = TensorDataset(normalized[val_indices], normalized[np.asarray(val_indices) + 1])
        else:
            train_indices = split[train_name][::window_stride]
            val_indices = split[val_name][::window_stride]
            val_indices = val_indices[::max(1, math.ceil(len(val_indices) / max_validation_windows))]
            train_data = TrajectoryWindowDataset(normalized, config, train_indices)
            val_data = TrajectoryWindowDataset(normalized, config, val_indices)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_data, batch_size=batch_size)
        rate = learning_rate * (joint_lr_factor if phase == "joint" else 1)
        if phase == "joint":
            groups = [{"params": model.manifold.parameters(), "lr": rate * encoder_lr_factor},
                      {"params": list(model.experts.parameters()) + list(model.gate.parameters())
                                  + list(model.history_encoder.parameters()), "lr": rate}]
        else:
            groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": rate}]
        optimizer = torch.optim.AdamW(groups, weight_decay=0)
        epochs = {"manifold": manifold_epochs, "specialize": expert_epochs, "joint": joint_epochs}[phase]
        phase_output = output if stage != "all" or phase == "joint" else output.with_name(output.stem + f".{phase}.pt")
        best = float("inf")
        training_generator = torch.Generator(device=device).manual_seed(seed)
        phase_metadata = {**base, "stage": phase, "train_split": train_name, "validation_split": val_name,
                          "train_window_count": len(train_indices), "validation_indices": val_indices,
                          "previous_checkpoint_sha256": _sha256(previous) if previous else None}
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(model, train_loader, device, training_generator, optimizer, options)
            validation = _epoch(model, val_loader, device,
                                torch.Generator(device=device).manual_seed(seed + 10000), None, options)
            score = validation["loss"] if phase == "manifold" else validation["energy"] + validation["crps"]
            rows.append({"stage": phase, "epoch": epoch, "train": train_metrics,
                         "validation": validation, "selection_score": score})
            print(f"stage={phase} epoch={epoch:04d} loss={train_metrics['loss']:.5f} val={score:.5f}", flush=True)
            if score < best:
                best = score
                selected = {**phase_metadata, "best_epoch": epoch, "best_selection_score": best,
                            "selection_metric": "pi_validation_loss" if phase == "manifold" else "energy_plus_crps"}
                _save(phase_output, model, schema, mean, scale, selected, rows)
            _json(phase_output.with_suffix(".metrics.json"), rows)
        payload = torch.load(phase_output, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if phase == "manifold":
            model.seal_manifold(normalized[:end].to(device))
            _save(phase_output, model, schema, mean, scale, payload["training"], rows)
        previous = phase_output
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="outputs/manifold-moe/model.pt")
    parser.add_argument("--stage", choices=("all", "manifold", "specialize", "joint"), default="all")
    parser.add_argument("--init-checkpoint")
    integer_model = ("history_steps", "history_stride", "horizon_steps", "num_experts", "manifold_dim",
                     "expert_latent_dim", "gate_hidden_dim", "hidden_dim", "context_dim")
    float_model = ("gate_temperature", "responsibility_temperature", "locality_weight",
                   "gate_correction_limit", "projection_ridge")
    for name in integer_model + float_model:
        parser.add_argument("--" + name.replace("_", "-"), type=int if name in integer_model else float)
    for name, default in (("manifold_epochs", 30), ("expert_epochs", 30), ("joint_epochs", 10),
                          ("batch_size", 8), ("window_stride", 4), ("purge_windows", 0),
                          ("ensemble_size", 4), ("integration_steps", 4), ("sampled_leads", 1),
                          ("max_validation_windows", 32), ("seed", 7)):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (("learning_rate", 1e-3), ("joint_lr_factor", 0.1), ("encoder_lr_factor", 0.1),
                          ("physics_weight", 0.1), ("invariant_weight", 0.05), ("metric_weight", 0.1),
                          ("dynamics_weight", 0.1), ("gate_weight", 0.2), ("balance_weight", 0.05),
                          ("diversity_weight", 0.001), ("projection_weight", 0.05), ("entropy_weight", 0.01),
                          ("energy_weight", 0.5), ("crps_weight", 0.5), ("joint_pi_weight", 0.5),
                          ("anchor_weight", 1.0)):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--device")
    args = vars(parser.parse_args(argv))
    model_options = {name: args.pop(name) for name in integer_model + float_model}
    print(train_manifold_moe(args.pop("archive"), args.pop("output"),
                             model_options={k: v for k, v in model_options.items() if v is not None}, **args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
