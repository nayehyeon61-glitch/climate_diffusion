"""Tiny CPU A/PINN -> frozen-A B -> C integration check using synthetic fields.

The pressure fields have meteorological units and plausible magnitudes, but are
not a primitive-equation solution or evidence of ERA5 forecast improvement.
Run from the repository root with PYTHONPATH=src and an unused --output path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import xarray as xr

from smoke_information_process import synthetic_information
from smoke_moe import synthetic_archive
from climate_diffusion.information_forecast import evaluate
from climate_diffusion.physical_information import prepare
from climate_diffusion.train_information_process import (
    load_checkpoint,
    main as train_main,
    write_json,
)


def synthetic_pinn_information(archive, directory):
    """Add matched pressure-level T/u/v/omega and true surface pressure.

    Preserve the existing synthetic-information helper and its Z/terrain inputs.
    The generated legacy sidecar also permits testing that missing PINN inputs
    are rejected instead of filled with zeros.
    """
    directory = Path(directory)
    synthetic_information(archive, directory)
    with xr.open_dataset(directory / "synthetic-information.nc") as source:
        fields = source.load()
    wave = (fields["z850"].values - 1500.0) / 80.0
    t500 = 255.0 + 3.0 * wave
    t850 = 280.0 + 4.0 * wave
    extras = {
        "t500": (t500, "K"),
        "t850": (t850, "K"),
        "u500": (1.25 * fields["u850"].values + 3.0, "m/s"),
        "v500": (1.15 * fields["v850"].values, "m/s"),
        "w500": (0.02 * wave, "Pa/s"),
        "w850": (0.03 * wave, "Pa/s"),
        "sp": (
            101000.0 - 11.7 * fields["terrain_height"].values[None]
            + 500.0 * wave,
            "Pa",
        ),
    }
    for name, (values, units) in extras.items():
        fields[name] = (("time", "lat", "lon"), values.astype(np.float32))
        fields[name].attrs["units"] = units
    # A reasonable dry layer thickness, not an exact discretized PDE solution.
    fields["z500"] = (
        ("time", "lat", "lon"),
        (fields["z850"].values
         + 287.05 / 9.80665 * 0.5 * (t500 + t850) * np.log(850.0 / 500.0))
        .astype(np.float32),
    )
    fields["z500"].attrs["units"] = "m"
    raw = directory / "synthetic-pinn-information.nc"
    fields.to_netcdf(raw, engine="scipy")
    return prepare(archive, raw, directory / "pinn-information.npz", pinn=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory for disposable smoke artifacts")
    parser.add_argument("--expanded", action="store_true",
                        help="Check the actual A64/B512 profile with hidden width 512")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    archive, _ = synthetic_archive(output, count=320)
    information = synthetic_pinn_information(archive, output)
    dimensions = dict(manifold_dim=64, hidden_dim=512, context_dim=64,
                      experts=4, expert_latent_dim=512, gate_hidden_dim=160) if args.expanded else dict(
                          manifold_dim=4, hidden_dim=24, context_dim=8,
                          experts=2, expert_latent_dim=8, gate_hidden_dim=12)
    common = [
        "--archive", str(archive), "--information", str(information),
        "--mode", "enriched", "--profile", "process", "--batch-size", "2",
        "--members", "2", "--tau-steps", "1", "--history-steps", "6",
        "--history-stride", "1", "--max-windows", "2", "--window-stride", "8",
        "--seed", "7", "--curriculum-interval", "1", "--gradient-audit",
    ]
    architecture_options = [item for key, value in dimensions.items()
                            for item in ("--" + key.replace("_", "-"), str(value))]
    parent = None
    checkpoints = {}
    for stage, epochs in (("A", 7), ("B", 1), ("C", 1)):
        checkpoint = output / f"pinn-{stage}.pt"
        options = common + ["--output", str(checkpoint), "--stage", stage,
                            "--epochs", str(epochs)]
        if stage == "A":
            options += architecture_options + ["--pinn", "--pinn-levels", "500", "850",
                        "--pinn-warmup-epochs", "1", "--pinn-ramp-epochs", "2",
                        "--pinn-weight", "0.1"]
        else:
            options += ["--init", str(parent)]
        train_main(options)
        checkpoints[stage] = checkpoint
        parent = checkpoint

    model_a, state_a = load_checkpoint(checkpoints["A"])
    model_b, state_b = load_checkpoint(checkpoints["B"])
    model_c, state_c = load_checkpoint(checkpoints["C"])
    if state_a["config"] != state_b["config"] or state_a["config"] != state_c["config"]:
        raise AssertionError("B/C did not inherit the exact A architecture")
    trainable_b_prefixes = ("core.experts.", "core.gate.correction.", "core.history_encoder.")
    frozen_b_equal = all(
        torch.equal(value, state_b["model"][name])
        for name, value in state_a["model"].items()
        if not name.startswith(trainable_b_prefixes)
    )
    pinn_frozen_equal = all(
        torch.equal(value, model.pinn.state_dict()[name])
        for model in (model_b, model_c)
        for name, value in model_a.pinn.state_dict().items()
    )
    if not frozen_b_equal or not pinn_frozen_equal:
        raise AssertionError("Separated B/C training modified frozen A/PINN parameters")
    training = json.loads(checkpoints["A"].with_suffix(".metrics.json").read_text())
    if ([row["train"]["pinn_warmup"] for row in training] != [1.0] + [0.0] * 6
            or not np.allclose([row["train"]["pinn_weight"] for row in training],
                               [0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1])
            or [row["eligible_for_best"] for row in training] != [False] * 6 + [True]
            or state_a["best_epoch"] != 7):
        raise AssertionError("A warmup/ramp/curriculum did not reach eligible joint training")
    gradients = training[-1]["gradient_first_batch"]
    gradient_groups = ("encoder", "decoder", "drift", "information", "info_decoder", "pinn")
    pinn_gradients = {name: gradients[f"gradient/pinn_total/{name}"] for name in gradient_groups}
    if not all(np.isfinite(value) and value > 0 for value in pinn_gradients.values()):
        raise AssertionError("Joint PINN gradient failed to reach an A component")
    report = evaluate(
        checkpoints["C"], archive, output / "heldout.json",
        information=information, max_cases=2, members=2, tau_steps=1,
        forecast_output=output / "forecast.npz",
    )
    summary = {
        "scope": "Synthetic integration smoke only; no ERA5 or forecast-skill claim",
        "profile": "expanded_a64_b512" if args.expanded else "tiny",
        "model_config": state_a["config"],
        "parameter_count": sum(parameter.numel() for parameter in model_a.parameters()),
        "parent_config_inherited": True,
        "epochs": {"A": 7, "B": 1, "C": 1},
        "pinn_config": state_a["pinn_config"],
        "members": 2,
        "physical_steps": 20,
        "training_windows_per_epoch": 2,
        "frozen_A_after_B": frozen_b_equal,
        "frozen_PINN_after_B_and_C": pinn_frozen_equal,
        "warmup_ramp_curriculum_verified": True,
        "pinn_gradient_norms_at_joint_epoch": pinn_gradients,
        "heldout": report["aggregate"],
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
