"""Manual/runner contracts: parse real CLIs without starting long training."""
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from inspect_joint_run import inspect
from prepare_temporal_120h import main as prepare_main
from smoke_moe import synthetic_archive


def dry(phase, tmp_path, **extra):
    env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable,
           "JOINT_DEVICE": "cpu", **extra}
    run = tmp_path / "new run with spaces"
    archive = tmp_path / "archive with spaces.npz"
    result = subprocess.run(["bash", "scripts/run_joint_ab_120h.sh", phase,
                             str(archive), str(run)], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert not run.exists()
    return result, [shlex.split(line) for line in result.stdout.splitlines()], archive, run


def parse_only(monkeypatch, command):
    """Use each module's actual ArgumentParser, stop before any side effect."""
    import argparse
    if command[1] == "-m":
        module = importlib.import_module(command[2])
    else:
        module = importlib.import_module(Path(command[1]).stem)
    captured = {}
    original = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def capture(parser, args=None, namespace=None):
        captured.update(vars(original(parser, args, namespace)))
        raise Parsed

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", capture)
        argv = command[3:] if command[1] == "-m" else command[2:]
        m.setattr(sys, "argv", [command[2] if command[1] == "-m" else command[1], *argv])
        with pytest.raises(Parsed):
            module.main()
    return captured


@pytest.mark.parametrize("phase", ["prepare", "warmup", "AB", "C", "validation", "forecast", "render", "test"])
def test_every_runner_phase_parses_actual_cli_without_writes(phase, tmp_path, monkeypatch):
    result, commands, archive, run = dry(phase, tmp_path, JOINT_TEST_CONFIRMED="1")
    assert result.returncode == 0, result.stderr
    assert commands
    for command in commands:
        parsed = parse_only(monkeypatch, command)
        if "archive" in parsed:
            assert parsed["archive"] == str(archive)
    assert list(tmp_path.iterdir()) == []


def test_warmup_and_ab_and_c_have_different_explicit_contracts(tmp_path, monkeypatch):
    _, commands, _, _ = dry("warmup", tmp_path)
    a = parse_only(monkeypatch, commands[0])
    assert (a["horizon_steps"], a["history_steps"], a["history_stride"]) == (20, 6, 4)
    assert a["forecast_dynamics"] == "recurrent_residual"
    assert a["ae_delta_weight"] > 0 and a["finite_step_drift_weight"] > 0
    _, commands, _, _ = dry("AB", tmp_path)
    ab = parse_only(monkeypatch, commands[0])
    assert ab["stage"] == "joint_ab" and ab["loss_profile"] == "v2_minimal"
    assert not ab["log_gradient_norms"]  # known upstream runtime defect; not repaired here
    assert ab["trajectory_edges"] == 0 and ab["delta_member_weight"] == 0
    _, commands, _, _ = dry("C", tmp_path)
    c = parse_only(monkeypatch, commands[0])
    assert c["stage"] == "joint" and "--loss-profile" not in commands[0]
    assert c["delta_weight"] > 0 and c["trajectory_weight"] > 0
    assert c["delta_member_weight"] == 0 and c["temporal_warmup_epochs"] == 5
    assert c["joint_lr_factor"] == c["encoder_lr_factor"] == .1


def test_one_forecast_two_prefix_renders_same_members(tmp_path, monkeypatch):
    _, commands, _, run = dry("forecast", tmp_path, JOINT_ORIGIN="2009-05-19T06:00:00Z")
    assert len(commands) == 1
    forecast = parse_only(monkeypatch, commands[0])
    assert forecast["forecast_steps"] == 20 and forecast["forecast_only"]
    _, commands, _, _ = dry("render", tmp_path)
    render = [parse_only(monkeypatch, command) for command in commands]
    assert [v["interval_hours"] for v in render] == [6, 12]
    assert all(v["forecast"] == forecast["forecast_output"] for v in render)
    assert all(v["members"] is None and v["horizon_hours"] == 120 for v in render)
    assert all("--checkpoint" not in c for c in commands)


def test_test_phase_requires_explicit_frozen_decision(tmp_path):
    result, _, _, _ = dry("test", tmp_path, JOINT_TEST_CONFIRMED="0")
    assert result.returncode != 0 and "Freeze choices" in result.stderr


def test_actual_preflight_calendar_shapes_hash_and_no_mutation(tmp_path):
    archive, _ = synthetic_archive(tmp_path, count=320)
    plan_path = tmp_path / "preflight.json"
    assert prepare_main(["--archive", str(archive), "--output", str(plan_path)]) == 0
    plan_before = plan_path.read_bytes()
    result = inspect(archive, plan_path)
    plan = json.loads(plan_before)
    assert result["state_dim"] == 4 * 4 * 8
    assert result["forecast_hours"] == 120 and result["history_hours"] == 120
    assert result["sample_shapes"]["trajectory_raw"] == [21, 128]
    assert result["sample_shapes"]["dt_hours"] == [20]
    assert result["train_unique_pairs"] == plan["normalization_span"][1] - 1
    previous_end = None
    for row in result["splits"].values():
        first, last = (np.datetime64(v.removesuffix("Z")) for v in row["future_targets_utc"])
        origin_last = np.datetime64(row["origins_utc"][-1].removesuffix("Z"))
        assert last - origin_last == np.timedelta64(120, "h")
        if previous_end is not None:
            assert first > previous_end
        previous_end = last
    assert plan_path.read_bytes() == plan_before
    # Tiny real warmup verifies the helper's checkpoint/manifest branch at H20.
    # This is neither AB/C training nor an ERA5 skill experiment.
    from climate_diffusion.train_manifold_moe import train_manifold_moe
    checkpoint = train_manifold_moe(archive, tmp_path / "warmup.pt", stage="manifold",
        model_options={"history_steps":6, "history_stride":4, "horizon_steps":20,
                       "forecast_dynamics":"recurrent_residual", "manifold_dim":3,
                       "hidden_dim":12, "context_dim":4, "num_experts":2,
                       "gate_hidden_dim":8, "expert_latent_dim":6},
        manifold_epochs=1, batch_size=4, window_stride=16,
        max_validation_windows=2, device="cpu", ae_delta_weight=.05,
        finite_step_drift_weight=.05)
    audited = inspect(archive, plan_path, checkpoint)["checkpoint"]
    assert audited["stage"] == "manifold" and audited["best_epoch"] == 1
    assert not audited["has_optimizer_state"]
    plan["archive_sha256"] = "wrong-hash"
    bad_plan = tmp_path / "bad.json"
    bad_plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Archive/schema changed"):
        inspect(archive, bad_plan)
