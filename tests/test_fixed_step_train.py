import json
import numpy as np
import torch

from climate_diffusion.fixed_step_train import train_runpod_fixed_step_flow
from climate_diffusion.inference import LatentFlowForecaster


def test_fixed_step_trainer_persists_360h_contract_and_resumes(tmp_path):
    archive = tmp_path / "fixed.npz"
    times = np.datetime64("2020-01-01T00:00:00") + np.arange(14) * np.timedelta64(360, "h")
    states = np.stack((np.linspace(0, 1, 14), np.linspace(1, 2, 14)), axis=1).astype(np.float32)
    np.savez_compressed(archive, states=states, observed_mask=np.ones_like(states, dtype=np.float32), times=times)
    schema = {
        "format": "climate_diffusion.fixed_step_state.v1",
        "state_dim": 2,
        "forecast_step_hours": 360,
        "variables": [{"name": "msl", "dims": ["lat", "lon"], "shape": [1, 2], "slice": [0, 2], "coords": {"lat": [30.0], "lon": [120.0, 140.0]}, "attrs": {}}],
    }
    archive.with_suffix(".schema.json").write_text(json.dumps(schema))
    root = tmp_path / "checkpoints"
    best = train_runpod_fixed_step_flow(archive, root, history_steps=2, epochs=1, batch_size=2, validation_fraction=.2, test_fraction=.2, purge_windows=0, latent_dim=2, hidden_dim=8)
    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["training"]["forecast_step_hours"] == 360
    assert json.loads((root / "best.manifest.json").read_text())["forecast_step_hours"] == 360
    forecaster = LatentFlowForecaster(best, device="cpu")
    assert forecaster.forecast_step_hours == 360
    assert all(not p.requires_grad for p in forecaster.model.parameters())
    resumed = train_runpod_fixed_step_flow(archive, root, history_steps=2, epochs=2, batch_size=2, validation_fraction=.2, test_fraction=.2, purge_windows=0, latent_dim=2, hidden_dim=8)
    latest = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed == best
    assert latest["training"]["epoch"] == 2
