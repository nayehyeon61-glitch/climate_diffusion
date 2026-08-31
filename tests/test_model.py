import numpy as np
import torch

from climate_diffusion.config import FlowModelConfig
from climate_diffusion.model import MonthlyLatentFlow


def test_flow_matching_loss_and_monthly_sampling_are_trainable():
    torch.manual_seed(3)
    model = MonthlyLatentFlow(
        FlowModelConfig(
            state_dim=12,
            history_months=3,
            latent_dim=4,
            hidden_dim=16,
            time_embedding_dim=8,
        )
    )
    history = torch.randn(5, 3, 12)
    target = torch.randn(5, 12)
    losses = model.loss(history, target)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    sample = model.sample(history[:2], integration_steps=3)
    assert sample.shape == (2, 12)
    assert np.isfinite(sample.detach().numpy()).all()

