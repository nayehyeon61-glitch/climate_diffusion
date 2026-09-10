import numpy as np
import torch

from climate_diffusion.config import FlowLossConfig, FlowModelConfig
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


def test_expanded_autoencoder_adds_capacity_without_changing_the_default():
    torch.manual_seed(5)
    legacy = MonthlyLatentFlow(FlowModelConfig(state_dim=12, latent_dim=4, hidden_dim=16))
    expanded = MonthlyLatentFlow(
        FlowModelConfig(
            state_dim=12,
            latent_dim=8,
            hidden_dim=16,
            autoencoder_hidden_dim=32,
            autoencoder_blocks=2,
            autoencoder_dropout=0.1,
        )
    )
    # The default still builds the original three-layer MLP, so earlier
    # checkpoints keep loading into it.
    assert [tuple(p.shape) for p in legacy.autoencoder.encoder.parameters()] == [
        (16, 12), (16,), (16,), (16,), (16, 16), (16,), (4, 16), (4,)
    ]
    legacy_size = sum(p.numel() for p in legacy.autoencoder.parameters())
    expanded_size = sum(p.numel() for p in expanded.autoencoder.parameters())
    assert expanded_size > 4 * legacy_size

    history = torch.randn(5, 6, 12)
    target = torch.randn(5, 12)
    losses = expanded.loss(history, target)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(
        parameter.grad is not None
        for parameter in expanded.autoencoder.parameters()
    )
    expanded.eval()
    assert expanded.sample(history[:2], integration_steps=2).shape == (2, 12)


def test_latent_normalization_puts_the_latent_on_the_flow_prior_scale():
    torch.manual_seed(11)
    config = FlowModelConfig(
        state_dim=12, latent_dim=6, hidden_dim=16, latent_normalization=True
    )
    model = MonthlyLatentFlow(config)
    # Force a decoder that shrinks its codes, the failure the buffer exists for.
    with torch.no_grad():
        for layer in model.autoencoder.encoder:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(0.05)
                layer.bias.mul_(0.05)
    history = torch.randn(64, 6, 12)
    target = torch.randn(64, 12)
    model.train()
    for _ in range(400):
        model.loss(history, target)
    raw = model.autoencoder.encode(target)
    normalized = model.encode_latent(target)
    assert float(raw.std()) < 0.3
    assert 0.7 < float(normalized.std()) < 1.4

    # Rescaling is undone on the way out, so reconstruction is unaffected.
    model.eval()
    round_trip = model.decode_latent(model.encode_latent(target))
    assert torch.allclose(round_trip, model.autoencoder(target), atol=1e-5)


def test_ensemble_crps_term_trains_and_is_fair():
    from climate_diffusion.model import fair_ensemble_crps

    # A perfect deterministic ensemble scores zero under the fair estimator.
    target = torch.randn(4, 12)
    identical = target[:, None, :].repeat(1, 5, 1)
    assert abs(float(fair_ensemble_crps(identical, target))) < 1e-6

    torch.manual_seed(13)
    model = MonthlyLatentFlow(
        FlowModelConfig(state_dim=12, latent_dim=4, hidden_dim=16)
    )
    losses = model.loss(
        torch.randn(5, 6, 12),
        torch.randn(5, 12),
        FlowLossConfig(ensemble_weight=1.0, ensemble_size=4, ensemble_steps=2),
    )
    assert "ensemble_crps" in losses and "ensemble_spread" in losses
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.vector_field.network.parameters()
    )


def test_torchdiffeq_solvers_match_the_builtin_midpoint_integrator():
    torch.manual_seed(0)
    base = dict(state_dim=64, latent_dim=8, hidden_dim=32, history_months=3)
    reference = MonthlyLatentFlow(FlowModelConfig(**base)).eval()
    history = torch.randn(4, 3, 64)
    with torch.no_grad():
        expected = reference.sample(
            history, integration_steps=64, generator=torch.Generator().manual_seed(1)
        )
    for solver in ("rk4", "dopri5"):
        model = MonthlyLatentFlow(FlowModelConfig(**base, flow_solver=solver))
        model.load_state_dict(reference.state_dict())
        model.eval()
        with torch.no_grad():
            actual = model.sample(
                history, integration_steps=64, generator=torch.Generator().manual_seed(1)
            )
        # Same ODE, different quadrature: agreement is a solver-accuracy check.
        assert torch.allclose(expected, actual, atol=2e-3), solver


def test_adjoint_backprops_through_a_rollout_longer_than_unrolling_allows():
    torch.manual_seed(0)
    model = MonthlyLatentFlow(
        FlowModelConfig(
            state_dim=64, latent_dim=8, hidden_dim=32, history_months=3,
            flow_solver="rk4", flow_adjoint=True,
        )
    )
    losses = model.loss(
        torch.randn(4, 3, 64),
        torch.randn(4, 64),
        FlowLossConfig(ensemble_weight=1.0, ensemble_size=4, ensemble_steps=32),
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(
        parameter.grad is not None
        for parameter in model.vector_field.network.parameters()
    )
