from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x)
        attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        hidden = self.norm1(hidden + attended)
        hidden = self.norm2(hidden + self.feed_forward(hidden))
        return hidden[:, -1, :]


class RULModel(nn.Module):
    model_kind = "data"

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FNNRUL(RULModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(torch.cat([x[:, -1, :], time], dim=1)))


class LSTMRUL(RULModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(x)
        return torch.sigmoid(
            self.head(torch.cat([hidden[:, -1, :], time], dim=1))
        )


class CNNRUL(RULModel):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return torch.sigmoid(self.head(torch.cat([hidden, time], dim=1)))


class AttentionRUL(RULModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TemporalAttentionEncoder(
            input_dim, hidden_dim, num_heads, dropout
        )
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict_from_hidden(
        self, hidden: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.rul_head(torch.cat([hidden, time], dim=1)))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.predict_from_hidden(self.encode(x), time)


class AttnPINNRUL(AttentionRUL):
    """Attention RUL model with the learned DeepHPM operator used by AttnPINN papers."""

    model_kind = "attnpinn"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(input_dim, hidden_dim, num_heads, dropout)
        operator_input_dim = hidden_dim + 3
        self.deep_hpm = nn.Sequential(
            nn.Linear(operator_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def operator_residual(
        self,
        hidden: torch.Tensor,
        time: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        dy_dt = torch.autograd.grad(
            prediction,
            time,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        dy_dh = torch.autograd.grad(
            prediction,
            hidden,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
            retain_graph=True,
        )[0]
        derivative_summary = torch.cat(
            [
                dy_dt,
                torch.linalg.vector_norm(dy_dh, dim=1, keepdim=True),
                torch.mean(dy_dh, dim=1, keepdim=True),
            ],
            dim=1,
        )
        learned_operator = self.deep_hpm(
            torch.cat([hidden, derivative_summary], dim=1)
        )
        return dy_dt - learned_operator


class WeakPhysicsPINNRUL(AttentionRUL):
    """PINN variant constrained by monotonic and condition-indicator priors."""

    model_kind = "weak_pinn"


class StrongPhysicsPINNRUL(AttentionRUL):
    """PINN variant constrained by explicit crack-growth and bearing-life laws."""

    model_kind = "strong_pinn"


MODEL_BUILDERS = {
    "fnn": FNNRUL,
    "cnn": CNNRUL,
    "lstm": LSTMRUL,
    "attnpinn": AttnPINNRUL,
    "weak_pinn": WeakPhysicsPINNRUL,
    "strong_pinn": StrongPhysicsPINNRUL,
}


@dataclass
class LossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _time_derivative(
    prediction: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    return torch.autograd.grad(
        prediction,
        time,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]


def weak_physics_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    health_indicator: torch.Tensor,
    x: torch.Tensor,
    meta: dict[str, torch.Tensor],
    feature_indices: dict[str, int],
) -> dict[str, torch.Tensor]:
    dy_dt = _time_derivative(prediction, time)
    damage = 1.0 - prediction
    components = {
        "data": F.mse_loss(prediction, target),
        "monotonic": torch.mean(F.relu(dy_dt) ** 2),
        "health_indicator": F.mse_loss(damage, health_indicator),
    }

    initial_mask = (target >= 0.95).float()
    terminal_mask = (target <= 0.05).float()
    boundary_count = initial_mask.sum() + terminal_mask.sum()
    if boundary_count > 0:
        boundary = (
            ((prediction - 1.0) ** 2 * initial_mask).sum()
            + (prediction**2 * terminal_mask).sum()
        ) / boundary_count.clamp_min(1.0)
    else:
        boundary = prediction.new_zeros(())
    components["boundary"] = boundary

    temperature_index = feature_indices["temperature_c"]
    temperature_slope = (
        x[:, -1, temperature_index : temperature_index + 1]
        - x[:, -2, temperature_index : temperature_index + 1]
    )
    temperature_mask = meta["temperature_available"]
    minimum_damage_rate = F.relu(temperature_slope)
    damage_rate = -dy_dt
    temperature_prior = (
        F.relu(minimum_damage_rate - damage_rate) ** 2 * temperature_mask
    )
    components["temperature_prior"] = temperature_prior.sum() / temperature_mask.sum().clamp_min(1.0)
    return components


def strong_physics_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
    health_indicator: torch.Tensor,
    x: torch.Tensor,
    meta: dict[str, torch.Tensor],
    feature_indices: dict[str, int],
    physics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    components = weak_physics_components(
        prediction,
        target,
        time,
        health_indicator,
        x,
        meta,
        feature_indices,
    )
    dy_dt = _time_derivative(prediction, time)
    damage_rate = F.relu(-dy_dt)
    damage = 1.0 - prediction

    crack = physics["crack_growth"]
    a0 = float(crack["initial_crack_m"])
    af = float(crack["critical_crack_m"])
    paris_c = float(crack["paris_coefficient"])
    paris_m = float(crack["paris_exponent"])
    crack_length = a0 + (af - a0) * damage
    crack_rate = torch.autograd.grad(
        crack_length,
        time,
        grad_outputs=torch.ones_like(crack_length),
        create_graph=True,
        retain_graph=True,
    )[0]

    temperature = meta["temperature_c"]
    thermo = physics["thermolubrication"]
    reference_temperature = float(thermo["reference_temperature_c"])
    beta = float(thermo["viscosity_temperature_beta"])
    viscosity_reference = meta["viscosity_ref_cst"].clamp_min(1e-6)
    viscosity_required = meta["viscosity_required_cst"].clamp_min(1e-6)
    viscosity = viscosity_reference * torch.exp(
        -beta * (temperature - reference_temperature)
    )
    kappa = viscosity / viscosity_required
    temperature_mask = meta['temperature_available']
    kappa = temperature_mask * kappa + (1.0 - temperature_mask)
    contamination = meta["contamination_factor"].clamp(0.05, 1.0)
    lubrication_modifier = (
        1.0
        + float(thermo["poor_lubrication_crack_multiplier"]) * F.relu(1.0 - kappa)
        + float(thermo["contamination_crack_multiplier"])
        * F.relu(1.0 - contamination)
    )
    temperature_mask = meta["temperature_available"]
    lubrication_modifier = (
        temperature_mask * lubrication_modifier + (1.0 - temperature_mask)
    )

    pressure = meta["contact_pressure_mpa"].clamp_min(1e-6)
    delta_k = pressure * torch.sqrt(crack_length.clamp_min(1e-12))
    paris_rate_per_cycle = paris_c * torch.pow(delta_k.clamp_min(1e-9), paris_m)
    paris_rate_per_life = (
        meta["cycles_per_time_unit"].clamp_min(1.0)
        * paris_rate_per_cycle
        * lubrication_modifier
    )
    crack_scale = max(af - a0, 1e-9)
    predicted_crack_rate = F.relu(crack_rate)
    crack_residual = F.smooth_l1_loss(
        torch.log1p(predicted_crack_rate / crack_scale),
        torch.log1p(paris_rate_per_life / crack_scale),
        reduction="none",
    )
    crack_mask = meta["contact_pressure_available"]
    components["paris_crack_growth"] = (
        crack_residual * crack_mask
    ).sum() / crack_mask.sum().clamp_min(1.0)

    bearing = physics["bearing_life"]
    load = meta["load_n"].clamp_min(1.0)
    capacity = meta["dynamic_capacity_n"].clamp_min(1.0)
    bearing_exponent = float(bearing["life_exponent"])
    askf = (
        float(bearing["askf_reference"])
        * torch.pow(kappa.clamp(0.05, 20.0), float(bearing["kappa_exponent"]))
        * torch.pow(
            contamination,
            float(bearing["contamination_exponent"]),
        )
    ).clamp(
        float(bearing["askf_minimum"]),
        float(bearing["askf_maximum"]),
    )
    life_cycles = (
        1e6
        * float(bearing["reliability_factor"])
        * askf
        * torch.pow(capacity / load, bearing_exponent)
    ).clamp_min(1.0)
    miner_rate = meta["cycles_per_time_unit"].clamp_min(1.0) / life_cycles
    miner_residual = F.smooth_l1_loss(
        torch.log1p(damage_rate),
        torch.log1p(miner_rate),
        reduction="none",
    )
    life_mask = (
        meta["load_available"]
        * (meta["dynamic_capacity_n"] > 0).float()
    )
    components["palmgren_miner"] = (
        miner_residual * life_mask
    ).sum() / life_mask.sum().clamp_min(1.0)

    components["crack_rate_positive"] = torch.mean(F.relu(-crack_rate) ** 2)
    return components


def calculate_loss(
    model: RULModel,
    batch: dict[str, Any],
    weights: dict[str, float],
    feature_indices: dict[str, int],
    physics: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    x = batch["x"]
    time = batch["time"].clone().detach().requires_grad_(True)
    target = batch["target"]
    health_indicator = batch["health_indicator"]
    meta = batch["meta"]

    if model.model_kind == "attnpinn":
        hidden = model.encode(x)
        prediction = model.predict_from_hidden(hidden, time)
        residual = model.operator_residual(hidden, time, prediction)
        components = {
            "data": F.mse_loss(prediction, target),
            "learned_operator": torch.mean(residual**2),
        }
    elif model.model_kind == "weak_pinn":
        prediction = model(x, time)
        components = weak_physics_components(
            prediction,
            target,
            time,
            health_indicator,
            x,
            meta,
            feature_indices,
        )
    elif model.model_kind == "strong_pinn":
        prediction = model(x, time)
        components = strong_physics_components(
            prediction,
            target,
            time,
            health_indicator,
            x,
            meta,
            feature_indices,
            physics,
        )
    else:
        prediction = model(x, time)
        components = {"data": F.mse_loss(prediction, target)}

    total = prediction.new_zeros(())
    for name, component in components.items():
        total = total + float(weights.get(name, 0.0)) * component
    return total, components, prediction


def build_model(
    name: str,
    input_dim: int,
    model_config: dict[str, Any],
) -> RULModel:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_BUILDERS)}")
    builder = MODEL_BUILDERS[name]
    kwargs = {
        "input_dim": input_dim,
        "hidden_dim": int(model_config.get("hidden_dim", 128)),
    }
    if name in {"attnpinn", "weak_pinn", "strong_pinn"}:
        kwargs.update(
            {
                "num_heads": int(model_config.get("num_heads", 4)),
                "dropout": float(model_config.get("dropout", 0.1)),
            }
        )
    return builder(**kwargs)
