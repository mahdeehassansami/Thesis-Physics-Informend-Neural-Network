from __future__ import annotations

import os
import random

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np

from thesis_work.config import (
    CNN_BASELINE_NAME,
    DATA_BASELINE_NAME,
    CNN_SEED,
    DATA_BASELINE_SEED,
    EKIN_COL,
    FEATURE_COLS_MULTI,
    LSTM_SEED,
    LSTM_BASELINE_NAME,
    PINN_SEED,
    PROPOSED_MODEL_NAME,
    TARGET_COL,
    TIME_COL,
)
from thesis_work.metrics import evaluation_row, regression_metrics

PINN_MONOTONIC_WEIGHT = 0.005
PINN_SPECTRAL_PRIOR_WEIGHT = 0.001
PINN_MONOTONIC_TOLERANCE = 0.01
PINN_SPECTRAL_PRIOR_STRENGTH = 0.15
PINN_SPECTRAL_PRIOR_MARGIN = 0.05


def _deepxde_and_torch():
    import deepxde as dde
    import torch

    return dde, torch


def _torch_nn():
    import torch
    import torch.nn as nn

    return torch, nn


def seed_python_numpy(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_deepxde(dde, torch, seed: int) -> None:
    seed_python_numpy(seed)
    if hasattr(dde.config, "set_random_seed"):
        dde.config.set_random_seed(seed)
    seed_torch(torch, seed)


def predict_dde_model(model, x: np.ndarray) -> np.ndarray:
    pred = model.predict(x)
    return np.clip(pred, 0, 1).ravel()


def train_data_only_baseline(ctx: dict[str, object], iterations: int, seed: int):
    dde, torch = _deepxde_and_torch()
    seed_deepxde(dde, torch, seed)
    data = dde.data.DataSet(
        X_train=ctx["X_train"],
        y_train=ctx["y_train"],
        X_test=ctx["X_val"],
        y_test=ctx["y_val"],
    )
    net = dde.nn.FNN([ctx["X_train"].shape[1], 64, 64, 64, 1], "tanh", "Glorot uniform")
    net.apply_output_transform(lambda x, y: torch.sigmoid(y))
    model = dde.Model(data, net)
    model.compile("adam", lr=5e-4, loss="MSE")
    model.train(iterations=iterations, display_every=max(1, min(1000, iterations)))
    return model


def weak_fault_frequency_physics_residual(x, y):
    dde, torch = _deepxde_and_torch()
    d_r_dt = dde.grad.jacobian(y, x, i=0, j=TIME_COL)
    ekin = x[:, EKIN_COL : EKIN_COL + 1]
    damage = 1.0 - y
    monotonicity_residual = torch.relu(d_r_dt - PINN_MONOTONIC_TOLERANCE)
    spectral_damage_floor = PINN_SPECTRAL_PRIOR_STRENGTH * ekin
    spectral_prior_residual = torch.relu(
        spectral_damage_floor - damage - PINN_SPECTRAL_PRIOR_MARGIN
    )
    return [monotonicity_residual, spectral_prior_residual]


def train_proposed_deepxde_model(ctx: dict[str, object], iterations: int, seed: int):
    dde, torch = _deepxde_and_torch()
    seed_deepxde(dde, torch, seed)
    geom = dde.geometry.PointCloud(ctx["X_train"])
    observe_rul = dde.icbc.PointSetBC(ctx["X_train"], ctx["y_train"], component=0)
    data = dde.data.PDE(
        geom,
        weak_fault_frequency_physics_residual,
        [observe_rul],
        num_domain=len(ctx["X_train"]),
        num_boundary=0,
        anchors=ctx["X_train"],
        num_test=None,
    )
    net = dde.nn.FNN([ctx["X_train"].shape[1], 64, 64, 64, 1], "tanh", "Glorot uniform")
    net.apply_output_transform(lambda x, y: torch.sigmoid(y))
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=5e-4,
        loss_weights=[PINN_MONOTONIC_WEIGHT, PINN_SPECTRAL_PRIOR_WEIGHT, 1.0],
    )
    model.train(iterations=iterations, display_every=max(1, min(1000, iterations)))
    return model


def make_sequence_arrays(
    df,
    scaler,
    run_ids: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_seq, y_seq, x_axis = [], [], []
    for run_id in run_ids:
        run = df[df["run_id"] == run_id].sort_values(["elapsed_scaled", "snapshot_index"]).copy()
        if len(run) <= sequence_length:
            continue
        x_run = scaler.transform(run[FEATURE_COLS_MULTI]).astype("float32")
        x_run[:, 1:] = np.clip(x_run[:, 1:], 0, 1)
        y_run = run[[TARGET_COL]].values.astype("float32")
        x_run_axis = np.linspace(0, 1, len(run), dtype=np.float32)
        for i in range(sequence_length, len(run)):
            x_seq.append(x_run[i - sequence_length : i])
            y_seq.append(y_run[i])
            x_axis.append(x_run_axis[i])
    if not x_seq:
        raise ValueError("Not enough samples to build sequence arrays.")
    return (
        np.stack(x_seq).astype("float32"),
        np.vstack(y_seq).astype("float32"),
        np.asarray(x_axis, dtype="float32"),
    )


def build_lstm_regressor(input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
    torch, nn = _torch_nn()

    class LSTMRULRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return LSTMRULRegressor()


def build_cnn_regressor(input_dim: int, hidden_dim: int = 64):
    torch, nn = _torch_nn()

    class CNNRULRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x.permute(0, 2, 1)
            return self.head(self.encoder(x))

    return CNNRULRegressor()


def predict_torch_sequence_model(model, x: np.ndarray, device, batch_size: int) -> np.ndarray:
    torch, _ = _torch_nn()
    model.eval()
    preds = []
    x_tensor = torch.from_numpy(x)
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            batch_x = x_tensor[start : start + batch_size].to(device)
            preds.append(model(batch_x).cpu().numpy())
    return np.clip(np.vstack(preds).reshape(-1), 0, 1)


def train_torch_sequence_model(
    model_builder,
    ctx: dict[str, object],
    spec: dict[str, object],
    model_name: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    sequence_length: int,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    torch, nn = _torch_nn()
    seed_python_numpy(seed)
    seed_torch(torch, seed)

    x_train, y_train, _ = make_sequence_arrays(
        ctx["train_df"], ctx["scaler"], spec["train_runs"], sequence_length
    )
    x_val, y_val, _ = make_sequence_arrays(
        ctx["val_df"], ctx["scaler"], spec["validation_runs"], sequence_length
    )
    x_test, y_test, x_test_axis = make_sequence_arrays(
        ctx["test_df"], ctx["scaler"], spec["test_runs"], sequence_length
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_builder(input_dim=x_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    n_train = len(x_train_tensor)
    best_loss = np.inf
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_x = x_train_tensor[batch_idx].to(device)
            batch_y = y_train_tensor[batch_idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        val_pred = predict_torch_sequence_model(model, x_val, device, batch_size)
        monitor_loss = float(np.mean((y_val.reshape(-1) - val_pred.reshape(-1)) ** 2))
        if monitor_loss < best_loss - 1e-6:
            best_loss = monitor_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        if epoch % 10 == 0 or epoch == 1:
            print(f'{spec["name"]} {model_name} epoch {epoch:03d} monitor MSE = {monitor_loss:.6f}')
        if patience_left <= 0:
            print(f'{spec["name"]} {model_name} early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred = predict_torch_sequence_model(model, x_test, device, batch_size)
    return model, y_test.reshape(-1), test_pred, x_test_axis


def train_all_models(
    contexts: dict[str, dict[str, object]],
    experiments: list[dict[str, object]],
    baseline_iterations: int,
    pinn_iterations: int,
    sequence_epochs: int,
    sequence_patience: int,
    sequence_batch_size: int,
    sequence_length: int,
    data_baseline_seed: int = DATA_BASELINE_SEED,
    pinn_seed: int = PINN_SEED,
    lstm_seed: int = LSTM_SEED,
    cnn_seed: int = CNN_SEED,
) -> tuple[object, dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    import pandas as pd

    core_rows = []
    lstm_rows = []
    cnn_rows = []
    prediction_store: dict[str, dict[str, object]] = {}
    trained_models: dict[str, dict[str, object]] = {}

    for spec in experiments:
        ctx = contexts[spec["name"]]
        full_x = np.linspace(0, 1, len(ctx["y_test"].reshape(-1)))
        prediction_store[spec["name"]] = {
            "true": ctx["y_test"].reshape(-1),
            "x_true": full_x,
            "x_by_model": {},
        }
        trained_models[spec["name"]] = {}

        print(f'\n=== {spec["name"]} data-only baseline ===')
        baseline_model = train_data_only_baseline(ctx, baseline_iterations, seed=data_baseline_seed)
        baseline_pred = predict_dde_model(baseline_model, ctx["X_test"])
        print(f"{DATA_BASELINE_NAME}: {regression_metrics(ctx['y_test'], baseline_pred)}")
        core_rows.append(evaluation_row(spec, DATA_BASELINE_NAME, ctx["y_test"], baseline_pred))
        prediction_store[spec["name"]][DATA_BASELINE_NAME] = baseline_pred
        prediction_store[spec["name"]]["x_by_model"][DATA_BASELINE_NAME] = full_x
        trained_models[spec["name"]][DATA_BASELINE_NAME] = baseline_model

        print(f'\n=== {spec["name"]} proposed DeepXDE model ===')
        proposed_model = train_proposed_deepxde_model(ctx, pinn_iterations, seed=pinn_seed)
        proposed_pred = predict_dde_model(proposed_model, ctx["X_test"])
        print(f"{PROPOSED_MODEL_NAME}: {regression_metrics(ctx['y_test'], proposed_pred)}")
        core_rows.append(evaluation_row(spec, PROPOSED_MODEL_NAME, ctx["y_test"], proposed_pred))
        prediction_store[spec["name"]][PROPOSED_MODEL_NAME] = proposed_pred
        prediction_store[spec["name"]]["x_by_model"][PROPOSED_MODEL_NAME] = full_x
        trained_models[spec["name"]][PROPOSED_MODEL_NAME] = proposed_model

        for model_name, builder, seed, rows in [
            (LSTM_BASELINE_NAME, build_lstm_regressor, lstm_seed, lstm_rows),
            (CNN_BASELINE_NAME, build_cnn_regressor, cnn_seed, cnn_rows),
        ]:
            print(f'\n=== {spec["name"]} {model_name} ===')
            model, y_seq, pred_seq, x_seq = train_torch_sequence_model(
                builder,
                ctx,
                spec,
                model_name,
                seed=seed,
                epochs=sequence_epochs,
                patience=sequence_patience,
                batch_size=sequence_batch_size,
                sequence_length=sequence_length,
            )
            print(f"{model_name}: {regression_metrics(y_seq, pred_seq)}")
            rows.append(evaluation_row(spec, model_name, y_seq, pred_seq))
            prediction_store[spec["name"]][model_name] = pred_seq
            prediction_store[spec["name"]]["x_by_model"][model_name] = x_seq
            trained_models[spec["name"]][model_name] = model

    final_results = pd.concat(
        [pd.DataFrame(core_rows), pd.DataFrame(lstm_rows), pd.DataFrame(cnn_rows)],
        ignore_index=True,
    )
    return final_results, prediction_store, trained_models
