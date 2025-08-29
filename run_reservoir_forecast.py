"""
Minimal reservoir computing runner: train + forecast only.
Outputs include only: config, system_params, forecast.
No error computation. No plotting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Tuple, Optional
import json
import os

import numpy as np
import scipy.sparse as sp

# -----------------------------
# Dynamical systems
# -----------------------------

def sprott_linz_deriv(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    a = params.get("a", 0.42)
    x_, y_, z_ = x
    return np.array([y_ + z_, -x_ + a * y_, x_**2 - z_])

def lorenz_deriv(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    sigma = params.get("sigma", 10.0)
    rho = params.get("rho", 28.0)
    beta = params.get("beta", 8.0 / 3.0)
    x_, y_, z_ = x
    return np.array([sigma * (y_ - x_), x_ * (rho - z_) - y_, x_ * y_ - beta * z_])

def get_system(system_name: str) -> Tuple[Callable[[np.ndarray, Dict[str, float]], np.ndarray], Dict[str, float]]:
    name = system_name.lower()
    if name in ["sprott-linz", "sprott_linz", "sprott"]:
        return sprott_linz_deriv, {"a": 0.42}
    if name in ["lorenz", "lorenz63"]:
        return lorenz_deriv, {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    raise ValueError(f"Unknown system '{system_name}'. Use 'sprott-linz' or 'lorenz'.")

# -----------------------------
# Integrators
# -----------------------------

def euler_step(f: Callable[[np.ndarray, Dict[str, float]], np.ndarray], x: np.ndarray, dt: float, params: Dict[str, float]) -> np.ndarray:
    return x + dt * f(x, params)

def rk4_step(f: Callable[[np.ndarray, Dict[str, float]], np.ndarray], x: np.ndarray, dt: float, params: Dict[str, float]) -> np.ndarray:
    k1 = f(x, params)
    k2 = f(x + 0.5 * dt * k1, params)
    k3 = f(x + 0.5 * dt * k2, params)
    k4 = f(x + dt * k3, params)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def get_stepper(method: str):
    method = method.lower()
    if method == "euler":
        return euler_step
    if method in ["rk4", "runge-kutta", "rk"]:
        return rk4_step
    raise ValueError("Integrator must be 'euler' or 'rk4'.")

# -----------------------------
# Reservoir
# -----------------------------

@dataclass
class Reservoir:
    A: sp.csr_matrix
    Win: np.ndarray
    bias: float

def build_reservoir(
    n: int,
    m: int,
    spectral_radius: float = 1.2,
    density: float = 6.0,
    input_scale: float = 0.1,
    bias: float = 1.0,
    seed: Optional[int] = None,
) -> Reservoir:
    rng = np.random.default_rng(seed)
    A_rand = sp.random(n, n, density=density / n, format="csr", random_state=seed)
    A_data = A_rand.toarray()
    A_data = A_data - 0.5 * np.sign(A_data)
    A_temp = sp.csr_matrix(A_data)
    # Estimate spectral radius (largest-magnitude eigenvalue) robustly
    lam = None
    try:
        # Prefer largest magnitude (LM); allow more iterations and a looser tolerance
        eigvals = sp.linalg.eigs(
            A_temp, k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-4
        )
        lam = float(np.abs(eigvals[0]))
    except Exception:
        lam = None

    if lam is None or not np.isfinite(lam) or lam <= 0:
        # Fallback 1: simple power iteration on the sparse matrix
        v = rng.random(n) - 0.5
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            v = np.ones(n)
            v_norm = np.linalg.norm(v)
        v /= v_norm
        est = 0.0
        for _ in range(300):
            v = A_temp.dot(v)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                break
            v /= v_norm
            est = v_norm
        lam = est if est > 0 else None

    if lam is None or not np.isfinite(lam) or lam <= 0:
        # Fallback 2: dense eigenvalue computation (feasible for typical n ~ 500)
        try:
            lam = float(np.max(np.abs(np.linalg.eigvals(A_temp.toarray()))))
        except Exception:
            lam = 1.0  # last-resort safe default to avoid division by zero

    if lam <= 0 or not np.isfinite(lam):
        lam = 1.0
    A_scaled = A_temp * (spectral_radius / lam)
    Win = input_scale * (2.0 * rng.random((n, m)) - 1.0)
    return Reservoir(A=A_scaled.tocsr(), Win=Win, bias=bias)

# -----------------------------
# Training and Forecast
# -----------------------------

def train_reservoir(
    deriv: Callable,
    params: Dict[str, float],
    reservoir: Reservoir,
    x0: np.ndarray,
    T: float,
    dt: float,
    lam: float = 1e-4,
    washout: int = 1000,
    integrator: str = "rk4",
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    stepper = get_stepper(integrator)
    n_steps = int(T / dt)
    m = len(x0)
    n = reservoir.A.shape[0]
    rng = np.random.default_rng(seed)
    R = np.zeros((n, n_steps))
    F = np.zeros((m, n_steps))
    r = 2.0 * rng.random(n) - 1.0
    x = x0.copy()
    for t in range(n_steps):
        u = x.copy()
        R[:, t] = r
        F[:, t] = u
        x = stepper(deriv, x, dt, params)
        r = np.tanh(reservoir.A.dot(r) + reservoir.Win.dot(u) + reservoir.bias)
    if washout >= n_steps:
        raise ValueError("Washout >= total steps; reduce washout or increase T.")
    R_eff = R[:, washout:]
    F_eff = F[:, washout:]
    Id = np.eye(n)
    Wout = F_eff.dot(R_eff.T).dot(np.linalg.inv(R_eff.dot(R_eff.T) + lam * Id))
    return {"Wout": Wout, "x_last": x, "r_last": r}

def forecast(
    deriv: Callable,
    params: Dict[str, float],
    reservoir: Reservoir,
    Wout: np.ndarray,
    x_start: np.ndarray,
    r_start: np.ndarray,
    T: float,
    dt: float,
    integrator_truth: str = "rk4",
) -> Dict[str, np.ndarray]:
    stepper_truth = get_stepper(integrator_truth)
    n_steps = int(T / dt)
    m = len(x_start)
    true_states = np.zeros((m, n_steps))
    pred_states = np.zeros((m, n_steps))
    x_true = x_start.copy()
    x_pred = x_start.copy()
    r = r_start.copy()
    for t in range(n_steps):
        true_states[:, t] = x_true
        pred_states[:, t] = x_pred
        x_true = stepper_truth(deriv, x_true, dt, params)
        r = np.tanh(reservoir.A.dot(r) + reservoir.Win.dot(x_pred) + reservoir.bias)
        x_pred = Wout.dot(r)
    times = np.linspace(0, T, n_steps)
    return {"true": true_states, "pred": pred_states, "times": times}

# -----------------------------
# Config, Run, Save
# -----------------------------

@dataclass
class Config:
    system_name: str = "sprott-linz"
    system_params: Optional[Dict[str, float]] = None
    training_time: float = 300.0
    dt: float = 0.002
    reservoir_size: int = 500
    spectral_radius: float = 1.2
    density: float = 6.0
    input_scale: float = 0.1
    bias: float = 1.0
    lam: float = 1e-4
    washout: int = 1000
    truth_integrator: str = "rk4"
    training_integrator: str = "rk4"
    forecast_time: float = 50.0
    seed: int = 42

def run(config: Config) -> Dict[str, Dict]:
    deriv, default_params = get_system(config.system_name)
    if config.system_params:
        default_params.update(config.system_params)
    params = default_params

    reservoir = build_reservoir(
        n=config.reservoir_size,
        m=3,
        spectral_radius=config.spectral_radius,
        density=config.density,
        input_scale=config.input_scale,
        bias=config.bias,
        seed=config.seed,
    )

    rng = np.random.default_rng(config.seed)
    x0 = rng.random(3) - 0.5

    tr = train_reservoir(
        deriv,
        params,
        reservoir,
        x0,
        T=config.training_time,
        dt=config.dt,
        lam=config.lam,
        washout=config.washout,
        integrator=config.training_integrator,
        seed=config.seed,
    )

    fc = forecast(
        deriv,
        params,
        reservoir,
        tr["Wout"],
        x_start=tr["x_last"],
        r_start=tr["r_last"],
        T=config.forecast_time,
        dt=config.dt,
        integrator_truth=config.truth_integrator,
    )

    return {
        "config": asdict(config),
        "system_params": params,
        "forecast": fc,
    }

def save_results(base_path: str, results: Dict[str, Dict]) -> None:
    """Save only config, system_params and forecast.
    Writes two files:
    - base_path + '.npz' with arrays: true, pred, times
    - base_path + '.json' with config and system_params
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fc = results["forecast"]
    np.savez(
        base_path + ".npz",
        true=fc["true"],
        pred=fc["pred"],
        times=fc["times"],
    )
    meta = {"config": results["config"], "system_params": results["system_params"]}
    with open(base_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    # Example minimal run; adapt or add CLI parsing as needed.
    cfg = Config(system_name="sprott-linz", forecast_time=20.0, training_time=200.0, seed=42)
    out = run(cfg)
    # Uncomment to save
    # save_results("./outputs/example_run", out)
    # Print a short confirmation
    print({k: (list(v.keys()) if isinstance(v, dict) else type(v)) for k, v in out.items()})
