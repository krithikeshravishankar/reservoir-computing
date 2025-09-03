"""Error evaluation utilities for reservoir computing forecasts.
Accepts outputs from run_reservoir_forecast.run (results dict with 'forecast').
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np

__all__ = [
    "trajectory_shortest_distance",
    "compute_errors",
    "evaluate_results",
]


def trajectory_shortest_distance(
    true: np.ndarray,
    pred: np.ndarray,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """Approximate shortest-average distance between two trajectories.
    Randomly sample n_samples points from pred; for each, find nearest point along true; average distances.
    Shapes: true, pred -> (dims, T).
    Returns a single float (np.nan if no samples)."""
    T_pred = pred.shape[1]
    if T_pred == 0:
        return float("nan")
    n_eff = min(n_samples, T_pred)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(T_pred, size=n_eff, replace=False)
    total = 0.0
    for idx in idxs:
        p = pred[:, idx][:, None]  # (dims,1)
        dists = np.linalg.norm(true - p, axis=0)  # (T_true,)
        total += float(dists.min())
    return total / n_eff


def compute_errors(
    true: np.ndarray,
    pred: np.ndarray,
    times: Optional[np.ndarray] = None,
    traj_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute MSE/NRMSE per-dimension and totals, trajectory distance, and divergence time.

    Inputs:
    - true, pred: arrays shaped (dims, T)
    - times: optional time array of shape (T,)
    - traj_samples: number of predicted points to sample for trajectory distance

    Returns keys:
    - mse_dim, nrmse_dim, mse_total, nrmse_total
    - trajectory_distance
    - divergence_time (float or None), divergence_threshold (float)
    """
    assert true.shape == pred.shape, "true and pred must have the same shape (dims, T)"
    diff = pred - true
    mse_dim = np.mean(diff**2, axis=1)
    var_dim = np.var(true, axis=1)
    # Guard against zero variance
    with np.errstate(divide="ignore", invalid="ignore"):
        nrmse_dim = np.sqrt(np.where(var_dim > 0, mse_dim / var_dim, np.nan))
    mse_total = float(np.mean(diff**2))
    var_total = float(np.var(true))
    nrmse_total = float(np.sqrt(mse_total / var_total)) if var_total > 0 else float("nan")

    # Trajectory distance
    traj_dist = float(trajectory_shortest_distance(true, pred, n_samples=traj_samples, seed=seed))

    # Divergence time: first time L2 error exceeds 5% of norm of initial true state
    base_norm = float(np.linalg.norm(true[:, 0]))
    threshold = 0.05 * base_norm
    err_time_series = np.linalg.norm(diff, axis=0)
    div_index = None
    for i, val in enumerate(err_time_series):
        if val > threshold:
            div_index = i
            break
    if times is None:
        divergence_time = float(div_index) if div_index is not None else None
    else:
        divergence_time = float(times[div_index]) if div_index is not None else None

    return {
        "mse_dim": mse_dim,
        "nrmse_dim": nrmse_dim,
        "mse_total": mse_total,
        "nrmse_total": nrmse_total,
        "trajectory_distance": traj_dist,
        "divergence_time": divergence_time,
        "divergence_threshold": threshold,
    }


def evaluate_results(results: Dict[str, Dict], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Convenience wrapper to compute errors from a run() results dict.
    Expects results["forecast"] with keys: true, pred, times."""
    fc = results["forecast"]
    errs = compute_errors(fc["true"], fc["pred"], times=fc.get("times"))
    if verbose:
        print("MSE dim:", errs["mse_dim"]) 
        print("NRMSE dim:", errs["nrmse_dim"]) 
        print("Total MSE:", errs["mse_total"]) 
        print("Total NRMSE:", errs["nrmse_total"]) 
        print("Trajectory distance:", errs["trajectory_distance"]) 
        print("Divergence threshold:", errs["divergence_threshold"]) 
        print("Divergence time:", errs["divergence_time"]) 
    return errs
