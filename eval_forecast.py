"""Thin wrappers to evaluate forecast dictionaries produced by run_reservoir_forecast.run.
Re-exports core metrics and provides evaluate_forecast_results(forecast,...).
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from eval_metrics import (
    trajectory_shortest_distance,
    compute_errors,
)

__all__ = [
    "trajectory_shortest_distance",
    "compute_errors",
    "evaluate_forecast_results",
]


def evaluate_forecast_results(
    forecast: Dict[str, np.ndarray],
    verbose: bool = True,
):
    """Evaluate a single forecast dict with keys: 'true', 'pred', 'times'."""
    errs = compute_errors(forecast["true"], forecast["pred"], times=forecast.get("times"))
    if verbose:
        print("MSE dim:", errs["mse_dim"]) 
        print("NRMSE dim:", errs["nrmse_dim"]) 
        print("Total MSE:", errs["mse_total"]) 
        print("Total NRMSE:", errs["nrmse_total"]) 
        print("Trajectory distance:", errs["trajectory_distance"]) 
        print("Divergence threshold:", errs["divergence_threshold"]) 
        print("Divergence time:", errs["divergence_time"]) 
    return errs
