"""Plotting utilities for reservoir computing forecasts.
Works with outputs from run_reservoir_forecast.run (results dict with 'forecast')
or with raw arrays (times, true, pred).
"""
from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "plot_forecast",
    "plot_error_over_time",
    "plot_all",
]


def plot_forecast(
    times: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    from_time: float = 0.0,
    to_time: Optional[float] = None,
) -> None:
    if to_time is None:
        to_time = float(times[-1])
    mask = (times >= from_time) & (times <= to_time)
    t_sel = times[mask]
    true_sel = true[:, mask]
    pred_sel = pred[:, mask]

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true_sel[0], true_sel[1], true_sel[2], "r", lw=0.5, label="True")
    ax.plot(pred_sel[0], pred_sel[1], pred_sel[2], "g", lw=0.5, label="Predicted")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Forecast Trajectory")
    ax.legend(); plt.tight_layout()

    components = ["X", "Y", "Z"]
    for i in range(true_sel.shape[0]):
        plt.figure(figsize=(6, 3))
        plt.plot(t_sel, true_sel[i], "r", label="True")
        plt.plot(t_sel, pred_sel[i], "b", label="Pred")
        plt.xlabel("Time"); plt.ylabel(components[i])
        plt.title(f"Forecast {components[i]}")
        plt.legend(); plt.tight_layout()


def plot_error_over_time(
    times: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    from_time: float = 0.0,
    to_time: Optional[float] = None,
) -> None:
    if to_time is None:
        to_time = float(times[-1])
    mask = (times >= from_time) & (times <= to_time)
    t_sel = times[mask]
    err = np.linalg.norm(pred - true, axis=0)[mask]
    plt.figure(figsize=(6, 3))
    plt.plot(t_sel, err)
    plt.xlabel("Time"); plt.ylabel("L2 Error")
    plt.title("Instantaneous Prediction Error Norm")
    plt.tight_layout()


def _extract_forecast_inputs(results_or_forecast: Union[Dict, np.ndarray]):
    if isinstance(results_or_forecast, dict) and "forecast" in results_or_forecast:
        fc = results_or_forecast["forecast"]
        return fc["times"], fc["true"], fc["pred"]
    elif isinstance(results_or_forecast, dict) and {"times", "true", "pred"}.issubset(results_or_forecast.keys()):
        return results_or_forecast["times"], results_or_forecast["true"], results_or_forecast["pred"]
    else:
        raise ValueError("Expected a results dict with 'forecast' or a forecast dict with keys times/true/pred")


def plot_all(
    results_or_forecast: Union[Dict, np.ndarray],
    from_time: float = 0.0,
    to_time: Optional[float] = None,
    error_plot: bool = True,
) -> None:
    times, true, pred = _extract_forecast_inputs(results_or_forecast)
    plot_forecast(times, true, pred, from_time=from_time, to_time=to_time)
    if error_plot:
        plot_error_over_time(times, true, pred, from_time=from_time, to_time=to_time)
