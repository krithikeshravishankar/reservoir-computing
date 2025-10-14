"""Plotting utilities for reservoir computing forecasts.
Works with outputs from run_reservoir_forecast.run (results dict with 'forecast')
or with raw arrays (times, true, pred).
"""
from __future__ import annotations

import matplotlib
# Set the backend to a non-interactive one before importing pyplot
matplotlib.use('Agg')

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


def save_plots_to_file(base_path: str, results: Dict) -> None:
    """Generates and saves all standard plots to files. Does not return paths."""
    fc = results["forecast"]
    times, true, pred = fc["times"], fc["true"], fc["pred"]

    # 3D Trajectory
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true[0], true[1], true[2], "r", lw=0.5, label="True")
    ax.plot(pred[0], pred[1], pred[2], "g", lw=0.5, label="Predicted")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Forecast Trajectory")
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{base_path}_3d.png")
    plt.close(fig)

    # Individual Components
    paths_components = {}
    components = ["X", "Y", "Z"]
    for i in range(true.shape[0]):
        fig = plt.figure(figsize=(6, 3))
        plt.plot(times, true[i], "r", label="True")
        plt.plot(times, pred[i], "b", label="Pred")
        plt.xlabel("Time"); plt.ylabel(components[i])
        plt.title(f"Forecast {components[i]}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{base_path}_comp_{components[i].lower()}.png")
        plt.close(fig)