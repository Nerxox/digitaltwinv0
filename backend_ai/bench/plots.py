from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


def plot_real_vs_pred(machine_id: str, algo: str, timestamps: Sequence, y_true: Sequence[float], y_pred: Sequence[float]) -> Path:
    """Plot real vs predicted values for the provided test window and save as PDF.

    Returns:
        Path to the saved PDF figure under docs/figures/.
    """
    out_dir = Path("docs") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"real_vs_pred_{machine_id}_{algo}.pdf"

    plt.figure(figsize=(10, 4))
    # X-axis: if timestamps provided, use them; else use range
    x = list(range(len(y_true))) if not timestamps else timestamps

    plt.plot(x, y_true, label="Actual", color="#1f77b4", linewidth=2)
    plt.plot(x, y_pred, label="Predicted", color="#ff7f0e", linewidth=2, linestyle="--")

    plt.title(f"Real vs Predicted — {machine_id} [{algo}]")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, format="pdf")
    plt.close()

    return out_path
