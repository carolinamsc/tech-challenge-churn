"""Plot the PSI drift baseline from the versioned monitoring report."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.monitoring.drift import ALERT_THRESHOLD, WARNING_THRESHOLD  # noqa: E402

REPORT = Path("reports/drift_baseline.csv")
OUTPUT = Path("docs/drift_psi.png")
STATUS_COLORS = {
    "stable": "#2e7d5b",
    "warning": "#c9a227",
    "alert": "#c0392b",
}

logger = logging.getLogger(__name__)


def plot_drift_baseline() -> Path:
    """Render the PSI of every feature against the warning and alert limits."""
    data = pd.read_csv(REPORT).sort_values("psi")
    colors = [STATUS_COLORS[status] for status in data["status"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["feature"], data["psi"], color=colors)
    ax.axvline(
        WARNING_THRESHOLD,
        color="#c9a227",
        linestyle="--",
        label=f"warning ({WARNING_THRESHOLD:.2f})",
    )
    ax.axvline(
        ALERT_THRESHOLD,
        color="#c0392b",
        linestyle="--",
        label=f"alert ({ALERT_THRESHOLD:.2f})",
    )
    ax.set_title("Drift de entrada — PSI por variável (treino × holdout)")
    ax.set_xlabel("Population Stability Index")
    ax.set_xlim(0, max(ALERT_THRESHOLD * 1.1, data["psi"].max() * 1.2))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_drift_baseline()
