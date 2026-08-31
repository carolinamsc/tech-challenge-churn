"""Plot the threshold trade-off curve from the versioned threshold report."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPORT = Path("reports/threshold_analysis.csv")
OUTPUT = Path("docs/threshold_curve.png")
CHOSEN_THRESHOLD = 0.55

logger = logging.getLogger(__name__)


def plot_threshold_curve() -> Path:
    """Render precision, recall, F1 and intervention rate across thresholds."""
    data = pd.read_csv(REPORT)

    fig, ax = plt.subplots(figsize=(9, 5))
    for column, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("intervention_rate", "Taxa de intervenção"),
    ]:
        ax.plot(data["threshold"], data[column], marker="o", label=label)

    ax.axvline(CHOSEN_THRESHOLD, color="grey", linestyle="--")
    ax.annotate(
        f"threshold = {CHOSEN_THRESHOLD}",
        xy=(CHOSEN_THRESHOLD, 0.95),
        xytext=(CHOSEN_THRESHOLD + 0.01, 0.95),
        color="grey",
    )
    ax.set_title("Trade-off do threshold — Regressão Logística (holdout)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_threshold_curve()
