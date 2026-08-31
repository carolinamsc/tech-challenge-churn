"""Helpers to persist experiment reports in a reproducible format."""

import logging
from pathlib import Path

import pandas as pd

METRIC_DECIMALS = 6
logger = logging.getLogger(__name__)


def write_report(result: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Round float metrics and persist the report so reruns produce identical files."""
    rounded = result.copy()
    float_columns = rounded.select_dtypes("float").columns
    rounded[float_columns] = rounded[float_columns].round(METRIC_DECIMALS)

    path.parent.mkdir(parents=True, exist_ok=True)
    rounded.to_csv(path, index=False)
    logger.info("%s", rounded.to_string(index=False))
    return rounded
