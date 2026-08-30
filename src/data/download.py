"""Download the reference Telco Customer Churn dataset.

The dataset is the IBM/Kaggle Telco Customer Churn file
`WA_Fn-UseC_-Telco-Customer-Churn.csv`. The raw dataset is intentionally
not committed to this repository.
"""

from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://huggingface.co/spaces/Wenny/Telco-Customer-Churn/raw/main/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def download_dataset() -> Path:
    """Download the dataset if it is not already present."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return DATA_PATH

    with urlopen(DATA_URL, timeout=60) as response:
        DATA_PATH.write_bytes(response.read())
    return DATA_PATH


if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset available at: {path}")
