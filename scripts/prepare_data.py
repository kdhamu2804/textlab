"""Phase 1: download Dreaddit, add topics, save processed data, print a summary.

Run from the project root:
    python -m scripts.prepare_data
"""

from ml.config import SPLITS
from ml.data import download_raw, prepare_split, topic_summary


def main() -> None:
    """Download raw data (if missing), process each split and print summaries."""
    download_raw()
    for split in SPLITS:
        df = prepare_split(split)
        print(f"\n=== {split.upper()} ({len(df)} posts) ===")
        print(topic_summary(df).to_string())


if __name__ == "__main__":
    main()
