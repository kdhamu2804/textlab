"""Phase 3: create data/label_check.csv with 50 random training posts.

It refuses to overwrite an existing file so filled-in labels are never lost.

Run from the project root:
    python -m scripts.make_label_check
"""

from ml.annotation import LABEL_CHECK_FILE, sample_for_labeling
from ml.data import load_split


def main() -> None:
    """Write the label-check sheet if it does not exist yet."""
    if LABEL_CHECK_FILE.exists():
        raise SystemExit(f"{LABEL_CHECK_FILE} already exists; delete it first to re-create it.")
    sample_for_labeling(load_split("train")).to_csv(LABEL_CHECK_FILE, index=False)
    print(f"Wrote 50 posts to {LABEL_CHECK_FILE}")


if __name__ == "__main__":
    main()
