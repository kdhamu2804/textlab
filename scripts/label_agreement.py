"""Phase 3: report how often each team member agrees with the dataset label.

Fill in data/label_check.csv first (1 = stress, 0 = no stress, blank = skipped).

Run from the project root:
    python -m scripts.label_agreement
"""

import pandas as pd

from ml.annotation import LABEL_CHECK_FILE, agreement_report
from ml.config import RESULTS_DIR

REPORT_FILE = RESULTS_DIR / "label_agreement.csv"


def main() -> None:
    """Compute agreement, save it to results/ and print it."""
    report = agreement_report(pd.read_csv(LABEL_CHECK_FILE, dtype={"id": str}))
    if report["n"].eq(0).all():
        raise SystemExit("No labels found yet. Fill in the dhamodhar/kushvanth/uday columns.")
    report.to_csv(REPORT_FILE, index=False)
    print(report.round(3).to_string(index=False))
    print(f"\nSaved {REPORT_FILE}")


if __name__ == "__main__":
    main()
