"""Extension: download GoEmotions and print a short summary.

Run from the project root:
    python -m scripts.prepare_emotions
"""

import pandas as pd

from ml.emotions import EMOTION_SPLITS, EMOTIONS, download_emotions, load_emotions


def main() -> None:
    """Download the three splits and show sizes and the most / least common emotions."""
    download_emotions()
    for split in EMOTION_SPLITS:
        df, y = load_emotions(split)
        print(f"{split:10s} {len(df):6d} comments · {y.sum(1).mean():.2f} emotions per comment")
    _, y = load_emotions("train")
    counts = pd.Series(y.sum(0), index=EMOTIONS).sort_values(ascending=False)
    print("\nMost common:", ", ".join(f"{e} {n}" for e, n in counts.head(5).items()))
    print("Least common:", ", ".join(f"{e} {n}" for e, n in counts.tail(5).items()))


if __name__ == "__main__":
    main()
