"""Project-wide paths and constants."""

from pathlib import Path

SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"
TFIDF_DIR: Path = MODELS_DIR / "tfidf"
ROBERTA_DIR: Path = MODELS_DIR / "roberta"

HF_DATASET: str = "andreagasparini/dreaddit"
SPLITS: tuple[str, ...] = ("train", "test")

COLUMNS: list[str] = ["id", "text", "label", "subreddit", "confidence"]

# Maps each Dreaddit subreddit to one of five topics.
SUBREDDIT_TO_TOPIC: dict[str, str] = {
    "domesticviolence": "abuse",
    "survivorsofabuse": "abuse",
    "anxiety": "anxiety",
    "stress": "anxiety",
    "assistance": "money",
    "homeless": "money",
    "almosthomeless": "money",
    "food_pantry": "money",
    "ptsd": "ptsd",
    "relationships": "relationships",
}
TOPICS: list[str] = sorted(set(SUBREDDIT_TO_TOPIC.values()))
