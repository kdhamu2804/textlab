"""API settings. Override any value with an environment variable or a ``.env`` file,
using the ``TEXTLAB_`` prefix (for example ``TEXTLAB_MAX_TEXT_CHARS=8000``)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, ROBERTA_DIR, ROBERTA_ONNX_DIR, TFIDF_DIR


class Settings(BaseSettings):
    """Configuration for the TextLab API."""

    model_config = SettingsConfigDict(env_prefix="TEXTLAB_", env_file=".env", extra="ignore")

    app_name: str = "TextLab Stress Detection API"
    roberta_dir: Path = ROBERTA_DIR
    # 8-bit ONNX RoBERTa, used when the full model (or PyTorch) is not available.
    onnx_dir: Path = ROBERTA_ONNX_DIR
    tfidf_dir: Path = TFIDF_DIR
    results_dir: Path = RESULTS_DIR
    # Manual annotation files (edited by the team, also through the labelling page).
    error_groups_file: Path = DATA_DIR / "error_groups.csv"
    label_check_file: Path = DATA_DIR / "label_check.csv"
    # The labelling page writes to label_check_file; switch it off on public deployments.
    labeling_enabled: bool = True
    # Built website (``npm run build``); served at / when it exists, e.g. on Hugging Face Spaces.
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"
    max_text_chars: int = 5000
    max_batch_size: int = 64
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, read once and cached."""
    return Settings()
