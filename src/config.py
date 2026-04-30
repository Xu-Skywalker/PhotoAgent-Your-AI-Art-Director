"""Central configuration for Photography Agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_console_encoding() -> None:
    """Keep Chinese and emoji logs from crashing on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_console_encoding()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_PHOTOS_DIR = DATA_DIR / "raw_photos"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# Perception
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
FEATURE_DIM = 768
PERCEPTION_BATCH_SIZE = 1

# Clustering
DBSCAN_EPS = 0.08
DBSCAN_MIN_SAMPLES = 2

# Scoring
SCORER_TOP_N = 3
AESTHETIC_MODEL_PATH = MODELS_DIR / "aesthetic_scorer.pkl"
MASTER_CENTROID_PATH = MODELS_DIR / "master_centroid.npy"

# LLM judge. Prefer project-specific variables, then OpenAI-compatible fallbacks.
LLM_API_KEY = os.getenv("LONGCAT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_BASE_URL = os.getenv("LONGCAT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
LLM_MODEL = os.getenv("LONGCAT_MODEL") or os.getenv("OPENAI_VISION_MODEL") or ""
LLM_TIMEOUT_SECONDS = float(os.getenv("PHOTO_AGENT_LLM_TIMEOUT", "90"))
MASTER_REFERENCE_IMAGE = Path(
    os.getenv("MASTER_REFERENCE_IMAGE", str(DATA_DIR / "master_reference.jpg"))
)


def ensure_project_dirs() -> None:
    """Create project directories that are safe for the app to own."""
    for path in (RAW_PHOTOS_DIR, CACHE_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_project_dirs()
    print("✅ 配置检查完成")
    print(f"📁 ROOT_DIR: {ROOT_DIR}")
    print(f"📷 RAW_PHOTOS_DIR: {RAW_PHOTOS_DIR}")
    print(f"🧠 CACHE_DIR: {CACHE_DIR}")
    print(f"🎯 CLIP_MODEL_ID: {CLIP_MODEL_ID}")
    print(f"📐 FEATURE_DIM: {FEATURE_DIM}")
