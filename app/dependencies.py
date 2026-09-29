from __future__ import annotations

from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "demand_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}. Run python train.py first.")
    return joblib.load(MODEL_PATH)
