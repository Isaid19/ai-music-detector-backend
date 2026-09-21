"""Utility helpers for the AI music detector."""

import math
import re
from typing import Any, Dict, List, Optional


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, strip and collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, str]:
    """Flatten a nested dict and convert leaf values to strings."""
    items: List[Any] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", str(item)))
        else:
            items.append((new_key, str(v)))
    return {k: v for k, v in items}


def search_signature_in_text(text: str, signatures: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return first matching signature dict or None."""
    normalized = normalize_text(text)
    for key, meta in signatures.items():
        for name in meta["names"]:
            if name in normalized:
                return {**meta, "key": key, "matched_name": name}
    return None


def contains_any(text: str, keywords: List[str]) -> bool:
    """Check if any keyword appears in normalized text."""
    normalized = normalize_text(text)
    return any(kw in normalized for kw in keywords)


def score_bar(value: float, length: int = 20) -> str:
    """Return a simple ASCII bar [||||||....]."""
    value = max(0.0, min(100.0, float(value)))
    filled = int(round(value / 100.0 * length))
    return "[" + "|" * filled + "." * (length - filled) + "]"


def safe_div(a: float, b: float) -> float:
    """Safe division returning 0 when denominator is zero."""
    return a / b if b != 0 else 0.0


def coefficient_of_variation(values: Any) -> float:
    """Compute CV for a numpy array / list."""
    try:
        import numpy as np
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return safe_div(std, abs(mean)) if mean != 0 else 0.0
    except Exception:
        return 0.0


def ai_flatness_score(values: Any) -> float:
    """
    Heuristic: very low coefficient of variation may indicate machine-like
    uniformity; very high variation may indicate noisy/unpolished audio.
    We map low CV toward a moderate AI indicator, high CV toward human.
    """
    cv = coefficient_of_variation(values)
    # CV < 0.10 -> likely flat / synthetic -> up to 70
    # CV > 0.50 -> likely organic / human -> down to 10
    if cv < 0.10:
        return 70.0
    if cv < 0.25:
        return 50.0
    if cv < 0.40:
        return 30.0
    return 10.0


def build_feature_summary(values: Any) -> Dict[str, Any]:
    """Build a summary dict for a numeric feature series."""
    try:
        import numpy as np
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        return {
            "mean": round(mean, 6),
            "ai_flatness_score": round(ai_flatness_score(arr), 2),
            "bar": score_bar(ai_flatness_score(arr)),
        }
    except Exception:
        return {"mean": 0.0, "ai_flatness_score": 50.0, "bar": score_bar(50.0)}


def extract_mean(values: Any) -> float:
    try:
        import numpy as np
        return float(np.mean(np.asarray(values, dtype=float)))
    except Exception:
        return 0.0


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
