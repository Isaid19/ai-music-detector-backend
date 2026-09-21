"""Lyrics extraction and heuristic AI-likelihood scoring."""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from app.utils import normalize_text


COMMON_AI_LYRIC_MARKERS = [
    "oh oh", "la la", "na na", "da da", "whoa oh",
    "endless dream", "digital heart", "silicon", "algorithm",
    "made of stars", "electric soul", "synthetic love",
]

POSITIVE_EMOTION_WORDS = [
    "love", "happy", "joy", "hope", "free", "light", "dream", "smile",
    "amor", "feliz", "alegría", "esperanza", "libre", "luz", "sueño", "sonrisa",
]

NEGATIVE_EMOTION_WORDS = [
    "sad", "pain", "cry", "tear", "dark", "alone", "hurt", "break",
    "triste", "dolor", "llorar", "lágrima", "oscuridad", "solo", "herir", "romper",
]


def extract_lyrics_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """Return lyrics if embedded in metadata."""
    return metadata.get("lyrics") or None


def _segment_into_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _unique_word_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", normalize_text(text))
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _repetition_score(text: str) -> float:
    """High literal repetition suggests AI-generated looped phrases."""
    lines = _segment_into_lines(text)
    if len(lines) < 4:
        return 30.0
    counter = Counter(normalize_text(line) for line in lines)
    most_common = counter.most_common(1)[0][1]
    repetition_ratio = most_common / len(lines)
    # Ratio > 0.25 -> strong AI signal
    if repetition_ratio > 0.25:
        return 75.0
    if repetition_ratio > 0.15:
        return 55.0
    if repetition_ratio > 0.08:
        return 35.0
    return 15.0


def _naturalness_score(text: str) -> float:
    """Penalize cliché AI markers and generic filler."""
    normalized = normalize_text(text)
    marker_hits = sum(1 for marker in COMMON_AI_LYRIC_MARKERS if marker in normalized)
    words = re.findall(r"\b\w+\b", normalized)
    if not words:
        return 50.0
    density = marker_hits / max(len(words) / 100, 1)
    ai_score = min(90.0, 30.0 + density * 40.0)
    return ai_score


def _thematic_originality_score(text: str) -> float:
    """Generic/vague themes are more common in AI output."""
    normalized = normalize_text(text)
    vague_phrases = [
        "in the night", "all the time", "every day", "my heart", "so high",
        "forever", "together", "in my mind", "let it go", "hold me tight",
    ]
    hits = sum(1 for phrase in vague_phrases if phrase in normalized)
    if hits > 5:
        return 70.0
    if hits > 2:
        return 50.0
    if hits > 0:
        return 35.0
    return 20.0


def _emotional_coherence_score(text: str) -> float:
    """Incoherent emotion mixing can indicate AI generation."""
    normalized = normalize_text(text)
    pos = sum(1 for w in POSITIVE_EMOTION_WORDS if w in normalized)
    neg = sum(1 for w in NEGATIVE_EMOTION_WORDS if w in normalized)
    total = pos + neg
    if total == 0:
        return 50.0  # Neutral / unknown
    # Both strong positive and negative -> possible incoherence
    balance = min(pos, neg) / max(total, 1)
    if balance > 0.35:
        return 65.0
    if balance > 0.20:
        return 45.0
    return 25.0


def _narrative_specificity_score(text: str) -> float:
    """Concrete entities (names, places, objects) reduce AI likelihood."""
    normalized = normalize_text(text)
    # Very rough heuristic: presence of proper nouns (capitalized words inside text)
    # We cannot rely on capitalization after normalization, so look for common concrete words.
    concrete = re.findall(r"\b(car|house|city|street|phone|rain|coffee|shoes|guitar|beach|mountain|river|road|school|room|door|window|bike|train|plane|money|watch|hat|dress|jacket)\b", normalized)
    density = len(concrete) / max(len(normalized.split()) / 100, 1)
    if density < 0.5:
        return 60.0
    if density < 1.5:
        return 40.0
    return 20.0


def _structural_variation_score(text: str) -> float:
    """Assess structural variation across lines / stanzas."""
    lines = _segment_into_lines(text)
    if len(lines) < 4:
        return 50.0
    lengths = [len(line.split()) for line in lines]
    if not lengths:
        return 50.0
    mean_len = sum(lengths) / len(lengths)
    variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
    cv = (variance ** 0.5) / mean_len if mean_len else 0.0
    # Very low variance -> repetitive / machine-like structure
    if cv < 0.15:
        return 70.0
    if cv < 0.30:
        return 50.0
    if cv < 0.50:
        return 35.0
    return 20.0


def analyze_lyrics(lyrics: Optional[str]) -> Dict[str, Any]:
    """Run all heuristic lyric analyses and return a structured report."""
    if not lyrics or not str(lyrics).strip():
        return {
            "available": False,
            "word_count": 0,
            "line_count": 0,
            "naturalness_ai_score": 50.0,
            "thematic_originality_ai_score": 50.0,
            "emotional_coherence_ai_score": 50.0,
            "repetition_patterns_ai_score": 50.0,
            "narrative_specificity_ai_score": 50.0,
            "structural_variation_ai_score": 50.0,
            "lyrics_ai_likelihood": 50.0,
            "source": "none",
        }

    text = str(lyrics).strip()
    words = re.findall(r"\b\w+\b", normalize_text(text))
    lines = _segment_into_lines(text)

    scores = {
        "naturalness_ai_score": _naturalness_score(text),
        "thematic_originality_ai_score": _thematic_originality_score(text),
        "emotional_coherence_ai_score": _emotional_coherence_score(text),
        "repetition_patterns_ai_score": _repetition_score(text),
        "narrative_specificity_ai_score": _narrative_specificity_score(text),
        "structural_variation_ai_score": _structural_variation_score(text),
    }

    weights = {
        "naturalness_ai_score": 0.20,
        "thematic_originality_ai_score": 0.15,
        "emotional_coherence_ai_score": 0.15,
        "repetition_patterns_ai_score": 0.20,
        "narrative_specificity_ai_score": 0.15,
        "structural_variation_ai_score": 0.15,
    }

    lyrics_ai = round(sum(scores[k] * weights[k] for k in scores), 2)

    return {
        "available": True,
        "word_count": len(words),
        "line_count": len(lines),
        **scores,
        "lyrics_ai_likelihood": lyrics_ai,
        "source": "embedded_metadata",
    }
