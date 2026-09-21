"""High-level orchestrator that combines all analyses into the final JSON."""

import os
from typing import Any, Dict, List, Optional

from app.config import AI_THRESHOLD, HUMAN_THRESHOLD
from app.services.acoustic import extract_acoustic_features
from app.services.lyrics import analyze_lyrics, extract_lyrics_from_metadata
from app.services.metadata import detect_ai_tool_signature, extract_metadata
from app.services.shazam_service import artist_matches_song, recognize
from app.services.youtube_service import check_cover_existence, verify_official_video
from app.utils import build_feature_summary, clamp, score_bar


def _feature_summary_from_score(mean_value: float, ai_score: float) -> Dict[str, Any]:
    """Build a top-level feature summary using the pre-computed AI score."""
    return {
        "mean": round(float(mean_value), 6),
        "ai_flatness_score": round(float(ai_score), 2),
        "bar": score_bar(ai_score),
    }


def _build_shazam_analysis(
    shazam_info: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Post-process Shazam results and detect cover/artist mismatches."""
    recognized = shazam_info.get("recognized", False)
    skipped = shazam_info.get("skipped", True)
    is_conclusive = shazam_info.get("is_conclusive", False)

    if not recognized:
        return {
            "skipped": skipped,
            "recognized": False,
            "is_conclusive": is_conclusive,
            "reasoning": shazam_info.get("reasoning", "Song not recognized by Shazam."),
            "title": shazam_info.get("title"),
            "artist": shazam_info.get("artist"),
            "album": shazam_info.get("album"),
            "year": shazam_info.get("year"),
            "genre": shazam_info.get("genre"),
        }

    meta_artist = metadata.get("artist")
    shazam_artist = shazam_info.get("artist")
    matches = artist_matches_song(meta_artist, shazam_artist)

    reasoning = shazam_info.get("reasoning", "")
    if meta_artist and not matches:
        reasoning += (
            f" Metadata artist '{meta_artist}' does not match Shazam artist '{shazam_artist}'; "
            "possible cover or unreleased/AI-generated track."
        )
    elif meta_artist and matches:
        reasoning += " Metadata artist matches Shazam artist."

    return {
        "skipped": False,
        "recognized": True,
        "is_conclusive": is_conclusive,
        "reasoning": reasoning,
        "title": shazam_info.get("title"),
        "artist": shazam_info.get("artist"),
        "album": shazam_info.get("album"),
        "year": shazam_info.get("year"),
        "genre": shazam_info.get("genre"),
        "artist_matches_metadata": matches,
    }


def _fuzzy_inference(
    metadata_ai: float,
    shazam_missing: float,
    youtube_missing: float,
    cover_mismatch: float,
    acoustic: float,
    lyrics: float,
) -> Dict[str, Any]:
    """
    Simple Mamdani-style fuzzy inference with weighted centroid defuzzification.
    Inputs are already crisp 0-100 scores; we fuzzify lightly with membership
    functions LOW / MEDIUM / HIGH and apply rule weights.
    """

    def membership_low(x: float) -> float:
        return clamp(1.0 - x / 40.0, 0.0, 1.0) if x <= 40 else 0.0

    def membership_medium(x: float) -> float:
        if x <= 30 or x >= 80:
            return 0.0
        if x <= 55:
            return (x - 30) / 25.0
        return (80 - x) / 25.0

    def membership_high(x: float) -> float:
        return clamp((x - 60) / 40.0, 0.0, 1.0) if x >= 60 else 0.0

    inputs = {
        "metadata_ai": metadata_ai,
        "shazam_missing": shazam_missing,
        "youtube_missing": youtube_missing,
        "cover_mismatch": cover_mismatch,
        "acoustic": acoustic,
        "lyrics": lyrics,
    }

    # Rule base: antecedent -> consequent AI likelihood
    # Each rule fires with weight = min(memberships) and contributes centroid at
    # the consequent level (LOW=25, MEDIUM=50, HIGH=80).
    rules = [
        # Metadata strongly conclusive
        ({"metadata_ai": "high"}, "high", 1.5),
        # No Shazam match and no YouTube official -> high AI
        ({"shazam_missing": "high", "youtube_missing": "high"}, "high", 1.2),
        # Acoustic strongly synthetic
        ({"acoustic": "high", "lyrics": "high"}, "high", 1.0),
        ({"acoustic": "high"}, "high", 0.9),
        # Recognized + official video -> low AI
        ({"shazam_missing": "low", "youtube_missing": "low"}, "low", 1.2),
        # Organic acoustic + human-like lyrics -> low AI
        ({"acoustic": "low", "lyrics": "low"}, "low", 1.0),
        # Medium evidence -> medium
        ({"metadata_ai": "medium"}, "medium", 0.8),
        ({"acoustic": "medium", "lyrics": "medium"}, "medium", 0.8),
    ]

    consequent_centroids = {"low": 20.0, "medium": 50.0, "high": 82.0}
    numerator = 0.0
    denominator = 0.0

    for antecedent, consequent_label, weight in rules:
        firing_strengths = []
        for var, level in antecedent.items():
            x = inputs[var]
            if level == "low":
                mu = membership_low(x)
            elif level == "medium":
                mu = membership_medium(x)
            else:
                mu = membership_high(x)
            firing_strengths.append(mu)
        if not firing_strengths:
            continue
        fire = min(firing_strengths) * weight
        numerator += fire * consequent_centroids[consequent_label]
        denominator += fire

    if denominator == 0.0:
        # No rule fired significantly; average inputs with emphasis on acoustic
        content_score = (acoustic * 0.45 + lyrics * 0.25 + metadata_ai * 0.20 + cover_mismatch * 0.10)
    else:
        content_score = numerator / denominator

    return {
        "method": "Mamdani-style weighted centroid defuzzification",
        "content_ai_score": round(clamp(content_score), 2),
        "inputs_used": {
            "acoustic_ai_likelihood": round(acoustic, 2),
            "lyrics_ai_likelihood": round(lyrics, 2),
            "metadata_ai_score": round(metadata_ai, 2),
            "shazam_missing_score": round(shazam_missing, 2),
            "youtube_missing_score": round(youtube_missing, 2),
            "cover_mismatch_score": round(cover_mismatch, 2),
        },
    }


def _compute_final_score_and_reasoning(
    metadata_ai: float,
    shazam_analysis: Dict[str, Any],
    youtube_analysis: Dict[str, Any],
    cover_analysis: Dict[str, Any],
    fuzzy: Dict[str, Any],
) -> Dict[str, Any]:
    """Blend all evidence into the final 0-100 AI score and classification."""
    recognized = shazam_analysis.get("recognized", False)
    artist_matches = shazam_analysis.get("artist_matches_metadata", False)
    official_found = youtube_analysis.get("is_official", False)
    cover_likely = cover_analysis.get("is_likely_cover", False) or youtube_analysis.get("is_cover", False)
    metadata_conclusive = metadata_ai >= 60

    # Base content score from fuzzy inference
    content_ai = fuzzy["content_ai_score"]

    # Decision layer adjustments
    decision_reasoning_parts: List[str] = []
    decision_layer = "content_based"

    # Strongest signal: explicit AI metadata signature
    if metadata_conclusive:
        content_ai = max(content_ai, metadata_ai)
        decision_layer = "metadata_ai_signature"
        decision_reasoning_parts.append(
            f"AI tool signature detected in metadata (score {metadata_ai})."
        )

    # Shazam recognized -> strong human evidence
    if recognized:
        decision_reasoning_parts.append(
            f"Shazam recognized the song as '{shazam_analysis.get('title')}' by '{shazam_analysis.get('artist')}'."
        )
        if artist_matches:
            content_ai = content_ai * 0.55  # strong human reduction
            decision_reasoning_parts.append("Metadata artist matches Shazam artist.")
        else:
            content_ai = content_ai * 0.75
            decision_reasoning_parts.append("Metadata artist does not match Shazam artist; possible cover/AI.")
        if official_found:
            content_ai = content_ai * 0.70
            decision_layer = "recognition_plus_official_video"
            decision_reasoning_parts.append("Official video found on YouTube.")
        else:
            decision_reasoning_parts.append("No clear official video found on YouTube.")
    else:
        decision_reasoning_parts.append("Shazam did not recognize the song; this increases AI likelihood.")
        content_ai = min(100.0, content_ai * 1.15)

    if cover_likely:
        decision_reasoning_parts.append("Cover-like results detected on YouTube; reduced AI likelihood if artist mismatch absent.")
        if not artist_matches:
            content_ai = min(100.0, content_ai * 1.05)

    final_score = round(clamp(content_ai), 2)

    if final_score < HUMAN_THRESHOLD:
        classification = "Humano"
    elif final_score > AI_THRESHOLD:
        classification = "Generado por IA"
    else:
        classification = "Indeterminado / Necesita revisión"

    return {
        "ai_score": final_score,
        "ai_score_bar": score_bar(final_score),
        "classification": classification,
        "decision_layer": decision_layer,
        "decision_reasoning": " ".join(decision_reasoning_parts),
    }


def _build_metadata_analysis_block(
    metadata: Dict[str, Any],
    ai_signature: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "title": metadata.get("title"),
        "artist": metadata.get("artist"),
        "album": metadata.get("album"),
        "year": metadata.get("year"),
        "genre": metadata.get("genre"),
        "comment": metadata.get("comment"),
        "encoder": metadata.get("encoder"),
        "ai_tool_detected": ai_signature.get("ai_tool_detected"),
        "generic_ai_hint": ai_signature.get("generic_ai_hint"),
        "matched_field": ai_signature.get("matched_field"),
        "matched_value": ai_signature.get("matched_value"),
        "is_conclusive": ai_signature.get("is_conclusive", False),
        "metadata_ai_score": ai_signature.get("metadata_ai_score", 0.0),
        "reasoning": ai_signature.get("reasoning", ""),
    }


async def analyze_audio(file_path: str, provided_lyrics: Optional[str] = None) -> Dict[str, Any]:
    """Run the complete AI music detection pipeline."""
    file_name = os.path.basename(file_path)

    # 1. Metadata
    metadata = extract_metadata(file_path)
    ai_signature = detect_ai_tool_signature(metadata)
    metadata_block = _build_metadata_analysis_block(metadata, ai_signature)

    # 2. Shazam
    shazam_info = await recognize(file_path)
    shazam_analysis = _build_shazam_analysis(shazam_info, metadata)

    # 3. YouTube official video / cover
    search_title = shazam_analysis.get("title") or metadata.get("title")
    search_artist = shazam_analysis.get("artist") or metadata.get("artist")
    youtube_analysis = await verify_official_video(search_title, search_artist)
    cover_analysis = await check_cover_existence(search_title, search_artist)

    # 4. Acoustic features
    acoustic = extract_acoustic_features(file_path)

    # 5. Lyrics
    lyrics_text = provided_lyrics or extract_lyrics_from_metadata(metadata)
    lyrics = analyze_lyrics(lyrics_text)

    # 6. Fuzzy inference inputs
    metadata_ai = float(ai_signature.get("metadata_ai_score", 0.0))
    shazam_missing = 0.0 if shazam_analysis.get("recognized") else 70.0
    youtube_missing = 0.0 if youtube_analysis.get("is_official") else 60.0
    cover_mismatch = 0.0
    if shazam_analysis.get("recognized") and not shazam_analysis.get("artist_matches_metadata", True):
        cover_mismatch = 65.0
    elif not shazam_analysis.get("recognized"):
        cover_mismatch = 45.0

    fuzzy = _fuzzy_inference(
        metadata_ai=metadata_ai,
        shazam_missing=shazam_missing,
        youtube_missing=youtube_missing,
        cover_mismatch=cover_mismatch,
        acoustic=acoustic.get("acoustic_ai_likelihood", 50.0),
        lyrics=lyrics.get("lyrics_ai_likelihood", 50.0),
    )

    # 7. Final decision
    final = _compute_final_score_and_reasoning(
        metadata_ai=metadata_ai,
        shazam_analysis=shazam_analysis,
        youtube_analysis=youtube_analysis,
        cover_analysis=cover_analysis,
        fuzzy=fuzzy,
    )

    # 8. Per-feature summary bars for top-level keys requested by user.
    # Re-use the already computed AI flatness/variability scores from the
    # acoustic analysis instead of recomputing them from a single scalar.
    zcr_summary = _feature_summary_from_score(
        acoustic["zcr"]["mean"], acoustic["zcr"]["ai_flatness_score"]
    )
    centroid_summary = _feature_summary_from_score(
        acoustic["spectral_centroid"]["mean"], acoustic["spectral_centroid"]["ai_flatness_score"]
    )
    bandwidth_summary = _feature_summary_from_score(
        acoustic["spectral_bandwidth"]["mean"], acoustic["spectral_bandwidth"]["ai_flatness_score"]
    )
    contrast_summary = _feature_summary_from_score(
        acoustic["spectral_contrast"]["mean"], acoustic["spectral_contrast"]["ai_flatness_score"]
    )
    rolloff_summary = _feature_summary_from_score(
        acoustic["spectral_rolloff"]["mean"], acoustic["spectral_rolloff"]["ai_flatness_score"]
    )
    flux_summary = _feature_summary_from_score(
        acoustic["spectral_flux"]["mean"], acoustic["spectral_flux"]["ai_flatness_score"]
    )
    mfcc_summary = _feature_summary_from_score(
        acoustic["mfcc"]["mean_of_means"], acoustic["mfcc"]["ai_variability_score"]
    )
    rmse_summary = _feature_summary_from_score(
        acoustic["rmse"]["mean"], acoustic["rmse"]["ai_flatness_score"]
    )

    return {
        "audio_file": file_name,
        "scale": {
            "0": "Humano",
            "100": "Generado por IA",
        },
        "ai_score": final["ai_score"],
        "ai_score_bar": final["ai_score_bar"],
        "classification": final["classification"],
        "decision_layer": final["decision_layer"],
        "decision_reasoning": final["decision_reasoning"],
        "shazam_analysis": shazam_analysis,
        "metadata_analysis": metadata_block,
        "acoustic_and_lyrics_analysis": {
            "acoustic_features": acoustic,
            "lyrics_analysis": lyrics,
            "fuzzy_inference": fuzzy,
        },
        "youtube_analysis": {
            "official_video": youtube_analysis,
            "cover_check": cover_analysis,
        },
        "Zero Crossing Rate": zcr_summary,
        "Spectral Centroid": centroid_summary,
        "Spectral Bandwidth": bandwidth_summary,
        "Spectral Contrast": contrast_summary,
        "Spectral Rolloff": rolloff_summary,
        "Spectral Flux": flux_summary,
        "MFCCs": mfcc_summary,
        "RMSE": rmse_summary,
    }
