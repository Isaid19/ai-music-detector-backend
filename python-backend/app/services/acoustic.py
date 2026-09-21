"""Acoustic feature extraction using librosa (with optional pyAudioAnalysis)."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import ANALYSIS_HOP_SECONDS, ANALYSIS_SEGMENT_SECONDS, HOP_LENGTH, N_FFT, N_MFCC, TARGET_SAMPLE_RATE
from app.utils import build_feature_summary, coefficient_of_variation, extract_mean, safe_div


def _load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio with librosa and resample."""
    import librosa
    y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    return y, sr


def _segment_for_analysis(y: np.ndarray, sr: int) -> np.ndarray:
    """Select a representative segment to keep analysis fast and consistent."""
    segment_samples = int(ANALYSIS_SEGMENT_SECONDS * sr)
    hop_samples = int(ANALYSIS_HOP_SECONDS * sr)

    if len(y) <= segment_samples:
        return y

    # Skip first 5% (possible silence/intro) and analyze middle segments
    start_offset = int(len(y) * 0.05)
    available = len(y) - start_offset

    segments: List[np.ndarray] = []
    pos = start_offset
    while pos + segment_samples <= len(y) and len(segments) < 3:
        segments.append(y[pos : pos + segment_samples])
        pos += hop_samples

    if not segments:
        segments.append(y[start_offset : start_offset + segment_samples])

    # Concatenate up to 3 segments (max 180s)
    return np.concatenate(segments)


def extract_acoustic_features(file_path: str) -> Dict[str, Any]:
    """Extract a comprehensive set of acoustic features."""
    result: Dict[str, Any] = {
        "duration_analyzed_seconds": 0.0,
        "sample_rate": TARGET_SAMPLE_RATE,
        "zcr": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "spectral_centroid": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "spectral_bandwidth": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "spectral_contrast": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "spectral_rolloff": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "spectral_flux": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "mfcc": {"mean_of_means": 0.0, "mean_of_stds": 0.0, "overall_std": 0.0, "ai_variability_score": 50.0},
        "rmse": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "coefficient_of_variation": 0.0, "ai_flatness_score": 50.0},
        "acoustic_ai_likelihood": 50.0,
        "pyaudioanalysis_available": False,
        "error": None,
    }

    try:
        import librosa
        y, sr = _load_audio(file_path)
        y_segment = _segment_for_analysis(y, sr)
        result["duration_analyzed_seconds"] = round(float(len(y_segment)) / sr, 3)
        result["sample_rate"] = sr

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y_segment, hop_length=HOP_LENGTH)[0]
        result["zcr"] = _stat_summary(zcr)

        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        result["spectral_centroid"] = _stat_summary(spec_cent)

        # Spectral Bandwidth
        spec_band = librosa.feature.spectral_bandwidth(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        result["spectral_bandwidth"] = _stat_summary(spec_band)

        # Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        result["spectral_contrast"] = _stat_summary(spec_contrast.flatten())

        # Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        result["spectral_rolloff"] = _stat_summary(spec_rolloff)

        # Spectral Flux (on magnitude spectrogram)
        S = np.abs(librosa.stft(y_segment, n_fft=N_FFT, hop_length=HOP_LENGTH))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        result["spectral_flux"] = _stat_summary(flux)

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        result["mfcc"] = _mfcc_summary(mfcc)

        # RMSE (RMS Energy)
        rms = librosa.feature.rms(y=y_segment, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        result["rmse"] = _stat_summary(rms)

        # Optional pyAudioAnalysis enrichment
        paa = _try_pyaudioanalysis(file_path)
        result["pyaudioanalysis_available"] = paa.get("available", False)
        if result["pyaudioanalysis_available"]:
            result["pyaudioanalysis_features"] = paa

        result["acoustic_ai_likelihood"] = round(_compute_acoustic_ai_likelihood(result), 2)
    except Exception as e:
        result["error"] = str(e)

    return result


def _stat_summary(values: np.ndarray) -> Dict[str, Any]:
    """Build statistical summary for a feature."""
    from app.utils import ai_flatness_score
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "coefficient_of_variation": round(coefficient_of_variation(arr), 6),
        "ai_flatness_score": round(ai_flatness_score(arr), 2),
    }


def _mfcc_summary(mfcc: np.ndarray) -> Dict[str, Any]:
    """Build MFCC summary."""
    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)
    overall_std = float(np.std(mfcc))
    mean_of_means = float(np.mean(means))
    mean_of_stds = float(np.mean(stds))

    # Heuristic: AI-generated music often has reduced timbral variability.
    # Lower overall std -> higher AI score.
    if overall_std < 8.0:
        ai_variability = 70.0
    elif overall_std < 15.0:
        ai_variability = 50.0
    elif overall_std < 25.0:
        ai_variability = 30.0
    else:
        ai_variability = 15.0

    return {
        "mean_of_means": round(mean_of_means, 6),
        "mean_of_stds": round(mean_of_stds, 6),
        "overall_std": round(overall_std, 6),
        "ai_variability_score": round(ai_variability, 2),
    }


def _try_pyaudioanalysis(file_path: str) -> Dict[str, Any]:
    """Attempt to extract pyAudioAnalysis mid-term features; fallback silently."""
    result: Dict[str, Any] = {"available": False}
    try:
        from pyAudioAnalysis import ShortTermFeatures as sF
        import soundfile as sf
        y, sr = sf.read(file_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Downsample if needed
        if sr != TARGET_SAMPLE_RATE:
            from scipy.signal import resample_poly
            factor = safe_div(TARGET_SAMPLE_RATE, sr)
            if factor != 1.0:
                y = resample_poly(y, TARGET_SAMPLE_RATE, sr)
                sr = TARGET_SAMPLE_RATE
        F, f_names = sF.feature_extraction(y, sr, int(0.050 * sr), int(0.025 * sr))
        result["available"] = True
        result["feature_names"] = f_names
        result["mean_features"] = [round(float(v), 6) for v in np.mean(F, axis=1)]
        result["std_features"] = [round(float(v), 6) for v in np.std(F, axis=1)]
    except Exception:
        pass
    return result


def _compute_acoustic_ai_likelihood(features: Dict[str, Any]) -> float:
    """Fuzzy-ish weighted heuristic combining all acoustic features."""
    scores = [
        features["zcr"]["ai_flatness_score"] * 0.10,
        features["spectral_centroid"]["ai_flatness_score"] * 0.15,
        features["spectral_bandwidth"]["ai_flatness_score"] * 0.10,
        features["spectral_contrast"]["ai_flatness_score"] * 0.10,
        features["spectral_rolloff"]["ai_flatness_score"] * 0.10,
        features["spectral_flux"]["ai_flatness_score"] * 0.10,
        features["mfcc"]["ai_variability_score"] * 0.25,
        features["rmse"]["ai_flatness_score"] * 0.10,
    ]
    return float(np.clip(sum(scores), 0.0, 100.0))
