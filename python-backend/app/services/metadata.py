"""Metadata extraction and AI tool signature detection."""

import os
from typing import Any, Dict, List, Optional

from mutagen import File
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

from app.config import AI_TOOL_SIGNATURES, GENERIC_AI_HINTS
from app.utils import normalize_text, search_signature_in_text


def _extract_id3_tags(audio: MP3) -> Dict[str, Any]:
    tags: Dict[str, Any] = {}
    if audio.tags is None:
        return tags
    for key in ["TIT2", "TPE1", "TPE2", "TALB", "TCON", "TDRC", "TYER", "COMM", "TENC", "TCOP", "TPUB"]:
        try:
            frame = audio.tags.get(key)
            if frame:
                tags[key] = str(frame)
        except Exception:
            pass
    # Lyrics (USLT)
    try:
        for k in audio.tags.keys():
            if k.startswith("USLT"):
                tags["LYRICS"] = str(audio.tags[k])
                break
    except Exception:
        pass
    return tags


def _extract_vorbis_tags(audio: Any) -> Dict[str, Any]:
    mapping = {
        "title": ["TITLE", "TITLE"],
        "artist": ["ARTIST"],
        "album": ["ALBUM"],
        "genre": ["GENRE"],
        "date": ["DATE", "YEAR"],
        "comment": ["COMMENT", "DESCRIPTION"],
        "encoder": ["ENCODER", "VENDOR"],
        "lyrics": ["LYRICS", "UNSYNCEDLYRICS"],
    }
    tags: Dict[str, Any] = {}
    for canonical, keys in mapping.items():
        for k in keys:
            val = audio.get(k)
            if val:
                tags[canonical] = val[0] if isinstance(val, list) else str(val)
                break
    return tags


def _extract_mp4_tags(audio: MP4) -> Dict[str, Any]:
    tags: Dict[str, Any] = {}
    mapping = {
        "title": "\xa9nam",
        "artist": "\xa9ART",
        "album": "\xa9alb",
        "genre": "\xa9gen",
        "year": "\xa9day",
        "comment": "\xa9cmt",
        "encoder": "\xa9too",
        "lyrics": "\xa9lyr",
    }
    for canonical, atom in mapping.items():
        val = audio.tags.get(atom) if audio.tags else None
        if val:
            tags[canonical] = val[0] if isinstance(val, list) else str(val)
    return tags


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract normalized metadata from an audio file."""
    result: Dict[str, Any] = {
        "title": None,
        "artist": None,
        "album": None,
        "year": None,
        "genre": None,
        "comment": None,
        "encoder": None,
        "lyrics": None,
        "duration_seconds": 0.0,
        "bitrate": None,
        "sample_rate": None,
        "channels": None,
        "format": None,
        "raw_tags": {},
    }

    try:
        audio = File(file_path)
        if audio is None:
            return result

        result["format"] = os.path.splitext(file_path)[1].lower().lstrip(".")
        result["duration_seconds"] = getattr(audio.info, "length", 0.0) or 0.0
        result["bitrate"] = getattr(audio.info, "bitrate", None)
        result["sample_rate"] = getattr(audio.info, "sample_rate", None)
        result["channels"] = getattr(audio.info, "channels", None)

        tags: Dict[str, Any] = {}
        if isinstance(audio, MP3):
            tags = _extract_id3_tags(audio)
            result["title"] = tags.get("TIT2")
            result["artist"] = tags.get("TPE1")
            result["album"] = tags.get("TALB")
            result["year"] = tags.get("TDRC") or tags.get("TYER")
            result["genre"] = tags.get("TCON")
            result["comment"] = tags.get("COMM")
            result["encoder"] = tags.get("TENC")
            result["lyrics"] = tags.get("LYRICS")
        elif isinstance(audio, (FLAC, OggVorbis)):
            tags = _extract_vorbis_tags(audio)
            result["title"] = tags.get("title")
            result["artist"] = tags.get("artist")
            result["album"] = tags.get("album")
            result["year"] = tags.get("date")
            result["genre"] = tags.get("genre")
            result["comment"] = tags.get("comment")
            result["encoder"] = tags.get("encoder")
            result["lyrics"] = tags.get("lyrics")
        elif isinstance(audio, MP4):
            tags = _extract_mp4_tags(audio)
            result["title"] = tags.get("title")
            result["artist"] = tags.get("artist")
            result["album"] = tags.get("album")
            result["year"] = tags.get("year")
            result["genre"] = tags.get("genre")
            result["comment"] = tags.get("comment")
            result["encoder"] = tags.get("encoder")
            result["lyrics"] = tags.get("lyrics")
        elif isinstance(audio, WAVE):
            pass

        result["raw_tags"] = tags
    except Exception as e:
        result["error"] = str(e)

    return result


def detect_ai_tool_signature(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Scan metadata fields for known AI music generator signatures."""
    fields_to_scan = [
        ("title", metadata.get("title")),
        ("artist", metadata.get("artist")),
        ("album", metadata.get("album")),
        ("genre", metadata.get("genre")),
        ("comment", metadata.get("comment")),
        ("encoder", metadata.get("encoder")),
        ("lyrics", metadata.get("lyrics")),
    ]

    haystack_parts: List[str] = []
    for field_name, value in fields_to_scan:
        if value:
            haystack_parts.append(normalize_text(str(value)))

    # Include raw tags as strings for thoroughness
    raw = metadata.get("raw_tags") or {}
    for k, v in raw.items():
        if v:
            haystack_parts.append(normalize_text(f"{k} {v}"))

    haystack = " ".join(haystack_parts)

    matched = search_signature_in_text(haystack, AI_TOOL_SIGNATURES)
    generic_match = any(hint in haystack for hint in GENERIC_AI_HINTS)

    if matched:
        return {
            "ai_tool_detected": matched["label"],
            "generic_ai_hint": generic_match,
            "matched_field": matched.get("matched_name"),
            "matched_value": matched.get("matched_name"),
            "metadata_ai_score": matched["score"],
            "is_conclusive": True,
            "reasoning": f"Detected signature of AI tool '{matched['label']}' in metadata ({matched.get('matched_name')}).",
        }

    if generic_match:
        return {
            "ai_tool_detected": None,
            "generic_ai_hint": True,
            "matched_field": None,
            "matched_value": None,
            "metadata_ai_score": 40.0,
            "is_conclusive": False,
            "reasoning": "No specific AI tool signature found, but generic AI-related hints detected in metadata.",
        }

    return {
        "ai_tool_detected": None,
        "generic_ai_hint": False,
        "matched_field": None,
        "matched_value": None,
        "metadata_ai_score": 0.0,
        "is_conclusive": False,
        "reasoning": "No AI generator signatures or hints found in metadata.",
    }
