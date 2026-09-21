"""ShazamIO-based recognition service."""

import asyncio
from typing import Any, Dict, Optional

from app.config import SHAZAM_TIMEOUT
from app.utils import normalize_text


def _extract_track_info(recognition: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ShazamIO recognition payload."""
    track = recognition.get("track", {}) if isinstance(recognition, dict) else {}
    if not track:
        track = recognition

    title = track.get("title") or track.get("heading", {}).get("title")
    subtitle = track.get("subtitle") or track.get("heading", {}).get("subtitle")
    sections = track.get("sections", [])
    metadata_list = []
    for section in sections:
        if section.get("type") == "SONG" or "metadata" in section:
            metadata_list.extend(section.get("metadata", []))

    album = None
    year = None
    genre = None
    for item in metadata_list:
        if item.get("title") == "Album":
            album = item.get("text")
        if item.get("title") == "Released":
            year = item.get("text")
        if item.get("title") == "Genre":
            genre = item.get("text")

    return {
        "recognized": True,
        "title": title,
        "artist": subtitle,
        "album": album,
        "year": year,
        "genre": genre,
        "shazam_url": track.get("url"),
        "cover_art": track.get("images", {}).get("coverarthq") or track.get("images", {}).get("coverart"),
        "raw": recognition,
    }


async def recognize(file_path: str) -> Dict[str, Any]:
    """Run ShazamIO recognition with timeout and robust error handling."""
    try:
        from shazamio import Shazam
    except Exception as e:
        return {
            "recognized": False,
            "skipped": True,
            "is_conclusive": False,
            "reasoning": f"ShazamIO library not available: {e}",
            "title": None,
            "artist": None,
            "album": None,
            "year": None,
            "genre": None,
        }

    async def _run() -> Dict[str, Any]:
        shazam = Shazam()
        try:
            # Prefer the newer rust-based recognize method
            if hasattr(shazam, "recognize"):
                result = await shazam.recognize(file_path)
            else:
                result = await shazam.recognize_song(file_path)

            if not result or not isinstance(result, dict):
                return {
                    "recognized": False,
                    "skipped": False,
                    "is_conclusive": False,
                    "reasoning": "Shazam returned an empty or malformed response.",
                    "title": None,
                    "artist": None,
                    "album": None,
                    "year": None,
                    "genre": None,
                }

            track = result.get("track", {})
            title = track.get("title") or track.get("heading", {}).get("title") if isinstance(track, dict) else None
            if not title:
                return {
                    "recognized": False,
                    "skipped": False,
                    "is_conclusive": False,
                    "reasoning": "Shazam did not identify the track.",
                    "title": None,
                    "artist": None,
                    "album": None,
                    "year": None,
                    "genre": None,
                }

            info = _extract_track_info(result)
            info["skipped"] = False
            info["is_conclusive"] = True
            info["reasoning"] = f"Recognized '{info.get('title')}' by '{info.get('artist')}'."
            return info
        except Exception as exc:
            return {
                "recognized": False,
                "skipped": True,
                "is_conclusive": False,
                "reasoning": f"Shazam recognition error: {exc}",
                "title": None,
                "artist": None,
                "album": None,
                "year": None,
                "genre": None,
            }

    try:
        return await asyncio.wait_for(_run(), timeout=SHAZAM_TIMEOUT)
    except asyncio.TimeoutError:
        return {
            "recognized": False,
            "skipped": True,
            "is_conclusive": False,
            "reasoning": "Shazam recognition timed out.",
            "title": None,
            "artist": None,
            "album": None,
            "year": None,
            "genre": None,
        }
    except Exception as e:
        return {
            "recognized": False,
            "skipped": True,
            "is_conclusive": False,
            "reasoning": f"Unexpected Shazam error: {e}",
            "title": None,
            "artist": None,
            "album": None,
            "year": None,
            "genre": None,
        }


def artist_matches_song(metadata_artist: Optional[str], shazam_artist: Optional[str]) -> bool:
    """Compare metadata artist with Shazam artist."""
    if not metadata_artist or not shazam_artist:
        return False
    a = normalize_text(metadata_artist)
    b = normalize_text(shazam_artist)
    return a == b or a in b or b in a
