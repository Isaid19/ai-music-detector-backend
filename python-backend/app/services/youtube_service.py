"""YouTube search service for official videos and cover detection."""

import asyncio
from typing import Any, Dict, List, Optional

from app.config import COVER_KEYWORDS, OFFICIAL_VIDEO_KEYWORDS, YOUTUBE_MAX_RESULTS, YOUTUBE_SEARCH_TIMEOUT
from app.utils import contains_any, normalize_text


def _pick_best_video(videos: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    """Pick the most relevant result based on title similarity and view count."""
    if not videos:
        return None

    query_tokens = set(normalize_text(query).split())

    def score(video: Dict[str, Any]) -> float:
        title = normalize_text(video.get("title", ""))
        title_tokens = set(title.split())
        overlap = len(query_tokens & title_tokens) / max(len(query_tokens), 1)
        is_official = 1.0 if contains_any(title, OFFICIAL_VIDEO_KEYWORDS) else 0.0
        views = str(video.get("viewCount", {}).get("text", "0")).replace(",", "").replace(" views", "")
        try:
            view_count = float(views) if views else 0.0
        except ValueError:
            view_count = 0.0
        # Normalize view count to 0-1 (log scale)
        view_score = min(1.0, math_log_view(view_count))
        return overlap * 3.0 + is_official * 2.0 + view_score * 1.0

    return max(videos, key=score)


def math_log_view(value: float) -> float:
    if value <= 0:
        return 0.0
    import math
    return math.log10(value) / 8.0  # 100M views -> 1.0


def _search_sync(query: str) -> List[Dict[str, Any]]:
    try:
        from youtubesearchpython import VideosSearch
        search = VideosSearch(query, limit=YOUTUBE_MAX_RESULTS)
        result = search.result()
        if isinstance(result, dict):
            return result.get("result", []) or []
        return []
    except Exception as e:
        return [{"error": str(e)}]


async def search_youtube(query: str) -> List[Dict[str, Any]]:
    """Async wrapper around youtubesearchpython."""
    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _search_sync, query),
            timeout=YOUTUBE_SEARCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return [{"error": "YouTube search timed out"}]
    except Exception as e:
        return [{"error": str(e)}]


async def verify_official_video(title: Optional[str], artist: Optional[str]) -> Dict[str, Any]:
    """Search for an official music video and return a structured verdict."""
    if not title:
        return {
            "found": False,
            "is_official": False,
            "is_cover": False,
            "best_video": None,
            "reasoning": "No title available to search YouTube.",
        }

    query_parts = [title]
    if artist:
        query_parts.append(artist)
    query_parts.append("official music video")
    query = " ".join(query_parts)

    videos = await search_youtube(query)
    if not videos or (len(videos) == 1 and "error" in videos[0]):
        return {
            "found": False,
            "is_official": False,
            "is_cover": False,
            "best_video": None,
            "reasoning": "Could not retrieve YouTube results.",
        }

    best = _pick_best_video(videos, query)
    if not best:
        return {
            "found": False,
            "is_official": False,
            "is_cover": False,
            "best_video": None,
            "reasoning": "No relevant YouTube results found.",
        }

    best_title = normalize_text(best.get("title", ""))
    is_official = contains_any(best_title, OFFICIAL_VIDEO_KEYWORDS)
    is_cover = contains_any(best_title, COVER_KEYWORDS) and not is_official

    return {
        "found": True,
        "is_official": is_official,
        "is_cover": is_cover,
        "best_video": {
            "title": best.get("title"),
            "link": best.get("link"),
            "duration": best.get("duration"),
            "view_count": best.get("viewCount", {}).get("text"),
            "published_time": best.get("publishedTime"),
            "channel_name": best.get("channel", {}).get("name"),
        },
        "reasoning": (
            "Official video found on YouTube." if is_official else
            "Cover or non-official version found on YouTube." if is_cover else
            "Video found but not clearly marked as official."
        ),
    }


async def check_cover_existence(title: Optional[str], artist: Optional[str]) -> Dict[str, Any]:
    """Search specifically for cover versions to estimate whether the input is a cover."""
    if not title:
        return {"is_likely_cover": False, "reasoning": "No title available.", "cover_results": []}

    query = f"{title} {artist or ''} cover".strip()
    videos = await search_youtube(query)
    cover_results = []
    for v in videos:
        if not isinstance(v, dict) or "error" in v:
            continue
        t = normalize_text(v.get("title", ""))
        if contains_any(t, COVER_KEYWORDS):
            cover_results.append({
                "title": v.get("title"),
                "link": v.get("link"),
                "channel": v.get("channel", {}).get("name"),
            })

    return {
        "is_likely_cover": len(cover_results) > 0,
        "reasoning": f"Found {len(cover_results)} cover-like results on YouTube." if cover_results else "No obvious cover results on YouTube.",
        "cover_results": cover_results[:3],
    }
