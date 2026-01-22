import os
import logging
from typing import Optional, Dict, List

try:
    from mutagen import File as MutagenFile
except Exception:
    MutagenFile = None
    logging.warning("mutagen not installed; install with: python -m pip install mutagen")

import requests
from rapidfuzz import fuzz
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Read Spotify credentials from environment variables if available
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

SUPPORTED_EXTENSIONS = (".mp3", ".flac", ".wav", ".aiff", ".m4a")


def _read_file_tags(file_path: str) -> Dict:
    """Return a dict with tag info for the given audio file using mutagen.

    Fields: title, artist, album, duration (seconds), genres (list), track_number
    """
    if MutagenFile is None:
        return {}
    try:
        mf = MutagenFile(file_path, easy=True)
    except Exception as e:
        logging.warning("Failed to open %s with mutagen: %s", file_path, e)
        return {}

    if mf is None:
        logging.warning("Unsupported or invalid audio file: %s", file_path)
        return {}

    tags = {}
    tags["title"] = mf.tags.get("title", [os.path.splitext(os.path.basename(file_path))[0]])[0] if mf.tags else os.path.splitext(os.path.basename(file_path))[0]
    tags["artist"] = mf.tags.get("artist", [None])[0] if mf.tags else None
    tags["album"] = mf.tags.get("album", [None])[0] if mf.tags else None
    tags["genres"] = mf.tags.get("genre", []) if mf.tags else []
    track = None
    if mf.tags:
        tr = mf.tags.get("tracknumber") or mf.tags.get("track")
        if tr:
            try:
                track = int(str(tr[0]).split("/")[0])
            except Exception:
                track = None
    tags["track_number"] = track

    try:
        tags["duration"] = int(mf.info.length) if mf.info and getattr(mf.info, "length", None) else None
    except Exception:
        tags["duration"] = None

    return tags


def _spotify_token() -> Optional[str]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        logging.warning("Spotify token fetch failed: %s", e)
        return None


def _search_spotify(artist: str, track: str, token: str) -> List[Dict]:
    try:
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": f"track:{track} artist:{artist}", "type": "track", "limit": 5}
        r = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("tracks", {}).get("items", [])
    except Exception as e:
        logging.warning("Spotify lookup failed: %s", e)
        return []


def _score_album(candidate: Dict, track: str, artist: str, duration: Optional[int] = None) -> float:
    score = 0.0
    title_score = fuzz.partial_ratio(candidate.get("title", ""), track)
    artist_score = fuzz.partial_ratio(candidate.get("artist", ""), artist)
    score += title_score * 0.4
    score += artist_score * 0.4
    if duration and "duration" in candidate:
        diff = abs((candidate.get("duration") or 0) - duration)
        score += max(0, 100 - diff) * 0.2
    return score


def identify_track(file_path: str) -> Optional[Dict]:
    """Identify track metadata for a source audio file.

    Returns a dict with fields: title, track_number, genres, styles.
    """
    filename = os.path.basename(file_path)

    # Read local tags first (mp3tag-like)
    tags = _read_file_tags(file_path)
    tag_artist = tags.get("artist") if tags else None
    tag_title = tags.get("title") if tags else None
    ref_duration = tags.get("duration") if tags else None

    candidates: List[Dict] = []
    if tags:
        candidates.append({
            "title": tag_title or filename,
            "artist": tag_artist or "",
            "duration": ref_duration,
            "source": "filetags",
        })

    # Spotify candidates if credentials exist and we have some tag info
    token = _spotify_token()
    if token and (tag_artist or tag_title):
        sp_items = _search_spotify(tag_artist or "", tag_title or "", token)
        for it in sp_items:
            cand_title = it.get("name")
            cand_artist = ", ".join([a.get("name") for a in it.get("artists", [])])
            dur = it.get("duration_ms")
            candidates.append({
                "title": cand_title or "",
                "artist": cand_artist or (tag_artist or ""),
                "duration": (dur // 1000) if dur else None,
                "source": "spotify",
            })

    if not candidates:
        return None

    # Score candidates
    best_album = None
    best_score = -1
    for cand in candidates:
        score = _score_album(cand, tag_title or filename, tag_artist or "", duration=ref_duration)
        if score > best_score:
            best_score = score
            best_album = cand

    if not best_album:
        return None

    return {
        "title": best_album.get("title"),
        "track_number": tags.get("track_number") if tags else None,
        "genres": tags.get("genres", []) if tags else [],
        "styles": [],  # placeholder; could be enriched with other sources
    }
