import json
import os
import hashlib
import time
from pathlib import Path

CACHE_DIR = Path("data/cache")
CACHE_EXPIRY_DAYS = 7  # refresh cache after 7 days


def get_cache_key(query: str, max_per_source: int) -> str:
    """Creates a unique key for this query."""
    raw = f"{query.lower().strip()}_{max_per_source}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_from_cache(query: str, max_per_source: int) -> list | None:
    """
    Returns cached papers if they exist and are less than 7 days old.
    Returns None if no valid cache exists.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = get_cache_key(query, max_per_source)
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)

        age_days = (time.time() - cached["timestamp"]) / 86400
        if age_days > CACHE_EXPIRY_DAYS:
            print(f"Cache expired ({age_days:.1f} days old)-fetching fresh")
            cache_file.unlink()
            return None

        print(
            f"Cache hit-loaded {len(cached['papers'])} papers (cached {age_days:.1f} days ago)")
        return cached["papers"]

    except Exception as e:
        print(f"Cache read error: {e}")
        return None


def save_to_cache(query: str, max_per_source: int, papers: list) -> None:
    """Saves retrieved papers to local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = get_cache_key(query, max_per_source)
    cache_file = CACHE_DIR / f"{key}.json"

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "query": query,
                "max_per_source": max_per_source,
                "timestamp": time.time(),
                "papers": papers
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(papers)} papers to cache")
    except Exception as e:
        print(f"Cache write error: {e}")


def list_cached_queries() -> list[dict]:
    """Returns all cached queries with their age and paper count."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = []

    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            age_days = (time.time() - data["timestamp"]) / 86400
            cached.append({
                "query": data["query"],
                "papers": len(data["papers"]),
                "age_days": round(age_days, 1),
                "file": cache_file.name
            })
        except Exception:
            continue

    return sorted(cached, key=lambda x: x["age_days"])


def clear_cache() -> int:
    """Deletes all cached results. Returns number of files deleted."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        cache_file.unlink()
        count += 1
    return count
