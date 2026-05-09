"""
neuronpedia.py — Persistent cache for Neuronpedia feature labels and categories.

Cache key: "{layer}_{feat_idx}" — position is intentionally excluded.
The cache file is written after every SAVE_EVERY new fetches to survive interruptions.

Public API:
    load_cache(path) -> dict
    save_cache(cache, path) -> None
    get_feature_info(layer, feat_idx, cache, cache_path, override_map) -> dict
    fetch_and_cache_batch(features, cache, cache_path, override_map) -> dict
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Neuronpedia constants (from plan) ──────────────────────────────────────────
NEURONPEDIA_MODEL = "gemma-2-2b"
NEURONPEDIA_SET = "gemmascope-transcoder-16k"
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api/feature"
RATE_LIMIT_DELAY = 0.5   # seconds between requests
SAVE_EVERY = 10           # save cache to disk after this many new fetches


# ── Cache IO ───────────────────────────────────────────────────────────────────

def load_cache(path: str | Path) -> dict:
    """Load the Neuronpedia cache from disk. Returns empty dict if file absent."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: dict, path: str | Path) -> None:
    """Persist the cache to disk (atomic-ish via write + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(path)


# ── Single feature fetch ───────────────────────────────────────────────────────

def _fetch_feature(layer: int, feat_idx: int) -> dict | None:
    """
    Fetch one feature from Neuronpedia API.
    Returns a parsed dict or None on failure.
    """
    try:
        import urllib.request
        url = (
            f"{NEURONPEDIA_BASE_URL}"
            f"/{NEURONPEDIA_MODEL}"
            f"/{layer}-{NEURONPEDIA_SET}"
            f"/{feat_idx}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "circuit-fingerprint/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except Exception:
        return None


def _parse_feature_response(data: dict, layer: int, feat_idx: int) -> dict:
    """Extract label and top_tokens from a Neuronpedia API response dict."""
    label = ""
    top_tokens: list[str] = []

    # Neuronpedia v1 API: data["explanations"][0]["description"] for auto-interp label
    explanations = data.get("explanations") or []
    if explanations:
        first = explanations[0]
        label = first.get("description") or first.get("label") or ""

    # Fallback: direct "label" field
    if not label:
        label = data.get("label") or data.get("description") or ""

    # Top activating tokens
    top_acts = data.get("topkActivations") or data.get("top_activations") or []
    for act in top_acts[:10]:
        tok = act.get("token") or act.get("token_str") or ""
        if tok:
            top_tokens.append(str(tok))

    return {
        "layer": layer,
        "feat_idx": feat_idx,
        "label": label,
        "top_tokens": top_tokens,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "category": None,  # filled in by categorizer
    }


# ── Core get_feature_info ──────────────────────────────────────────────────────

def get_feature_info(
    layer: int,
    feat_idx: int,
    cache: dict,
    cache_path: str | Path | None = None,
    override_map: dict | None = None,
) -> dict:
    """
    Return feature info from cache, fetching from Neuronpedia if needed.

    Applies category overrides if override_map is provided.
    Saves cache to cache_path immediately after a new fetch if path given.
    """
    from utils.feature_categorizer import categorize_label, apply_category_overrides

    key = f"{layer}_{feat_idx}"
    if key not in cache:
        time.sleep(RATE_LIMIT_DELAY)
        raw = _fetch_feature(layer, feat_idx)
        if raw is not None:
            entry = _parse_feature_response(raw, layer, feat_idx)
        else:
            entry = {
                "layer": layer,
                "feat_idx": feat_idx,
                "label": "",
                "top_tokens": [],
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "category": "other",
            }
        # Assign category from label
        entry["category"] = categorize_label(entry["label"])
        # Apply manual override if present
        if override_map:
            entry["category"] = apply_category_overrides(
                layer, feat_idx, entry["category"], override_map
            )
        cache[key] = entry
        if cache_path is not None:
            save_cache(cache, cache_path)

    return cache[key]


# ── Batch fetch ────────────────────────────────────────────────────────────────

def fetch_and_cache_batch(
    features: list[tuple[int, int]],
    cache: dict,
    cache_path: str | Path | None = None,
    override_map: dict | None = None,
) -> dict:
    """
    Fetch all uncached (layer, feat_idx) pairs with rate limiting.
    Saves to cache_path every SAVE_EVERY new fetches.

    Returns the updated cache.
    """
    new_fetches = 0
    for layer, feat_idx in features:
        key = f"{layer}_{feat_idx}"
        if key not in cache:
            get_feature_info(layer, feat_idx, cache, cache_path=None, override_map=override_map)
            new_fetches += 1
            if cache_path is not None and new_fetches % SAVE_EVERY == 0:
                save_cache(cache, cache_path)
            time.sleep(RATE_LIMIT_DELAY)

    if cache_path is not None and new_fetches > 0:
        save_cache(cache, cache_path)

    return cache
