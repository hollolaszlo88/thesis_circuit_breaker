"""
feature_categorizer.py — Keyword-based feature category classification.

Priority order: geometry > boolean-logic > language-comparative > math-general
                > format-template > other

Public API:
    categorize_label(label) -> str
    apply_category_overrides(layer, feat_idx, category, override_map) -> str
    compute_fingerprint(stats, cache) -> dict
    compute_position_heatmap(stats, cache) -> dict
    get_top_features_by_category(stats, cache, category, n) -> list
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# ── Category keyword lists (from plan) ────────────────────────────────────────
CATEGORIES: dict[str, list[str]] = {
    "geometry": [
        "triangle", "triangles", "triangular", "side", "sides", "angle", "angles",
        "polygon", "polygons", "hypotenuse", "vertex", "vertices", "geometric",
        "equilateral", "isosceles", "shape", "shapes", "spatial",
    ],
    "boolean-logic": [
        "true", "false", "boolean", "logical", "logic", "conditional",
        "assertion", "assert", "truth", "falsity", "predicate", "valid", "invalid",
    ],
    "language-comparative": [
        "greater", "less", "more", "fewer", "larger", "smaller", "than",
        "comparison", "comparative", "ordinal", "ordering", "rank", "ranking",
        "exceed", "exceeds", "above", "below", "higher", "lower",
    ],
    "math-general": [
        "equation", "equations", "formula", "arithmetic", "calculation",
        "mathematical", "math", "number", "integer", "expression", "notation",
        "algebra", "inequality", "inequalities", "operator", "symbol", "symbols",
    ],
    "format-template": [
        "answer", "statement", "delimiter", "punctuation", "colon", "period",
        "template", "format", "instruction", "prompt", "question", "syntax",
    ],
    "other": [],  # catch-all
}

# Priority order: first match wins
_PRIORITY = [
    "geometry",
    "boolean-logic",
    "language-comparative",
    "math-general",
    "format-template",
    "other",
]


# ── Core classification ────────────────────────────────────────────────────────

def categorize_label(label: str) -> str:
    """
    Classify a Neuronpedia label string into one of six categories.
    Returns the highest-priority category whose keywords appear in the label.
    Falls back to 'other' if no match.
    """
    lower = label.lower()
    for cat in _PRIORITY:
        keywords = CATEGORIES[cat]
        if any(kw in lower for kw in keywords):
            return cat
    return "other"


def apply_category_overrides(
    layer: int,
    feat_idx: int,
    category: str,
    override_map: dict,
) -> str:
    """
    Apply a manual category override for a specific (layer, feat_idx) pair.
    override_map keys are "{layer}_{feat_idx}" strings.
    Returns the override category if present, otherwise returns category unchanged.
    """
    key = f"{layer}_{feat_idx}"
    return override_map.get(key, category)


# ── Fingerprint ────────────────────────────────────────────────────────────────

def compute_fingerprint(
    stats: list[dict],
    cache: dict,
    override_map: dict | None = None,
) -> dict[str, float]:
    """
    Aggregate top-50 feature appearances across all prompts into category proportions.

    Returns dict: {category: proportion} summing to 1.0.
    Only uses successfully attributed entries that have top50_features.
    """
    from utils.neuronpedia import get_feature_info

    category_counts: dict[str, int] = defaultdict(int)
    total = 0

    for entry in stats:
        if not entry.get("attribution_succeeded"):
            continue
        for (layer, _pos, feat_idx) in (entry.get("top50_features") or []):
            info = get_feature_info(layer, feat_idx, cache, override_map=override_map)
            category_counts[info["category"]] += 1
            total += 1

    if total == 0:
        return {cat: 0.0 for cat in _PRIORITY}

    return {cat: category_counts.get(cat, 0) / total for cat in _PRIORITY}


# ── Position heatmap ──────────────────────────────────────────────────────────

def compute_position_heatmap(
    stats: list[dict],
    cache: dict,
    override_map: dict | None = None,
) -> dict[str, dict[int, int]]:
    """
    Returns {category: {token_position: count}} for all successfully attributed prompts.
    """
    from utils.neuronpedia import get_feature_info

    position_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for entry in stats:
        if not entry.get("attribution_succeeded"):
            continue
        for (layer, pos, feat_idx) in (entry.get("top50_features") or []):
            info = get_feature_info(layer, feat_idx, cache, override_map=override_map)
            position_counts[info["category"]][pos] += 1

    return {cat: dict(positions) for cat, positions in position_counts.items()}


# ── Top features per category ──────────────────────────────────────────────────

def get_top_features_by_category(
    stats: list[dict],
    cache: dict,
    category: str,
    n: int = 10,
    override_map: dict | None = None,
) -> list[dict]:
    """
    Return the top-n most frequently appearing (layer, feat_idx) pairs
    for a given category, with their Neuronpedia labels.

    Returns list of dicts: {layer, feat_idx, label, count}.
    """
    from utils.neuronpedia import get_feature_info

    freq: dict[tuple[int, int], int] = defaultdict(int)

    for entry in stats:
        if not entry.get("attribution_succeeded"):
            continue
        for (layer, _pos, feat_idx) in (entry.get("top50_features") or []):
            info = get_feature_info(layer, feat_idx, cache, override_map=override_map)
            if info["category"] == category:
                freq[(layer, feat_idx)] += 1

    sorted_feats = sorted(freq.items(), key=lambda x: -x[1])[:n]
    result = []
    for (layer, feat_idx), count in sorted_feats:
        info = cache.get(f"{layer}_{feat_idx}", {})
        result.append({
            "layer": layer,
            "feat_idx": feat_idx,
            "label": info.get("label", ""),
            "count": count,
        })
    return result
