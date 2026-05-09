"""
graph_statistics.py — Compute structural statistics from circuit-tracer attribution graphs.

All thresholds are fixed constants (never change between phases):
    NODE_THRESHOLD = 0.8
    EDGE_THRESHOLD = 0.98
    TOP_K = 50
    N_LAYERS = 26

Public API:
    compute_statistics(graph, prompt_entry, phase) -> dict
    load_statistics(path) -> list[dict]
    save_statistics(stats, path) -> None
    aggregate_statistics(stats) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Fixed constants (plan §Critical Constants) ─────────────────────────────────
N_LAYERS = 26
TOP_K = 50
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98


# ── Statistics computation ─────────────────────────────────────────────────────

def compute_statistics(
    graph: Any,
    prompt_entry: dict,
    phase: str,
) -> dict:
    """
    Compute all structural statistics for one attribution graph.

    Parameters
    ----------
    graph       : circuit_tracer attribution graph object
    prompt_entry: the JSONL entry for this prompt (used for metadata fields)
    phase       : 'base' or 'lora_triangle'

    Returns the full output schema dict as specified in the plan.
    Missing / failed metrics are stored as None.
    """
    from circuit_tracer.utils.demo_utils import get_top_features

    result: dict[str, Any] = {
        "prompt_id": prompt_entry["prompt_id"],
        "phase": phase,
        "label": prompt_entry["label"],
        "label_token": prompt_entry["label_token"],
        "template_id": prompt_entry["template_id"],
        "attribution_succeeded": False,
        "prob_true": None,
        "prob_false": None,
        "logit_gap": None,
        "n_active_features": None,
        "layer_distribution": None,
        "edge_density": None,
        "mean_top50_score": None,
        "top10_over_top50": None,
        "layer_entropy": None,
        "mean_error_node_weight": None,
        "top50_features": None,
    }

    try:
        # ── Probabilities and logit gap ────────────────────────────────────────
        probs = graph.logit_probabilities.tolist()
        targets = [t.token_str for t in graph.logit_targets]
        prob_true = prob_false = None
        for token_str, p in zip(targets, probs):
            if token_str.strip() == "True":
                prob_true = p
            elif token_str.strip() == "False":
                prob_false = p
        result["prob_true"] = prob_true
        result["prob_false"] = prob_false
        if prob_true is not None and prob_false is not None:
            # logit_gap is log(p_true/p_false) ≈ logit(True) - logit(False)
            # Use the raw logit if accessible, otherwise approximate from probs
            try:
                logits_attr = graph.logit_attributions  # shape: (n_targets,)
                # fallback: difference of prob logits
                import math
                eps = 1e-12
                result["logit_gap"] = float(
                    math.log(prob_true + eps) - math.log(prob_false + eps)
                )
            except Exception:
                pass

        # ── Active feature count ───────────────────────────────────────────────
        n_active = int(graph.active_features.shape[0])
        result["n_active_features"] = n_active

        # ── Top-K features and scores ──────────────────────────────────────────
        features, scores = get_top_features(graph, n=TOP_K)
        scores_arr = np.array(scores, dtype=float)

        # Store top-50 features as [[layer, pos, feat_idx], ...]
        result["top50_features"] = [list(f) for f in features]

        # Layer distribution (normalised histogram over N_LAYERS)
        layer_counts = [0] * N_LAYERS
        for (layer, _pos, _feat) in features:
            if 0 <= layer < N_LAYERS:
                layer_counts[layer] += 1
        total = sum(layer_counts)
        if total > 0:
            layer_dist = [c / total for c in layer_counts]
        else:
            layer_dist = [0.0] * N_LAYERS
        result["layer_distribution"] = layer_dist

        # Mean top-50 influence score
        result["mean_top50_score"] = float(np.mean(np.abs(scores_arr))) if len(scores_arr) else 0.0

        # Influence concentration: top-10 / top-50 (by absolute value)
        abs_scores = np.abs(scores_arr)
        top50_sum = float(np.sum(abs_scores))
        if len(abs_scores) >= 10:
            top10_sum = float(np.sum(abs_scores[:10]))
        else:
            top10_sum = top50_sum
        result["top10_over_top50"] = (top10_sum / top50_sum) if top50_sum > 0 else 0.0

        # Layer entropy of top-50 distribution
        eps = 1e-12
        dist_arr = np.array(layer_dist, dtype=float)
        result["layer_entropy"] = float(
            -np.sum(dist_arr * np.log(dist_arr + eps))
        )

        # ── Edge density after pruning ─────────────────────────────────────────
        try:
            from circuit_tracer.graph import prune_graph
            node_mask, edge_mask, _ = prune_graph(
                graph,
                node_threshold=NODE_THRESHOLD,
                edge_threshold=EDGE_THRESHOLD,
            )
            n_nodes = int(node_mask.sum().item())
            n_edges = int(edge_mask.sum().item())
            max_edges = n_nodes * (n_nodes - 1)
            result["edge_density"] = (n_edges / max_edges) if max_edges > 0 else 0.0
        except Exception:
            # prune_graph API may differ; leave as None if unavailable
            result["edge_density"] = None

        # ── Mean error node weight ─────────────────────────────────────────────
        try:
            from circuit_tracer.graph import prune_graph
            _, _, node_scores = prune_graph(
                graph,
                node_threshold=1.0,
                edge_threshold=1.0,
            )
            # Identify error nodes by type attribute
            error_weight = _compute_error_node_weight(graph, node_scores)
            result["mean_error_node_weight"] = error_weight
        except Exception:
            result["mean_error_node_weight"] = None

        result["attribution_succeeded"] = True

    except Exception as exc:
        result["attribution_succeeded"] = False
        result["_error"] = str(exc)

    return result


def _compute_error_node_weight(graph: Any, node_scores: Any) -> float | None:
    """
    Extract error node weight. Tries multiple approaches for API resilience.
    Error nodes represent the residual discrepancy between CLT reconstruction
    and true MLP output.
    """
    scores_arr = node_scores.numpy() if hasattr(node_scores, "numpy") else np.array(node_scores)

    # Approach 1: graph.node_types attribute
    if hasattr(graph, "node_types"):
        error_indices = [
            i for i, t in enumerate(graph.node_types)
            if str(t).lower() in ("error", "error_node")
        ]
        if error_indices:
            return float(np.mean(np.abs(scores_arr[error_indices])))

    # Approach 2: graph has an 'error_node_indices' attribute
    if hasattr(graph, "error_node_indices"):
        idx = graph.error_node_indices
        if len(idx) > 0:
            return float(np.mean(np.abs(scores_arr[idx])))

    # Approach 3: error nodes are the last N_LAYERS nodes
    if len(scores_arr) >= N_LAYERS:
        error_scores = scores_arr[-N_LAYERS:]
        return float(np.mean(np.abs(error_scores)))

    return None


# ── IO helpers ─────────────────────────────────────────────────────────────────

def load_statistics(path: str | Path) -> list[dict]:
    """Load a stats JSON file (list of dicts) from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list at {path}, got {type(data)}")


def save_statistics(stats: list[dict], path: str | Path) -> None:
    """Write a stats list to disk as a JSON array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def append_statistic(stat: dict, path: str | Path) -> None:
    """
    Append a single stat entry to a JSON array file, creating it if needed.
    Enables checkpoint-style saving: call after each prompt to survive interruptions.
    """
    path = Path(path)
    if path.exists():
        existing = load_statistics(path)
    else:
        existing = []
    existing.append(stat)
    save_statistics(existing, path)


# ── Aggregation ────────────────────────────────────────────────────────────────

_SCALAR_METRICS = [
    "n_active_features",
    "edge_density",
    "mean_top50_score",
    "top10_over_top50",
    "layer_entropy",
    "mean_error_node_weight",
    "logit_gap",
]


def aggregate_statistics(stats: list[dict]) -> dict:
    """
    Compute mean / std / median / IQR for each scalar metric across all
    successfully attributed prompts.
    """
    succeeded = [s for s in stats if s.get("attribution_succeeded")]
    result: dict[str, Any] = {
        "n_total": len(stats),
        "n_succeeded": len(succeeded),
        "success_rate": len(succeeded) / len(stats) if stats else 0.0,
    }
    for metric in _SCALAR_METRICS:
        vals = [s[metric] for s in succeeded if s.get(metric) is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            q1, q3 = np.percentile(arr, [25, 75])
            result[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "iqr": float(q3 - q1),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(vals),
            }
        else:
            result[metric] = None
    return result
