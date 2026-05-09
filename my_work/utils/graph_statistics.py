"""
graph_statistics.py — Compute structural statistics from circuit-tracer attribution graphs.

Superset diagnostic schema (backward-compatible):
  All original fields preserved + supervisor-requested fingerprint block.

Fixed constants (never change between phases):
    NODE_THRESHOLD = 0.8
    EDGE_THRESHOLD = 0.98
    TOP_K = 50
    TOP_K_SUPERVISOR = 20
    PRUNE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    N_LAYERS = 26

Public API:
    compute_statistics(graph, prompt_entry, phase) -> dict
    load_statistics(path) -> list[dict]
    save_statistics(stats, path) -> None
    append_statistic(stat, path) -> None
    aggregate_statistics(stats) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Fixed constants ────────────────────────────────────────────────────────────
N_LAYERS = 26
TOP_K = 50               # used for legacy top50_features field
TOP_K_SUPERVISOR = 20   # used for supervisor topk20 block
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
PRUNE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


# ── Statistics computation ─────────────────────────────────────────────────────

def compute_statistics(
    graph: Any,
    prompt_entry: dict,
    phase: str,
) -> dict:
    """
    Compute full superset structural statistics for one attribution graph.

    Parameters
    ----------
    graph       : circuit_tracer attribution graph object
    prompt_entry: the JSONL entry for this prompt
    phase       : 'base' or 'lora_triangle'

    Returns the output schema dict. Missing / failed metrics stored as None.
    Schema is extend-only: all legacy keys preserved, supervisor keys appended.
    """
    from circuit_tracer.utils.demo_utils import get_top_features

    task_type = prompt_entry.get("task_type", "binary")

    result: dict[str, Any] = {
        # ── metadata ───────────────────────────────────────────────────────────
        "prompt_id": prompt_entry["prompt_id"],
        "phase": phase,
        "task_type": task_type,
        "label": prompt_entry["label"],
        "label_token": prompt_entry["label_token"],
        "template_id": prompt_entry["template_id"],
        "attribution_succeeded": False,

        # ── verdict probabilities (original) ──────────────────────────────────
        "prob_true": None,
        "prob_false": None,
        "prob_target": None,
        "logit_gap": None,

        # ── active features (original) ─────────────────────────────────────────
        "n_active_features": None,

        # ── layer distribution (original) ─────────────────────────────────────
        "layer_distribution": None,   # length-26 normalised histogram

        # ── top-50 legacy block (original) ────────────────────────────────────
        "top50_features": None,       # [[layer, pos, feat_idx], ...]
        "mean_top50_score": None,
        "top10_over_top50": None,
        "layer_entropy": None,        # nats

        # ── pruned graph density at fixed threshold (original) ─────────────────
        "edge_density": None,         # at NODE_THRESHOLD=0.8, EDGE_THRESHOLD=0.98

        # ── error diagnostic (original) ───────────────────────────────────────
        "mean_error_node_weight": None,

        # ── NEW: layer scalar summary ──────────────────────────────────────────
        # Scalars derived from the active-feature layer histogram
        "layer_stats": None,          # {mean, std, median, entropy_bits}

        # ── NEW: pruning survival curve ───────────────────────────────────────
        # 7-entry list: one per threshold in PRUNE_THRESHOLDS
        # Each entry: {threshold, n_nodes_kept, n_edges_kept, edge_density}
        "prune_curve": None,

        # ── NEW: supervisor top-K=20 block ────────────────────────────────────
        # {features: [[layer,pos,feat_idx,score], ...],
        #  score_total, score_gini, layer_hist: length-26 vector}
        "topk20": None,
    }

    try:
        import math

        # ── Probabilities and logit gap ────────────────────────────────────────
        probs = graph.logit_probabilities.tolist()
        targets = [t.token_str for t in graph.logit_targets]
        eps = 1e-12

        if task_type == "binary":
            prob_true = prob_false = None
            for token_str, p in zip(targets, probs):
                if token_str.strip() == "True":
                    prob_true = p
                elif token_str.strip() == "False":
                    prob_false = p
            result["prob_true"] = prob_true
            result["prob_false"] = prob_false
            result["prob_target"] = prob_true if prompt_entry["label"] else prob_false
            if prob_true is not None and prob_false is not None:
                result["logit_gap"] = float(
                    math.log(prob_true + eps) - math.log(prob_false + eps)
                )
        else:
            label_tok = prompt_entry["label_token"].strip()
            prob_target = None
            for token_str, p in zip(targets, probs):
                if token_str.strip() == label_tok:
                    prob_target = p
                    break
            result["prob_target"] = prob_target
            if prob_target is not None:
                result["logit_gap"] = float(math.log(prob_target + eps))

        # ── Active feature count ───────────────────────────────────────────────
        n_active = int(graph.active_features.shape[0])
        result["n_active_features"] = n_active

        # ── Top-50 legacy block ────────────────────────────────────────────────
        features50, scores50 = get_top_features(graph, n=TOP_K)
        scores50_arr = np.array(scores50, dtype=float)

        result["top50_features"] = [list(f) for f in features50]

        layer_counts = [0] * N_LAYERS
        for (layer, _pos, _feat) in features50:
            if 0 <= layer < N_LAYERS:
                layer_counts[layer] += 1
        total_k = sum(layer_counts)
        layer_dist = [c / total_k for c in layer_counts] if total_k > 0 else [0.0] * N_LAYERS
        result["layer_distribution"] = layer_dist

        result["mean_top50_score"] = float(np.mean(np.abs(scores50_arr))) if len(scores50_arr) else 0.0

        abs50 = np.abs(scores50_arr)
        top50_sum = float(np.sum(abs50))
        top10_sum = float(np.sum(abs50[:10])) if len(abs50) >= 10 else top50_sum
        result["top10_over_top50"] = (top10_sum / top50_sum) if top50_sum > 0 else 0.0

        eps_e = 1e-12
        dist_arr = np.array(layer_dist, dtype=float)
        result["layer_entropy"] = float(-np.sum(dist_arr * np.log(dist_arr + eps_e)))

        # ── NEW: layer scalar summary ──────────────────────────────────────────
        # Build from active layer histogram using full active_features tensor
        # (not just top-K) for faithful representation of the whole graph
        all_layers = graph.active_features[:, 0].cpu().numpy().astype(int)
        layer_full_counts = np.bincount(all_layers, minlength=N_LAYERS).astype(float)
        layer_full_norm = layer_full_counts / layer_full_counts.sum() if layer_full_counts.sum() > 0 else layer_full_counts
        layer_indices = np.arange(N_LAYERS, dtype=float)

        layer_mean = float(np.sum(layer_indices * layer_full_norm))
        layer_std = float(np.sqrt(np.sum(layer_full_norm * (layer_indices - layer_mean) ** 2)))
        # weighted median
        cum = np.cumsum(layer_full_norm)
        median_layer = float(layer_indices[np.searchsorted(cum, 0.5)])
        # entropy in bits
        ent_bits = float(-np.sum(layer_full_norm * np.log2(layer_full_norm + 1e-12)))
        result["layer_stats"] = {
            "mean": layer_mean,
            "std": layer_std,
            "median": median_layer,
            "entropy_bits": ent_bits,
        }

        # ── Edge density at fixed threshold (original) ─────────────────────────
        try:
            from circuit_tracer.graph import prune_graph
            pres = prune_graph(graph, node_threshold=NODE_THRESHOLD, edge_threshold=EDGE_THRESHOLD)
            node_mask, edge_mask = pres.node_mask, pres.edge_mask
            n_nodes = int(node_mask.sum().item())
            n_edges = int(edge_mask.sum().item())
            max_edges = n_nodes * (n_nodes - 1)
            result["edge_density"] = (n_edges / max_edges) if max_edges > 0 else 0.0
        except Exception:
            result["edge_density"] = None

        # ── NEW: pruning survival curve ───────────────────────────────────────
        try:
            from circuit_tracer.graph import prune_graph
            curve = []
            for thresh in PRUNE_THRESHOLDS:
                try:
                    pr = prune_graph(graph, node_threshold=thresh, edge_threshold=thresh)
                    nm, em = pr.node_mask, pr.edge_mask
                    n_n = int(nm.sum().item())
                    n_e = int(em.sum().item())
                    mx = n_n * (n_n - 1)
                    density = (n_e / mx) if mx > 0 else 0.0
                    curve.append({
                        "threshold": thresh,
                        "n_nodes_kept": n_n,
                        "n_edges_kept": n_e,
                        "edge_density": density,
                    })
                except Exception:
                    curve.append({
                        "threshold": thresh,
                        "n_nodes_kept": None,
                        "n_edges_kept": None,
                        "edge_density": None,
                    })
            result["prune_curve"] = curve
        except Exception:
            result["prune_curve"] = None

        # ── NEW: supervisor top-K=20 block ────────────────────────────────────
        try:
            features20, scores20 = get_top_features(graph, n=TOP_K_SUPERVISOR)
            scores20_arr = np.array(scores20, dtype=float)
            abs20 = np.abs(scores20_arr)

            score_total = float(np.sum(abs20))
            # Gini coefficient of absolute scores
            if len(abs20) > 1 and abs20.sum() > 0:
                sorted_abs = np.sort(abs20)
                n = len(sorted_abs)
                gini_num = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_abs)
                gini = float(gini_num / (n * sorted_abs.sum()))
            else:
                gini = 0.0

            # layer histogram for top-20
            lh = [0] * N_LAYERS
            for (layer, _pos, _feat) in features20:
                if 0 <= layer < N_LAYERS:
                    lh[layer] += 1
            lh_total = sum(lh)
            lh_norm = [c / lh_total for c in lh] if lh_total > 0 else [0.0] * N_LAYERS

            result["topk20"] = {
                "features": [[layer, pos, feat, float(sc)]
                              for (layer, pos, feat), sc in zip(features20, scores20)],
                "score_total": score_total,
                "score_gini": gini,
                "layer_hist": lh_norm,
            }
        except Exception:
            result["topk20"] = None

        # ── Mean error node weight (original) ─────────────────────────────────
        try:
            from circuit_tracer.graph import prune_graph
            pr_all = prune_graph(graph, node_threshold=1.0, edge_threshold=1.0)
            node_scores = pr_all.cumulative_scores
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
    Extract error node weight. Tries multiple attribute-access patterns.
    Error nodes represent the CLT reconstruction residual.
    """
    scores_arr = node_scores.numpy() if hasattr(node_scores, "numpy") else np.array(node_scores)

    if hasattr(graph, "node_types"):
        error_indices = [
            i for i, t in enumerate(graph.node_types)
            if str(t).lower() in ("error", "error_node")
        ]
        if error_indices:
            return float(np.mean(np.abs(scores_arr[error_indices])))

    if hasattr(graph, "error_node_indices"):
        idx = graph.error_node_indices
        if len(idx) > 0:
            return float(np.mean(np.abs(scores_arr[idx])))

    # Fallback: last N_LAYERS entries are error nodes by convention
    if len(scores_arr) >= N_LAYERS:
        return float(np.mean(np.abs(scores_arr[-N_LAYERS:])))

    return None


# ── IO helpers ─────────────────────────────────────────────────────────────────

def load_statistics(path: str | Path) -> list[dict]:
    """
    Load stats from disk.

    Backward/robust behavior:
    - Missing file -> []
    - Empty/whitespace file -> []
    - Standard JSON array -> list[dict]
    - JSONL fallback (one JSON object per line) -> list[dict]
    """
    path = Path(path)
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    # Primary format: JSON array.
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON list at {path}, got {type(data)}")
    except json.JSONDecodeError:
        # Fallback: JSONL (useful if an interrupted write left line-based data).
        rows: list[dict] = []
        for i, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse stats file {path} as JSON array or JSONL; "
                    f"first invalid line={i}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid JSONL entry at {path}:{i}; expected object, got {type(obj)}"
                )
            rows.append(obj)
        return rows


def save_statistics(stats: list[dict], path: str | Path) -> None:
    """Write a stats list to disk as a JSON array (atomic replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.flush()
    tmp.replace(path)


def append_statistic(stat: dict, path: str | Path) -> None:
    """
    Append a single stat entry to a JSON array file, creating it if needed.
    Enables checkpoint-style saving: call after each prompt.
    """
    path = Path(path)
    existing = load_statistics(path)
    existing.append(stat)
    save_statistics(existing, path)


# ── Aggregation ────────────────────────────────────────────────────────────────

_SCALAR_METRICS = [
    # original
    "n_active_features",
    "edge_density",
    "mean_top50_score",
    "top10_over_top50",
    "layer_entropy",
    "mean_error_node_weight",
    "logit_gap",
    "prob_target",
    # new scalars derived from nested blocks
    "layer_stats_mean",
    "layer_stats_std",
    "layer_stats_median",
    "layer_stats_entropy_bits",
    "topk20_score_total",
    "topk20_score_gini",
]


def _flatten_nested(stat: dict) -> dict:
    """Flatten nested supervisor blocks into scalar keys for aggregation."""
    flat = dict(stat)
    ls = stat.get("layer_stats") or {}
    flat["layer_stats_mean"] = ls.get("mean")
    flat["layer_stats_std"] = ls.get("std")
    flat["layer_stats_median"] = ls.get("median")
    flat["layer_stats_entropy_bits"] = ls.get("entropy_bits")

    t20 = stat.get("topk20") or {}
    flat["topk20_score_total"] = t20.get("score_total")
    flat["topk20_score_gini"] = t20.get("score_gini")

    return flat


def aggregate_statistics(stats: list[dict]) -> dict:
    """
    Compute mean / std / median / IQR for each scalar metric across all
    successfully attributed prompts. Handles both original and new nested fields.
    """
    succeeded = [s for s in stats if s.get("attribution_succeeded")]
    result: dict[str, Any] = {
        "n_total": len(stats),
        "n_succeeded": len(succeeded),
        "success_rate": len(succeeded) / len(stats) if stats else 0.0,
    }

    flat_succeeded = [_flatten_nested(s) for s in succeeded]

    for metric in _SCALAR_METRICS:
        vals = [s[metric] for s in flat_succeeded if s.get(metric) is not None]
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

    # Aggregate prune_curve per threshold across all prompts
    all_curves = [s.get("prune_curve") for s in succeeded if s.get("prune_curve")]
    if all_curves:
        curve_agg = {}
        for entry in all_curves:
            for pt in entry:
                t = pt["threshold"]
                if t not in curve_agg:
                    curve_agg[t] = {"n_nodes": [], "n_edges": [], "density": []}
                if pt["n_nodes_kept"] is not None:
                    curve_agg[t]["n_nodes"].append(pt["n_nodes_kept"])
                if pt["n_edges_kept"] is not None:
                    curve_agg[t]["n_edges"].append(pt["n_edges_kept"])
                if pt["edge_density"] is not None:
                    curve_agg[t]["density"].append(pt["edge_density"])
        result["prune_curve_agg"] = {
            str(t): {
                "mean_n_nodes": float(np.mean(v["n_nodes"])) if v["n_nodes"] else None,
                "mean_n_edges": float(np.mean(v["n_edges"])) if v["n_edges"] else None,
                "mean_density": float(np.mean(v["density"])) if v["density"] else None,
            }
            for t, v in sorted(curve_agg.items())
        }

    return result
