"""
structural_classification.py — Shared classifier helpers for multi-phase
structural analysis, aligned with §5 / §5b of 03_baseline_structural_analysis.

Public API
----------
TF_FAMILIES
    frozenset of family names used for binary T/F classification.

build_feature_cols_layer_stats(sample_flat) -> list[str]
    §5-style feature vector: aggregate scalars + dynamic prune-curve columns.

build_feature_cols_layer_hist(feature_cols_layer_stats) -> list[str]
    §5b-style: same but replaces four layer_stats_* aggregates with
    per-layer active-feature counts layer_hist_0 … layer_hist_{N_LAYERS-1}.

build_matrix(stats_raw, stats_flat, feature_cols, mode) -> (X, y_label, y_tail, families)
    Build numpy arrays from matched raw/flat stat pairs (binary-TF rows only).

run_lofo(X, y, groups, is_multiclass, n_splits, random_state) -> dict
    StandardScaler + LogisticRegression with 5-fold stratified CV and
    LeaveOneGroupOut(groups). Returns a result dict with cv5_mean, loo_details, etc.

format_lofo_result(result, phase_name, feature_variant, target) -> str
    Human-readable block for printing.

eval_phase_lofo(phase_raw, phase_flat, common_ids, phase_name) -> dict
    Run all four combinations (layer_stats / layer_hist) × (label / tail)
    on the rows in common_ids.  Returns nested dict keyed by (feat_mode, target).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.graph_statistics import N_LAYERS, _binary_label_true, _flatten_nested

# ── Constants ──────────────────────────────────────────────────────────────────

TF_FAMILIES: frozenset[str] = frozenset({"numeric_validity", "geometry_claim"})

_BASE_FEATURE_COLS: list[str] = [
    "n_active_features",
    "edge_density",
    "mean_top50_score",
    "top10_over_top50",
    "layer_entropy",
    "mean_error_node_weight",
    "layer_stats_mean",
    "layer_stats_std",
    "layer_stats_median",
    "layer_stats_entropy_bits",
    "topk20_score_total",
    "topk20_score_gini",
]

_PRUNE_PREFIXES: tuple[str, ...] = (
    "density_at_",
    "n_kept_at_",
    "n_edges_at_",
    "n_nodes_total_at_",
    "n_edges_total_at_",
)

_AGG_LAYER_HOLDOUT: frozenset[str] = frozenset({
    "layer_stats_mean",
    "layer_stats_std",
    "layer_stats_median",
    "layer_stats_entropy_bits",
})


# ── Feature column builders ────────────────────────────────────────────────────

def build_feature_cols_layer_stats(sample_flat: dict) -> list[str]:
    """Column list for §5-style feature vector (aggregate layer stats + prune columns).

    Parameters
    ----------
    sample_flat : one flattened stat dict; used only to discover available prune keys.
    """
    prune_cols = sorted(k for k in sample_flat if k.startswith(_PRUNE_PREFIXES))
    return _BASE_FEATURE_COLS + prune_cols


def build_feature_cols_layer_hist(feature_cols_layer_stats: list[str]) -> list[str]:
    """Column list for §5b-style feature vector.

    Identical to `feature_cols_layer_stats` except the four aggregate
    ``layer_stats_*`` entries are replaced by ``layer_hist_0 … layer_hist_{N-1}``,
    inserted after ``mean_error_node_weight``.
    """
    out: list[str] = []
    for col in feature_cols_layer_stats:
        if col in _AGG_LAYER_HOLDOUT:
            continue
        out.append(col)
        if col == "mean_error_node_weight":
            out.extend(f"layer_hist_{i}" for i in range(N_LAYERS))
    return out


# ── Matrix builder ─────────────────────────────────────────────────────────────

def build_matrix(
    stats_raw: list[dict],
    stats_flat: list[dict],
    feature_cols: list[str],
    mode: Literal["layer_stats", "layer_hist"] = "layer_stats",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y_label, y_tail, families) from binary-T/F rows.

    Parameters
    ----------
    stats_raw   : raw stat dicts (one per prompt, already filtered to common_ids if desired)
    stats_flat  : corresponding flattened dicts (from _flatten_nested)
    feature_cols: list of column names (from build_feature_cols_layer_stats or _layer_hist)
    mode        : 'layer_stats' reads every column from flat dict;
                  'layer_hist'  additionally reads layer_hist array from raw stat for
                  layer_hist_* columns (raw stat may have length != N_LAYERS — padded/truncated).

    Returns
    -------
    X         : float array (n_valid, n_features)
    y_label   : int array (n_valid,)  — 1=True, 0=False
    y_tail    : object array (n_valid,) — tail string
    families  : object array (n_valid,) — family string

    All four arrays are empty (shape (0, …)) when no valid rows exist.
    """
    X_rows: list[list[float]] = []
    y_label_list: list[int] = []
    y_tail_list: list[str] = []
    families_list: list[str] = []

    for raw, flat in zip(stats_raw, stats_flat):
        # Precompute layer_hist padding for layer_hist mode
        lh_pad: list[float] | None = None
        if mode == "layer_hist":
            lh = raw.get("layer_hist")
            if lh is None:
                continue
            lh = list(lh)
            if len(lh) > N_LAYERS:
                lh = lh[:N_LAYERS]
            lh_pad = [float(x) for x in lh] + [0.0] * (N_LAYERS - len(lh))

        row: list[float] = []
        ok = True
        for col in feature_cols:
            if mode == "layer_hist" and col.startswith("layer_hist_"):
                idx = int(col.rsplit("_", 1)[-1])
                row.append(lh_pad[idx])  # type: ignore[index]
            else:
                v = flat.get(col)
                if v is None:
                    ok = False
                    break
                try:
                    row.append(float(v))
                except (TypeError, ValueError):
                    ok = False
                    break
        if not ok:
            continue

        try:
            lbl = int(_binary_label_true(raw["label"]))
        except Exception:
            continue

        X_rows.append(row)
        y_label_list.append(lbl)
        y_tail_list.append(str(raw.get("tail", "?")))
        families_list.append(str(raw.get("family", "?")))

    n_feats = len(feature_cols)
    if not X_rows:
        return (
            np.empty((0, n_feats), dtype=float),
            np.empty(0, dtype=int),
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
        )

    return (
        np.array(X_rows, dtype=float),
        np.array(y_label_list, dtype=int),
        np.array(y_tail_list, dtype=object),
        np.array(families_list, dtype=object),
    )


# ── LOFO runner ────────────────────────────────────────────────────────────────

def run_lofo(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    is_multiclass: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run StandardScaler + LogisticRegression with 5-fold CV and LOFO.

    Parameters
    ----------
    X           : feature matrix (n_samples, n_features)
    y           : target array — int for binary label, object/str for tail
    groups      : group array for LeaveOneGroupOut (family strings)
    is_multiclass: True → encode string labels, compute chance=1/n_classes
    n_splits    : k for stratified k-fold CV (capped at minimum class size)
    random_state: reproducibility seed

    Returns
    -------
    dict with keys: n, majority, cv5_mean, cv5_std, loo_details (list of tuples),
    mean_loo, error (present only on failure).
    """
    if len(X) < 4:
        return {"n": len(X), "error": f"Too few rows ({len(X)})"}

    # Encode string labels for multiclass (tail strings → int codes)
    if is_multiclass:
        classes, y_enc = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        majority = float(np.max(np.bincount(y_enc)) / len(y_enc))
        y_fit = y_enc
    else:
        y_fit = y.astype(int)
        n_classes = 2
        counts = np.bincount(y_fit, minlength=2)
        majority = float(counts.max() / counts.sum())
        classes = None

    unique_in_y = np.unique(y_fit)
    if len(unique_in_y) < 2:
        return {"n": len(X), "majority": majority, "error": "Single class in labels"}

    k = max(2, min(n_splits, int(np.min(np.bincount(y_fit))), len(X) - 1))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0)),
    ])

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    try:
        cv_scores = cross_val_score(pipe, X, y_fit, cv=cv, scoring="accuracy")
        cv5_mean = float(cv_scores.mean())
        cv5_std = float(cv_scores.std())
    except Exception as exc:
        cv5_mean = float("nan")
        cv5_std = float("nan")

    loo = LeaveOneGroupOut()
    loo_scores: list[float] = []
    loo_details: list[tuple] = []

    for train_idx, test_idx in loo.split(X, y_fit, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_fit[train_idx], y_fit[test_idx]
        held = str(groups[test_idx][0])

        if len(np.unique(y_tr)) < 2:
            loo_details.append((held, None, "skipped: single class in train"))
            continue
        try:
            p = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=1.0)),
            ])
            p.fit(X_tr, y_tr)
            acc = float(p.score(X_te, y_te))
            loo_scores.append(acc)
            loo_details.append((held, acc, f"n_test={len(y_te)}"))
        except Exception as exc:
            loo_details.append((held, None, f"error: {exc}"))

    return {
        "n": len(X),
        "majority": majority,
        "cv5_mean": cv5_mean,
        "cv5_std": cv5_std,
        "loo_details": loo_details,
        "mean_loo": float(np.mean(loo_scores)) if loo_scores else float("nan"),
        "n_classes": n_classes,
    }


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_lofo_result(
    result: dict,
    phase_name: str,
    feature_variant: str,
    target: str,
) -> str:
    """Return a human-readable block for one LOFO result."""
    lines: list[str] = []
    lines.append(f"\n  [{phase_name}]  features={feature_variant}  target={target}")

    if "error" in result:
        lines.append(f"    ERROR: {result['error']}")
        return "\n".join(lines)

    n = result.get("n", "?")
    majority = result.get("majority", float("nan"))
    cv5_mean = result.get("cv5_mean", float("nan"))
    cv5_std = result.get("cv5_std", float("nan"))
    mean_loo = result.get("mean_loo", float("nan"))

    lines.append(f"    n={n}  majority={majority:.1%}  CV5={cv5_mean:.1%}±{cv5_std:.1%}  mean_LOO={mean_loo:.1%}")

    for held, acc, note in result.get("loo_details", []):
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"
        lines.append(f"      held-out={held:<22} acc={acc_str}  ({note})")

    return "\n".join(lines)


# ── Phase-level evaluation ─────────────────────────────────────────────────────

def eval_phase_lofo(
    phase_raw: list[dict],
    phase_flat: list[dict],
    common_ids: set[str],
    phase_name: str,
) -> dict[str, dict]:
    """Run all four LOFO combinations for one phase.

    Filters phase_raw/phase_flat to common_ids × binary-TF rows, then
    runs {layer_stats, layer_hist} × {label, tail}.

    Parameters
    ----------
    phase_raw   : list of raw stat dicts for this phase
    phase_flat  : list of flattened dicts (parallel to phase_raw)
    common_ids  : set of prompt_ids to restrict to
    phase_name  : human-readable label for printing

    Returns
    -------
    dict keyed by ("layer_stats"|"layer_hist", "label"|"tail") → lofo result dict.
    Also includes "n_total_filtered" and "feature_cols_layer_stats" for reference.
    """
    # Filter to common_ids × binary-TF
    tf_raw: list[dict] = []
    tf_flat: list[dict] = []
    for raw, flat in zip(phase_raw, phase_flat):
        pid = raw.get("prompt_id")
        if pid not in common_ids:
            continue
        if raw.get("task_type", "binary") != "binary":
            continue
        if raw.get("family") not in TF_FAMILIES:
            continue
        tf_raw.append(raw)
        tf_flat.append(flat)

    results: dict = {"n_total_filtered": len(tf_raw)}

    if not tf_flat:
        results["error"] = "No binary-TF rows after filtering to common_ids"
        return results

    # Build feature column lists from the first available flat dict
    feature_cols_ls = build_feature_cols_layer_stats(tf_flat[0])
    feature_cols_lh = build_feature_cols_layer_hist(feature_cols_ls)
    results["feature_cols_layer_stats"] = feature_cols_ls
    results["feature_cols_layer_hist"] = feature_cols_lh

    for mode, fcols in [("layer_stats", feature_cols_ls), ("layer_hist", feature_cols_lh)]:
        X, y_label, y_tail, families = build_matrix(tf_raw, tf_flat, fcols, mode=mode)

        if len(X) == 0:
            for tgt in ("label", "tail"):
                results[(mode, tgt)] = {"n": 0, "error": "No complete rows"}
            continue

        results[(mode, "label")] = run_lofo(
            X, y_label, families, is_multiclass=False
        )
        results[(mode, "tail")] = run_lofo(
            X, y_tail, families, is_multiclass=True
        )

    return results
