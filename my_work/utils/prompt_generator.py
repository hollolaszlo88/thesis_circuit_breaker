"""
prompt_generator.py — Triangle inequality dataset generation.

Constants (from plan, never change):
    TOKEN_TRUE  = " True"   (vocab 5569)
    TOKEN_FALSE = " False"  (vocab 7662)

Public API:
    verify_triangle_claim(a, b, c, claim_type) -> bool
    generate_triangle_prompts(n, seed) -> list[dict]
    split_dataset(prompts, train_frac, eval_frac, analysis_frac, seed) -> dict
    generate_math_broad_prompts(n, seed) -> list[dict]  # optional Stage 5
"""

from __future__ import annotations

import random
from typing import Literal

# ── Constants ──────────────────────────────────────────────────────────────────
TOKEN_TRUE = " True"
TOKEN_FALSE = " False"

# Templates are phrased as triangle-inequality feasibility claims and always
# terminate with "Answer:". We intentionally avoid "True or False" instructions
# in the prompt text to reduce formatting pressure on the model.
TEMPLATES: dict[int | str, str] = {
    1: "There {can_phrase} a triangle with side lengths {a}, {b}, and {c}. Answer:",
    2: "A triangle with sides {a}, {b}, and {c} is {possible_phrase}. Answer:",
    3: "For a triangle with sides {a} and {b}, the third side {can_phrase} {c}. Answer:",
    4: "Given sides {a} and {b}, {c} is {allowable_phrase} third side for a triangle. Answer:",
    5: "For sides {a} and {b}, the third side {can_phrase} {c} (maximum boundary case). Answer:",
    6: "For sides {a} and {b}, the third side {can_phrase} {c} (just below maximum). Answer:",
    7: "For sides {a} and {b}, the third side {can_phrase} {c} (minimum boundary case). Answer:",
    8: "For sides {a} and {b}, the third side {can_phrase} {c} (just above minimum). Answer:",
}

TEMPLATES_GENERAL = [1, 2, 3, 4]
TEMPLATES_BOUNDARY = [5, 6, 7, 8]


# ── Ground-truth verification ──────────────────────────────────────────────────

def _triangle_valid(a: float, b: float, c: float) -> bool:
    """Returns True if (a, b, c) form a valid (strict) triangle."""
    return a + b > c and a + c > b and b + c > a


def verify_triangle_claim(
    a: float,
    b: float,
    c: float,
    claim_type: Literal["holds", "does_not_hold"],
) -> bool:
    """
    Programmatic ground truth for triangle inequality claims.

    claim_type='holds'         → claim: the inequality holds → True iff triangle is valid
    claim_type='does_not_hold' → claim: the inequality does NOT hold → True iff triangle invalid
    """
    valid = _triangle_valid(a, b, c)
    if claim_type == "holds":
        return valid
    elif claim_type == "does_not_hold":
        return not valid
    else:
        raise ValueError(f"Unknown claim_type: {claim_type!r}")


# ── Side-length generators ─────────────────────────────────────────────────────

# Curated valid triangle triples (diverse shapes, integer sides)
_VALID_TRIPLES: list[tuple[int, int, int]] = [
    (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
    (2, 3, 4), (3, 5, 6), (4, 6, 7), (5, 7, 8),
    (6, 8, 9), (7, 9, 11), (9, 10, 12), (10, 12, 15),
    (6, 6, 6), (5, 5, 5), (4, 4, 4), (7, 7, 7),
    (5, 5, 8), (6, 6, 10), (7, 7, 12),
    (1, 2, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5),
    (10, 10, 1), (15, 15, 2), (8, 8, 1),
    (3, 4, 6), (5, 6, 9), (4, 7, 9), (6, 10, 13),
    (1, 3, 3), (2, 4, 5), (3, 6, 7), (5, 8, 11),
    (2, 5, 6), (4, 9, 10), (6, 11, 14), (3, 7, 8),
    (10, 24, 26), (9, 40, 41), (12, 35, 37),
]

# Degenerate / invalid triples (triangle inequality violated)
_DEGENERATE_TRIPLES: list[tuple[int, int, int]] = [
    (1, 2, 3),   # 1+2=3, degenerate (not strict)
    (1, 1, 3),   # 1+1<3
    (2, 3, 6),   # 2+3<6
    (1, 4, 6),   # 1+4<6
    (3, 5, 9),   # 3+5<9
    (2, 2, 5),   # 2+2<5
    (1, 5, 7),   # 1+5<7
    (4, 4, 9),   # 4+4<9
    (1, 1, 5),
    (2, 4, 8),
    (3, 3, 7),
    (1, 6, 8),
    (2, 6, 9),
    (4, 6, 11),
    (3, 4, 8),
    (5, 5, 11),
    (1, 2, 4),
    (2, 3, 7),
    (3, 5, 10),
    (4, 7, 12),
    (5, 8, 14),
    (6, 9, 16),
    (2, 8, 11),
    (3, 9, 13),
    (4, 10, 15),
    (5, 12, 18),
    (6, 14, 21),
    (1, 3, 5),
    (2, 5, 8),
    (3, 6, 10),
]


def _generate_valid_triple(rng: random.Random) -> tuple[int, int, int]:
    return rng.choice(_VALID_TRIPLES)


def _generate_degenerate_triple(rng: random.Random) -> tuple[int, int, int]:
    return rng.choice(_DEGENERATE_TRIPLES)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(
    a: int,
    b: int,
    c: int,
    template_id: int | str,
    claim_type: Literal["holds", "does_not_hold"],
) -> str:
    tmpl = TEMPLATES[template_id]
    can_phrase = "can be" if claim_type == "holds" else "cannot be"
    possible_phrase = "possible" if claim_type == "holds" else "impossible"
    allowable_phrase = "an allowable" if claim_type == "holds" else "not an allowable"
    return tmpl.format(
        a=a,
        b=b,
        c=c,
        can_phrase=can_phrase,
        possible_phrase=possible_phrase,
        allowable_phrase=allowable_phrase,
    )


def _make_entry(
    prompt_id: str,
    a: int,
    b: int,
    c: int,
    template_id: int | str,
    claim_type: Literal["holds", "does_not_hold"],
    triangle_valid: bool,
) -> dict:
    label = verify_triangle_claim(a, b, c, claim_type)
    prompt = _build_prompt(a, b, c, template_id, claim_type)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "label": label,
        "label_token": TOKEN_TRUE if label else TOKEN_FALSE,
        "sides": [a, b, c],
        "triangle_valid": triangle_valid,
        "template_id": template_id,
        "claim_direction": "possible" if claim_type == "holds" else "not_possible",
        "split": None,  # filled in by split_dataset
    }


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_triangle_prompts(n: int = 300, seed: int = 42) -> list[dict]:
    """
    Generate exactly n balanced (50% True, 50% False) triangle inequality prompts.

    Composition (balanced and compatible with existing pipeline):
    - Exactly 10% degenerate / invalid triples across the whole set.
    - 50% label=True and 50% label=False.
    - Prompts are feasibility-style claims that end with "Answer:" and do not
      include explicit "True or False" formatting instructions.
    """
    rng = random.Random(seed)

    n_true = n // 2
    n_false = n - n_true
    n_degenerate = round(n * 0.10)  # fixed 10%

    entries: list[dict] = []
    counter = 1

    def next_id() -> str:
        nonlocal counter
        pid = f"tri_{counter:03d}"
        counter += 1
        return pid

    # ── True prompts ────────────────────────────────────────────────────────────
    n_true_boundary = max(1, n_true // 10)
    n_false_boundary = max(1, n_false // 10)
    n_true_via_degenerate = n_degenerate // 2
    n_false_via_degenerate = max(0, n_degenerate - n_true_via_degenerate - n_false_boundary)
    n_true_via_valid = n_true - n_true_boundary - n_true_via_degenerate
    n_false_via_valid = n_false - n_false_boundary - n_false_via_degenerate

    # True regular (valid triple, positive claim)
    for _ in range(n_true_via_valid):
        a, b, c = _generate_valid_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "holds", True))

    # True via degenerate (invalid triple, negative claim)
    for _ in range(n_true_via_degenerate):
        a, b, c = _generate_degenerate_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "does_not_hold", False))

    # Boundary true: c=a+b-1 and c=|a-b|+1
    for _ in range(n_true_boundary):
        a, b, _ = _generate_valid_triple(rng)
        if rng.random() < 0.5:
            c = a + b - 1
            t_id = 6
        else:
            c = abs(a - b) + 1
            t_id = 8
        entries.append(_make_entry(next_id(), a, b, c, t_id, "holds", _triangle_valid(a, b, c)))

    # ── False prompts ────────────────────────────────────────────────────────────
    # False regular (valid triple, negative claim)
    for _ in range(n_false_via_degenerate):
        a, b, c = _generate_degenerate_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "holds", False))

    for _ in range(n_false_via_valid):
        a, b, c = _generate_valid_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "does_not_hold", True))

    # Boundary false: c=a+b and c=|a-b|
    for _ in range(n_false_boundary):
        a, b, _ = _generate_valid_triple(rng)
        if rng.random() < 0.5:
            c = a + b
            t_id = 5
        else:
            c = abs(a - b)
            t_id = 7
        entries.append(_make_entry(next_id(), a, b, c, t_id, "holds", _triangle_valid(a, b, c)))

    # ── Shuffle and renumber ────────────────────────────────────────────────────
    rng.shuffle(entries)
    for i, entry in enumerate(entries, start=1):
        entry["prompt_id"] = f"tri_{i:03d}"

    assert len(entries) == n, f"Expected {n} prompts, got {len(entries)}"
    n_true_actual = sum(1 for e in entries if e["label"])
    n_false_actual = sum(1 for e in entries if not e["label"])
    assert n_true_actual == n_true, f"True count mismatch: {n_true_actual} != {n_true}"
    assert n_false_actual == n_false, f"False count mismatch: {n_false_actual} != {n_false}"

    return entries


# ── Dataset splitting ──────────────────────────────────────────────────────────

def split_dataset(
    prompts: list[dict],
    train_frac: float = 0.5,
    eval_frac: float = 0.25,
    analysis_frac: float = 0.25,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Split prompts into train / eval / analysis sets, stratified by label.

    Returns dict with keys 'train', 'eval', 'analysis', each a list of entries
    with 'split' field set.
    """
    assert abs(train_frac + eval_frac + analysis_frac - 1.0) < 1e-9, \
        "Fractions must sum to 1.0"

    rng = random.Random(seed)

    true_prompts = [e for e in prompts if e["label"]]
    false_prompts = [e for e in prompts if not e["label"]]

    rng.shuffle(true_prompts)
    rng.shuffle(false_prompts)

    def _split_list(lst: list, f_train: float, f_eval: float) -> tuple:
        n = len(lst)
        n_train = round(n * f_train)
        n_eval = round(n * f_eval)
        return lst[:n_train], lst[n_train:n_train + n_eval], lst[n_train + n_eval:]

    t_train, t_eval, t_analysis = _split_list(true_prompts, train_frac, eval_frac)
    f_train, f_eval, f_analysis = _split_list(false_prompts, train_frac, eval_frac)

    result: dict[str, list[dict]] = {"train": [], "eval": [], "analysis": []}
    for split_name, true_part, false_part in [
        ("train", t_train, f_train),
        ("eval", t_eval, f_eval),
        ("analysis", t_analysis, f_analysis),
    ]:
        combined = true_part + false_part
        rng.shuffle(combined)
        for entry in combined:
            entry["split"] = split_name
        result[split_name] = combined

    return result


# ── Optional Stage 5 stub ─────────────────────────────────────────────────────

def generate_math_broad_prompts(n: int = 200, seed: int = 42) -> list[dict]:
    """Optional Stage 5 expansion — not in scope for core run."""
    raise NotImplementedError(
        "Math-broad dataset generation is optional Stage 5 expansion. "
        "Implement when the triangle-only pipeline is complete."
    )
