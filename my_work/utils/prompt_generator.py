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

# Templates 1–3 and 5–7 support both True and False claim directions.
# Template 4 is always True (triangle inequality always holds for valid triangles).
# Template 4b is the negated variant (for False direction on degenerate triples).
TEMPLATES: dict[int | str, str] = {
    1: "Statement: For a triangle with sides {a}, {b}, and {c}, "
       "the sum of any two sides is {comparison} the third side. Answer:",

    2: "Statement: For a triangle with sides {a}, {b}, and {c}, "
       "the sum of any two sides is {comparison} the third side. "
       "Answer with exactly one word, True or False:",

    3: "Is it true that for a triangle with sides {a}, {b}, and {c}, "
       "the sum of any two sides is {comparison} the third side? Answer:",

    4: "Mathematical claim: Given side lengths {a}, {b}, {c}, "
       "the triangle inequality holds: each side is strictly less than "
       "the sum of the other two. True or False?",

    "4b": "Mathematical claim: Given side lengths {a}, {b}, {c}, "
          "the triangle inequality does NOT hold. True or False?",

    5: "A triangle has sides of length {a}, {b}, and {c}. "
       "Someone claims that the sum of any two sides is {comparison} the third. "
       "Is this claim correct? Answer:",

    6: "Consider a triangle with side lengths {a}, {b}, and {c}. "
       "Is it correct that the sum of any two sides is {comparison} the third side? "
       "Respond True or False:",

    7: "Mathematical statement: For side lengths {a}, {b}, {c}, "
       "we have that the sum of any two sides is {comparison} the third. "
       "Is this statement true? Answer:",
}

# Templates that can produce False prompts (exclude template 4 which is always True).
# Template 4b can produce True prompts on degenerate triples.
TEMPLATES_BIDIRECTIONAL = [1, 2, 3, 5, 6, 7]
TEMPLATES_TRUE_ONLY = [4]
TEMPLATES_4B = ["4b"]

COMPARISON_GREATER = "greater than"
COMPARISON_LESS = "less than"


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
    comparison: str,
) -> str:
    tmpl = TEMPLATES[template_id]
    if "{comparison}" in tmpl:
        return tmpl.format(a=a, b=b, c=c, comparison=comparison)
    return tmpl.format(a=a, b=b, c=c)


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
    comparison = COMPARISON_GREATER if claim_type == "holds" else COMPARISON_LESS
    prompt = _build_prompt(a, b, c, template_id, comparison)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "label": label,
        "label_token": TOKEN_TRUE if label else TOKEN_FALSE,
        "sides": [a, b, c],
        "triangle_valid": triangle_valid,
        "template_id": template_id,
        "claim_direction": "greater_than" if claim_type == "holds" else "less_than",
        "split": None,  # filled in by split_dataset
    }


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_triangle_prompts(n: int = 300, seed: int = 42) -> list[dict]:
    """
    Generate exactly n balanced (50% True, 50% False) triangle inequality prompts.

    Composition:
    - Exactly 10% degenerate / invalid triples across the whole set.
    - True prompts: valid triples with claim_type='holds' (template 1-7, 4).
    - False prompts: two sub-types:
        * Valid triple + claim_type='does_not_hold' (templates 1-3, 5-7 only)
        * Degenerate triple + claim_type='holds' (templates 1-3, 5-7 only)
    - Template 4b: degenerate triple + claim_type='does_not_hold'
      → label=True (the inequality does NOT hold is a true claim for degenerate).
      These are counted under True prompts.

    The pool is shuffled after generation. Exactly n entries are returned.
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
    # Most True prompts: valid triple + claim holds
    # Reserve ~half of degenerate budget for True side via template 4b
    n_true_via_4b = n_degenerate // 2
    n_true_direct = n_true - n_true_via_4b

    # True via template 4b (degenerate triple, claim: inequality does NOT hold)
    tmpl4b_pool = TEMPLATES_4B
    for _ in range(n_true_via_4b):
        a, b, c = _generate_degenerate_triple(rng)
        t_id = rng.choice(tmpl4b_pool)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "does_not_hold", False))

    # True via valid triples, templates 1-7 and 4
    all_true_templates = TEMPLATES_BIDIRECTIONAL + TEMPLATES_TRUE_ONLY
    for _ in range(n_true_direct):
        a, b, c = _generate_valid_triple(rng)
        t_id = rng.choice(all_true_templates)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "holds", True))

    # ── False prompts ────────────────────────────────────────────────────────────
    # Sub-type 1: valid triple + claim 'does_not_hold' (False because inequality does hold)
    # Sub-type 2: degenerate triple + claim 'holds' (False because inequality does NOT hold)
    n_false_via_degenerate = n_degenerate - n_true_via_4b  # remaining degenerate budget
    n_false_via_valid = n_false - n_false_via_degenerate

    for _ in range(n_false_via_degenerate):
        a, b, c = _generate_degenerate_triple(rng)
        t_id = rng.choice(TEMPLATES_BIDIRECTIONAL)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "holds", False))

    for _ in range(n_false_via_valid):
        a, b, c = _generate_valid_triple(rng)
        t_id = rng.choice(TEMPLATES_BIDIRECTIONAL)
        entries.append(_make_entry(next_id(), a, b, c, t_id, "does_not_hold", True))

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
