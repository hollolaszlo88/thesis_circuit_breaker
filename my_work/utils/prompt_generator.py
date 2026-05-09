"""
prompt_generator.py — Triangle inequality dataset generation.

Constants (from plan, never change for binary labels):
    TOKEN_TRUE  = " True"   (vocab 5569)
    TOKEN_FALSE = " False"  (vocab 7662)

Schema fields in every generated entry:
    task_type    : "binary"  → label is True/False, attribution targets [True, False]
                   "numeric" → label is an integer (e.g. 9), single target
    label_token  : the correct next token (e.g. " True", " False", "9")

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
    "N1": "For a triangle with sides {a} and {b}, what is the largest integer value the third side can take? Answer:",
}

TEMPLATES_GENERAL = [1, 2, 3, 4]
TEMPLATES_BOUNDARY = [5, 6, 7, 8]
TEMPLATES_NUMERIC = ["N1"]
TOKEN_NUMERIC_9 = "9"


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
    if template_id in TEMPLATES_NUMERIC:
        return tmpl.format(a=a, b=b)
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


def _make_numeric_entry(
    prompt_id: str,
    a: int,
    b: int,
    template_id: int | str = "N1",
) -> dict:
    # Largest integer third side for strict triangle inequality is a+b-1.
    c_max = a + b - 1
    prompt = _build_prompt(a, b, c_max, template_id, "holds")
    label_token = f"{c_max}"
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "task_type": "numeric",
        "label": c_max,
        "label_token": label_token,
        "sides": [a, b, None],
        "triangle_valid": None,
        "template_id": template_id,
        "claim_direction": "max_third_side",
        "split": None,  # filled in by split_dataset
    }


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
        "task_type": "binary",
        "label": label,
        "label_token": TOKEN_TRUE if label else TOKEN_FALSE,
        "sides": [a, b, c],
        "triangle_valid": triangle_valid,
        "template_id": template_id,
        "claim_direction": "possible" if claim_type == "holds" else "not_possible",
        "split": None,  # filled in by split_dataset
    }


# ── Main generator ─────────────────────────────────────────────────────────────

def _generate_binary_prompts(n: int, rng: random.Random, id_fn) -> list[dict]:
    """Generate only binary feasibility prompts, balanced by True/False labels."""
    n_true = n // 2
    n_false = n - n_true
    n_degenerate = round(n * 0.10)  # fixed 10% among binary prompts
    entries: list[dict] = []

    n_true_boundary = max(1, n_true // 10)
    n_false_boundary = max(1, n_false // 10)
    n_true_via_degenerate = n_degenerate // 2
    n_false_via_degenerate = max(0, n_degenerate - n_true_via_degenerate - n_false_boundary)
    n_true_via_valid = n_true - n_true_boundary - n_true_via_degenerate
    n_false_via_valid = n_false - n_false_boundary - n_false_via_degenerate

    for _ in range(n_true_via_valid):
        a, b, c = _generate_valid_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(id_fn(), a, b, c, t_id, "holds", True))

    for _ in range(n_true_via_degenerate):
        a, b, c = _generate_degenerate_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(id_fn(), a, b, c, t_id, "does_not_hold", False))

    for _ in range(n_true_boundary):
        a, b, _ = _generate_valid_triple(rng)
        if rng.random() < 0.5:
            c = a + b - 1
            t_id = 6
        else:
            c = abs(a - b) + 1
            t_id = 8
        entries.append(_make_entry(id_fn(), a, b, c, t_id, "holds", _triangle_valid(a, b, c)))

    for _ in range(n_false_via_degenerate):
        a, b, c = _generate_degenerate_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(id_fn(), a, b, c, t_id, "holds", False))

    for _ in range(n_false_via_valid):
        a, b, c = _generate_valid_triple(rng)
        t_id = rng.choice(TEMPLATES_GENERAL)
        entries.append(_make_entry(id_fn(), a, b, c, t_id, "does_not_hold", True))

    for _ in range(n_false_boundary):
        a, b, _ = _generate_valid_triple(rng)
        if rng.random() < 0.5:
            c = a + b
            t_id = 5
        else:
            c = abs(a - b)
            t_id = 7
        entries.append(_make_entry(id_fn(), a, b, c, t_id, "holds", _triangle_valid(a, b, c)))

    assert len(entries) == n, f"Expected {n} binary prompts, got {len(entries)}"
    n_true_actual = sum(1 for e in entries if e["label"])
    n_false_actual = sum(1 for e in entries if not e["label"])
    assert n_true_actual == n_true, f"True count mismatch: {n_true_actual} != {n_true}"
    assert n_false_actual == n_false, f"False count mismatch: {n_false_actual} != {n_false}"
    return entries


def _generate_numeric_prompts(n: int, rng: random.Random, id_fn) -> list[dict]:
    """
    Generate numeric prompts with controlled single-token target '9'.
    We pick (a,b) pairs with a+b=10 so max integer third side is always 9.
    """
    entries: list[dict] = []
    ab_pairs = [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
    for _ in range(n):
        a, b = rng.choice(ab_pairs)
        entry = _make_numeric_entry(id_fn(), a, b, "N1")
        assert entry["label_token"] == TOKEN_NUMERIC_9, f"Unexpected numeric target: {entry['label_token']}"
        entries.append(entry)
    return entries


def generate_triangle_prompts(
    n: int = 300,
    seed: int = 42,
    numeric_frac: float = 0.2,
) -> list[dict]:
    """
    Generate exactly n mixed triangle-inequality prompts:
      - binary feasibility prompts (True/False token targets)
      - numeric max-third-side prompts (single-token numeric targets)

    Composition:
    - numeric_frac controls the numeric share (default 20%).
    - Binary subset is exactly 50/50 True/False.
    - Binary subset keeps ~10% degenerate / invalid triples.
    - Numeric subset currently uses fixed target token '9' for controlled
      attribution comparisons.
    """
    rng = random.Random(seed)

    entries: list[dict] = []
    counter = 1

    def next_id() -> str:
        nonlocal counter
        pid = f"tri_{counter:03d}"
        counter += 1
        return pid

    n_numeric = int(round(n * numeric_frac))
    n_binary = n - n_numeric
    if n_binary < 2:
        raise ValueError("numeric_frac too high: need at least 2 binary prompts")
    # Keep binary perfectly balanced.
    if n_binary % 2 != 0:
        n_binary -= 1
        n_numeric += 1

    entries.extend(_generate_binary_prompts(n_binary, rng, next_id))
    entries.extend(_generate_numeric_prompts(n_numeric, rng, next_id))

    # ── Shuffle and renumber ────────────────────────────────────────────────────
    rng.shuffle(entries)
    for i, entry in enumerate(entries, start=1):
        entry["prompt_id"] = f"tri_{i:03d}"

    assert len(entries) == n, f"Expected {n} prompts, got {len(entries)}"

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
    Split prompts into train / eval / analysis sets, stratified by task type.
    Within binary prompts, keep True/False balance stratified as well.

    Returns dict with keys 'train', 'eval', 'analysis', each a list of entries
    with 'split' field set.
    """
    assert abs(train_frac + eval_frac + analysis_frac - 1.0) < 1e-9, \
        "Fractions must sum to 1.0"

    rng = random.Random(seed)

    binary_true = [e for e in prompts if e.get("task_type", "binary") == "binary" and bool(e["label"])]
    binary_false = [e for e in prompts if e.get("task_type", "binary") == "binary" and not bool(e["label"])]
    numeric = [e for e in prompts if e.get("task_type", "binary") == "numeric"]

    rng.shuffle(binary_true)
    rng.shuffle(binary_false)
    rng.shuffle(numeric)

    def _split_list(lst: list, f_train: float, f_eval: float) -> tuple:
        n = len(lst)
        n_train = round(n * f_train)
        n_eval = round(n * f_eval)
        return lst[:n_train], lst[n_train:n_train + n_eval], lst[n_train + n_eval:]

    t_train, t_eval, t_analysis = _split_list(binary_true, train_frac, eval_frac)
    f_train, f_eval, f_analysis = _split_list(binary_false, train_frac, eval_frac)
    n_train, n_eval, n_analysis = _split_list(numeric, train_frac, eval_frac)

    result: dict[str, list[dict]] = {"train": [], "eval": [], "analysis": []}
    for split_name, true_part, false_part, numeric_part in [
        ("train", t_train, f_train, n_train),
        ("eval", t_eval, f_eval, n_eval),
        ("analysis", t_analysis, f_analysis, n_analysis),
    ]:
        combined = true_part + false_part + numeric_part
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


# ── v2 dataset: families × tails (supervisor-aligned) ─────────────────────────
#
# Schema additions vs v1:
#   family       : "numeric_validity" | "geometry_claim" | "numeric_open"
#   tail         : "answer_colon" | "true_or_false" | "the_answer_is"
#   claim_type   : "holds" | "does_not_hold" | None  (None for geometry)
#   template_id  : prefixed — num_1…num_8, geom_triangle_classic…, open_max_third
#   open_kind    : "max_third" | "min_third" | None

# Tail suffix strings (keys match supervisor prompts.py TAILS dict).
_TAILS_V2: dict[str, str] = {
    "answer_colon":  " Answer:",
    "true_or_false": " Answer with exactly one word: True or False.",
    "the_answer_is": " The answer is",
}

# ── geometry_claim templates (supervisor-sourced, prefixed geom_) ──────────────

_GEOM_TRUE_TEMPLATES: list[tuple[str, str]] = [
    ("geom_triangle_classic",
     "Statement: For any triangle, the sum of any two sides is greater than the third side."),
    ("geom_triangle_each_side",
     "Statement: In every triangle, each side is shorter than the sum of the other two."),
    ("geom_triangle_two_sum",
     "Statement: For any triangle, the sum of two sides is greater than the remaining side."),
    ("geom_triangle_abc_symbolic",
     "Statement: For any triangle ABC, |AB| + |BC| > |AC|."),
    ("geom_triangle_valid_property",
     "Statement: A valid triangle has the property that the sum of any two sides exceeds the third."),
    ("geom_triangle_longest",
     "Statement: In any triangle the longest side is shorter than the sum of the other two sides."),
]

_GEOM_FALSE_TEMPLATES: list[tuple[str, str]] = [
    ("geom_triangle_classic_neg",
     "Statement: For any triangle, the sum of any two sides is less than the third side."),
    ("geom_triangle_each_side_neg",
     "Statement: In every triangle, each side is longer than the sum of the other two."),
    ("geom_triangle_two_sum_eq",
     "Statement: For any triangle, the sum of two sides equals the remaining side."),
    ("geom_triangle_abc_symbolic_neg",
     "Statement: For any triangle ABC, |AB| + |BC| < |AC|."),
    ("geom_triangle_valid_property_neg",
     "Statement: A valid triangle has the property that the sum of any two sides equals the third."),
    ("geom_triangle_longest_neg",
     "Statement: In any triangle the longest side is greater than the sum of the other two sides."),
]


def _make_geom_entry(
    prompt_id: str,
    template_id: str,
    body: str,
    label: bool,
    tail_key: str,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt": body + _TAILS_V2[tail_key],
        "task_type": "binary",
        "family": "geometry_claim",
        "tail": tail_key,
        "label": label,
        "label_token": TOKEN_TRUE if label else TOKEN_FALSE,
        "template_id": template_id,
        "claim_type": None,
        "claim_direction": None,
        "open_kind": None,
        "sides": None,
        "triangle_valid": None,
        "split": None,
    }


def _generate_geometry_prompts_v2(id_fn) -> list[dict]:
    """Generate geometry_claim rows: 6 true + 6 false templates × 3 tails."""
    entries = []
    for tail_key in _TAILS_V2:
        for tid, body in _GEOM_TRUE_TEMPLATES:
            entries.append(_make_geom_entry(id_fn(), tid, body, True, tail_key))
        for tid, body in _GEOM_FALSE_TEMPLATES:
            entries.append(_make_geom_entry(id_fn(), tid, body, False, tail_key))
    return entries  # 36 entries


def _make_numeric_validity_entry_v2(
    prompt_id: str,
    a: int,
    b: int,
    c: int,
    template_id: int | str,
    claim_type: str,
    triangle_valid: bool,
    tail_key: str,
) -> dict:
    label = verify_triangle_claim(a, b, c, claim_type)
    prompt = _build_prompt(a, b, c, template_id, claim_type)
    # Strip v1's hardcoded "Answer:" suffix — tails are now appended separately.
    # The v1 TEMPLATES already embed "Answer:" at the end; we replace it with
    # the chosen tail.
    if prompt.endswith(" Answer:"):
        prompt = prompt[: -len(" Answer:")] + _TAILS_V2[tail_key]
    else:
        prompt = prompt + _TAILS_V2[tail_key]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "task_type": "binary",
        "family": "numeric_validity",
        "tail": tail_key,
        "label": label,
        "label_token": TOKEN_TRUE if label else TOKEN_FALSE,
        "template_id": f"num_{template_id}",
        "claim_type": claim_type,
        "claim_direction": "possible" if claim_type == "holds" else "not_possible",
        "open_kind": None,
        "sides": [a, b, c],
        "triangle_valid": triangle_valid,
        "split": None,
    }


def _generate_numeric_validity_prompts_v2(
    n_per_tail_half: int,
    rng: random.Random,
    id_fn,
) -> list[dict]:
    """Generate numeric_validity rows balanced 50/50 per (tail) cell.

    n_per_tail_half: true examples per tail (false will be equal).
    Each tail cell uses TEMPLATES_GENERAL and a mix of valid/invalid triples.
    ~10% degenerate cases are included within the false half.
    """
    entries = []
    for tail_key in _TAILS_V2:
        n_true = n_per_tail_half
        n_false = n_per_tail_half
        n_degen = max(1, round(n_false * 0.10))

        # True rows: valid triple + "holds" claim
        for _ in range(n_true):
            a, b, c = _generate_valid_triple(rng)
            tid = rng.choice(TEMPLATES_GENERAL)
            entries.append(
                _make_numeric_validity_entry_v2(
                    id_fn(), a, b, c, tid, "holds", True, tail_key
                )
            )

        # False rows: degenerate (holds → False) + invalid (does_not_hold on valid → False)
        # Degenerate: claim "holds" on a degenerate/invalid triple → label False
        for _ in range(n_degen):
            a, b, c = _generate_degenerate_triple(rng)
            tid = rng.choice(TEMPLATES_GENERAL)
            entries.append(
                _make_numeric_validity_entry_v2(
                    id_fn(), a, b, c, tid, "holds", False, tail_key
                )
            )
        # Remaining false: claim "does_not_hold" on a valid triple → label False
        for _ in range(n_false - n_degen):
            a, b, c = _generate_valid_triple(rng)
            tid = rng.choice(TEMPLATES_GENERAL)
            entries.append(
                _make_numeric_validity_entry_v2(
                    id_fn(), a, b, c, tid, "does_not_hold", True, tail_key
                )
            )

    return entries


def _make_numeric_open_entry_v2(
    prompt_id: str,
    a: int,
    b: int,
    kind: str,
) -> dict:
    """Open-ended max/min third-side question; tail fixed to answer_colon."""
    if kind == "max_third":
        body = (
            f"Question: Two sides of a triangle have lengths {a} and {b}. "
            f"The largest possible integer length of the third side is"
        )
        c_answer = a + b - 1
    else:
        body = (
            f"Question: Two sides of a triangle have lengths {a} and {b}. "
            f"The smallest possible integer length of the third side is"
        )
        c_answer = abs(a - b) + 1

    prompt = body + _TAILS_V2["answer_colon"]
    label_token = str(c_answer)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "task_type": "numeric",
        "family": "numeric_open",
        "tail": "answer_colon",
        "label": c_answer,
        "label_token": label_token,
        "template_id": f"open_{kind}",
        "claim_type": None,
        "claim_direction": None,
        "open_kind": kind,
        "sides": [a, b],
        "triangle_valid": None,
        "split": None,
    }


# Pairs (a, b) reused from supervisor; c_max = a+b-1, c_min = |a-b|+1
_OPEN_PAIRS_V2: list[tuple[int, int]] = [
    (3, 4), (5, 7), (8, 9), (10, 13), (4, 5), (6, 7), (12, 15),
    (2, 3), (5, 6), (7, 8), (9, 10), (11, 12), (14, 15), (3, 5),
    (6, 8),
]


def _generate_numeric_open_prompts_v2(n: int, rng: random.Random, id_fn) -> list[dict]:
    """Generate n numeric_open rows (max and min third side, answer_colon tail)."""
    entries = []
    pairs = list(_OPEN_PAIRS_V2)
    rng.shuffle(pairs)
    for i in range(n):
        a, b = pairs[i % len(pairs)]
        kind = "max_third" if i % 2 == 0 else "min_third"
        entries.append(_make_numeric_open_entry_v2(id_fn(), a, b, kind))
    return entries


def generate_triangle_prompts_v2(
    n: int = 300,
    seed: int = 42,
    numeric_open_frac: float = 0.10,
    geometry_frac: float = 0.20,
) -> list[dict]:
    """Generate v2 triangle-inequality prompt set.

    Design: family × tail factorial with balanced binary labels per cell.

    Families
    --------
    - numeric_validity  (~70%): concrete triples, 3 tails, balanced T/F per tail
    - geometry_claim    (~20%): abstract statements, 3 tails, 6 true + 6 false templates
    - numeric_open      (~10%): max/min third side, answer_colon only, excluded from
                                the T/F structural classifier in analysis

    Metadata on every row (enables slice-without-rejoin in notebooks):
      family, tail, claim_type, claim_direction, template_id, open_kind,
      sides, triangle_valid, task_type, label, label_token, split.

    template_id namespace (prefixed, no collisions across families):
      num_1…num_8       numeric_validity (existing TEMPLATES keys)
      geom_*            geometry_claim  (12 IDs from supervisor prompts.py)
      open_max_third    numeric_open max question
      open_min_third    numeric_open min question
    """
    rng = random.Random(seed)
    entries: list[dict] = []
    counter = [1]

    def next_id() -> str:
        pid = f"tri_v2_{counter[0]:03d}"
        counter[0] += 1
        return pid

    # -- geometry_claim (fixed structure: 36 rows regardless of n)
    geom_rows = _generate_geometry_prompts_v2(next_id)
    n_geom = len(geom_rows)  # 36

    # -- numeric_open
    n_open = max(2, round(n * numeric_open_frac))
    if n_open % 2 != 0:
        n_open += 1
    open_rows = _generate_numeric_open_prompts_v2(n_open, rng, next_id)

    # -- numeric_validity: fill the remainder
    n_numeric = n - n_geom - n_open
    if n_numeric < 6:
        raise ValueError(
            f"n={n} too small for geometry ({n_geom}) + open ({n_open}) rows."
        )
    # Divide evenly across 3 tails; each tail gets n_per_tail_half true + same false
    n_per_tail = n_numeric // 3
    if n_per_tail % 2 != 0:
        n_per_tail -= 1
    n_per_tail_half = n_per_tail // 2
    numeric_rows = _generate_numeric_validity_prompts_v2(n_per_tail_half, rng, next_id)

    entries = geom_rows + open_rows + numeric_rows
    rng.shuffle(entries)

    # Renumber sequentially after shuffle.
    counter[0] = 1
    for entry in entries:
        entry["prompt_id"] = next_id()

    return entries


def summarize_v2(prompts: list[dict]) -> None:
    """Print a breakdown by (family, tail, label) for a v2 prompt set."""
    from collections import Counter
    c: Counter = Counter()
    for p in prompts:
        key = (
            p.get("family", "?"),
            p.get("tail", "?"),
            str(p.get("label", "?")),
        )
        c[key] += 1
    print(f"{'family':<22} {'tail':<18} {'label':<8} {'n':>5}")
    print("-" * 60)
    for (family, tail, label), cnt in sorted(c.items()):
        print(f"{family:<22} {tail:<18} {label:<8} {cnt:>5}")
    print(f"\nTotal: {len(prompts)}")
