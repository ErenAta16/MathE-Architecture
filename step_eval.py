"""
Pure-Python metric library for the STEP keyword pipeline.

This module is intentionally free of I/O and free of any dependency on the
Gemini / Together / sentence-transformers stacks. Tools and notebooks can
import it without paying the cost of loading the rest of the pipeline. All
functions accept already-extracted predictions and gold annotations and
return plain ``float``s.

Metric definitions
------------------

Let ``P`` be the predicted ranked keyword list and ``G`` the (unordered) set
of gold keywords. Define ``P_k`` as the first ``k`` items of ``P`` and use a
case-insensitive, whitespace-collapsed equality on keyword strings.

* **Precision@k**
      ``|P_k ∩ G| / |P_k|`` if ``P_k`` is non-empty, else ``0``.
* **Recall@k**
      ``|P_k ∩ G| / |G|`` if ``G`` is non-empty, else ``0``.
* **F1@k**
      Harmonic mean of Precision@k and Recall@k.
* **Jaccard**
      ``|P ∩ G| / |P ∪ G|``. Uses the full ``P``, not ``P_k``. Returns
      ``1.0`` when both sets are empty (vacuous match) and ``0.0`` when only
      one is empty.

A small self-test runs when this module is executed as a script
(``python step_eval.py``); it verifies the formulas on hand-crafted fixtures
without touching any LLM or embedding model.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize_keyword(s: str) -> str:
    """Lowercase + collapse whitespace + strip surrounding punctuation.

    Used so that ``"Partial fractions decomposition"``, ``"partial  fractions
    decomposition"`` and ``" Partial Fractions Decomposition. "`` are treated
    as the same keyword.
    """
    if s is None:
        return ""
    out = str(s).strip().strip(".,;:")
    out = _WS.sub(" ", out)
    return out.lower()


def _normset(items: Iterable[str]) -> set[str]:
    return {normalize_keyword(x) for x in items if x and str(x).strip()}


def _normlist(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        n = normalize_keyword(x)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Set-based metrics
# ---------------------------------------------------------------------------

def precision_at_k(predicted: Sequence[str], gold: Iterable[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top = _normlist(list(predicted)[:k])
    if not top:
        return 0.0
    g = _normset(gold)
    if not g:
        return 0.0
    hits = sum(1 for p in top if p in g)
    return hits / len(top)


def recall_at_k(predicted: Sequence[str], gold: Iterable[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    g = _normset(gold)
    if not g:
        return 0.0
    top = set(_normlist(list(predicted)[:k]))
    return len(top & g) / len(g)


def f1_at_k(predicted: Sequence[str], gold: Iterable[str], k: int = 5) -> float:
    p = precision_at_k(predicted, gold, k)
    r = recall_at_k(predicted, gold, k)
    if p + r <= 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def jaccard(predicted: Iterable[str], gold: Iterable[str]) -> float:
    a = _normset(predicted)
    b = _normset(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dcg_at_k(relevances: Sequence[float], k: int = 5) -> float:
    """Discounted cumulative gain for an already-ranked relevance list."""
    if k <= 0:
        return 0.0
    score = 0.0
    for i, rel in enumerate(list(relevances)[:k]):
        try:
            gain = float(rel)
        except (TypeError, ValueError):
            gain = 0.0
        if gain <= 0:
            continue
        score += gain / math.log2(i + 2)
    return score


def ndcg_at_k(
    relevances: Sequence[float],
    ideal_relevances: Sequence[float] | None = None,
    k: int = 5,
) -> float:
    """Normalized DCG in ``[0, 1]``.

    ``relevances`` follows the system's rank order. ``ideal_relevances`` is the
    reference set of available graded gains; if omitted, the sorted system gains
    are used as the ideal order.
    """
    if k <= 0:
        return 0.0
    rels = list(relevances or [])[:k]
    ideals = list(ideal_relevances if ideal_relevances is not None else rels)
    ideals = sorted((float(x) for x in ideals), reverse=True)[:k]
    ideal = dcg_at_k(ideals, k=k)
    if ideal <= 0:
        return 0.0
    return dcg_at_k(rels, k=k) / ideal


def label_match(predicted: str | None, gold: str | None) -> bool:
    """Case/whitespace-insensitive equality used for topic / subtopic accuracy."""
    if not predicted or not gold:
        return False
    return normalize_keyword(predicted) == normalize_keyword(gold)


def diversity_at_k(keywords: Sequence[str], *, k: int = 5,
                    encoder=None) -> float:
    """Inter-keyword diversity: 1 - mean pairwise cosine among the top-k.

    Higher = more diverse selection. Range [0, 1]. Returns 1.0 when fewer
    than 2 keywords are provided (vacuously diverse). Follows the diversity
    axis of KPEval (Wu et al., ACL 2024).

    If no ``encoder`` is passed, uses sentence-transformers/all-MiniLM-L6-v2
    (lazy-loaded). Pass a callable ``encoder(texts) -> np.ndarray`` to
    override.
    """
    import numpy as np

    items = list(keywords)[:k]
    if len(items) < 2:
        return 1.0

    if encoder is None:
        from similarity_compare import _encode_bert
        encoder = _encode_bert

    vecs = encoder(items)
    n = len(vecs)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(vecs[i], vecs[j])) / (
                float(np.linalg.norm(vecs[i])) * float(np.linalg.norm(vecs[j])) + 1e-10
            )
            total += sim
            count += 1
    mean_sim = total / count if count > 0 else 0.0
    return round(1.0 - mean_sim, 4)


# ---------------------------------------------------------------------------
# Per-item record + aggregate
# ---------------------------------------------------------------------------

@dataclass
class ItemMetrics:
    """Metric values for a single (predicted, gold) pair."""

    item_id: str
    n_pred: int
    n_gold: int
    precision_at_5: float
    recall_at_5: float
    f1_at_5: float
    jaccard: float
    topic_match: bool
    subtopic_match: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "n_pred": self.n_pred,
            "n_gold": self.n_gold,
            "precision_at_5": round(self.precision_at_5, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "f1_at_5": round(self.f1_at_5, 4),
            "jaccard": round(self.jaccard, 4),
            "topic_match": bool(self.topic_match),
            "subtopic_match": bool(self.subtopic_match),
            "notes": list(self.notes),
        }


def score_item(
    item_id: str,
    predicted_keywords: Sequence[str],
    gold_keywords: Iterable[str],
    *,
    predicted_topic: str | None = None,
    predicted_subtopic: str | None = None,
    gold_topic: str | None = None,
    gold_subtopic: str | None = None,
    k: int = 5,
) -> ItemMetrics:
    """Build the metric record for one PDF / video."""
    notes: list[str] = []
    pred_list = list(predicted_keywords or [])
    gold_list = list(gold_keywords or [])
    if not gold_list:
        notes.append("ungraded: empty gold_keywords")
    return ItemMetrics(
        item_id=item_id,
        n_pred=len(_normlist(pred_list)),
        n_gold=len(_normset(gold_list)),
        precision_at_5=precision_at_k(pred_list, gold_list, k=k),
        recall_at_5=recall_at_k(pred_list, gold_list, k=k),
        f1_at_5=f1_at_k(pred_list, gold_list, k=k),
        jaccard=jaccard(pred_list, gold_list),
        topic_match=label_match(predicted_topic, gold_topic),
        subtopic_match=label_match(predicted_subtopic, gold_subtopic),
        notes=notes,
    )


def aggregate(items: Sequence[ItemMetrics]) -> dict:
    """Mean metrics + topic/subtopic accuracy across rows.

    Items with empty gold are excluded from the kw means but still counted in
    ``n_total`` so the report shows how many entries are still ungraded.
    """
    n_total = len(items)
    graded = [m for m in items if m.n_gold > 0]
    n = len(graded) or 1
    return {
        "n_total": n_total,
        "n_graded": len(graded),
        "precision_at_5_mean": round(sum(m.precision_at_5 for m in graded) / n, 4),
        "recall_at_5_mean": round(sum(m.recall_at_5 for m in graded) / n, 4),
        "f1_at_5_mean": round(sum(m.f1_at_5 for m in graded) / n, 4),
        "jaccard_mean": round(sum(m.jaccard for m in graded) / n, 4),
        "topic_accuracy": round(
            sum(1 for m in graded if m.topic_match) / n, 4,
        ),
        "subtopic_accuracy": round(
            sum(1 for m in graded if m.subtopic_match) / n, 4,
        ),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _self_test() -> None:
    pred = ["Trigonometric substitution", "Direct integrals", "Substitution"]
    gold = ["Trigonometric substitution", "Trigonometric functions",
            "Fundamental theorem of Calculus"]

    p5 = precision_at_k(pred, gold, k=5)
    r5 = recall_at_k(pred, gold, k=5)
    j = jaccard(pred, gold)

    assert _close(p5, 1 / 3), f"precision_at_5={p5}"
    assert _close(r5, 1 / 3), f"recall_at_5={r5}"
    assert _close(j, 1 / 5), f"jaccard={j}"

    # Empty handling.
    assert precision_at_k([], gold) == 0.0
    assert recall_at_k(pred, []) == 0.0
    assert jaccard([], []) == 1.0
    assert jaccard(pred, []) == 0.0

    # Case-insensitive normalization.
    assert label_match("Definite Integrals", "definite integrals")

    # Aggregate over a tiny fixture.
    item_a = score_item(
        "A", pred, gold,
        predicted_topic="Integration", predicted_subtopic="Definite Integrals",
        gold_topic="Integration", gold_subtopic="Definite Integrals",
    )
    item_b = score_item(
        "B", ["Substitution"], ["Substitution", "Direct integrals"],
        predicted_topic="Integration", predicted_subtopic="Indefinite Integrals",
        gold_topic="Integration", gold_subtopic="Definite Integrals",
    )
    agg = aggregate([item_a, item_b])
    assert agg["n_graded"] == 2, agg
    assert agg["topic_accuracy"] == 1.0, agg
    assert agg["subtopic_accuracy"] == 0.5, agg
    print("step_eval self-test OK:", agg)


if __name__ == "__main__":
    _self_test()
