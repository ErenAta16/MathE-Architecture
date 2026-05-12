"""
Pure abstain / low-confidence helper for the STEP keyword pipeline.

Two pieces of information may be available for each keyword that the system
proposes:

* **rule_hits** — how many regular-expression patterns fired in the
  taxonomy module for that keyword. Always ``>= 0``. ``None`` is treated as
  "unknown".
* **cosine_score** — the cosine similarity between the problem text and the
  keyword in the embedding space (Layer 7 / per-scene similarity output).
  Always in ``[-1, 1]`` after L2-normalisation; ``None`` is treated as
  "unknown".

The abstain rule (mirroring the equation written in
``pipeline_report.docx`` section 3 (e)) is::

    uncertain(k)  =  ( rule_hits == 0 )  AND  ( cosine_score < tau_low )

where unknown signals are treated conservatively: a keyword is *not* marked
uncertain when nothing is known, so the system never silently downgrades a
chip just because a score happens to be missing.

The module is import-safe (no LLM, no network, no file I/O) and ships with
its own self-test that runs when the file is executed directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceConfig:
    """Thresholds for the three-band keyword confidence display.

    * ``tau_low``    cosine below which a rule-less keyword is *uncertain*.
    * ``tau_high``   cosine at or above which a keyword is *high* confidence.
    * ``min_rule_hits``  minimum regex hits to override a low cosine score.
    """

    tau_low: float = 0.30
    tau_high: float = 0.50
    min_rule_hits: int = 1


def default_confidence_config() -> ConfidenceConfig:
    """Build a config from optional environment overrides (no surprises if unset)."""
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, "").strip() or default)
        except ValueError:
            return default

    def _i(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, "").strip() or default)
        except ValueError:
            return default

    return ConfidenceConfig(
        tau_low=_f("STEP_CONFIDENCE_TAU_LOW", 0.30),
        tau_high=_f("STEP_CONFIDENCE_TAU_HIGH", 0.50),
        min_rule_hits=_i("STEP_CONFIDENCE_MIN_HITS", 1),
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def is_uncertain(
    rule_hits: int | None,
    cosine_score: float | None,
    *,
    cfg: ConfidenceConfig | None = None,
) -> bool:
    """Return True when the abstain condition holds.

    The decision uses the conservative reading: if neither signal is known,
    the keyword is *not* flagged. A known rule hit (>= ``min_rule_hits``)
    overrides a low cosine score; a known cosine at or above ``tau_low``
    overrides zero rule hits.
    """
    cfg = cfg or default_confidence_config()
    has_rules = isinstance(rule_hits, int) and rule_hits >= cfg.min_rule_hits
    if has_rules:
        return False
    if cosine_score is None:
        return False
    return cosine_score < cfg.tau_low


def confidence_band(
    rule_hits: int | None,
    cosine_score: float | None,
    *,
    cfg: ConfidenceConfig | None = None,
) -> str:
    """Return ``"high"``, ``"medium"`` or ``"low"`` for UI styling."""
    cfg = cfg or default_confidence_config()
    if is_uncertain(rule_hits, cosine_score, cfg=cfg):
        return "low"
    if cosine_score is not None and cosine_score >= cfg.tau_high:
        return "high"
    if isinstance(rule_hits, int) and rule_hits >= cfg.min_rule_hits:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Bulk annotation (used by run.py / run_video.py)
# ---------------------------------------------------------------------------

def annotate_keywords(
    keywords: Iterable[str],
    *,
    cosine_scores: Mapping[str, float] | None = None,
    rule_hits: Mapping[str, int] | None = None,
    cfg: ConfidenceConfig | None = None,
) -> list[dict]:
    """Return one ``chip`` dict per keyword with ``score``, ``hits``,
    ``uncertain`` and ``band`` fields. Order is preserved.

    Lookup keys are lowercased internally so callers do not have to worry
    about case differences between sources.
    """
    cfg = cfg or default_confidence_config()
    scores_lc = {str(k).lower(): float(v) for k, v in (cosine_scores or {}).items()}
    hits_lc = {str(k).lower(): int(v) for k, v in (rule_hits or {}).items()}

    out: list[dict] = []
    for raw in keywords:
        if not raw:
            continue
        name = str(raw)
        lc = name.lower()
        s = scores_lc.get(lc)
        h = hits_lc.get(lc)
        chip = {
            "keyword": name,
            "score": (round(s, 6) if isinstance(s, (int, float)) else None),
            "hits": (int(h) if isinstance(h, (int, float)) else None),
            "uncertain": is_uncertain(h, s, cfg=cfg),
            "band": confidence_band(h, s, cfg=cfg),
        }
        out.append(chip)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    cfg = ConfidenceConfig(tau_low=0.30, tau_high=0.50, min_rule_hits=1)

    # Strong cosine → high.
    assert confidence_band(0, 0.66, cfg=cfg) == "high"
    # Mid cosine, no rules → medium.
    assert confidence_band(0, 0.40, cfg=cfg) == "medium"
    # Low cosine, no rules → low (uncertain).
    assert is_uncertain(0, 0.18, cfg=cfg) is True
    assert confidence_band(0, 0.18, cfg=cfg) == "low"
    # Rule hit overrides low cosine.
    assert is_uncertain(2, 0.10, cfg=cfg) is False
    assert confidence_band(2, 0.10, cfg=cfg) == "high"
    # Unknown signals never abstain.
    assert is_uncertain(None, None, cfg=cfg) is False
    assert is_uncertain(None, 0.10, cfg=cfg) is True   # cosine known, low → abstain
    assert is_uncertain(0, None, cfg=cfg) is False     # cosine unknown → no abstain

    chips = annotate_keywords(
        ["Trigonometric substitution", "Volume of revolution", "Total area"],
        cosine_scores={
            "trigonometric substitution": 0.66,
            "volume of revolution": 0.40,
            "total area": 0.05,
        },
        rule_hits={
            "trigonometric substitution": 1,
            "volume of revolution": 0,
            "total area": 0,
        },
        cfg=cfg,
    )
    assert chips[0]["band"] == "high" and not chips[0]["uncertain"]
    assert chips[1]["band"] == "medium" and not chips[1]["uncertain"]
    assert chips[2]["band"] == "low" and chips[2]["uncertain"]
    print("confidence self-test OK:", chips)


if __name__ == "__main__":
    _self_test()
