"""
Run the STEP pipeline on the MathE handmade materials and report
keyword-level evaluation metrics against ``data/gold/handmade.json``.

Usage from the project root::

    python Step_Project/tools/eval_handmade.py            # run + report
    python Step_Project/tools/eval_handmade.py --limit 3
    python Step_Project/tools/eval_handmade.py --dry-run  # no LLM, oracle predictions
    python Step_Project/tools/eval_handmade.py --report-out reports/handmade.json

The runner is intentionally thin: heavy work happens in ``STEPSolver`` and
in ``step_eval``. This file only wires them together and prints the table
the supervisor expects to see in the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Make the project root importable when invoking from anywhere.
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from step_eval import (  # noqa: E402
    ItemMetrics,
    aggregate,
    score_item,
)

_log = logging.getLogger("eval_handmade")


# ---------------------------------------------------------------------------
# Locating PDFs
# ---------------------------------------------------------------------------

_REPO_ROOT = _ROOT.parent  # Step_Project / .. = workspace root
_SEARCH_ROOTS = (
    _REPO_ROOT,
    _ROOT,
    _ROOT / "Materials shared with EREN" / "mathE handmade materials",
    _ROOT / "uploads",
)


def find_pdf(rel_path: str, item_id: str) -> Path | None:
    """Search the standard locations for the gold-listed PDF.

    Falls back to ``<id>.pdf`` if the literal relative path is not found, so
    the harness still works after the user reorganises folders.
    """
    candidates: list[Path] = []
    rel = Path(rel_path)
    for root in _SEARCH_ROOTS:
        candidates.append(root / rel)
        candidates.append(root / rel.name)
        candidates.append(root / f"{item_id}.pdf")
    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_with_solver(pdf_path: Path) -> dict[str, Any]:
    """Run ``STEPSolver`` on a PDF and return the relevant fields for scoring.

    Lazy-imports the solver so ``--dry-run`` and ``--help`` stay fast and do
    not depend on the LLM stack being installed.

    Returns a diagnostic-friendly dict that includes the LLM ordering kept
    for cross-reference (``task2_llm``), the cosine ranking that drove the
    final selection (``cosine_top``), and any per-step error so the report
    can pinpoint which layer dropped a keyword on the floor.
    """
    from run import STEPSolver  # type: ignore

    solver = STEPSolver(use_nougat=False, use_vlm=True)
    result = solver.solve(pdf_path, verbose=False)

    keval = result.get("keyword_eval") or {}
    tax = result.get("taxonomy") or {}

    sim = keval.get("keyword_similarity") or {}
    ranked = sim.get("ranked") if isinstance(sim, dict) else None
    cosine_top = []
    if isinstance(ranked, list):
        for row in ranked[:10]:
            if isinstance(row, dict) and row.get("keyword"):
                cosine_top.append({
                    "keyword": row.get("keyword"),
                    "score": row.get("score"),
                })

    return {
        "task2": list(keval.get("task2") or []),
        "task1": list(keval.get("task1") or []),
        "task2_llm": list(keval.get("task2_llm") or []),
        "task2_source": keval.get("task2_source"),
        "cosine_query_source": keval.get("cosine_query_source"),
        "cosine_top": cosine_top,
        "task2_chips": keval.get("task2_chips"),
        "hybrid_ranking": keval.get("hybrid_ranking"),
        "rerank_weights": keval.get("rerank_weights"),
        "topic": tax.get("topic"),
        "subtopic": tax.get("subtopic"),
        "elapsed_s": result.get("elapsed_s"),
        "model_used": result.get("model_used"),
        "error": result.get("error") or keval.get("error"),
        "keval_skipped": bool(keval.get("skipped")),
        "similarity_error": sim.get("error") if isinstance(sim, dict) else None,
    }


def _predict_dry_run(item: dict) -> dict[str, Any]:
    """Use the gold itself as predictions. Validates the harness without LLM."""
    return {
        "task2": list(item.get("gold_keywords") or []),
        "task1": list(item.get("gold_keywords") or []),
        "topic": item.get("topic"),
        "subtopic": item.get("subtopic"),
        "elapsed_s": 0.0,
        "model_used": "dry-run",
        "error": None,
        "keyword_similarity": None,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _row_for_item(item: dict, pred: dict[str, Any]) -> ItemMetrics:
    return score_item(
        item_id=str(item.get("id", "?")),
        predicted_keywords=pred.get("task2") or [],
        gold_keywords=item.get("gold_keywords") or [],
        predicted_topic=pred.get("topic"),
        predicted_subtopic=pred.get("subtopic"),
        gold_topic=item.get("topic"),
        gold_subtopic=item.get("subtopic"),
    )


def _print_table(metrics: list[ItemMetrics]) -> None:
    cols = (
        ("id", 6),
        ("n_pred", 6),
        ("n_gold", 6),
        ("P@5", 6),
        ("R@5", 6),
        ("F1@5", 6),
        ("Jacc", 6),
        ("topic", 5),
        ("subt.", 5),
    )
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = (
            m.item_id.ljust(6),
            str(m.n_pred).ljust(6),
            str(m.n_gold).ljust(6),
            f"{m.precision_at_5:.2f}".ljust(6),
            f"{m.recall_at_5:.2f}".ljust(6),
            f"{m.f1_at_5:.2f}".ljust(6),
            f"{m.jaccard:.2f}".ljust(6),
            ("Y" if m.topic_match else "-").ljust(5),
            ("Y" if m.subtopic_match else "-").ljust(5),
        )
        print("  ".join(row))


def _build_report(items: list[dict], rows: list[ItemMetrics],
                    preds: list[dict[str, Any]],
                    dry_run: bool) -> dict:
    return {
        "schema_version": "v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": dry_run,
        "aggregate": aggregate(rows),
        "items": [
            {
                **m.to_dict(),
                "predicted": {
                    "task2": preds[i].get("task2"),
                    "task2_llm": preds[i].get("task2_llm"),
                    "task2_source": preds[i].get("task2_source"),
                    "task1": preds[i].get("task1"),
                    "task2_chips": preds[i].get("task2_chips"),
                    "cosine_query_source": preds[i].get("cosine_query_source"),
                    "cosine_top": preds[i].get("cosine_top"),
                    "hybrid_ranking": preds[i].get("hybrid_ranking"),
                    "rerank_weights": preds[i].get("rerank_weights"),
                    "topic": preds[i].get("topic"),
                    "subtopic": preds[i].get("subtopic"),
                    "elapsed_s": preds[i].get("elapsed_s"),
                    "model_used": preds[i].get("model_used"),
                    "error": preds[i].get("error"),
                    "similarity_error": preds[i].get("similarity_error"),
                    "keval_skipped": preds[i].get("keval_skipped"),
                },
                "gold": {
                    "topic": items[i].get("topic"),
                    "subtopic": items[i].get("subtopic"),
                    "gold_keywords": items[i].get("gold_keywords"),
                    "bootstrap": items[i].get("bootstrap", False),
                },
            }
            for i, m in enumerate(rows)
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        default=str(_ROOT / "data" / "gold" / "handmade.json"),
        help="Path to the gold JSON (default: data/gold/handmade.json).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Run on at most this many graded items (0 = all).",
    )
    parser.add_argument(
        "--ids", default="",
        help="Comma-separated list of item ids to evaluate (overrides --limit).",
    )
    parser.add_argument(
        "--include-ungraded", action="store_true",
        help="Solve items with empty gold_keywords too (they are still ungraded in metrics).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use gold_keywords as predictions (no LLM call). Validates the harness only.",
    )
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional path to write a JSON report. Disabled when empty.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging from the runner (does not affect STEPSolver verbosity).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    gold_path = Path(args.gold).resolve()
    if not gold_path.is_file():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        return 2

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    items_all: list[dict] = list(gold.get("items") or [])
    if not items_all:
        print(f"No items in {gold_path}", file=sys.stderr)
        return 2

    if args.include_ungraded:
        items = items_all
    else:
        items = [it for it in items_all if (it.get("gold_keywords") or [])]

    wanted_ids = [s.strip() for s in (args.ids or "").split(",") if s.strip()]
    if wanted_ids:
        wanted = set(wanted_ids)
        items = [it for it in items_all if str(it.get("id")) in wanted]
    elif args.limit and args.limit > 0:
        items = items[: args.limit]

    if not items:
        print("Nothing to evaluate (no graded items). Use --include-ungraded to run anyway.")
        return 0

    print(f"Gold file: {gold_path}")
    print(f"Items to evaluate: {len(items)}  (graded only={not args.include_ungraded})")
    print(f"Mode: {'DRY-RUN (oracle predictions)' if args.dry_run else 'STEPSolver'}")
    print()

    rows: list[ItemMetrics] = []
    preds: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        item_id = str(item.get("id"))
        rel_pdf = item.get("pdf") or f"{item_id}.pdf"

        if args.dry_run:
            pred = _predict_dry_run(item)
            note = "dry-run"
        else:
            pdf_path = find_pdf(rel_pdf, item_id)
            if pdf_path is None:
                pred = {"task2": [], "task1": [], "topic": None, "subtopic": None,
                        "elapsed_s": 0.0, "model_used": "", "error": "pdf not found"}
                note = f"PDF not found ({rel_pdf})"
            else:
                t0 = time.time()
                try:
                    pred = _predict_with_solver(pdf_path)
                    note = f"solved in {round(time.time() - t0, 1)}s"
                except Exception as e:
                    pred = {"task2": [], "task1": [], "topic": None, "subtopic": None,
                            "elapsed_s": 0.0, "model_used": "", "error": str(e)[:160]}
                    note = f"FAILED: {str(e)[:120]}"

        if args.verbose:
            print(f"[{i}/{len(items)}] id={item_id} :: {note}")

        rows.append(_row_for_item(item, pred))
        preds.append(pred)

    print()
    _print_table(rows)
    print()
    agg = aggregate(rows)
    print("Aggregate:")
    for k in ("n_total", "n_graded", "precision_at_5_mean", "recall_at_5_mean",
              "f1_at_5_mean", "jaccard_mean", "topic_accuracy", "subtopic_accuracy"):
        print(f"  {k:>22} = {agg[k]}")

    if args.report_out:
        out_path = Path(args.report_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = _build_report(items, rows, preds, args.dry_run)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

