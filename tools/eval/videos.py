"""
Run the STEP video pipeline on 6 YouTube URLs and score keyword predictions
against ``data/gold/videos.json``.

Usage::

    python Step_Project/tools/eval_videos.py                    # full run
    python Step_Project/tools/eval_videos.py --dry-run          # oracle
    python Step_Project/tools/eval_videos.py --ids 7R_DTzVHw5Y
    python Step_Project/tools/eval_videos.py --report-out reports/videos_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from step_eval import ItemMetrics, aggregate, score_item  # noqa: E402

_log = logging.getLogger("eval_videos")


def _predict_with_analyzer(url: str) -> dict[str, Any]:
    from run_video import VideoAnalyzer  # type: ignore

    analyzer = VideoAnalyzer()
    result = analyzer.analyze_youtube(url, mode="quick")

    return {
        "keywords": list(result.get("keywords") or []),
        "keywords_llm": list(result.get("keywords_llm") or []),
        "keywords_source": result.get("keywords_source"),
        "hybrid_ranking": result.get("hybrid_ranking"),
        "rerank_weights": result.get("rerank_weights"),
        "topic": (result.get("taxonomy") or {}).get("topic"),
        "subtopic": (result.get("taxonomy") or {}).get("subtopic"),
        "title": result.get("title"),
        "summary": result.get("summary"),
        "elapsed_s": result.get("elapsed_s"),
        "model_used": result.get("model_used"),
        "cached": result.get("cached"),
        "error": result.get("error"),
    }


def _predict_dry_run(item: dict) -> dict[str, Any]:
    return {
        "keywords": list(item.get("gold_keywords") or []),
        "keywords_llm": [],
        "keywords_source": "dry-run",
        "hybrid_ranking": None,
        "rerank_weights": None,
        "topic": item.get("topic"),
        "subtopic": item.get("subtopic"),
        "title": item.get("title_hint"),
        "summary": "",
        "elapsed_s": 0.0,
        "model_used": "dry-run",
        "cached": False,
        "error": None,
    }


def _row_for_item(item: dict, pred: dict[str, Any]) -> ItemMetrics:
    return score_item(
        item_id=str(item.get("id", "?")),
        predicted_keywords=pred.get("keywords") or [],
        gold_keywords=item.get("gold_keywords") or [],
        predicted_topic=pred.get("topic"),
        predicted_subtopic=pred.get("subtopic"),
        gold_topic=item.get("topic"),
        gold_subtopic=item.get("subtopic"),
    )


def _print_table(metrics: list[ItemMetrics]) -> None:
    cols = (("id", 14), ("n_pred", 6), ("n_gold", 6), ("P@5", 6),
            ("R@5", 6), ("F1@5", 6), ("Jacc", 6), ("topic", 5), ("subt.", 5))
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = (
            m.item_id.ljust(14),
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


def _build_report(items, rows, preds, dry_run):
    return {
        "schema_version": "v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dry_run": dry_run,
        "modality": "video",
        "aggregate": aggregate(rows),
        "items": [
            {
                **m.to_dict(),
                "predicted": {
                    "keywords": preds[i].get("keywords"),
                    "keywords_llm": preds[i].get("keywords_llm"),
                    "keywords_source": preds[i].get("keywords_source"),
                    "hybrid_ranking": preds[i].get("hybrid_ranking"),
                    "rerank_weights": preds[i].get("rerank_weights"),
                    "title": preds[i].get("title"),
                    "topic": preds[i].get("topic"),
                    "subtopic": preds[i].get("subtopic"),
                    "elapsed_s": preds[i].get("elapsed_s"),
                    "model_used": preds[i].get("model_used"),
                    "cached": preds[i].get("cached"),
                    "error": preds[i].get("error"),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(_ROOT / "data" / "gold" / "videos.json"))
    parser.add_argument("--ids", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-out", default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    gold_path = Path(args.gold).resolve()
    if not gold_path.is_file():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        return 2
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    items_all = list(gold.get("items") or [])

    wanted_ids = [s.strip() for s in (args.ids or "").split(",") if s.strip()]
    items = [it for it in items_all if str(it.get("id")) in set(wanted_ids)] if wanted_ids else items_all

    if not items:
        print("Nothing to evaluate.")
        return 0

    print(f"Gold file: {gold_path}")
    print(f"Items: {len(items)}   Mode: {'DRY-RUN' if args.dry_run else 'VideoAnalyzer (quick)'}")
    print()

    rows: list[ItemMetrics] = []
    preds: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        item_id = str(item.get("id"))
        url = item.get("url", "")
        if args.dry_run:
            pred = _predict_dry_run(item)
            note = "dry-run"
        else:
            t0 = time.time()
            try:
                pred = _predict_with_analyzer(url)
                note = f"done in {round(time.time() - t0, 1)}s"
            except Exception as e:
                pred = _predict_dry_run(item)
                pred["error"] = str(e)[:200]
                pred["keywords"] = []
                pred["keywords_source"] = "error"
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

