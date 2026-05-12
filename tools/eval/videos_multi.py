"""
Run the 6-video keyword evaluation across all (model × metric) combinations
and produce a comparison report + visualization.

Combinations:
  Models:  BERT (all-MiniLM-L6-v2), Word2Vec (google-news-300), GloVe (glove-wiki-gigaword-300)
  Metrics: Cosine Similarity, Euclidean Similarity (1/(1+d))

For each combo the script:
  1. Loads the video's title+summary from the existing quick-mode cache
  2. Ranks the pool using similarity_compare.rank_pool
  3. Picks top-5
  4. Scores against gold using step_eval

Usage::
    python tools/eval_videos_multi.py
    python tools/eval_videos_multi.py --models bert,word2vec
    python tools/eval_videos_multi.py --report-out reports/multi_model_comparison.json
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

from step_eval import ItemMetrics, aggregate, score_item, diversity_at_k  # noqa: E402
from similarity_compare import rank_pool, rank_pool_bm25, top_n_mmr, ENCODERS, METRICS  # noqa: E402
from keyword_eval import DEFAULT_KEYWORD_POOL  # noqa: E402

_log = logging.getLogger("eval_multi")

GOLD_PATH = _ROOT / "data" / "gold" / "videos.json"
VIDEO_CACHE_DIR = _ROOT / "step_pipeline" / "video_cache"


def _load_video_text(vid_id: str) -> str:
    """Load title+summary from quick-mode cache (already computed by prior runs)."""
    import glob
    pattern = str(VIDEO_CACHE_DIR / f"yt_{vid_id}_quick_*.json")
    files = sorted(Path(VIDEO_CACHE_DIR).glob(f"yt_{vid_id}_quick_*.json"))
    if not files:
        files = sorted(Path(VIDEO_CACHE_DIR).glob(f"yt_{vid_id}_*.json"))
    if not files:
        return ""
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    title = data.get("title", "")
    summary = data.get("summary", "")
    return f"{title}\n\n{summary}".strip()


def _score_combo(items: list[dict], model: str, metric: str,
                  pool: list[str]) -> dict:
    """Run one (model, metric) combo on all items."""
    rows: list[dict] = []
    for item in items:
        vid_id = item["id"]
        query = _load_video_text(vid_id)
        if not query:
            query = item.get("title_hint", vid_id)

        try:
            ranked = rank_pool(query, pool, model=model, metric=metric)
            top5 = [r["keyword"] for r in ranked[:5]]
            scores = {r["keyword"]: r["score"] for r in ranked[:10]}
        except Exception as e:
            top5 = []
            scores = {}
            _log.warning(f"  [{model}×{metric}] {vid_id} FAILED: {e!s:.120}")

        m = score_item(
            item_id=vid_id,
            predicted_keywords=top5,
            gold_keywords=item.get("gold_keywords") or [],
            predicted_topic=item.get("topic"),
            predicted_subtopic=item.get("subtopic"),
            gold_topic=item.get("topic"),
            gold_subtopic=item.get("subtopic"),
        )
        rows.append({
            **m.to_dict(),
            "top5": top5,
            "scores_top5": {k: scores.get(k) for k in top5},
        })
    agg = aggregate([
        score_item(r["item_id"], r["top5"], items[i].get("gold_keywords") or [],
                   gold_topic=items[i].get("topic"), gold_subtopic=items[i].get("subtopic"),
                   predicted_topic=items[i].get("topic"), predicted_subtopic=items[i].get("subtopic"))
        for i, r in enumerate(rows)
    ])
    return {
        "model": model,
        "metric": metric,
        "aggregate": agg,
        "items": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="bert,word2vec,glove",
                        help="Comma-separated model list")
    parser.add_argument("--metrics", default="cosine,euclidean",
                        help="Comma-separated metric list")
    parser.add_argument("--report-out", default="reports/multi_model_comparison.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    items = gold["items"]
    pool = list(DEFAULT_KEYWORD_POOL)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print(f"Videos: {len(items)}   Models: {models}   Metrics: {metrics}")
    print(f"Total combos: {len(models) * len(metrics)} + BM25 + BERT-MMR + Hybrid\n")

    results: list[dict] = []
    for model in models:
        for metric in metrics:
            print(f"  Running {model} × {metric}...", end=" ", flush=True)
            t0 = time.time()
            combo = _score_combo(items, model, metric, pool)
            elapsed = round(time.time() - t0, 1)
            combo["elapsed_s"] = elapsed
            results.append(combo)
            agg = combo["aggregate"]
            print(f"P@5={agg['precision_at_5_mean']:.2f}  "
                  f"R@5={agg['recall_at_5_mean']:.2f}  "
                  f"F1={agg['f1_at_5_mean']:.2f}  "
                  f"Jacc={agg['jaccard_mean']:.2f}  ({elapsed}s)")

    # ── BM25 baseline ──
    print(f"  Running BM25 baseline...", end=" ", flush=True)
    t0 = time.time()
    bm25_rows = []
    for item in items:
        query = _load_video_text(item["id"]) or item.get("title_hint", "")
        ranked = rank_pool_bm25(query, pool)
        top5 = [r["keyword"] for r in ranked[:5]]
        m = score_item(item["id"], top5, item.get("gold_keywords") or [],
                       predicted_topic=item.get("topic"), predicted_subtopic=item.get("subtopic"),
                       gold_topic=item.get("topic"), gold_subtopic=item.get("subtopic"))
        bm25_rows.append({"top5": top5, **m.to_dict()})
    bm25_agg = aggregate([score_item(r["item_id"], r["top5"], items[i].get("gold_keywords") or [],
                          gold_topic=items[i].get("topic"), gold_subtopic=items[i].get("subtopic"),
                          predicted_topic=items[i].get("topic"), predicted_subtopic=items[i].get("subtopic"))
                          for i, r in enumerate(bm25_rows)])
    elapsed = round(time.time() - t0, 1)
    results.append({"model": "BM25", "metric": "term-freq", "aggregate": bm25_agg,
                    "items": bm25_rows, "elapsed_s": elapsed})
    print(f"P@5={bm25_agg['precision_at_5_mean']:.2f}  R@5={bm25_agg['recall_at_5_mean']:.2f}  "
          f"F1={bm25_agg['f1_at_5_mean']:.2f}  Jacc={bm25_agg['jaccard_mean']:.2f}  ({elapsed}s)")

    # ── BERT + MMR (diversity-aware) ──
    print(f"  Running BERT + MMR (λ=0.3)...", end=" ", flush=True)
    t0 = time.time()
    mmr_rows = []
    for item in items:
        query = _load_video_text(item["id"]) or item.get("title_hint", "")
        top5 = top_n_mmr(query, pool, model="bert", n=5, diversity=0.3)
        m = score_item(item["id"], top5, item.get("gold_keywords") or [],
                       predicted_topic=item.get("topic"), predicted_subtopic=item.get("subtopic"),
                       gold_topic=item.get("topic"), gold_subtopic=item.get("subtopic"))
        mmr_rows.append({"top5": top5, **m.to_dict()})
    mmr_agg = aggregate([score_item(r["item_id"], r["top5"], items[i].get("gold_keywords") or [],
                         gold_topic=items[i].get("topic"), gold_subtopic=items[i].get("subtopic"),
                         predicted_topic=items[i].get("topic"), predicted_subtopic=items[i].get("subtopic"))
                         for i, r in enumerate(mmr_rows)])
    elapsed = round(time.time() - t0, 1)
    results.append({"model": "BERT+MMR", "metric": "cosine+diversity", "aggregate": mmr_agg,
                    "items": mmr_rows, "elapsed_s": elapsed})
    print(f"P@5={mmr_agg['precision_at_5_mean']:.2f}  R@5={mmr_agg['recall_at_5_mean']:.2f}  "
          f"F1={mmr_agg['f1_at_5_mean']:.2f}  Jacc={mmr_agg['jaccard_mean']:.2f}  ({elapsed}s)")

    # ── Hybrid (from existing report if available) ──
    hybrid_path = Path(_ROOT / "reports" / "videos_hybrid_v1.json")
    if hybrid_path.exists():
        hdata = json.loads(hybrid_path.read_text(encoding="utf-8"))
        results.append({"model": "BERT+LLM", "metric": "hybrid(ours)",
                        "aggregate": hdata["aggregate"],
                        "items": hdata.get("items", []), "elapsed_s": 0})

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<12} {'Metric':<12} {'P@5':>6} {'R@5':>6} {'F1@5':>6} {'Jacc':>6} {'Time':>6}")
    print("-" * 70)
    for r in results:
        a = r["aggregate"]
        print(f"{r['model']:<12} {r['metric']:<12} "
              f"{a['precision_at_5_mean']:>6.2f} {a['recall_at_5_mean']:>6.2f} "
              f"{a['f1_at_5_mean']:>6.2f} {a['jaccard_mean']:>6.2f} "
              f"{r.get('elapsed_s', 0):>5.1f}s")
    print("=" * 70)

    # Save
    if args.report_out:
        out = Path(args.report_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": "v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_videos": len(items),
            "pool_size": len(pool),
            "combos": results,
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

