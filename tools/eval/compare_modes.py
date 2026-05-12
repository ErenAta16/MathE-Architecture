"""
Compare three keyword-selection modes on the 6 YouTube videos:

  Mode 1 — PURE SEMANTIC SIMILARITY  (Beatriz's exact proposal)
           top-5 = argmax cos(BERT(title+summary), BERT(keyword))
           weights: (1.0, 0, 0)

  Mode 2 — ALGORITHMIC HYBRID  (math-only: cosine + regex rule hits)
           Both signals are deterministic and mathematically grounded.
           weights: (0.8, 0, 0.2)

  Mode 3 — FULL HYBRID  (math + LLM augmentation, current default)
           weights: (0.6, 0.3, 0.1)

The VLM (Gemini) is used identically in all three modes ONLY for video
understanding (title + summary extraction). It is NOT used as a ranking
signal in modes 1 and 2.

Results are scored against data/gold/videos.json.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from step_eval import score_item, aggregate
from semantic_similarity import default_similarity_config, top_k_keywords
from reranker import rerank_pool, RerankConfig
from taxonomy import keyword_hit_counts
from keyword_eval import DEFAULT_KEYWORD_POOL

VIDEO_CACHE = _ROOT / "step_pipeline" / "video_cache"
GOLD = json.loads((_ROOT / "data" / "gold" / "videos.json").read_text(encoding="utf-8"))
pool = list(DEFAULT_KEYWORD_POOL)


def _load_cache(vid_id):
    files = sorted(VIDEO_CACHE.glob(f"yt_{vid_id}_*_v4.json")) or sorted(VIDEO_CACHE.glob(f"yt_{vid_id}_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def evaluate_mode(label, weights):
    """Score the 6 videos with the given (w_cos, w_llm, w_rule) weights."""
    cfg_sim = default_similarity_config()
    cfg_re = RerankConfig(w_cos=weights[0], w_llm=weights[1], w_rule=weights[2])

    rows = []
    for item in GOLD["items"]:
        cache = _load_cache(item["id"])
        if cache is None:
            continue
        title = cache.get("title", "")
        summary = cache.get("summary", "")
        query = f"{title}\n\n{summary}".strip()
        llm_kws = list(cache.get("keywords_llm") or [])

        # Step 1: Cosine similarity for every keyword in the pool
        sim = top_k_keywords(query, pool, cfg=cfg_sim, query_source="title+summary")
        ranked = sim.get("ranked") or []
        cosine_scores = {r["keyword"]: float(r["score"]) for r in ranked if isinstance(r, dict)}

        # Step 2: Rule hits (regex) — also mathematical, deterministic
        try:
            hits = keyword_hit_counts(query, list(cosine_scores))
        except Exception:
            hits = {}

        # Step 3: Weighted combination
        if weights[1] > 0:
            llm_input = llm_kws  # use LLM ranking
        else:
            llm_input = []      # IGNORE LLM ranking signal
        if weights[2] == 0:
            hits = {}           # ignore rule signal

        ranked_h = rerank_pool(
            cosine_scores=cosine_scores,
            llm_ranking=llm_input,
            rule_hits=hits,
            cfg=cfg_re,
        )
        top5 = [r["keyword"] for r in ranked_h[:5]]

        m = score_item(
            item["id"], top5, item.get("gold_keywords") or [],
            predicted_topic=item.get("topic"),
            predicted_subtopic=item.get("subtopic"),
            gold_topic=item.get("topic"),
            gold_subtopic=item.get("subtopic"),
        )
        rows.append((item, top5, m))

    agg = aggregate([r[2] for r in rows])
    return rows, agg


def main():
    modes = [
        ("Mode 1: PURE COSINE (Beatriz's proposal)", (1.0, 0.0, 0.0)),
        ("Mode 2: ALGORITHMIC (cosine + rule, no LLM)", (0.8, 0.0, 0.2)),
        ("Mode 3: FULL HYBRID (with LLM augmentation)", (0.6, 0.3, 0.1)),
    ]

    print("=" * 90)
    print(f"{'Selection method':<50} {'P@5':>6} {'R@5':>6} {'F1@5':>6} {'Jacc':>6} {'Topic':>6} {'Sub':>5}")
    print("-" * 90)

    all_results = []
    for label, weights in modes:
        rows, agg = evaluate_mode(label, weights)
        all_results.append((label, weights, rows, agg))
        print(f"{label:<50} {agg['precision_at_5_mean']:>6.2f} "
              f"{agg['recall_at_5_mean']:>6.2f} {agg['f1_at_5_mean']:>6.2f} "
              f"{agg['jaccard_mean']:>6.2f} {agg['topic_accuracy']:>6.2f} "
              f"{agg['subtopic_accuracy']:>5.2f}")

    print("=" * 90)

    # Per-video detail for Mode 1 (Beatriz's exact proposal)
    print("\nMode 1 (PURE COSINE) — per-video detail:")
    print(f"{'Video ID':<14}  {'P@5':>5} {'R@5':>5} {'F1@5':>5} {'Top-5'}")
    print("-" * 90)
    for item, top5, m in all_results[0][2]:
        kws = ", ".join(top5)
        if len(kws) > 60:
            kws = kws[:57] + "..."
        print(f"{item['id']:<14}  {m.precision_at_5:>5.2f} {m.recall_at_5:>5.2f} {m.f1_at_5:>5.2f}  {kws}")

    # Save report
    out = _ROOT / "reports" / "selection_modes_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([
        {
            "label": r[0],
            "weights": {"w_cos": r[1][0], "w_llm": r[1][1], "w_rule": r[1][2]},
            "aggregate": r[3],
            "items": [{"id": ri[0]["id"], "top5": ri[1], **ri[2].to_dict()} for ri in r[2]],
        }
        for r in all_results
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()

