"""Grid search over (w_embed, w_llm) on the 6 video gold set.

Goal: find the sweet spot where embedding-based semantic similarity remains
the main decision signal while the Gemini/VLM keyword order can still support it.

Constraints during sweep:
  w_embed + w_llm = 1.0
  w_rule = 0.0  (rule-hit score removed from the decision)
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from step_eval import score_item, aggregate
from semantic_similarity import default_similarity_config, top_k_keywords
from reranker import rerank_pool, RerankConfig
from keyword_eval import DEFAULT_KEYWORD_POOL

VIDEO_CACHE = _ROOT / "step_pipeline" / "video_cache"
GOLD = json.loads((_ROOT / "data" / "gold" / "videos.json").read_text(encoding="utf-8"))
pool = list(DEFAULT_KEYWORD_POOL)


def _load(vid_id):
    files = sorted(VIDEO_CACHE.glob(f"yt_{vid_id}_quick_*.json"))
    if not files:
        files = sorted(VIDEO_CACHE.glob(f"yt_{vid_id}_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def evaluate(weights):
    cfg_sim = default_similarity_config()
    cfg_re = RerankConfig(w_cos=weights[0], w_llm=weights[1], w_rule=0.0)
    rows = []
    for item in GOLD["items"]:
        cache = _load(item["id"])
        if cache is None:
            continue
        query = f"{cache.get('title','')}\n\n{cache.get('summary','')}".strip()
        llm_kws = list(cache.get("keywords_llm") or [])

        sim = top_k_keywords(query, pool, cfg=cfg_sim, query_source="title+summary")
        cosine_scores = {r["keyword"]: float(r["score"]) for r in (sim.get("ranked") or [])}

        ranked = rerank_pool(
            cosine_scores=cosine_scores,
            llm_ranking=llm_kws if weights[1] > 0 else [],
            cfg=cfg_re,
        )
        top5 = [r["keyword"] for r in ranked[:5]]
        m = score_item(item["id"], top5, item.get("gold_keywords") or [],
                       predicted_topic=item.get("topic"), predicted_subtopic=item.get("subtopic"),
                       gold_topic=item.get("topic"), gold_subtopic=item.get("subtopic"))
        rows.append(m)
    return aggregate(rows)


# Search grid: two-signal fusion only, summing to 1.
candidates = [(round(w_embed / 100, 2), round(1 - w_embed / 100, 2))
              for w_embed in range(0, 101, 5)]

# De-duplicate
seen = set()
unique = []
for c in candidates:
    if c not in seen:
        seen.add(c)
        unique.append(c)

print(f"Testing {len(unique)} weight combinations on 6 videos\n")
print(f"{'w_emb':>6} {'w_llm':>6}  {'P@5':>5} {'R@5':>5} {'F1@5':>5} {'Jacc':>5}")
print("-" * 50)

best = None
results = []
for w in unique:
    agg = evaluate(w)
    results.append((w, agg))
    f1 = agg["f1_at_5_mean"]
    print(f"{w[0]:>6.2f} {w[1]:>6.2f}  "
          f"{agg['precision_at_5_mean']:>5.2f} {agg['recall_at_5_mean']:>5.2f} "
          f"{f1:>5.2f} {agg['jaccard_mean']:>5.2f}")
    # Primary objective: F1@5. Tie-breaker: keep the highest embedding weight
    # among equally scoring configurations so the model does not collapse to
    # LLM-only when a mixed signal performs identically on the small gold set.
    if (
        best is None
        or f1 > best[1]["f1_at_5_mean"]
        or (f1 == best[1]["f1_at_5_mean"] and w[0] > best[0][0])
    ):
        best = (w, agg)

print("-" * 50)
print(f"\nBEST F1@5 configuration: w_embed={best[0][0]}, w_llm={best[0][1]}, w_rule=0.0")
print(f"   F1@5 = {best[1]['f1_at_5_mean']:.3f}")
print(f"   P@5  = {best[1]['precision_at_5_mean']:.3f}")
print(f"   R@5  = {best[1]['recall_at_5_mean']:.3f}")
print(f"   Jacc = {best[1]['jaccard_mean']:.3f}")

# Save
out = _ROOT / "reports" / "weight_sweep.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([
    {"weights": {"w_embed": w[0], "w_llm": w[1], "w_rule": 0.0}, "aggregate": agg}
    for w, agg in results
], ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {out}")

