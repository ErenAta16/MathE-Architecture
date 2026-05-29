# Tools

CLI utilities that consume the production pipeline (run.py / run_video.py)
without modifying it.

## tools/eval/ — evaluation harnesses

| Script | Purpose |
|--------|---------|
| `videos.py` | Run `VideoAnalyzer` on the 6-video gold set, score against `data/gold/videos.json`. |
| `handmade.py` | Same harness for the MathE handmade-PDF gold set. |
| `videos_multi.py` | Compare BERT, Word2Vec, and GloVe semantic keyword ranking on a fixed video set. Supports custom 15-video gold files, custom 100+ keyword pools, cosine-first evaluation, optional BM25/MMR baselines, and Word2Vec/GloVe OOV coverage. |
| `sweep_weights.py` | Grid search over `(w_cos, w_llm)` for the hybrid video re-ranker with `w_rule=0.0`. |
| `compare_modes.py` | Pure-cosine vs algorithmic-only vs full-hybrid ablation. |

All eval scripts follow the same pattern: read gold JSON → run the pipeline
(or use cached results) → compute metrics via `step_eval.score_item` →
optionally write a JSON report under `reports/`.

## tools/viz/ — figure generation

| Script | Purpose |
|--------|---------|
| `generate_figures.py` | Produce the six canonical `step_fig0X.png` figures used in supervisor reports. |
| `visualize_eval.py` | Quick per-report visualisation (per-item bars, signal stack, aggregate compare). Useful for ad-hoc reports. |

Outputs go to `reports/figures/`.

## Usage

Always invoke from the project root so `_ROOT` resolves correctly:

```
python tools/eval/videos.py --report-out reports/final_video_evaluation.json
python tools/eval/videos_multi.py --gold data/gold/videos_15.json --pool data/keyword_pool_v2.json --metrics cosine --report-out reports/semantic_model_comparison_15videos.json --strict-queries
python tools/eval/sweep_weights.py
python tools/viz/generate_figures.py
```

For a VLM-reference agreement run on the 15-video set (not human gold):

```
python tools/eval/build_video_reference.py --urls-file data/videos_15_urls.txt --mode quick --gold-from-vlm --out data/gold/videos_15_vlm_reference.json
python tools/eval/videos_multi.py --gold data/gold/videos_15_vlm_reference.json --metrics cosine --strict-queries --report-out reports/semantic_model_comparison_15videos.json
python tools/viz/semantic_model_report.py reports/semantic_model_comparison_15videos.json --prefix semantic_model_comparison_15videos
```

`videos_multi.py` expects a gold/query JSON with an `items` array. Each item
needs an `id` plus `gold_keywords`; for fixed VLM text, add either
`query_text` directly or `title` and `summary`:

```
{
  "items": [
    {
      "id": "YouTubeId",
      "url": "https://www.youtube.com/watch?v=YouTubeId",
      "title": "VLM title",
      "summary": "VLM summary",
      "topic": "Integration",
      "subtopic": "Definite Integrals",
      "gold_keywords": ["Substitution", "Fundamental theorem of Calculus"]
    }
  ]
}
```

The keyword pool file can be a JSON list, a JSON object with `keywords`,
`pool`, or `items`, or a plain text file with one keyword per line.
