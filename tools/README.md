# Tools

CLI utilities that consume the production pipeline (run.py / run_video.py)
without modifying it.

## tools/eval/ — evaluation harnesses

| Script | Purpose |
|--------|---------|
| `videos.py` | Run `VideoAnalyzer` on the 6-video gold set, score against `data/gold/videos.json`. |
| `handmade.py` | Same harness for the MathE handmade-PDF gold set. |
| `videos_multi.py` | Compare BERT, Word2Vec, GloVe × cosine, Euclidean, Manhattan, dot, soft-cosine on the videos. Also runs BM25 baseline and BERT+MMR. |
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
python tools/eval/sweep_weights.py
python tools/viz/generate_figures.py
```
