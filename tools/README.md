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
| `word_vector_top5.py` | Export separate/static Word2Vec and GloVe top-5 lists plus production-style VLM hybrids for the 15-video set. |

All eval scripts follow the same pattern: read gold JSON → run the pipeline
(or use cached results) → compute metrics via `step_eval.score_item` →
optionally write a JSON report under `reports/`.

## tools/viz/ — figure generation

| Script | Purpose |
|--------|---------|
| `generate_figures.py` | Produce the six canonical `step_fig0X.png` figures used in supervisor reports. |
| `visualize_eval.py` | Quick per-report visualisation (per-item bars, signal stack, aggregate compare). Useful for ad-hoc reports. |
| `semantic_model_report.py` | Convert semantic evaluation JSON into compact CSV/Markdown summaries and figures. |
| `export_semantic_comparison_xlsx.py` | Export the semantic model comparison report to a supervisor-facing Excel workbook. |
| `export_individual_model_top5.py` | Export BERT, GloVe, and Word2Vec top-5 keywords in separate columns without combined systems. |

Outputs go to `reports/figures/`.

## transcript utilities

| Script | Purpose |
|--------|---------|
| `transcribe_youtube_captions.py` | Download English YouTube captions and convert VTT to plain text. |
| `transcribe_youtube_whisper.py` | Run local faster-whisper transcription on downloaded YouTube media. |
| `transcribe_youtube_gemini.py` | Transcribe YouTube media through Gemini audio/video input. |
| `build_transcript_comparison.py` | Build a Markdown side-by-side comparison of Whisper, YouTube captions, and VLM summaries. |

Downloaded media and raw VTT caption files are ignored by git. Keep reusable
pilot evidence as `.txt` / `.json` under `data/transcripts/`.

## Usage

Always invoke from the project root so `_ROOT` resolves correctly:

```
python tools/eval/videos.py --report-out reports/final_video_evaluation.json
python tools/eval/videos_multi.py --gold data/gold/videos_15.json --pool data/keyword_pool_v2.json --metrics cosine --report-out reports/semantic_model_comparison_15videos.json --strict-queries
python tools/eval/sweep_weights.py
python tools/viz/generate_figures.py
python tools/viz/export_individual_model_top5.py
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

For transcript pilots:

```
python tools/transcribe_youtube_captions.py --urls-file data/transcripts/pilot_3_urls.txt
python tools/transcribe_youtube_whisper.py --urls-file data/transcripts/pilot_3_urls.txt --model medium.en
python tools/transcribe_youtube_gemini.py --urls-file data/transcripts/pilot_3_urls.txt --mode upload
python tools/build_transcript_comparison.py --gold data/gold/videos_15_graded_vlm_reference_nocache.json --ids KTNcYYHuBTY LVLuqNH5iWw uKfcS7-O6UE
```
