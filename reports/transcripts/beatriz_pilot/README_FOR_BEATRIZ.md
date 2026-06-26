# MathE pilot — spoken transcripts for Beatriz

Generated: 2026-05-29 21:21 UTC

## Recommended files (primary deliverable)

Use the **`gemini_primary/`** folder. These are the best-quality transcripts we can produce today with **Gemini 3 Flash** (`gemini-3-flash-preview`) and the optimized math-education prompt (`mathe-transcribe-v4-readable`).

| File | Description |
|------|-------------|
| `gemini_primary/KTNcYYHuBTY_gemini_transcript.txt` | Powers of the Imaginary Unit i — clean transcript |
| `gemini_primary/KTNcYYHuBTY_gemini_full.txt` | Same video — transcript + run metadata |
| `gemini_primary/LVLuqNH5iWw_gemini_transcript.txt` | Local Extrema of a Multivariable Function — clean transcript |
| `gemini_primary/LVLuqNH5iWw_gemini_full.txt` | Same video — transcript + run metadata |
| `gemini_primary/uKfcS7-O6UE_gemini_transcript.txt` | Quotient Rule for Differentiation — clean transcript |
| `gemini_primary/uKfcS7-O6UE_gemini_full.txt` | Same video — transcript + run metadata |

## Baselines (optional comparison)

- `baseline_whisper/` — local **distil-large-v3** (optimized Whisper)
- `baseline_youtube_captions/` — YouTube auto subtitles

## How the Gemini transcripts were produced

1. Download YouTube audio/video with yt-dlp.
2. Upload media to Gemini Files API.
3. Transcribe with `gemini-3-flash-preview`, temperature 0, math-aware system prompt.
4. Formulas appear as inline LaTeX where helpful; spoken words are kept verbatim.

## Quality snapshot (Gemini primary)

| Video | Words | Duration (s) | Keyword recall* | Notes |
|-------|------:|-------------:|----------------:|-------|
| KTNcYYHuBTY | 659 | 279.2 | 20.0% | imaginary unit ✓ |
| LVLuqNH5iWw | 564 | 441.9 | 80.0% | saddle point ✓ |
| uKfcS7-O6UE | 455 | 252.1 | 20.0% | quotient rule ✓ |

*Keyword recall = share of graded VLM reference keywords literally present in the transcript (proxy for MathE keyword-pool alignment, not human annotation).

## Note on a separate pipeline step (not included here)

These files are **spoken transcripts**. A different internal step feeds text into the MathE **keyword ranking** pipeline as `query_text` (today that input is usually a VLM video summary). Using transcripts instead is a possible follow-up experiment; this package is the transcription deliverable only.

## Video URLs

- **Powers of the Imaginary Unit i** — https://www.youtube.com/watch?v=KTNcYYHuBTY
- **Local Extrema of a Multivariable Function** — https://www.youtube.com/watch?v=LVLuqNH5iWw
- **Quotient Rule for Differentiation** — https://www.youtube.com/watch?v=uKfcS7-O6UE
