# Summary Context Ablation

Same 15 videos, same closed keyword pool, same fixed VLM title/summary.
Scores below use the UI-style hybrid ranker: embedding score + VLM keyword rank.

| Model | Mode | P@5 | R@5 | F1@5 | Jaccard | nDCG@5 | Weighted P@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bert | current | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.9054 | 0.8355 |
| bert | scene_only | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8960 | 0.8355 |
| bert | summary_late_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8998 | 0.8355 |
| bert | summary_shortlist | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.9054 | 0.8355 |
| bert | summary_shortlist_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8998 | 0.8355 |
| word2vec | current | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8927 | 0.8355 |
| word2vec | scene_only | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.9042 | 0.8355 |
| word2vec | summary_late_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.9005 | 0.8355 |
| word2vec | summary_shortlist | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8927 | 0.8355 |
| word2vec | summary_shortlist_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.9005 | 0.8355 |
| glove | current | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8810 | 0.8355 |
| glove | scene_only | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8996 | 0.8355 |
| glove | summary_late_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8990 | 0.8355 |
| glove | summary_shortlist | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8810 | 0.8355 |
| glove | summary_shortlist_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8990 | 0.8355 |
| word2vec_glove | current | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8810 | 0.8355 |
| word2vec_glove | scene_only | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8996 | 0.8355 |
| word2vec_glove | summary_late_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8990 | 0.8355 |
| word2vec_glove | summary_shortlist | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8810 | 0.8355 |
| word2vec_glove | summary_shortlist_fusion | 0.9067 | 0.4764 | 0.6223 | 0.0000 | 0.8990 | 0.8355 |
