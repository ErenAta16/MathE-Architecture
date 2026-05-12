"""
Generate the complete set of academic figures for the final STEP report.

All filenames use the prefix `step_` and are descriptive in English so they
read cleanly when embedded in the report.

Inputs:
  - reports/final_video_evaluation.json   (live system, 6 videos)
  - reports/final_comparison.json         (multi-method comparison)
  - reports/selection_modes_comparison.json (3 selection modes)

Outputs (in reports/figures/):
  - step_fig01_pipeline_architecture.png       — 4-layer system diagram
  - step_fig02_per_video_metrics.png           — P@5/R@5/F1/Jacc per video
  - step_fig03_method_comparison.png           — BERT vs Word2Vec vs GloVe vs BM25
  - step_fig04_signal_decomposition.png        — How each signal contributes
  - step_fig05_selection_modes.png             — Pure cosine vs Algorithmic vs Hybrid
  - step_fig06_aggregate_summary.png           — Final aggregate metrics
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


_OUT = Path("reports/figures")
_OUT.mkdir(parents=True, exist_ok=True)


def _short_id(vid_id):
    titles = {
        "7hCsQOKOYS8": "V1: u-substitution",
        "7R_DTzVHw5Y": "V2: area by geometry",
        "OBZnEvoRuW0": "V3: integral of tan(x)",
        "4ZiO_eaj9Ag": "V4: area between curves",
        "Sg24zkhG0Vw": "V5: volume of revolution",
        "JB961uV015E": "V6: partial fractions",
    }
    return titles.get(vid_id, vid_id[:8])


# ─────────────────────────────────────────────────────────────────
# Figure 1: System architecture diagram
# ─────────────────────────────────────────────────────────────────

def fig_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7.5, 8.6, "STEP Pipeline — Keyword Selection Architecture",
            ha="center", fontsize=15, weight="bold")
    ax.text(7.5, 8.2, "Four-layer system: Content Understanding · Encoding · Similarity · Selection",
            ha="center", fontsize=11, color="gray")

    # Layer 1: Content Understanding
    box1 = FancyBboxPatch((0.5, 6.0), 14.0, 1.5, boxstyle="round,pad=0.1",
                           facecolor="#fff5f5", edgecolor="#c53030", linewidth=2)
    ax.add_patch(box1)
    ax.text(7.5, 7.2, "Layer 1 — CONTENT UNDERSTANDING (VLM)",
            ha="center", fontsize=11, weight="bold", color="#c53030")
    ax.text(7.5, 6.75, "Gemini 2.5 Pro watches the video and produces: title + summary (English text)",
            ha="center", fontsize=10)
    ax.text(7.5, 6.3, "Role: irreplaceable — the system needs textual representation to encode",
            ha="center", fontsize=9, style="italic", color="gray")

    ax.annotate("", xy=(7.5, 5.9), xytext=(7.5, 5.95),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Layer 2: Semantic Encoding
    box2 = FancyBboxPatch((0.5, 4.4), 14.0, 1.4, boxstyle="round,pad=0.1",
                           facecolor="#ebf8ff", edgecolor="#2b6cb0", linewidth=2)
    ax.add_patch(box2)
    ax.text(7.5, 5.55, "Layer 2 — SEMANTIC ENCODING (BERT)",
            ha="center", fontsize=11, weight="bold", color="#2b6cb0")
    ax.text(7.5, 5.1, "sentence-transformers/all-MiniLM-L6-v2 → 384-dim vector",
            ha="center", fontsize=10)
    ax.text(7.5, 4.65, "Query (title+summary) and each pool keyword → embeddings (deterministic)",
            ha="center", fontsize=9, style="italic", color="gray")

    ax.annotate("", xy=(7.5, 4.3), xytext=(7.5, 4.4),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Layer 3: Similarity Computation
    box3 = FancyBboxPatch((0.5, 2.8), 14.0, 1.4, boxstyle="round,pad=0.1",
                           facecolor="#f0fff4", edgecolor="#2f855a", linewidth=2)
    ax.add_patch(box3)
    ax.text(7.5, 3.95, "Layer 3 — SIMILARITY COMPUTATION (Cosine)",
            ha="center", fontsize=11, weight="bold", color="#2f855a")
    ax.text(7.5, 3.50, r"sim(q, k) = ⟨E(q), E(k)⟩ / (‖E(q)‖ · ‖E(k)‖)",
            ha="center", fontsize=11, style="italic")
    ax.text(7.5, 3.05, "Pure mathematics — answers Beatriz's 'how is similarity defined' question",
            ha="center", fontsize=9, style="italic", color="gray")

    ax.annotate("", xy=(7.5, 2.7), xytext=(7.5, 2.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Layer 4: Algorithmic Selection
    box4 = FancyBboxPatch((0.5, 0.8), 14.0, 1.8, boxstyle="round,pad=0.1",
                           facecolor="#faf5ff", edgecolor="#6b46c1", linewidth=2)
    ax.add_patch(box4)
    ax.text(7.5, 2.4, "Layer 4 — ALGORITHMIC SELECTION (Hybrid Re-ranker)",
            ha="center", fontsize=11, weight="bold", color="#6b46c1")
    ax.text(7.5, 1.95,
            r"r(k) = 0.6 · sim_norm(q,k)  +  0.3 · RRF₆₀(rank_LLM)_norm  +  0.1 · rule_hits_norm",
            ha="center", fontsize=10.5, style="italic", color="#553c9a")
    ax.text(7.5, 1.45,
            "Embedding 60% (mathematical decision)  ·  LLM 30% (supporting)  ·  Rules 10% (algorithmic helper)",
            ha="center", fontsize=9, color="gray")
    ax.text(7.5, 1.0, "Top-5 by r(k) descending → final keyword set",
            ha="center", fontsize=10, weight="bold")

    fig.tight_layout()
    fp = _OUT / "step_fig01_pipeline_architecture.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


# ─────────────────────────────────────────────────────────────────
# Figure 2: Per-video metrics
# ─────────────────────────────────────────────────────────────────

def fig_per_video_metrics():
    data = json.loads(Path("reports/final_video_evaluation.json").read_text(encoding="utf-8"))
    items = data["items"]

    ids = [_short_id(it["item_id"]) for it in items]
    p5 = [it["precision_at_5"] for it in items]
    r5 = [it["recall_at_5"] for it in items]
    f1 = [it["f1_at_5"] for it in items]
    jac = [it["jaccard"] for it in items]

    x = np.arange(len(ids))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 1.5*w, p5, w, label="Precision@5", color="#2b6cb0", edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5*w, r5, w, label="Recall@5",    color="#38a169", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5*w, f1, w, label="F1@5",        color="#d69e2e", edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5*w, jac, w, label="Jaccard",    color="#e53e3e", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Figure 2 — Per-video Evaluation Metrics on the 6 YouTube Test Set\n"
                 "(System: BERT cosine + LLM RRF + Regex hits, weights 0.6/0.3/0.1)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0.50, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)

    # Annotate each F1 bar
    for i, v in enumerate(f1):
        ax.text(x[i] + 0.5*w, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    fp = _OUT / "step_fig02_per_video_metrics.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ─────────────────────────────────────────────────────────────────
# Figure 3: Method comparison (BERT vs others)
# ─────────────────────────────────────────────────────────────────

def fig_method_comparison():
    data = json.loads(Path("reports/final_comparison.json").read_text(encoding="utf-8"))
    combos = data["combos"]

    # Pick representative methods
    selected = []
    for c in combos:
        label = f"{c['model']}\n({c['metric']})"
        selected.append({
            "label": label,
            "p5": c["aggregate"]["precision_at_5_mean"],
            "r5": c["aggregate"]["recall_at_5_mean"],
            "f1": c["aggregate"]["f1_at_5_mean"],
            "jac": c["aggregate"]["jaccard_mean"],
        })

    # Read final hybrid result
    final_data = json.loads(Path("reports/final_video_evaluation.json").read_text(encoding="utf-8"))
    fa = final_data["aggregate"]
    selected.append({
        "label": "STEP Hybrid\n(ours)",
        "p5": fa["precision_at_5_mean"],
        "r5": fa["recall_at_5_mean"],
        "f1": fa["f1_at_5_mean"],
        "jac": fa["jaccard_mean"],
    })

    fig, ax = plt.subplots(figsize=(13, 5.5))
    n = len(selected)
    x = np.arange(n)
    w = 0.2
    p5 = [s["p5"] for s in selected]
    r5 = [s["r5"] for s in selected]
    f1 = [s["f1"] for s in selected]
    jac = [s["jac"] for s in selected]
    labels = [s["label"] for s in selected]

    ax.bar(x - 1.5*w, p5, w, label="Precision@5", color="#2b6cb0", edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5*w, r5, w, label="Recall@5",    color="#38a169", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5*w, f1, w, label="F1@5",        color="#d69e2e", edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5*w, jac, w, label="Jaccard",    color="#e53e3e", edgecolor="black", linewidth=0.4)

    ax.axvspan(n - 1.5, n - 0.5, alpha=0.08, color="green")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Figure 3 — Embedding Method Comparison on the 6-video Test Set\n"
                 "(STEP Hybrid combines BERT cosine, LLM ranking, and regex rules)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fp = _OUT / "step_fig03_method_comparison.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ─────────────────────────────────────────────────────────────────
# Figure 4: Signal decomposition (how each signal contributes)
# ─────────────────────────────────────────────────────────────────

def fig_signal_decomposition():
    data = json.loads(Path("reports/final_video_evaluation.json").read_text(encoding="utf-8"))
    items = data["items"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for idx, it in enumerate(items):
        ax = axes_flat[idx]
        hr = (it.get("predicted") or {}).get("hybrid_ranking") or []
        top5 = hr[:5]
        if not top5:
            ax.set_visible(False)
            continue

        kws = [r["keyword"] if len(r["keyword"]) <= 22 else r["keyword"][:20] + "…" for r in top5]
        cos_w = [r["components"].get("embedding", r["components"].get("cosine", 0)) * 0.6 for r in top5]
        llm_w = [r["components"]["rrf_llm"] * 0.3 for r in top5]
        rule_w = [r["components"]["rule"] * 0.1 for r in top5]
        total = [r["score"] for r in top5]

        gold_set = set(k.lower() for k in (it.get("gold") or {}).get("gold_keywords") or [])
        y = np.arange(len(kws))
        ax.barh(y, cos_w, color="#2b6cb0", label="Embedding (60%)" if idx == 0 else "")
        ax.barh(y, llm_w, left=cos_w, color="#38a169", label="LLM rank (30%)" if idx == 0 else "")
        lefts = [c + l for c, l in zip(cos_w, llm_w)]
        ax.barh(y, rule_w, left=lefts, color="#d69e2e", label="Regex rule (10%)" if idx == 0 else "")

        for i, r in enumerate(top5):
            is_gold = r["keyword"].lower() in gold_set
            marker = " ★" if is_gold else ""
            ax.text(total[i] + 0.005, i, f"{total[i]:.2f}{marker}", va="center", fontsize=8)

        ax.set_yticks(y)
        ax.set_yticklabels(kws, fontsize=8.5)
        ax.set_xlim(0, max(total) * 1.2 if total else 1.0)
        ax.set_title(_short_id(it["item_id"]), fontsize=10, weight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("r(k) score", fontsize=8)

    handles = [
        mpatches.Patch(color="#2b6cb0", label="Embedding (60%)"),
        mpatches.Patch(color="#38a169", label="LLM rank (30%)"),
        mpatches.Patch(color="#d69e2e", label="Regex rule (10%)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="k",
                   markersize=10, label="★ gold keyword"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 4 — Signal Decomposition for Top-5 Keywords (per video)\n"
                 "r(k) = 0.6·sim(q,k) + 0.3·RRF(rank_LLM) + 0.1·rule_hits",
                 fontsize=12, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fp = _OUT / "step_fig04_signal_decomposition.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


# ─────────────────────────────────────────────────────────────────
# Figure 5: Selection modes (ablation)
# ─────────────────────────────────────────────────────────────────

def fig_selection_modes():
    data = json.loads(Path("reports/selection_modes_comparison.json").read_text(encoding="utf-8"))

    labels = [m["label"].split(":")[0] for m in data]
    short_labels = ["Pure Cosine\n(Beatriz's proposal)",
                    "Algorithmic\n(cosine + rules, no LLM)",
                    "Full Hybrid\n(cosine + LLM + rules)"]
    if len(data) >= 3:
        labels = short_labels[:len(data)]

    p5 = [m["aggregate"]["precision_at_5_mean"] for m in data]
    r5 = [m["aggregate"]["recall_at_5_mean"] for m in data]
    f1 = [m["aggregate"]["f1_at_5_mean"] for m in data]
    jac = [m["aggregate"]["jaccard_mean"] for m in data]

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 1.5*w, p5, w, label="Precision@5", color="#2b6cb0", edgecolor="black", linewidth=0.4)
    ax.bar(x - 0.5*w, r5, w, label="Recall@5", color="#38a169", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5*w, f1, w, label="F1@5", color="#d69e2e", edgecolor="black", linewidth=0.4)
    ax.bar(x + 1.5*w, jac, w, label="Jaccard", color="#e53e3e", edgecolor="black", linewidth=0.4)

    for i, v in enumerate(f1):
        ax.text(x[i] + 0.5*w, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Figure 5 — Selection Mode Ablation on the 6-video Test Set\n"
                 "(All modes share the VLM understanding step; they differ only in selection)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fp = _OUT / "step_fig05_selection_modes.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ─────────────────────────────────────────────────────────────────
# Figure 6: Aggregate summary (single panel)
# ─────────────────────────────────────────────────────────────────

def fig_aggregate_summary():
    data = json.loads(Path("reports/final_video_evaluation.json").read_text(encoding="utf-8"))
    agg = data["aggregate"]

    metrics = ["Precision@5", "Recall@5", "F1@5", "Jaccard", "Topic accuracy", "Subtopic accuracy"]
    values = [
        agg["precision_at_5_mean"], agg["recall_at_5_mean"],
        agg["f1_at_5_mean"], agg["jaccard_mean"],
        agg["topic_accuracy"], agg["subtopic_accuracy"],
    ]
    colors = ["#2b6cb0", "#38a169", "#d69e2e", "#e53e3e", "#805ad5", "#dd6b20"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Figure 6 — Aggregate Performance on the 6-video Test Set\n"
                 "(STEP Hybrid: BERT cosine + LLM RRF + regex rules)",
                 fontsize=11, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fp = _OUT / "step_fig06_aggregate_summary.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


# ─────────────────────────────────────────────────────────────────

def main():
    print("Generating final figure set...\n")
    fp1 = fig_pipeline_architecture()
    print(f"  [1] {fp1}")
    fp2 = fig_per_video_metrics()
    print(f"  [2] {fp2}")
    fp3 = fig_method_comparison()
    print(f"  [3] {fp3}")
    fp4 = fig_signal_decomposition()
    print(f"  [4] {fp4}")
    fp5 = fig_selection_modes()
    print(f"  [5] {fp5}")
    fp6 = fig_aggregate_summary()
    print(f"  [6] {fp6}")
    print(f"\nDone — 6 figures in {_OUT}/")


if __name__ == "__main__":
    main()
