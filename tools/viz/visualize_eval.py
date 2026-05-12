"""
Generate evaluation figures from JSON reports produced by eval_handmade.py
and eval_videos.py.

Usage::

    python tools/visualize_eval.py reports/videos_baseline.json
    python tools/visualize_eval.py reports/handmade_baseline_v3_hybrid.json
    python tools/visualize_eval.py reports/videos_baseline.json reports/handmade_baseline_v3_hybrid.json

Outputs PNG figures to ``reports/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_OUT_DIR = Path("reports/figures")


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_dir():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_per_item_bars(report: dict, tag: str) -> Path:
    """Grouped bar chart: P@5, R@5, F1@5, Jaccard per item."""
    items = report.get("items") or []
    ids = [it["item_id"] for it in items]
    p5 = [it["precision_at_5"] for it in items]
    r5 = [it["recall_at_5"] for it in items]
    f1 = [it["f1_at_5"] for it in items]
    jac = [it["jaccard"] for it in items]

    x = np.arange(len(ids))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(7, len(ids) * 1.5), 4.5))
    ax.bar(x - 1.5 * w, p5, w, label="Precision@5", color="#2b6cb0")
    ax.bar(x - 0.5 * w, r5, w, label="Recall@5", color="#38a169")
    ax.bar(x + 0.5 * w, f1, w, label="F1@5", color="#d69e2e")
    ax.bar(x + 1.5 * w, jac, w, label="Jaccard", color="#e53e3e")

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    modality = report.get("modality", "pdf").upper()
    ax.set_title(f"Per-item keyword evaluation ({modality})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _ensure_dir()
    fp = _OUT_DIR / f"per_item_{tag}.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_aggregate_comparison(*reports_with_labels: tuple[dict, str]) -> Path:
    """Side-by-side aggregate bar chart for multiple reports (e.g. PDF vs video)."""
    metrics = ("precision_at_5_mean", "recall_at_5_mean", "f1_at_5_mean",
               "jaccard_mean", "topic_accuracy", "subtopic_accuracy")
    labels_short = ("P@5", "R@5", "F1@5", "Jacc", "Topic", "Subtopic")

    n = len(reports_with_labels)
    x = np.arange(len(metrics))
    w = 0.8 / max(n, 1)
    colors = ["#2b6cb0", "#38a169", "#d69e2e", "#e53e3e", "#805ad5", "#dd6b20"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (rep, lbl) in enumerate(reports_with_labels):
        agg = rep.get("aggregate") or {}
        vals = [agg.get(m, 0) for m in metrics]
        offset = (i - (n - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=lbl, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Aggregate evaluation metrics")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _ensure_dir()
    tag = "_vs_".join(lbl.replace(" ", "_") for _, lbl in reports_with_labels)
    fp = _OUT_DIR / f"aggregate_{tag}.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def plot_signal_stack(report: dict, tag: str) -> Path:
    """Stacked bar showing cosine / rrf_llm / rule contributions per item (top-1 keyword).

    Only available when hybrid_ranking is present in the report.
    """
    items = report.get("items") or []
    ids: list[str] = []
    cos_vals: list[float] = []
    llm_vals: list[float] = []
    rule_vals: list[float] = []

    for it in items:
        hr = (it.get("predicted") or {}).get("hybrid_ranking")
        if not hr or not isinstance(hr, list) or not hr:
            continue
        top1 = hr[0]
        comp = top1.get("components") or {}
        ids.append(it["item_id"])
        cos_vals.append(comp.get("cosine", 0))
        llm_vals.append(comp.get("rrf_llm", 0))
        rule_vals.append(comp.get("rule", 0))

    if not ids:
        return Path("")

    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(max(7, len(ids) * 1.3), 4))
    ax.bar(x, cos_vals, label="Cosine", color="#2b6cb0")
    ax.bar(x, llm_vals, bottom=cos_vals, label="RRF LLM", color="#38a169")
    bottoms = [c + l for c, l in zip(cos_vals, llm_vals)]
    ax.bar(x, rule_vals, bottom=bottoms, label="Rule hits", color="#d69e2e")

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Normalized signal contribution")
    modality = report.get("modality", "pdf").upper()
    ax.set_title(f"Signal decomposition — top-1 keyword ({modality})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _ensure_dir()
    fp = _OUT_DIR / f"signal_stack_{tag}.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    return fp


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/visualize_eval.py report1.json [report2.json ...]")
        return 1

    reports = [(json.loads(Path(p).read_text(encoding="utf-8")), Path(p).stem) for p in sys.argv[1:]]
    generated: list[Path] = []

    for rep, tag in reports:
        fp = plot_per_item_bars(rep, tag)
        print(f"  {fp}")
        generated.append(fp)
        sp = plot_signal_stack(rep, tag)
        if sp and sp != Path(""):
            print(f"  {sp}")
            generated.append(sp)

    if len(reports) >= 2:
        fp = plot_aggregate_comparison(*[(r, t) for r, t in reports])
        print(f"  {fp}")
        generated.append(fp)

    print(f"\nGenerated {len(generated)} figure(s) in {_OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
