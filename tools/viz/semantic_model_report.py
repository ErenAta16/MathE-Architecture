"""
Create table and chart artifacts from tools/eval/videos_multi.py reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent


def _fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""


def _summary_rows(report: dict) -> list[dict]:
    rows = []
    for combo in report.get("combos") or []:
        agg = combo.get("aggregate") or {}
        cov = combo.get("coverage_summary") or {}
        rows.append({
            "model": combo.get("model"),
            "metric": combo.get("metric"),
            "precision_at_5": agg.get("precision_at_5_mean"),
            "recall_at_5": agg.get("recall_at_5_mean"),
            "f1_at_5": agg.get("f1_at_5_mean"),
            "jaccard": agg.get("jaccard_mean"),
            "ndcg_at_5": agg.get("ndcg_at_5_mean"),
            "weighted_precision_at_5": agg.get("weighted_precision_at_5"),
            "mean_relevance_at_5": agg.get("mean_relevance_at_5"),
            "mrr_relevance_ge_2": agg.get("mrr_relevance_ge_2"),
            "n_graded": agg.get("n_graded"),
            "elapsed_s": combo.get("elapsed_s"),
            "query_coverage_mean": cov.get("query_coverage_mean"),
            "keyword_coverage_mean": cov.get("keyword_coverage_mean"),
            "n_zero_vector_keywords": cov.get("n_zero_vector_keywords"),
        })
    return rows


def _write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Model | Metric | P@5 | R@5 | F1@5 | Jaccard | nDCG@5 | W-P@5 | Mean Rel. | MRR | Query Cov. | Keyword Cov. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['metric']} | {_fmt(r['precision_at_5'])} | "
            f"{_fmt(r['recall_at_5'])} | {_fmt(r['f1_at_5'])} | {_fmt(r['jaccard'])} | "
            f"{_fmt(r['ndcg_at_5'])} | {_fmt(r['weighted_precision_at_5'])} | "
            f"{_fmt(r['mean_relevance_at_5'])} | {_fmt(r['mrr_relevance_ge_2'])} | "
            f"{_fmt(r['query_coverage_mean'])} | {_fmt(r['keyword_coverage_mean'])} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_chart(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{r['model']}\n{r['metric']}" for r in rows]
    if any(r.get("ndcg_at_5") is not None for r in rows):
        metrics = [
            ("P@5", "precision_at_5"),
            ("F1@5", "f1_at_5"),
            ("nDCG@5", "ndcg_at_5"),
            ("W-P@5", "weighted_precision_at_5"),
        ]
    else:
        metrics = [
            ("P@5", "precision_at_5"),
            ("R@5", "recall_at_5"),
            ("F1@5", "f1_at_5"),
            ("Jaccard", "jaccard"),
        ]
    x = np.arange(len(labels))
    width = 0.18
    colors = ["#355C7D", "#2A9D8F", "#E9C46A", "#E76F51"]

    fig_w = max(9, len(labels) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))
    for i, (name, key) in enumerate(metrics):
        vals = [float(r.get(key) or 0.0) for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=colors[i])

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean score")
    ax.set_title("Semantic Model Comparison on Fixed Video Set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="JSON report from tools/eval/videos_multi.py")
    parser.add_argument("--out-dir", default="reports/figures")
    parser.add_argument("--prefix", default="semantic_model_comparison")
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = (_ROOT / report_path).resolve()
    if not report_path.is_file():
        print(f"Report not found: {report_path}", file=sys.stderr)
        return 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _summary_rows(report)
    if not rows:
        print("No combos found in report.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()
    csv_path = out_dir / f"{args.prefix}_summary.csv"
    md_path = out_dir / f"{args.prefix}_summary.md"
    png_path = out_dir / f"{args.prefix}_bars.png"

    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path)
    _write_chart(rows, png_path)

    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Chart: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
