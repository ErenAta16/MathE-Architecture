import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..");
const reportPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "reports", "summary_context_ablation_15videos.json");
const outPath = process.argv[3]
  ? path.resolve(process.argv[3])
  : path.join(root, "reports", "summary_context_ablation_15videos.xlsx");

const report = JSON.parse(await fs.readFile(reportPath, "utf8"));

const modeLabels = {
  current: "Current",
  scene_only: "Title/Scene only",
  summary_late_fusion: "Summary late fusion",
  summary_shortlist: "Summary shortlist",
  summary_shortlist_fusion: "Shortlist + fusion",
};
const modelLabels = {
  bert: "BERT",
  word2vec: "Word2Vec",
  glove: "GloVe",
  word2vec_glove: "Word2Vec + GloVe",
};

function num(x) {
  return x === null || x === undefined || x === "" ? null : Number(x);
}

function top5Text(item) {
  return (item.top_keywords || []).join(", ");
}

function scoredTop5(item) {
  return (item.ranking || [])
    .slice(0, 5)
    .map((row, i) => `${i + 1}. ${row.keyword} (${Number(row.score || 0).toFixed(3)})`)
    .join("; ");
}

function aggRows() {
  return (report.combos || []).map((combo) => {
    const agg = combo.aggregate || {};
    return [
      modelLabels[combo.model] || combo.model,
      modeLabels[combo.mode] || combo.mode,
      num(agg.precision_at_5_mean),
      num(agg.recall_at_5_mean),
      num(agg.f1_at_5_mean),
      num(agg.jaccard_at_5_mean),
      num(agg.ndcg_at_5_mean),
      num(agg.mean_relevance_at_5),
      num(agg.weighted_precision_at_5),
      num(agg.mrr_relevance_ge_2),
    ];
  });
}

function bestRows() {
  const byModel = new Map();
  for (const combo of report.combos || []) {
    const model = combo.model;
    const agg = combo.aggregate || {};
    const score = Number(agg.ndcg_at_5_mean || 0);
    if (!byModel.has(model) || score > byModel.get(model).score) {
      byModel.set(model, { combo, score });
    }
  }
  return [...byModel.values()].map(({ combo }) => {
    const agg = combo.aggregate || {};
    return [
      modelLabels[combo.model] || combo.model,
      modeLabels[combo.mode] || combo.mode,
      num(agg.ndcg_at_5_mean),
      num(agg.weighted_precision_at_5),
      num(agg.precision_at_5_mean),
      num(agg.f1_at_5_mean),
    ];
  });
}

function longRows() {
  const rows = [];
  for (const combo of report.combos || []) {
    for (const item of combo.items || []) {
      const gm = item.graded_metrics || {};
      rows.push([
        item.item_id,
        item.title || "",
        item.topic || "",
        item.subtopic || "",
        modelLabels[combo.model] || combo.model,
        modeLabels[combo.mode] || combo.mode,
        item.query_source || "",
        item.pool_scope || "",
        num(item.candidate_pool_size),
        top5Text(item),
        scoredTop5(item),
        (item.gold_keywords || []).join(", "),
        (item.vlm_order || []).join(", "),
        num(gm.ndcg_at_5),
        num(gm.weighted_precision_at_5),
      ]);
    }
  }
  return rows;
}

function separateRows() {
  const firstCombo = (report.combos || [])[0] || {};
  const baseItems = firstCombo.items || [];
  const byCombo = new Map();
  for (const combo of report.combos || []) {
    const key = `${combo.model}__${combo.mode}`;
    byCombo.set(key, new Map((combo.items || []).map((item) => [item.item_id, item])));
  }
  const comboKeys = [];
  for (const model of report.models || []) {
    for (const mode of report.modes || []) {
      comboKeys.push([model, mode, `${model}__${mode}`]);
    }
  }
  const headers = ["Video ID", "Title", "Topic", "Subtopic"];
  for (const [model, mode] of comboKeys) {
    headers.push(`${modelLabels[model] || model} | ${modeLabels[mode] || mode}`);
  }
  const rows = [headers];
  for (const item of baseItems) {
    const row = [item.item_id, item.title || "", item.topic || "", item.subtopic || ""];
    for (const [, , key] of comboKeys) {
      row.push(top5Text((byCombo.get(key) || new Map()).get(item.item_id) || {}));
    }
    rows.push(row);
  }
  return rows;
}

function modelRows(model) {
  const combos = new Map();
  for (const combo of report.combos || []) {
    if (combo.model === model) {
      combos.set(combo.mode, new Map((combo.items || []).map((item) => [item.item_id, item])));
    }
  }
  const firstMode = (report.modes || [])[0];
  const baseItems = [...(combos.get(firstMode) || new Map()).values()];
  const headers = ["Video ID", "Title", "Topic", "Subtopic"];
  for (const mode of report.modes || []) {
    headers.push(modeLabels[mode] || mode);
  }
  const rows = [headers];
  for (const item of baseItems) {
    const row = [item.item_id, item.title || "", item.topic || "", item.subtopic || ""];
    for (const mode of report.modes || []) {
      row.push(top5Text((combos.get(mode) || new Map()).get(item.item_id) || {}));
    }
    rows.push(row);
  }
  return rows;
}

function setValues(sheet, startRow, startCol, values) {
  if (!values.length || !values[0].length) return;
  sheet.getRangeByIndexes(startRow, startCol, values.length, values[0].length).values = values;
}

function styleHeader(sheet, row, colCount) {
  const r = sheet.getRangeByIndexes(row, 0, 1, colCount);
  r.format.fill = { color: "#D9EAF7" };
  r.format.font = { color: "#1F4E78", bold: true };
  r.format.wrapText = true;
}

function styleTable(sheet, rows, cols) {
  const r = sheet.getRangeByIndexes(0, 0, rows, cols);
  r.format.font = { name: "Aptos", size: 10 };
  r.format.wrapText = true;
  r.format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
}

function finishSheet(sheet, widths = []) {
  for (let i = 0; i < widths.length; i++) {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidth = widths[i];
  }
}

const workbook = Workbook.create();

const summary = workbook.worksheets.add("Summary");
const summaryValues = [
  ["Summary Context Ablation - 15 Videos", "", "", "", "", ""],
  ["Purpose", "Compare keyword-selection strategies that use the VLM summary as context.", "", "", "", ""],
  ["Input", "Same 15 videos, same closed keyword pool, fixed VLM title/summary.", "", "", "", ""],
  ["Metric", report.metric || "cosine", "", "", "", ""],
  ["Rerank weights", `Embedding ${report.rerank_weights?.embedding ?? ""} / VLM rank ${report.rerank_weights?.vlm_rank ?? ""}`, "", "", "", ""],
  ["Summary fusion", `Primary ${report.summary_fusion_weights?.primary ?? ""} / Summary ${report.summary_fusion_weights?.summary ?? ""}`, "", "", "", ""],
  [],
  ["Model", "Best strategy by nDCG@5", "nDCG@5", "Weighted P@5", "P@5", "F1@5"],
  ...bestRows(),
  [],
  ["Model", "Mode", "P@5", "R@5", "F1@5", "Jaccard", "nDCG@5", "Mean relevance@5", "Weighted P@5", "MRR"],
  ...aggRows(),
];
setValues(summary, 0, 0, summaryValues.map((row) => {
  const width = Math.max(...summaryValues.map((r) => r.length));
  return [...row, ...Array(width - row.length).fill("")];
}));
summary.getRange("A1:F1").merge();
summary.getRange("A1").format.font = { bold: true, size: 16, color: "#1F4E78" };
styleHeader(summary, 7, 6);
styleHeader(summary, 7 + bestRows().length + 2, 10);
styleTable(summary, summaryValues.length, Math.max(...summaryValues.map((r) => r.length)));
finishSheet(summary, [24, 34, 12, 14, 12, 12, 12, 18, 16, 12]);
summary.getRange("C9:F12").format.numberFormat = [["0.000"], ["0.000"], ["0.000"], ["0.000"]];

const top5 = workbook.worksheets.add("Top5 Matrix");
const sep = separateRows();
setValues(top5, 0, 0, sep);
styleHeader(top5, 0, sep[0].length);
styleTable(top5, sep.length, sep[0].length);
finishSheet(top5, [14, 34, 18, 22, ...Array(sep[0].length - 4).fill(34)]);

for (const model of report.models || []) {
  const sheetName = `${modelLabels[model] || model} Top5`.slice(0, 31);
  const sheet = workbook.worksheets.add(sheetName);
  const rows = modelRows(model);
  setValues(sheet, 0, 0, rows);
  styleHeader(sheet, 0, rows[0].length);
  styleTable(sheet, rows.length, rows[0].length);
  finishSheet(sheet, [14, 34, 18, 22, ...Array(rows[0].length - 4).fill(42)]);
}

const long = workbook.worksheets.add("Per Video Details");
const longHeaders = [
  "Video ID", "Title", "Topic", "Subtopic", "Model", "Mode", "Query source",
  "Pool scope", "Candidate pool size", "Top 5", "Top 5 with scores",
  "Reference keywords", "VLM keyword order", "nDCG@5", "Weighted P@5",
];
const longValues = [longHeaders, ...longRows()];
setValues(long, 0, 0, longValues);
styleHeader(long, 0, longHeaders.length);
styleTable(long, longValues.length, longHeaders.length);
finishSheet(long, [14, 34, 18, 22, 16, 24, 24, 22, 12, 42, 58, 48, 48, 12, 14]);

const method = workbook.worksheets.add("Methodology");
const methodValues = [
  ["Field", "Value"],
  ["current", "Existing global query strategy: title + summary, then embedding/VLM-rank hybrid."],
  ["scene_only", "No summary context in this global harness: title-only query, then hybrid rerank."],
  ["summary_late_fusion", "Title score and summary score are computed separately and fused before VLM-rank hybrid."],
  ["summary_shortlist", "Summary first narrows the 100+ keyword pool; title+summary then scores candidates."],
  ["summary_shortlist_fusion", "Summary narrows candidates, then title and summary scores are fused."],
  ["BERT", "Transformers all-MiniLM-L6-v2 from local HuggingFace cache."],
  ["Word2Vec", "Google News 300d vectors from local gensim-data file."],
  ["GloVe", "glove-wiki-gigaword-300 vectors from local gensim-data file."],
  ["Reference", "Graded VLM reference, not independent human gold."],
];
setValues(method, 0, 0, methodValues);
styleHeader(method, 0, 2);
styleTable(method, methodValues.length, 2);
finishSheet(method, [26, 100]);

await fs.mkdir(path.dirname(outPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outPath);
console.log(`[OK] XLSX: ${outPath}`);
