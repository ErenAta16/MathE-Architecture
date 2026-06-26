import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [reportPath = "reports/summary_context_ablation_15videos.json", outputPath = "reports/beatriz_style_summary_context_top5_15videos.xlsx"] = process.argv.slice(2);

const SELECTED_MODE = "summary_shortlist_fusion";
const METHODS = [
  { model: "bert", key: "bert_vlm", label: "BERT + VLM" },
  { model: "glove", key: "glove_vlm", label: "GloVe + VLM" },
  { model: "word2vec", key: "word2vec_vlm", label: "Word2Vec + VLM" },
  { model: "word2vec_glove", key: "word2vec_glove_vlm", label: "Word2Vec + GloVe + VLM" },
];

function columnName(n) {
  let out = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function a1(row, col) {
  return `${columnName(col)}${row}`;
}

function rangeA1(row1, col1, row2, col2) {
  return `${a1(row1, col1)}:${a1(row2, col2)}`;
}

function fmtScore(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toFixed(3);
}

function topRanking(item, k = 5) {
  const ranking = Array.isArray(item?.ranking) ? item.ranking : [];
  const top = ranking.slice(0, k).map((row) => ({
    keyword: String(row.keyword ?? "").trim(),
    score: Number(row.score),
  }));
  while (top.length < k) top.push({ keyword: "", score: NaN });
  return top;
}

function keywordsOnly(top) {
  return top
    .map((x, i) => (x.keyword ? `${i + 1}. ${x.keyword}` : ""))
    .filter(Boolean)
    .join("\n");
}

function keywordsWithScores(top) {
  return top
    .filter((x) => x.keyword)
    .map((x, i) => `${i + 1}. ${x.keyword}${Number.isFinite(x.score) ? ` (${fmtScore(x.score)})` : ""}`)
    .join("; ");
}

function makeComboIndex(report) {
  const index = new Map();
  for (const combo of report.combos || []) {
    const key = `${combo.model}::${combo.mode}`;
    const items = new Map();
    for (const item of combo.items || []) items.set(String(item.item_id || ""), item);
    index.set(key, { combo, items });
  }
  return index;
}

function getSelected(index, model) {
  const found = index.get(`${model}::${SELECTED_MODE}`);
  if (!found) throw new Error(`Missing combo for ${model} / ${SELECTED_MODE}`);
  return found;
}

function collectRows(report) {
  const index = makeComboIndex(report);
  const primary = getSelected(index, METHODS[0].model);
  const ids = [...primary.items.keys()];
  const selected = Object.fromEntries(METHODS.map((m) => [m.key, getSelected(index, m.model)]));

  return ids.map((id, idx) => {
    const base = primary.items.get(id) || {};
    const row = {
      video_no: idx + 1,
      video_id: id,
      url: base.url || "",
      title: base.title || "",
      topic: base.topic || "",
      subtopic: base.subtopic || "",
    };
    for (const method of METHODS) {
      const item = selected[method.key].items.get(id) || {};
      const top = topRanking(item);
      row[`${method.key}_top5`] = keywordsOnly(top);
      row[`${method.key}_top5_with_scores`] = keywordsWithScores(top);
      row[`${method.key}_meta`] = { item, top };
    }
    return row;
  });
}

function aggregateRows(report) {
  const index = makeComboIndex(report);
  return METHODS.map((method) => {
    const combo = getSelected(index, method.model).combo;
    const agg = combo.aggregate || {};
    return [
      method.label,
      method.model,
      SELECTED_MODE,
      agg.precision_at_5_mean ?? "",
      agg.recall_at_5_mean ?? "",
      agg.f1_at_5_mean ?? "",
      agg.ndcg_at_5_mean ?? "",
      agg.weighted_precision_at_5 ?? "",
      agg.jaccard_mean ?? "",
    ];
  });
}

function writeMatrix(sheet, startRow, startCol, rows) {
  if (!rows.length || !rows[0].length) return;
  sheet.getRange(rangeA1(startRow, startCol, startRow + rows.length - 1, startCol + rows[0].length - 1)).values = rows;
}

function styleBlock(sheet, rowCount, colCount) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(rangeA1(1, 1, 1, colCount)).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(rangeA1(1, 1, rowCount, colCount)).format.borders = {
    preset: "all",
    style: "thin",
    color: "#D9E2F3",
  };
  sheet.getRange(rangeA1(2, 1, rowCount, colCount)).format = {
    wrapText: true,
    verticalAlignment: "top",
  };
  sheet.getRange(rangeA1(1, 1, rowCount, colCount)).format.autofitColumns();
  sheet.getRange(rangeA1(1, 1, rowCount, colCount)).format.autofitRows();
}

function addTable(sheet, rowCount, colCount, name) {
  try {
    const table = sheet.tables.add(rangeA1(1, 1, rowCount, colCount), true, name);
    table.style = "TableStyleMedium2";
    table.showFilterButton = true;
  } catch {
    // Formatting still leaves the sheet readable if table creation is unavailable.
  }
}

function capColumns(sheet, widths) {
  for (const [col, width] of Object.entries(widths)) {
    sheet.getRange(`${col}:${col}`).format.columnWidth = width;
  }
}

function setDataRowHeight(sheet, firstRow, lastRow, height) {
  if (lastRow >= firstRow) {
    sheet.getRange(rangeA1(firstRow, 1, lastRow, 1)).format.rowHeight = height;
  }
}

const report = JSON.parse(await fs.readFile(reportPath, "utf8"));
const rows = collectRows(report);
const workbook = Workbook.create();

const compactHeaders = [
  "video_no",
  "video_id",
  "url",
  "title",
  "topic",
  "subtopic",
  ...METHODS.map((m) => `${m.key}_top5`),
];
const compactRows = [
  compactHeaders,
  ...rows.map((row) => compactHeaders.map((h) => row[h] ?? "")),
];
const compact = workbook.worksheets.add("Top5 Compact");
writeMatrix(compact, 1, 1, compactRows);
styleBlock(compact, compactRows.length, compactHeaders.length);
addTable(compact, compactRows.length, compactHeaders.length, "Top5Compact");
capColumns(compact, { A: 10, B: 16, C: 44, D: 36, G: 48, H: 48, I: 48, J: 54 });
setDataRowHeight(compact, 2, compactRows.length, 88);

const separateHeaders = ["video_no", "video_id", "url", "title", "topic", "subtopic"];
for (const method of METHODS) {
  for (let i = 1; i <= 5; i += 1) separateHeaders.push(`${method.key}_keyword_${i}`);
}
const separateRows = [
  separateHeaders,
  ...rows.map((row) => {
    const out = [row.video_no, row.video_id, row.url, row.title, row.topic, row.subtopic];
    for (const method of METHODS) {
      for (const item of row[`${method.key}_meta`].top) out.push(item.keyword);
    }
    return out;
  }),
];
const separate = workbook.worksheets.add("Top5 Separate Columns");
writeMatrix(separate, 1, 1, separateRows);
styleBlock(separate, separateRows.length, separateHeaders.length);
addTable(separate, separateRows.length, separateHeaders.length, "Top5Separate");
capColumns(separate, { A: 10, B: 16, C: 44, D: 36 });
setDataRowHeight(separate, 2, separateRows.length, 30);

const scoredHeaders = [
  "video_no",
  "video_id",
  "title",
  ...METHODS.map((m) => `${m.key}_top5_with_scores`),
];
const scoredRows = [
  scoredHeaders,
  ...rows.map((row) => scoredHeaders.map((h) => row[h] ?? "")),
];
const scored = workbook.worksheets.add("Top5 With Scores");
writeMatrix(scored, 1, 1, scoredRows);
styleBlock(scored, scoredRows.length, scoredHeaders.length);
addTable(scored, scoredRows.length, scoredHeaders.length, "Top5Scores");
capColumns(scored, { A: 10, B: 16, C: 38, D: 58, E: 58, F: 58, G: 62 });
setDataRowHeight(scored, 2, scoredRows.length, 104);

const methodSheet = workbook.worksheets.add("Methodology");
const notes = [
  ["Field", "Value"],
  ["Purpose", "Top-5 keywords per video, separated by semantic model/method."],
  ["Selected context mode", SELECTED_MODE],
  ["Meaning", "The candidate pool is first narrowed with summary context, then final ranking uses the VLM signal plus the selected semantic model."],
  ["Input", "Same 15-video set and the same closed keyword pool."],
  ["Score", "Hybrid normalized score used by the application ranking pipeline."],
  ["Output format", "One row per video; each method contributes exactly five keywords."],
  ["Included methods", METHODS.map((m) => m.label).join(", ")],
];
writeMatrix(methodSheet, 1, 1, notes);
methodSheet.getRange("A1:B1").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
methodSheet.getRange(`A2:A${notes.length}`).format = { fill: "#EAF2F8", font: { bold: true }, wrapText: true };
methodSheet.getRange(`B2:B${notes.length}`).format = { wrapText: true };
methodSheet.getRange(`A1:B${notes.length}`).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
capColumns(methodSheet, { A: 26, B: 110 });

const metricsStart = notes.length + 3;
const metrics = [
  ["Method", "Model key", "Context mode", "Precision@5", "Recall@5", "F1@5", "NDCG@5", "Weighted P@5", "Jaccard"],
  ...aggregateRows(report),
];
writeMatrix(methodSheet, metricsStart, 1, metrics);
methodSheet.getRange(rangeA1(metricsStart, 1, metricsStart, metrics[0].length)).format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
methodSheet.getRange(rangeA1(metricsStart, 1, metricsStart + metrics.length - 1, metrics[0].length)).format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2F3",
};
methodSheet.getRange(rangeA1(metricsStart + 1, 4, metricsStart + metrics.length - 1, metrics[0].length)).format.numberFormat = "0.000";
methodSheet.getRange(rangeA1(1, 1, metricsStart + metrics.length - 1, metrics[0].length)).format.autofitRows();
methodSheet.freezePanes.freezeRows(1);
methodSheet.showGridLines = false;

const output = await SpreadsheetFile.exportXlsx(workbook);
await fs.mkdir(outputPath.split(/[\\/]/).slice(0, -1).join("/") || ".", { recursive: true });
await output.save(outputPath);
console.log(`[OK] ${outputPath}`);
