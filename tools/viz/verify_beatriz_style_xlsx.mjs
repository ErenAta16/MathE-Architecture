import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [inputPath, outputDir = "reports/beatriz_style_summary_context_top5_preview"] = process.argv.slice(2);

if (!inputPath) {
  console.error("Usage: node verify_beatriz_style_xlsx.mjs <input.xlsx> [preview_dir]");
  process.exit(2);
}

await fs.mkdir(outputDir, { recursive: true });

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name,address",
  maxChars: 5000,
});
console.log(sheets.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
  maxChars: 5000,
});
console.log(errors.ndjson);

const names = [];
for (const line of sheets.ndjson.trim().split(/\n+/)) {
  if (!line) continue;
  const row = JSON.parse(line);
  if (row.kind === "sheet") names.push(row.name);
}

for (const sheetName of names) {
  const safe = sheetName.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "") || "sheet";
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/${safe}.png`, new Uint8Array(await preview.arrayBuffer()));
}

console.log(`[OK] verified ${inputPath}`);
