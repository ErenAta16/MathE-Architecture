import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [inputPath, outputDir] = process.argv.slice(2);

if (!inputPath || !outputDir) {
  console.error("Usage: node inspect_reference_workbook.mjs <input.xlsx> <output_dir>");
  process.exit(2);
}

await fs.mkdir(outputDir, { recursive: true });

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 8,
  tableMaxCols: 12,
  tableMaxCellChars: 120,
});
await fs.writeFile(`${outputDir}/reference_inspect.ndjson`, summary.ndjson, "utf8");
console.log(summary.ndjson);

const sheets = JSON.parse(`[${summary.ndjson.trim().split(/\n+/).filter(Boolean).join(",")}]`)
  .filter((row) => row.kind === "sheet")
  .map((row) => row.name);

for (const sheetName of sheets) {
  const safe = sheetName.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "") || "sheet";
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/${safe}.png`, new Uint8Array(await preview.arrayBuffer()));
}
