import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.resolve("reports", "summary_context_ablation_15videos.xlsx");
const previewDir = path.resolve("reports", "summary_context_ablation_preview");

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheetInfo = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 4000,
});
console.log(sheetInfo.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
  maxChars: 2000,
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["Summary", "Top5 Matrix", "BERT Top5", "Word2Vec Top5", "GloVe Top5", "Word2Vec + GloVe Top5", "Per Video Details", "Methodology"]) {
  const blob = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`), bytes);
}
console.log(`[OK] previews: ${previewDir}`);
