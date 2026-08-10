import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath, previewPath] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error(
    "Usage: node hbvcam_rt_excel_export.mjs <results.json> <output.xlsx> [preview.png]",
  );
}

const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const rows = Array.isArray(payload.rows) ? payload.rows : [];
const frameSelection =
  payload.frame_selection ??
  `Fixed zero-based frame index: ${payload.reference_frame_index ?? ""}`;
const extrinsicUsage = payload.frame_selection
  ? "JSON R/T are not fed into any per-frame RT estimation. The JSON baseline is used only to select the winning frame after all frames are solved."
  : "JSON R/T are ground truth only and are not used by RT estimation.";
const workbook = Workbook.create();
const results = workbook.worksheets.add("RT Results");
const summary = workbook.worksheets.add("Summary");
const protocol = workbook.worksheets.add("Protocol");

results.showGridLines = false;
summary.showGridLines = false;
protocol.showGridLines = false;

const headers = [
  "Run #",
  "Video File",
  "Video Path",
  "Status",
  "Failure Reason",
  "Frame Index",
  "Total Frames",
  "Rotation Error (deg)",
  "Translation L2 Error (mm)",
  "Translation Direction Error (deg)",
  "Algorithm Baseline (mm)",
  "JSON Baseline (mm)",
  "Baseline Delta (mm)",
  "Baseline Error (%)",
  "RT Internal Error (px)",
  "Marker Bidirectional RMS (px)",
  "Feature Median (px)",
  "RT Reliable",
  "Processing Time (s)",
  "Absolute Baseline Delta (mm)",
];

results.getRange("A1:T1").values = [headers];
if (rows.length > 0) {
  const values = rows.map((row) => [
    row.run_number,
    row.video_file,
    row.video_path,
    row.status,
    row.failure_reason,
    row.frame_index,
    row.total_frames,
    row.rotation_error_deg,
    row.translation_l2_error_mm,
    row.translation_direction_error_deg,
    row.algorithm_baseline_mm,
    row.json_baseline_mm,
    row.baseline_delta_mm,
    null,
    row.rt_internal_error_px,
    row.marker_bidirectional_rms_px,
    row.feature_median_px,
    row.rt_reliable,
    row.processing_time_s,
    null,
  ]);
  results.getRange(`A2:T${rows.length + 1}`).values = values;

  results.getRange("N2").formulas = [["=IFERROR(ABS(M2)/L2,\"\")"]];
  results.getRange("T2").formulas = [["=IF(M2=\"\",\"\",ABS(M2))"]];
  if (rows.length > 1) {
    results.getRange(`N2:N${rows.length + 1}`).fillDown();
    results.getRange(`T2:T${rows.length + 1}`).fillDown();
  }
}

const lastRow = Math.max(rows.length + 1, 2);
const resultsTable = results.tables.add(
  `A1:T${lastRow}`,
  true,
  "HBVCAMRTResults",
);
resultsTable.style = "TableStyleMedium2";
results.freezePanes.freezeRows(1);

results.getRange("A1:T1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
  wrapText: true,
};
results.getRange("A1:T1").format.rowHeight = 36;
results.getRange(`A2:T${lastRow}`).format.verticalAlignment = "center";
results.getRange(`H2:M${lastRow}`).format.numberFormat = "0.0000";
results.getRange(`N2:N${lastRow}`).format.numberFormat = "0.00%";
results.getRange(`O2:Q${lastRow}`).format.numberFormat = "0.000";
results.getRange(`S2:S${lastRow}`).format.numberFormat = "0.00";
results.getRange(`T2:T${lastRow}`).format.numberFormat = "0.0000";
results.getRange(`A1:T${lastRow}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  bottom: { style: "thin", color: "#9EADBA" },
};

const widths = {
  A: 8,
  B: 28,
  C: 52,
  D: 18,
  E: 44,
  F: 12,
  G: 13,
  H: 20,
  I: 22,
  J: 27,
  K: 22,
  L: 19,
  M: 20,
  N: 19,
  O: 21,
  P: 28,
  Q: 20,
  R: 13,
  S: 19,
  T: 25,
};
for (const [column, width] of Object.entries(widths)) {
  results.getRange(`${column}:${column}`).format.columnWidth = width;
}
results.getRange(`C2:C${lastRow}`).format.wrapText = true;
results.getRange(`E2:E${lastRow}`).format.wrapText = true;

if (rows.length > 0) {
  const statusRange = results.getRange(`D2:D${rows.length + 1}`);
  statusRange.conditionalFormats.add("containsText", {
    text: "FAILED",
    format: { fill: "#FDE9E7", font: { color: "#B91C1C", bold: true } },
  });
  statusRange.conditionalFormats.add("containsText", {
    text: "QUALITY_WARNING",
    format: { fill: "#FFF2CC", font: { color: "#9C6500", bold: true } },
  });
  statusRange.conditionalFormats.add("containsText", {
    text: "OK",
    format: { fill: "#E2F0D9", font: { color: "#375623", bold: true } },
  });
  results.getRange(`N2:N${rows.length + 1}`).conditionalFormats.add(
    "colorScale",
    {
      colors: ["#E2F0D9", "#FFF2CC", "#F4CCCC"],
      thresholds: ["min", "50%", "max"],
    },
  );
}

summary.getRange("A1:B1").values = [["Metric", "Value"]];
summary.getRange("A2:A10").values = [
  ["Total Videos"],
  ["RT Results Produced"],
  ["Quality OK"],
  ["Quality Warnings"],
  ["Failed"],
  ["Mean Rotation Error (deg)"],
  ["Mean Absolute Baseline Delta (mm)"],
  ["Mean Baseline Error (%)"],
  ["Maximum Baseline Error (%)"],
];
summary.getRange("B2:B10").formulas = [
  [`=COUNTA('RT Results'!B2:B${lastRow})`],
  [`=COUNT('RT Results'!K2:K${lastRow})`],
  [`=COUNTIF('RT Results'!D2:D${lastRow},"OK")`],
  [`=COUNTIF('RT Results'!D2:D${lastRow},"QUALITY_WARNING")`],
  [`=COUNTIF('RT Results'!D2:D${lastRow},"FAILED")`],
  [`=IFERROR(AVERAGE('RT Results'!H2:H${lastRow}),"")`],
  [`=IFERROR(AVERAGE('RT Results'!T2:T${lastRow}),"")`],
  [`=IFERROR(AVERAGE('RT Results'!N2:N${lastRow}),"")`],
  [`=IFERROR(MAX('RT Results'!N2:N${lastRow}),"")`],
];
summary.getRange("A1:B1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
summary.getRange("A1:B10").format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2F3",
};
summary.getRange("A2:A10").format.font = { bold: true, color: "#1F2937" };
summary.getRange("B7:B8").format.numberFormat = "0.0000";
summary.getRange("B9:B10").format.numberFormat = "0.00%";
summary.getRange("A:A").format.columnWidth = 38;
summary.getRange("B:B").format.columnWidth = 22;
summary.freezePanes.freezeRows(1);

protocol.getRange("A1:B1").values = [["Parameter", "Value"]];
protocol.getRange("A2:B8").values = [
  ["Generated At", payload.generated_at ?? ""],
  ["Source Folder", payload.source_folder ?? ""],
  ["Calibration File", payload.calibration_file ?? ""],
  ["Frame Selection", frameSelection],
  ["Recursive Search", Boolean(payload.recursive)],
  [
    "Baseline Error Formula",
    "ABS(Algorithm Baseline - JSON Baseline) / JSON Baseline",
  ],
  [
    "Extrinsic Usage",
    extrinsicUsage,
  ],
];
protocol.getRange("A1:B1").format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
};
protocol.getRange("A2:A8").format.font = { bold: true, color: "#1F2937" };
protocol.getRange("A1:B8").format.borders = {
  preset: "all",
  style: "thin",
  color: "#D9E2F3",
};
protocol.getRange("A:A").format.columnWidth = 38;
protocol.getRange("B:B").format.columnWidth = 78;
protocol.getRange("B2:B8").format.wrapText = true;
protocol.freezePanes.freezeRows(1);

const inspect = await workbook.inspect({
  kind: "table",
  range: `RT Results!A1:T${Math.min(lastRow, 12)}`,
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 20,
});
console.log(inspect.ndjson);

const summaryInspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:B10",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 2,
});
console.log(summaryInspect.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

if (previewPath) {
  const resultPreview = await workbook.render({
    sheetName: "RT Results",
    range: `A1:T${Math.min(lastRow, 12)}`,
    scale: 1,
    format: "png",
  });
  const extension = path.extname(previewPath);
  const stem = previewPath.slice(0, -extension.length);
  const summaryPreviewPath = `${stem}_summary${extension}`;
  const protocolPreviewPath = `${stem}_protocol${extension}`;
  const summaryPreview = await workbook.render({
    sheetName: "Summary",
    range: "A1:B10",
    scale: 1.5,
    format: "png",
  });
  const protocolPreview = await workbook.render({
    sheetName: "Protocol",
    range: "A1:B8",
    scale: 1.25,
    format: "png",
  });
  await fs.mkdir(path.dirname(previewPath), { recursive: true });
  await fs.writeFile(
    previewPath,
    new Uint8Array(await resultPreview.arrayBuffer()),
  );
  await fs.writeFile(
    summaryPreviewPath,
    new Uint8Array(await summaryPreview.arrayBuffer()),
  );
  await fs.writeFile(
    protocolPreviewPath,
    new Uint8Array(await protocolPreview.arrayBuffer()),
  );
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved workbook: ${outputPath}`);
