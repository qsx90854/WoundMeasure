import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = path.join(root, "outputs", "opencv_function_inventory_20260902");
const sources = [
  ["主程式", "depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py"],
  ["Stereo matching", "Algorithm/stereo_matching.py"],
  ["RT 估計", "Algorithm/video_pose_analysis_temporal_unified_pattern_guided_local_window.py"],
  ["Aruco/平面支援", "Algorithm/aruco_pose.py"],
  ["相機前處理支援", "Algorithm/camera_preprocess.py"],
  ["反光遮罩支援", "Algorithm/specular_detection.py"],
];

const rows = [
  // 1. Input / output
  ["01 影像與影片輸入輸出", "cv2.VideoCapture()", "開啟相機或影片來源，供錄影、指定影格解碼與 RT 影格分析使用。", "cv2.VideoCapture"],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.isOpened()", "確認相機或影片來源是否成功開啟。", "\\.isOpened\\("],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.read()", "讀取並解碼下一個影格。", "\\.read\\("],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.grab()", "只抓取下一個影格、暫不解碼；RT 批次取幀時用來降低不必要的解碼成本。", "\\.grab\\("],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.retrieve()", "將先前 grab 的影格解碼取回。", "\\.retrieve\\("],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.get()", "讀取影格尺寸、FPS、總影格數、目前位置、曝光等擷取屬性。", "\\.(?:get)\\(cv2\\.CAP_PROP_"],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.set()", "設定影格位置、解析度、FPS、FOURCC、buffer 等擷取屬性。", "\\.(?:set)\\(cv2\\.CAP_PROP_"],
  ["01 影像與影片輸入輸出", "cv2.VideoCapture.release()", "釋放相機或影片擷取資源。", "\\.release\\(\\)"],
  ["01 影像與影片輸入輸出", "cv2.VideoWriter_fourcc()", "建立影片編碼 FOURCC 代碼，例如 YUY2、MJPG 或 mp4v。", "cv2.VideoWriter_fourcc"],
  ["01 影像與影片輸入輸出", "cv2.VideoWriter()", "建立錄影輸出物件。", "cv2.VideoWriter\\("],
  ["01 影像與影片輸入輸出", "cv2.VideoWriter.write()", "把目前影格寫入錄製影片。", "video_writer\\.write\\("],
  ["01 影像與影片輸入輸出", "cv2.VideoWriter.release()", "完成錄影並釋放影片寫入資源。", "video_writer\\.release\\("],
  ["01 影像與影片輸入輸出", "cv2.imwrite()", "輸出 RT 診斷用的觀測/重投影影像。", "cv2.imwrite"],

  // 2. Camera and image preprocessing
  ["02 相機校正與影像前處理", "cv2.cvtColor()", "在 BGR、RGB、灰階與 HSV 之間轉換，供偵測、特徵、色彩比較與顯示使用。", "cv2.cvtColor"],
  ["02 相機校正與影像前處理", "cv2.resize()", "調整影像或遮罩解析度；也用於 RT/SIFT 的降尺度處理。", "cv2.resize"],
  ["02 相機校正與影像前處理", "cv2.createCLAHE()", "建立局部對比增強器，改善低對比影格的 ArUco 與特徵偵測。", "cv2.createCLAHE"],
  ["02 相機校正與影像前處理", "cv2.CLAHE.apply()", "把 CLAHE 套用到灰階影像。", "clahe\\.apply\\("],
  ["02 相機校正與影像前處理", "cv2.getOptimalNewCameraMatrix()", "依內參與畸變係數計算去畸變後的新相機矩陣。", "cv2.getOptimalNewCameraMatrix"],
  ["02 相機校正與影像前處理", "cv2.initUndistortRectifyMap()", "預先建立去畸變/校正映射表，供 remap 快速重映射。", "cv2.initUndistortRectifyMap"],
  ["02 相機校正與影像前處理", "cv2.remap()", "依映射表進行去畸變，或將 patch 依局部映射重採樣。", "cv2.remap"],
  ["02 相機校正與影像前處理", "cv2.undistortPoints()", "把 SIFT/角點座標去畸變並映回指定內參座標系。", "cv2.undistortPoints"],
  ["02 相機校正與影像前處理", "cv2.GaussianBlur()", "平滑灰階影像以估計亮度背景，輔助反光區域判定。", "cv2.GaussianBlur"],
  ["02 相機校正與影像前處理", "cv2.split()", "拆分 BGR 或 HSV 通道，供 opponent-SIFT、色彩直方圖與高光偵測使用。", "cv2.split"],
  ["02 相機校正與影像前處理", "cv2.morphologyEx()", "以開/閉運算去除反光遮罩雜點並補齊小孔洞。", "cv2.morphologyEx"],
  ["02 相機校正與影像前處理", "cv2.dilate()", "擴張反光區域，避免特徵點落在高光邊界。", "cv2.dilate"],
  ["02 相機校正與影像前處理", "cv2.bitwise_or()", "合併空間與時間反光遮罩，或合併排除區。", "cv2.bitwise_or"],
  ["02 相機校正與影像前處理", "cv2.addWeighted()", "把不同類型反光遮罩以半透明色疊加到顯示影像。", "cv2.addWeighted"],
  ["02 相機校正與影像前處理", "cv2.absdiff()", "計算對齊影像的絕對差，產生診斷或時間差異訊號。", "cv2.absdiff"],

  // 3. ArUco
  ["03 ArUco 偵測與角點精修", "cv2.aruco.getPredefinedDictionary()", "取得 DICT_4X4_100 標記字典。", "cv2.aruco.getPredefinedDictionary"],
  ["03 ArUco 偵測與角點精修", "cv2.aruco.DetectorParameters()", "建立新版 ArUcoDetector 的偵測參數。", "cv2.aruco.DetectorParameters\\("],
  ["03 ArUco 偵測與角點精修", "cv2.aruco.DetectorParameters_create()", "建立舊版 OpenCV ArUco 偵測參數，作為版本相容路徑。", "cv2.aruco.DetectorParameters_create"],
  ["03 ArUco 偵測與角點精修", "cv2.aruco.ArucoDetector()", "建立新版 ArUco 標記偵測器。", "cv2.aruco.ArucoDetector"],
  ["03 ArUco 偵測與角點精修", "cv2.aruco.ArucoDetector.detectMarkers()", "新版 API：偵測標記角點與 ID。", "(?:detector|local_detector|preview_detector)\\.detectMarkers"],
  ["03 ArUco 偵測與角點精修", "cv2.aruco.detectMarkers()", "舊版 API：偵測標記角點與 ID，作為版本相容路徑。", "cv2.aruco.detectMarkers"],
  ["03 ArUco 偵測與角點精修", "cv2.cornerSubPix()", "將 ArUco 角點精修到亞像素精度，以降低 PnP 與 RT 誤差。", "cv2.cornerSubPix"],

  // 4. Features/matching
  ["04 特徵擷取與立體匹配", "cv2.SIFT_create()", "建立 SIFT 特徵器，用於 RT 輔助配對與局部 stereo matching。", "cv2.SIFT_create"],
  ["04 特徵擷取與立體匹配", "cv2.ORB_create()", "建立 ORB 特徵器，作為較快的局部二值特徵匹配選項。", "cv2.ORB_create"],
  ["04 特徵擷取與立體匹配", "cv2.KeyPoint()", "在梯度/角點候選位置建立 OpenCV KeyPoint，交給 SIFT 或 ORB 計算描述子。", "cv2.KeyPoint"],
  ["04 特徵擷取與立體匹配", "Feature2D.detectAndCompute()", "同時偵測關鍵點並計算描述子；用在 RT 的 SIFT 輔助與跨幀/跨視角特徵。", "detectAndCompute\\("],
  ["04 特徵擷取與立體匹配", "Feature2D.compute()", "對指定/注入的 KeyPoint 計算 SIFT、Opponent-SIFT 或 ORB 描述子。", "(?:sift|sift_detector|orb|extractor)\\.compute\\("],
  ["04 特徵擷取與立體匹配", "cv2.BFMatcher()", "建立暴力描述子匹配器；SIFT 使用 L2，ORB 使用 Hamming。", "cv2.BFMatcher"],
  ["04 特徵擷取與立體匹配", "DescriptorMatcher.knnMatch()", "取得每個描述子的 k 個最近鄰，供 Lowe ratio 與雙向一致性檢查。", "bf\\.knnMatch"],
  ["04 特徵擷取與立體匹配", "DescriptorMatcher.match()", "取得一對一最佳描述子匹配，供局部注入特徵流程使用。", "bf\\.match\\("],
  ["04 特徵擷取與立體匹配", "cv2.DMatch()", "建立經過局部約束挑選後的自訂 match 記錄。", "cv2.DMatch"],
  ["04 特徵擷取與立體匹配", "cv2.norm()", "計算 ORB 描述子間的 Hamming 距離。", "cv2.norm"],
  ["04 特徵擷取與立體匹配", "cv2.Sobel()", "計算局部影像 x/y 梯度，找出紋理強的位置。", "cv2.Sobel"],
  ["04 特徵擷取與立體匹配", "cv2.magnitude()", "由 x/y Sobel 梯度計算梯度強度。", "cv2.magnitude"],
  ["04 特徵擷取與立體匹配", "cv2.sqrt()", "計算梯度平方和的平方根，取得梯度幅值。", "cv2.sqrt"],
  ["04 特徵擷取與立體匹配", "cv2.cornerMinEigenVal()", "計算角點最小特徵值響應，挑選有辨識力的注入特徵點。", "cv2.cornerMinEigenVal"],
  ["04 特徵擷取與立體匹配", "cv2.goodFeaturesToTrack()", "在 RT 跨幀追蹤中選取 Shi-Tomasi 角點。", "cv2.goodFeaturesToTrack"],
  ["04 特徵擷取與立體匹配", "cv2.calcOpticalFlowPyrLK()", "以金字塔 Lucas–Kanade 光流追蹤點，並做前後向一致性檢查。", "cv2.calcOpticalFlowPyrLK"],
  ["04 特徵擷取與立體匹配", "cv2.matchTemplate()", "以 TM_CCOEFF_NORMED 計算 ZNCC 模板匹配分數，沿候選區/極線尋找對應點。", "cv2.matchTemplate"],
  ["04 特徵擷取與立體匹配", "cv2.minMaxLoc()", "從模板匹配分數圖中找最佳位置與分數。", "cv2.minMaxLoc"],
  ["04 特徵擷取與立體匹配", "cv2.pyrDown()", "建立較低解析度影像，先粗略執行 ECC 再回到原尺寸精修。", "cv2.pyrDown"],
  ["04 特徵擷取與立體匹配", "cv2.findTransformECC()", "以 ECC 最佳化局部平移，將初始匹配點精修到亞像素位置。", "cv2.findTransformECC"],
  ["04 特徵擷取與立體匹配", "cv2.calcHist()", "計算局部 HSV 二維直方圖，評估左右 patch 色彩一致性。", "cv2.calcHist"],
  ["04 特徵擷取與立體匹配", "cv2.normalize()", "將色彩直方圖正規化至固定範圍，便於跨 patch 比較。", "cv2.normalize"],
  ["04 特徵擷取與立體匹配", "cv2.compareHist()", "以 Bhattacharyya 距離比較左右 patch 的 HSV 直方圖。", "cv2.compareHist"],

  // 5. Geometry and RT
  ["05 RT、幾何估計與三角化", "cv2.solvePnP()", "由 3D 標記/平面點與 2D 觀測估計相機旋轉和平移；含 IPPE_SQUARE 與 ITERATIVE 精修。", "cv2.solvePnP\\("],
  ["05 RT、幾何估計與三角化", "cv2.solvePnPGeneric()", "取得平面/IPPE PnP 的多組候選解，再依重投影與幾何條件選解。", "cv2.solvePnPGeneric"],
  ["05 RT、幾何估計與三角化", "cv2.solvePnPRansac()", "以 RANSAC 排除錯誤 SIFT 對應並估計穩健的相對姿態初值。", "cv2.solvePnPRansac"],
  ["05 RT、幾何估計與三角化", "cv2.Rodrigues()", "在旋轉向量與 3×3 旋轉矩陣間互轉，供姿態合成、平均、平滑與投影使用。", "cv2.Rodrigues"],
  ["05 RT、幾何估計與三角化", "cv2.projectPoints()", "把 3D 點投影回影像，計算重投影誤差並驗證/最佳化 RT。", "cv2.projectPoints"],
  ["05 RT、幾何估計與三角化", "cv2.findEssentialMat()", "由正規化的左右/跨幀對應點，以 RANSAC 估計本質矩陣。", "cv2.findEssentialMat"],
  ["05 RT、幾何估計與三角化", "cv2.recoverPose()", "由本質矩陣恢復相對旋轉與平移方向，作為 RT 候選或檢核。", "cv2.recoverPose"],
  ["05 RT、幾何估計與三角化", "cv2.findHomography()", "以 RANSAC 估計平面單應矩陣，用於局部匹配、跨幀平面運動與模型競爭檢核。", "cv2.findHomography"],
  ["05 RT、幾何估計與三角化", "cv2.estimateAffinePartial2D()", "以 RANSAC 估計局部相似/部分仿射變換，將特徵群映射到右圖。", "cv2.estimateAffinePartial2D"],
  ["05 RT、幾何估計與三角化", "cv2.correctMatches()", "依 fundamental matrix 做 Hartley–Sturm 最佳對應點修正，再進行三角化。", "cv2.correctMatches"],
  ["05 RT、幾何估計與三角化", "cv2.triangulatePoints()", "由左右投影矩陣與對應點求 3D 齊次座標，用於深度、平面與尺度檢核。", "cv2.triangulatePoints"],
  ["05 RT、幾何估計與三角化", "cv2.warpPerspective()", "依 homography 將影像或有效區遮罩對齊，供局部匹配與時間反光分析。", "cv2.warpPerspective"],

  // 6. Shapes and masks
  ["06 輪廓、區域與品質評估", "cv2.findContours()", "從傷口/遮罩找出外部輪廓。", "cv2.findContours"],
  ["06 輪廓、區域與品質評估", "cv2.contourArea()", "計算輪廓或標記四邊形面積，供最大區域選擇與覆蓋率/品質評分。", "cv2.contourArea"],
  ["06 輪廓、區域與品質評估", "cv2.minAreaRect()", "求輪廓的最小面積旋轉矩形。", "cv2.minAreaRect"],
  ["06 輪廓、區域與品質評估", "cv2.boxPoints()", "把旋轉矩形轉成四個角點，供傷口尺寸量測。", "cv2.boxPoints"],
  ["06 輪廓、區域與品質評估", "cv2.convexHull()", "求點集合凸包，以估算 ArUco/特徵在影像中的空間覆蓋率。", "cv2.convexHull"],
  ["06 輪廓、區域與品質評估", "cv2.fillConvexPoly()", "在特徵遮罩中填滿標記區域，避免在 ArUco 內部抽取自然特徵。", "cv2.fillConvexPoly"],
  ["06 輪廓、區域與品質評估", "cv2.connectedComponentsWithStats()", "把高光遮罩分成連通區並取得面積，以針對性擴張較大的高光區。", "cv2.connectedComponentsWithStats"],
  ["06 輪廓、區域與品質評估", "cv2.Laplacian()", "以 Laplacian 變異數評估影格清晰度，供 RT 影格配對評分。", "cv2.Laplacian"],

  // 7. UI only
  ["07 UI 顯示與診斷繪圖", "cv2.rectangle()", "在預覽或診斷影像畫 ROI、標籤底框與偵測框；不參與 RT/深度數值計算。", "cv2.rectangle"],
  ["07 UI 顯示與診斷繪圖", "cv2.putText()", "在影像上顯示錄影狀態、標記 ID、ROI 與診斷文字；不參與數值計算。", "cv2.putText"],
  ["07 UI 顯示與診斷繪圖", "cv2.getTextSize()", "計算文字框尺寸以排版標籤背景；僅供 UI。", "cv2.getTextSize"],
  ["07 UI 顯示與診斷繪圖", "cv2.circle()", "畫選點、量測點或錄影提示圓點；僅供 UI/診斷。", "cv2.circle"],
  ["07 UI 顯示與診斷繪圖", "cv2.line()", "畫量測線、輔助線或方向線；僅供 UI/診斷。", "cv2.line"],
  ["07 UI 顯示與診斷繪圖", "cv2.polylines()", "畫 ArUco 四邊形、傷口輪廓與觀測/投影框；僅供 UI/診斷。", "cv2.polylines"],
  ["07 UI 顯示與診斷繪圖", "cv2.drawContours()", "把遮罩輪廓疊加到顯示影像；僅供 UI/診斷。", "cv2.drawContours"],
  ["07 UI 顯示與診斷繪圖", "cv2.imshow()", "顯示相機錄影預覽視窗。", "cv2.imshow"],
  ["07 UI 顯示與診斷繪圖", "cv2.waitKey()", "處理 OpenCV 視窗鍵盤事件，例如開始/停止錄影與離開。", "cv2.waitKey"],
  ["07 UI 顯示與診斷繪圖", "cv2.destroyAllWindows()", "關閉所有 OpenCV 預覽視窗。", "cv2.destroyAllWindows"],
];

const sourceText = {};
for (const [label, rel] of sources) {
  sourceText[rel] = (await fs.readFile(path.join(root, rel), "utf8")).split(/\r?\n/);
}

function locate(patternText) {
  const re = new RegExp(patternText);
  const hits = [];
  const labels = [];
  for (const [label, rel] of sources) {
    const lines = sourceText[rel];
    const lineNos = [];
    for (let i = 0; i < lines.length; i++) if (re.test(lines[i])) lineNos.push(i + 1);
    if (lineNos.length) {
      labels.push(label);
      hits.push(`${rel}:${lineNos.slice(0, 4).join(",")}${lineNos.length > 4 ? "…" : ""}`);
    }
  }
  return [labels.join("；"), hits.join("\n")];
}

const data = rows.map((r, i) => {
  const [usedIn, locations] = locate(r[3]);
  return [i + 1, r[0], r[1], r[2], usedIn || "—", locations || "—"];
});

const workbook = Workbook.create();
const summary = workbook.worksheets.add("範圍與摘要");
const list = workbook.worksheets.add("OpenCV Function List");
summary.showGridLines = false;
list.showGridLines = false;

summary.getRange("A1:F1").merge();
summary.getRange("A1").values = [["OpenCV Function 使用盤點"]];
summary.getRange("A2:F2").merge();
summary.getRange("A2").values = [["depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py／Stereo Matching／RT 估計"]];
summary.getRange("A4:B8").values = [
  ["統計日期", "2026-09-02"],
  ["Function 筆數", data.length],
  ["分類數", 7],
  ["主分析範圍", "主程式、stereo_matching.py、video_pose_analysis_temporal_unified_pattern_guided_local_window.py"],
  ["支援範圍", "aruco_pose.py、camera_preprocess.py、specular_detection.py（僅因主/RT 流程直接呼叫）"],
];
summary.getRange("A10:B17").values = [
  ["分類", "Function 數"],
  ...[...new Set(data.map(r => r[1]))].map(cat => [cat, data.filter(r => r[1] === cat).length]),
];
summary.getRange("D4:F4").merge();
summary.getRange("D4").values = [["盤點原則"]];
summary.getRange("D5:F10").merge();
summary.getRange("D5").values = [[
  "以實際程式呼叫為準，包含 cv2.* 與由 OpenCV 建立物件的方法（例如 VideoCapture.read、BFMatcher.knnMatch、Feature2D.compute）。不列入 cv2 常數、NumPy/SciPy/Matplotlib/ONNX Runtime API。新版與舊版 ArUco 相容路徑分列；同一 function 若在多個模組使用則合併為一列，並保留代表位置。"
]];
summary.getRange("A20:B26").values = [
  ["範圍標籤", "檔案"],
  ...sources.map(x => x),
];

list.getRange("A1:F1").merge();
list.getRange("A1").values = [["OpenCV Function List（依用途排序）"]];
list.getRange("A2:F2").merge();
list.getRange("A2").values = [["Function name 與功能描述分欄；第 07 區為 UI／診斷顯示用途。"]];
list.getRange("A4:F4").values = [["No.", "用途分類", "Function name", "用途描述", "使用範圍", "代表程式位置（行號）"]];
list.getRange(`A5:F${data.length + 4}`).values = data;

// General styling
for (const sheet of [summary, list]) {
  sheet.getRange("A1:F1").format = {
    fill: "#17365D", font: { bold: true, color: "#FFFFFF", size: 18 },
    horizontalAlignment: "left", verticalAlignment: "center"
  };
  sheet.getRange("A2:F2").format = {
    fill: "#D9EAF7", font: { color: "#17365D", italic: true, size: 11 },
    verticalAlignment: "center"
  };
  sheet.getRange("A1:F1").format.rowHeight = 30;
  sheet.getRange("A2:F2").format.rowHeight = 24;
}

summary.getRange("A4:A8").format = { fill: "#EAF2F8", font: { bold: true, color: "#17365D" } };
summary.getRange("A10:B10").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
summary.getRange("A20:B20").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
summary.getRange("D4:F4").format = { fill: "#4472C4", font: { bold: true, color: "#FFFFFF" } };
summary.getRange("D5:F10").format = { fill: "#FFF4CC", wrapText: true, verticalAlignment: "top" };
summary.getRange("A4:B8").format.borders = { preset: "outside", style: "thin", color: "#9EADBA" };
summary.getRange("A10:B17").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
summary.getRange("A20:B26").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
summary.getRange("A1:F26").format.font = { name: "Aptos", size: 10 };
summary.getRange("A1:F1").format.font = { name: "Aptos Display", bold: true, color: "#FFFFFF", size: 18 };
summary.getRange("A:A").format.columnWidth = 24;
summary.getRange("B:B").format.columnWidth = 72;
summary.getRange("C:C").format.columnWidth = 3;
summary.getRange("D:F").format.columnWidth = 22;
summary.getRange("D5:F10").format.rowHeight = 42;
summary.freezePanes.freezeRows(2);

list.getRange("A4:F4").format = {
  fill: "#4472C4", font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#2F5597" }
};
list.getRange(`A5:F${data.length + 4}`).format = {
  font: { name: "Aptos", size: 10 }, verticalAlignment: "top",
  borders: { insideHorizontal: { style: "thin", color: "#D9E2F3" } }
};
list.getRange(`D5:F${data.length + 4}`).format.wrapText = true;
list.getRange(`A5:A${data.length + 4}`).format.horizontalAlignment = "center";
list.getRange(`A5:A${data.length + 4}`).format.numberFormat = "0";
list.getRange("A:A").format.columnWidth = 7;
list.getRange("B:B").format.columnWidth = 30;
list.getRange("C:C").format.columnWidth = 36;
list.getRange("D:D").format.columnWidth = 72;
list.getRange("E:E").format.columnWidth = 38;
list.getRange("F:F").format.columnWidth = 58;
list.getRange("A4:F4").format.rowHeight = 30;
list.getRange(`A5:F${data.length + 4}`).format.rowHeight = 42;
list.freezePanes.freezeRows(4);
list.freezePanes.freezeColumns(2);

const categoryFills = ["#EAF2F8", "#E2F0D9", "#FFF2CC", "#FCE4D6", "#E4DFEC", "#DDEBF7", "#F2F2F2"];
const cats = [...new Set(data.map(r => r[1]))];
for (let i = 0; i < data.length; i++) {
  const catIndex = cats.indexOf(data[i][1]);
  list.getRange(`B${i + 5}`).format = { fill: categoryFills[catIndex], font: { bold: true, color: "#244062" }, wrapText: true };
  if (i === 0 || data[i - 1][1] !== data[i][1]) {
    list.getRange(`A${i + 5}:F${i + 5}`).format.borders = { top: { style: "medium", color: "#4472C4" } };
  }
}

list.tables.add(`A4:F${data.length + 4}`, true, "OpenCVFunctionTable").style = "TableStyleMedium2";

await fs.mkdir(outputDir, { recursive: true });
const summaryPng = await workbook.render({ sheetName: "範圍與摘要", range: "A1:F26", scale: 1.2, format: "png" });
await fs.writeFile(path.join(outputDir, "summary_preview.png"), new Uint8Array(await summaryPng.arrayBuffer()));
const listPng = await workbook.render({ sheetName: "OpenCV Function List", range: `A1:F${Math.min(data.length + 4, 28)}`, scale: 1.0, format: "png" });
await fs.writeFile(path.join(outputDir, "list_preview.png"), new Uint8Array(await listPng.arrayBuffer()));

const inspection = await workbook.inspect({
  kind: "table", range: "OpenCV Function List!A1:F14", include: "values,formulas",
  tableMaxRows: 14, tableMaxCols: 6, maxChars: 5000
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan"
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
const outPath = path.join(outputDir, "opencv_function_inventory.xlsx");
await output.save(outPath);
console.log(`OUTPUT=${outPath}`);
console.log(`ROWS=${data.length}`);
