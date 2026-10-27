# Region-SIFT 精簡版（對外 demo 用）

只保留 Region-SIFT 量測流程的精簡程式，結果與主程式（MAIN）逐位元相同。

- `regionsift_lite_core.py`：無 UI 的核心（Session 準備、點擊量測、融合、高度、量測紀錄）。不 import MAIN。
- `depth_measure_regionsift_lite.py`：matplotlib（TkAgg）UI。
- MAIN = `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py`。

## 啟動

```bat
cd D:\Lightglue\lg_env
Scripts\python.exe depth_measure_regionsift_lite.py              :: 跳出選檔視窗
Scripts\python.exe depth_measure_regionsift_lite.py 影片.avi      :: 直接載入
Scripts\python.exe depth_measure_regionsift_lite.py 影片.avi --no-record   :: 不寫量測紀錄
```

無 GUI 使用核心：

```python
import regionsift_lite_core as core
session = core.build_session(r"D:\...\video.avi")
res = core.measure_click(session, u, v, core.REGION_SIFT_CONFIG)   # u, v 須先 int(round())
res['depth'], res['height_display_mm'], res['pt']
```

## 操作

| 動作 | 說明 |
|---|---|
| 開啟影片 | 選影片檔 → 背景分析（進度條）→ 顯示左右圖。只支援影片檔，不錄影 |
| 左圖點擊 | 量測；靠近 ArUco 角點 15 px 內自動吸附，座標 `int(round())`。計算中畫面有提示且不凍結 |
| 右圖顯示 | 匹配點（綠圈）與極線（黃線）；下方文字為相機/世界座標、品質指標、多配對融合 |
| 滾輪／左鍵拖曳／右鍵或 R | 縮放／平移／重設視圖 |
| Descriptor | 切換 OpenCV SIFT／自製遮罩式 SIFT（自製模式使用 block-adaptive 空間遮罩，第一次用到才計算） |
| SIFT 參數 | 開啟 Region-SIFT 參數視窗（`Algorithm/region_sift_settings.py`，本次執行有效） |
| 量測紀錄 | 開：每次點擊在影片旁同名 `.txt` 追加一筆（格式同 MAIN）；關：不寫 |

固定設定：Reject SpecPts 關閉、CLAHE 關閉、視差微調預設關（UI 無按鈕；參數視窗內仍可改）。
高度參考平面：Shared Pattern Plane，不可用時退回 Legacy Plane。

已移除：自訂平面、Grad-SIFT／Epi-band／ECC／Precise／Top-2 等舊匹配、temporal/V2/wound 遮罩、傷口 AI、
block accuracy、拖曳高度剖面、metric block、RT warp view、multi_dedrift/multi_pure、debug 下方面板、錄影。

## 與 MAIN 的關係（維護規則）

- **演算法仍以 MAIN 為主體開發**；精簡版只是 demo 的複本，不獨立演進。
- `Algorithm/` 下的模組是兩邊共用（只 import），改了自動兩邊同步，不用動精簡版。
- 下列 MAIN 內容**被複製到 core**（每段 core 內註明來源 MAIN 行號），MAIN 若改動需同步改 `regionsift_lite_core.py`：
  - 頂層函式：`triangulate_point_3d`、`compute_fundamental_matrix`、`fuse_candidate_results`、
    `compute_shared_marker_corner_plane`、`average_rotations_svd`、`analyze_video_frames` 包裝（及它注入 POSE 的常數）、
    `configure_world_marker_sift_rt_refine` 中影響數值的旗標、`preprocess_gray`、`compute_global_plane`、
    `snap_to_aruco_corner`、`save_measurement_to_txt`。
  - 常數：`REGION_SIFT_CONFIG`、`SPATIAL_BLOCK_CONFIG`、基線／深度／融合常數、`PARAMS_JSON_PATH`、
    POSE 注入常數（`RT_SIFT_*`、`PAIR_SCORE_*`、選幀模式等）、stereo_matching 可調常數。
  - MAIN 巢狀函式（GUI 閉包）的邏輯：`compute_measure` 的 Region-SIFT 分支與後處理、`do_measure`（多配對融合、高度）、
    `main()` 的 Session 準備（cand 組法、高度平面選擇、遮罩來源）。
- 其餘 MAIN 的 UI／debug／舊匹配改動不用同步。
- 這份複本的邏輯與 `D:\Lightglue\RegionSIFT_Cpp\golden\export_golden.py` 同源；同步 C++ 移植版時兩者一併檢查。
- 驗證：`D:\Lightglue\RegionSIFT_Cpp\golden\verify_lite_core.py`（同一進程、IPP 開著，逐位元比對 core 與 MAIN／export_golden）。
  MAIN 改動並同步 core 後請重跑；全部「逐位元一致」才算同步完成。
- UI 自測（無 GUI）：`python depth_measure_regionsift_lite.py --selftest <輸出資料夾>`，存截圖與紀錄到該資料夾。

## 注意

- IPP 與 MAIN 相同，保持開啟（不設 `OPENCV_IPP`、不呼叫 `setUseIPP(False)`）。
- POSE 分析會在影片旁寫 `<影片名>_rt_sift_diagnostics.txt`（與 MAIN 相同）。
- 相對路徑的標定檔 `calibration_result_Zebra_1_monocular.json` 以 core 所在資料夾為基準。
