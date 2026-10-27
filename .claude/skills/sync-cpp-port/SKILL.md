---
name: sync-cpp-port
description: 修改 lg_env 的 Python 量測程式（GradSIFTDebug.py 主程式、Algorithm/ 下 Region-SIFT 或 POSE 模組）之後，把改動同步到 D:\Lightglue\RegionSIFT_Cpp 的 C++ library，並用黃金資料驗證 C++ 與 Python 一致。使用者說「Python 改了，同步到 C++」「更新 C++ 版」「重新驗證 C++」時使用。
---

# Python 改動同步到 C++（RegionSIFT_Cpp）

C++ 版是 Python 版的逐位元移植，驗證方式是「Python 產生黃金資料 → C++ 重算比對」。
Python 一改，黃金資料就要重產，C++ 對應模組要照改，再跑測試直到全 PASS。

先讀 `D:\Lightglue\RegionSIFT_Cpp\STATUS.md` 最上方的「目前總結」與 `README.md`，掌握現況再動手。

## 固定前提（不要改，除非使用者明說）

- **比對基準**：關閉 IPP 的 Python。必須在 `import cv2` **之前**設 `OPENCV_IPP=disabled`（`cv2.ipp.setUseIPP(False)` 只影響呼叫的執行緒，POSE 的工作執行緒仍會用 IPP）。tools/ 腳本與各 trace 腳本已處理；新寫的腳本也必須這樣做。
- **測試影片**：只用 `golden/scan_pairs.log` 中 `pairs=1` 的 43 支。POSE 失敗的 17 支是其他實驗用，不測。
- **編譯環境**：公司無 MSVC 授權。用 `D:\Lightglue\third_party\winlibs-gcc16.2-ucrt`（GCC）＋自編 OpenCV 4.13.0（無 IPP、`-ffp-contract=off`）。
- **Reject SpecPts 關閉**；descriptor 兩種（OpenCV／自製遮罩式）都要支援。
- **不修改 lg_env 的程式**（使用者自己改 Python）；不動 venv 的 `Lib/ Scripts/ Include/ share/`。改既有檔案前備份到 `C:\Users\88880445\.claude\backups\`。

## 流程

### 1. 找出改了什麼、要不要同步

```bash
cd /d/Lightglue/lg_env && git status && git diff --stat   # 或問使用者改了哪些函式
```

把改動分成三類：
- **只動 UI／log／計時／除錯輸出** → 不用同步，回報即可（Click 層與 POSE 的 C 類程式碼本來就沒移植）。
- **重構，數值不該變** → 只重產黃金資料並跑測試；測試若 FAIL 代表 Python 重構其實改了數值，先回報使用者。
- **改了演算法或參數（數值會變）** → 完整走 2～6 步。

不確定改動有沒有落在生產路徑上：POSE 可用 `golden/pose_coverage.py` 重跑覆蓋率確認（見 `golden/pose_analysis/`）。

### 2. 對應到 C++ 模組、黃金資料目標、測試

| Python（以函式名為準，行號會偏移） | C++（include/rsift、src） | regen 目標 | 測試（run_all_tests 名稱） |
|---|---|---|---|
| MAIN 的三角化、融合、世界座標、品質指標、高度（compute_measure 後段、do_measure）；stereo_matching.get_patch/score_patch_match | postprocess | data、fusion | postprocess、fusion |
| Region_SIFT_Matching.RegionSIFTConfig；MAIN 的 REGION_SIFT_CONFIG | config.hpp（`production_config()`） | data | 全部 Click 測試 |
| Region_SIFT_Matching：_validate_config、select_region_points、單應性、build_epipolar_band、ROI、eroded mask | region_geometry | region_geometry | region_geometry |
| region_sift_frames.py | frames | frames | frames |
| region_sift_scoring.py；_keep_count、score_descriptor_group | scoring | scoring | scoring |
| region_sift_search.py；run_region_sift_matching 主流程 | region_search、region_sift、descriptor_backend | search | search |
| masked_sift_descriptor.py | masked_sift | masked_sift | masked_sift |
| specular_detection.compute_specular_mask_bgr_block_adaptive；MAIN SPATIAL_BLOCK_CONFIG | specular_mask | specular_mask | specular_mask |
| POSE 模組常數；MAIN analyze_video_frames 注入值 | pose_config.hpp | pose_frames | 全部 POSE 測試 |
| POSE：影格/probe/ArUco（LazyVideoFrames、detect_frame_markers…） | pose_frames | pose_frames（含 pose_numpy、pose_session） | pose_frames |
| POSE：marker map、逐幀候選、observability | pose_marker_map | pose_marker_map | pose_marker_map |
| POSE：時序 DP、pattern-guided 補 probe | pose_temporal | pose_temporal | pose_temporal |
| POSE：配對枚舉/評分/TopK | pose_pairs | pose_pairs | pose_pairs |
| POSE：SIFT＋Essential 重排 | pose_feature_rerank | pose_feature_rerank | pose_feature_rerank |
| POSE：世界聯合最佳化；scipy least_squares 用法 | least_squares、pose_world_opt | least_squares、pose_world_opt | least_squares、pose_world_opt |
| POSE 輸出/參考平面；MAIN 的 F、Shared Pattern Plane、去畸變；camera_preprocess | session | pose_session | session |

另外：**改 Click 層 → 一定再跑 click、pipeline；改 POSE → 一定再跑 session、pipeline**。共用工具 numpy_compat／numpy_blas／lapack_svd 改了 → 跑全部。

各 C++ 函式的註解標有對應 Python 檔案:行號；Python 行號已偏移時以函式名搜尋。trace 腳本與黃金資料格式說明在 `golden/trace_*.py` 開頭的 docstring、`golden/pose/README.md`、`README.md`。

### 3. 用新 Python 重產黃金資料

```bash
cd /d/Lightglue/RegionSIFT_Cpp
tools/regen_golden.sh --list                 # 目標清單
tools/regen_golden.sh data search            # 只重產受影響的目標（data 永遠要重產）
tools/regen_golden.sh all                    # 不確定時全部重產（約 1～2 小時）
```

- 舊資料自動移到 `golden/_prev/<目標>`，log 在 `golden/logs/regen_<目標>.log`。
- **最容易漏掉：主程式巢狀函式內的邏輯不會自動反映。** MAIN 的 `compute_measure`、`do_measure`、`main()` 的 Session 準備是 GUI 閉包，無法從外部呼叫，`golden/export_golden.py` 是**照抄**這些邏輯（`build_session`、`compute_measure_region`、`do_measure`，每段都註明 MAIN 行號）。MAIN 的這些區段若有改動（例如品質指標、世界座標、高度平面選擇、cand 組法、遮罩來源、Region-SIFT 呼叫參數），必須先把 export_golden.py 對應段落照改，否則黃金資料仍是舊邏輯，C++ 測試照樣 PASS 卻與新 GUI 不一致。判斷方法：`git diff` 看 MAIN 的改動行號是否落在上述函式內。MAIN 的頂層函式與模組常數（如 `triangulate_point_3d`、`fuse_candidate_results`、`REGION_SIFT_CONFIG`）是直接 import 使用，會自動反映。
- **精簡版也要同步**：`lg_env/regionsift_lite_core.py`（Demo 精簡版，見 `lg_env/README_regionsift_lite.md`）**複製**了 MAIN 的頂層函式／常數（`triangulate_point_3d`、`fuse_candidate_results`、`compute_shared_marker_corner_plane`、`REGION_SIFT_CONFIG`、`SPATIAL_BLOCK_CONFIG`、POSE 注入常數、`save_measurement_to_txt` 等），邏輯與 export_golden.py 同源。MAIN 的這些頂層函式／常數，或 `compute_measure`／`do_measure`／`main()` Session 準備改動時，要連同 core 一起改，再跑 `golden/verify_lite_core.py` 確認仍逐位元一致。
- 改完 export_golden.py 後，用 `golden/crosscheck_gui_records.py` 拿使用者在新版 GUI 存下的量測紀錄（影片旁的 `.txt`）交叉比對，確認匯出腳本仍忠實重現 GUI（需關閉 Reject SpecPts 的紀錄才可比；開 IPP 的 GUI 與基準約有 1% 點擊差異屬已知）。
- trace 腳本是 monkeypatch Python 函式名稱攔截輸入輸出的。**Python 若改了函式名稱、拆分或改簽名，對應 trace 腳本會失敗或錄不到**，要先修 trace 腳本（改前備份）。
- 新增 Python 功能、沒有對應 trace 時，照既有 trace 腳本的寫法新增一支，並在 `tools/regen_golden.sh` 的 ORDER/OUT 加上目標。

### 4. 先跑測試看差在哪

```bash
tools/run_all_tests.sh --no-clean <受影響的測試...>
```

數值有變的改動，此時 FAIL 是預期的；看 `golden/logs/test_<名稱>.log` 的差異，確認差異範圍跟 Python 改動一致。

### 5. 移植到 C++

- 在對應 C++ 檔照 Python 改，註解標新的 Python 檔案:行號。
- 結構變動大或要寫新模組：照既有模組的模式（標頭＋實作＋`tests/golden_<名>_test.cpp`，CMake 會自動收錄）。
- 反覆 `tools/run_all_tests.sh --no-clean <名稱>` 直到 PASS。

### 6. 全面驗收與紀錄

```bash
tools/run_all_tests.sh        # 乾淨重建，零警告＋全部 PASS 才算完成（約 20～25 分鐘）
```

- **驗收標準**：離散結果（匹配點、成功/失敗與原因、選中的影格、排序、旗標）必須完全一致；連續值逐位元一致，做不到要寫明原因與容差。已知例外見 STATUS.md：P3.7 世界最佳化（SVD 實作不同，R/t ≤1e-5 量級）、OpenCV SIFT descriptor 極少數元素差 1。不要為了通過而放寬容差。
- 更新 `STATUS.md`（目前總結的日期、對應模組列），必要時更新 `README.md` 與 `docs/progress.html`。
- 回報使用者：改了哪些 C++ 檔、重產了哪些黃金資料、測試結果表（`golden/logs/test_summary.txt`）、任何新的差異與原因。

## 已踩過的陷阱（移植時逐項檢查）

- **IPP**：見固定前提。比對有微小差異時，先確認 Python 端所有執行緒的 `cv2.ipp.useIPP()` 都是 False。
- **FMA**：GCC 預設會把 a*b+c 合併成 FMA；本專案與 OpenCV 都用 `-ffp-contract=off`。Android（clang）也必須如此。
- **捨入**：Python `round()`／`np.rint` 是 half-even，用 `rsift::np::round_half_even`，不可用 `std::round`。
- **取模**：Python `%` 符號跟隨除數，用 `np::py_mod`／`py_mod_f32`。
- **NumPy 2 NEP 50**：Python float 是 weak type，float32 陣列 − Python float 仍是 float32。逐步追蹤每個變數的 dtype。
- **加總順序**：`np.sum/mean` 是 pairwise；F-contiguous 切片沿軸 mean 是逐欄循序；`np.dot/linalg.norm` 走 OpenBLAS（numpy_blas.hpp 已重現本機 SkylakeX 核心順序）。
- **`x ** 2`**：Python float 的平方走 UCRT pow，約 0.2% 不等於 x*x，用 `py_pow2`。
- **`cv::Vec / 純量`** 是乘以倒數，不等於 NumPy 除法，要逐元素除。
- **SVD**：`np.linalg.svd` 走 LAPACK dgesdd（float32 輸入也用 double 算），用 lapack_svd 的移植版；OpenCV 的 SVD 差 1e-15 就可能翻轉後續排序。
- **排序**：Python `sorted`／`list.sort` 是穩定排序（`std::stable_sort`）；`np.argsort` 預設不穩定；argmin/argmax/min() 取第一個；dict 依插入順序。
- **OpenCV 內部的 STL 差異**：SIFT 的 nfeatures 截斷用 `std::nth_element`，MSVC 與 libstdc++/libc++ 結果不同；pose_feature_rerank 已改為自己實作 MSVC 版本，Click 層 SIFT 只做 compute 不受影響。
- **cv2 對 numpy view**：切片傳進 cv2 與獨立陣列的邊界處理不同；Python 先 astype/copy 的地方 C++ 也要先 clone。
- **就地改寫的副作用**：例如 POSE `_pattern_guided_marker_observability` 會就地正規化 marker_map 的第 3 欄，C++ 要在相同時機重現。
- **FileStorage** 會把 -0.0 寫成 0；比對時要容許這點。
- **環境版本**：黃金資料依賴 venv 的 numpy／scipy／opencv-python 版本與本機 CPU（OpenBLAS 核心）。升級套件或換電腦後，先確認全部測試仍 PASS，再開始同步。

## 派工建議（省 token、避免用量上限）

- 重產黃金資料、跑測試、彙整結果是機械工作，主對話直接執行腳本，不用開 agent。
- 一般移植用 sonnet 起步；只有「為什麼不是逐位元一致」這類難題才用 opus。
- 同時開多個 opus agent 很容易撞到 API 用量上限（本專案發生過多次）；平行開 agent 時每個都用獨立的 build 目錄（`build_<名稱>`），且明確分配誰可以改共用檔案（numpy_compat、pose_config、CMakeLists）。
