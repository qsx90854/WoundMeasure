# Region-SIFT 量測演算法 — Android C++ Library 打包規劃

寫於 2026-09-30。依據:對 `depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py`
(下稱 MAIN,9406 行)與 `Algorithm/` 的三路原始碼追蹤(量測主流程、幾何與輸入準備、
Region-SIFT 模組依賴)。行號以 2026-09-30 工作樹為準。

## 已確認的前提

| 項目 | 決定 |
|---|---|
| 量測流程 | 以目前 UI 的 Region-SIFT 為主(`view_state['region_sift']=True`) |
| Reject SpecPts | **不啟用**(`reject_specular_candidates=False`) |
| Descriptor | **OpenCV 與自製遮罩式都要,可切換** |
| 平台 | **Android**(C++ 核心 + JNI,NDK 建置,OpenCV for Android) |
| Library 輸出 | **只輸出最終量測結果**(不輸出除錯/視覺化資料) |
| 與 Python 版一致性 | **容許誤差內一致**,以 Python 版為黃金參照 |

---

## 0. 會影響整個規劃的四個事實

1. **這不是 SBS 雙鏡頭,是單鏡頭影片。** 檔名與檔頭 docstring 的「SBS」是過期遺跡。
   左圖 = 影片**後半段**選出的最佳一格(`frame_B`),右圖 = **前半段**選出的最佳一格
   (`frame_A`),兩格是手持相機在不同時間、不同位置拍的(MAIN:2566-2567、
   `FRAME_RANGE_MODE="half_half"` MAIN:128)。左右共用同一組內參(camera_preprocess.py:325-330)。
   R_rel/t_rel **不是雙目校正外參**,是每次由 ArUco 標記位姿反解出來的,基線限制 35~220 mm。
   → 手機端的「擷取影像」其實是**錄一段移動中、畫面含 ArUco 標記的影片**。

2. **計算分兩層。**
   - **Session 層**(每段影片一次):位姿分析選出左圖與 1+最多 5 張右圖、算出各配對的
     R_rel/t_rel/基線/參考平面,並做去畸變、反光遮罩、高度參考平面。
   - **Click 層**(每次點擊):對每個配對跑 Region-SIFT → 三角化 → 檢查 → 多配對融合 → 高度。
   真正產生 R_rel/t_rel/平面的是
   `Algorithm/video_pose_analysis_temporal_unified_pattern_guided_local_window.py`
   (下稱 POSE,**6929 行**),不是 MAIN 裡的 `get_joint_relative_pose`(那是沒有呼叫點的死碼)。

3. **Reject SpecPts 關閉的實際效果取決於 Descriptor:**
   - OpenCV descriptor:點擊路徑**完全不用反光遮罩**(Region_SIFT_Matching.py:702-703 把遮罩設為 None)。
   - 自製遮罩式 descriptor:`run_region_sift_matching` 內部**強制** `reject_specular=True`
     (Region_SIFT_Matching.py:682-685),**一定要空間反光遮罩**,不受按鈕控制。
   → 因為兩種都要支援,**空間反光遮罩(block-adaptive)必須打包**,但只在自製模式時計算。
   → 注意:目前 Python 版**預設 Reject SpecPts 是開的**(MAIN:3337,按鈕顯示 `[X]`)。
   關掉會讓 OpenCV 模式的行為跟現在預設不同,**黃金參照必須用「關閉」的設定重新產生**。

4. **Region-SIFT 本體沒有用到 `stereo_matching.py`**(那是 Grad-SIFT 專用)。唯一例外是
   click 層的品質指標 MaskedScore 用到其中的 `score_patch_match`(stereo_matching.py:129-134),
   只需單獨移植這一個函式。

---

## 1. 資料流總覽

```
手機 APP                              C++ Library
────────                              ───────────
錄影(畫面含 ArUco 標記) ──影格──▶ [Session 層,每段影片一次]
                                       ① 相機內參/畸變 → 去畸變對照表
                                       ② 影片位姿分析(POSE)
                                          → 左圖 + 右圖(1 主 + 最多 5 次佳)
                                          → 每對 R_rel, t_rel, baseline
                                          → 參考平面 plane_n/c、左右 ArUco 角點
                                       ③ 選中影格去畸變、轉灰階、算 F 矩陣
                                       ④ 空間反光遮罩(僅自製模式,惰性計算)
                                       ⑤ 高度參考平面(Shared Pattern Plane 等)
                    ◀── 去畸變左圖 ──  回傳給 UI 顯示,供使用者點擊
使用者點擊 (u,v)  ──────────────▶ [Click 層,每次點擊]
                                       對每個配對:
                                         基線檢查 → Region-SIFT 匹配
                                         → 三角化 → 深度檢查
                                         → 世界座標、重投影誤差、品質指標
                                       多配對融合 → 高度
顯示結果          ◀── 結果 struct ──
```

點擊座標 (u,v) 的座標系是**去畸變後的左圖**,所以 Library 必須把去畸變左圖回傳給 UI 顯示。

---

## 2. 要打包的部分

### 2.1 Session 層

| # | 功能 | Python 位置 | 移植重點 |
|---|---|---|---|
| S1 | 相機參數與去畸變 | camera_preprocess.py:313-345;MAIN:2491-2494 | `getOptimalNewCameraMatrix(alpha=1)` + `initUndistortRectifyMap(CV_16SC2)` + `remap(INTER_LINEAR)`。對照表可依解析度快取。**Region-SIFT 用的是去畸變後的新內參 `KL`**,原始 K/dist 只用於 ArUco 角點與特徵點去畸變 |
| S2 | **影片位姿分析** | POSE:2893-6929(由 MAIN:1215 包裝、MAIN:2499 呼叫) | 見 2.1.1。**整個專案最大、最難的一塊** |
| S3 | 選中影格處理 | MAIN:2566-2567、2641-2663、2704-2729 | 去畸變、轉灰階;每個配對組成 `cand`;`F = compute_fundamental_matrix(KL,KL,R_rel,t_rel)`(MAIN:534-539) |
| S4 | 空間反光遮罩(**僅自製模式**) | specular_detection.py:91-174 + `SPATIAL_BLOCK_CONFIG`(MAIN:92-107) | block-adaptive:64×64 tile 百分位閾值(可雙線性內插)、三分支 OR、突出度閘門、強反光例外、OPEN/CLOSE。純 OpenCV 影像運算,約 80 行,好移植。左圖一張、每張右圖各一張,**用到才算** |
| S5 | 高度參考平面 | `compute_shared_marker_corner_plane` MAIN:703-837;`compute_marker_pose_plane` MAIN:1282;選擇順序 `get_selected_height_plane` MAIN:3368-3378 | UI 顯示的高度預設用 **Shared Pattern Plane**(共視 marker 四角三角化 + SVD),不可用時退回 legacy 平面;Marker Pose Plane 只有在 UI 切換 `use_pose_plane` 時才用(預設關,MAIN:3359-3362)。**注意:這跟餵給 Region-SIFT 的 `cand['plane_n/c']` 是不同的平面** |

#### 2.1.1 影片位姿分析(POSE)包含的步驟

依 POSE 執行順序(細節見原始碼行號):

1. 按需解碼影格(`LazyVideoFrames`,POSE:2773-2882)。
2. 端點候選取樣:前段 0.80/1.00、後段 0.10/0.22/0.40(POSE:121-122, 3399-3415)。
3. ArUco 偵測:`DICT_4X4_100`、邊長 8.25 mm(MAIN:168),CLAHE → `ArucoDetector.detectMarkers` →
   `cornerSubPix`(POSE:3190-3226)。在**原始未去畸變**影像上偵測,再 `undistortPoints` 到新內參。
4. **Marker map 自標定**:沒有預先定義的板子佈局,marker 相對位置從影片自動推算
   (`_build_temporal_marker_map_graph` POSE:746-883,`solvePnPGeneric(IPPE_SQUARE)` 雙分支共識)。
5. 逐幀位姿假設 + 時序動態規劃解 IPPE 雙解歧義(POSE:1285, 1616)。
6. Pattern-guided 補 probe、local window + `calcOpticalFlowPyrLK` 重排端點(POSE:5009-5400)。
7. 配對枚舉與評分:基線硬閘門 37~220 mm、重投影/模糊/覆蓋率/幾何分數(POSE:4619-4751)。
8. Top-K 配對的 SIFT + `findEssentialMat` + `recoverPose` + `findHomography` 驗證(POSE:3705-4141)。
9. **World 聯合最佳化**:`scipy.optimize.least_squares`(trf + Huber)同時優化兩端點位姿,
   殘差含 marker 角點重投影與 SIFT Sampson(POSE:2621-2771)。`ENABLE_WORLD_MARKER_SIFT_RT_REFINE=True`
   (MAIN:148),所以 SIFT 部分也要移植。**C++ 沒有 scipy,需改用 Ceres Solver 或自寫 LM**。
10. 次佳配對(同左圖、不同右圖,最多 5 個,POSE:6005-6111)與可能的主對升格(POSE:6128-6203)。
11. 參考平面:ref marker 四角三角化 + SVD,退路為 marker z=0 平面或單張 PnP 平面(POSE:6721-6751)。

**尚未完成的評估**:POSE 有多種模式與參數,目前預設設定下**實際會執行到的子集**還沒逐行界定
(例如 `FRAME_PAIR_SELECTION_MODE="angle_guided"` 路徑預設不走)。移植前應先做一輪專門的範圍界定,
見第 7 節 Phase 3。

### 2.2 Click 層

| # | 功能 | Python 位置 | 移植重點 |
|---|---|---|---|
| C1 | 基線與位姿檢查 | MAIN:4719-4733 | `baseline ∈ [35, 220] mm` |
| C2 | **Region-SIFT 匹配引擎** | 見 2.3 | 兩階段粗細搜尋(`two_stage_search=True`) |
| C3 | 三角化 | MAIN:541-561 | `cv2.correctMatches(F)` 修正左右點 → `cv2.triangulatePoints`,P0=K[I\|0]、P1=K[R\|t] |
| C4 | 深度檢查與距離 | MAIN:5229-5242 | z≤0 或 z>2000 mm 為失敗;`d=‖p3d‖` |
| C5 | 世界座標 | MAIN:5244-5263 | 以最小 ID 的 ArUco 為原點,`solvePnP(IPPE_SQUARE)` + 左圖角點 |
| C6 | 重投影誤差 | MAIN:5270-5276 | p3d 投到右圖與匹配點的距離 |
| C7 | 品質指標 | MAIN:5278-5304;`score_patch_match` stereo_matching.py:129-134 | `d_epi`(點到極線距離)、ZNCC(31×31)、MaskedScore、`confidence = max(0,zncc)·max(0,masked)·exp(−d_epi²/(2·1.5²))` |
| C8 | 多配對融合 | `fuse_candidate_results` MAIN:604-666;常數 MAIN:161-162 | 深度偏離主對 `max(5mm, 3%·z)` 剔除;≥3 個候選再做 MAD 離群剔除;依 `(baseline/z²)²` 加權平均 |
| C9 | 高度 | MAIN:5605-5629 | `n·(p3d_best − c)`,平面為 S5 的選擇結果。**注意用的是主對的 p3d_best,不是融合值** |

### 2.3 Region-SIFT 引擎(C2)的模組範圍

閉包只有 6 個檔案,Python 程式碼約 2728 行,**其中約 55-60% 是演算法核心**,其餘是驗證、
錯誤訊息、計時與除錯欄位。

| 模組 | 用途 | 需要的 backend | Python 核心行(估計) |
|---|---|---|---|
| `Region_SIFT_Matching.py` | 主入口、左圖 28 點取樣、平面單應性、極線帶、warp、候選篩選、選最佳/次佳、拒絕判定 | 共用 | ~480 |
| `region_sift_search.py` | 兩階段粗細搜尋、低谷分析、候選批次評估 | 共用 | ~280 |
| `region_sift_frames.py` | 逐點尺度與主方向估計(**自製 Python,不是 OpenCV 內建**)、OpenCV 模式的高斯金字塔、`SIFT.compute` 包裝 | 共用(金字塔與 compute 僅 OpenCV 模式) | ~330 |
| `region_sift_scoring.py` | GroupScore = 0.65·TrimK + 0.35·CellAll,加尺度/角度一致性懲罰 | 共用 | ~100 |
| `region_sift_refinement.py` | 「視差微調」(`local_plane_refine`,UI 可切換,預設關) | 共用(選用) | ~130 |
| `masked_sift_descriptor.py` | 遮罩式金字塔、自製 128 維 descriptor、共同覆蓋率比較、群組有效性 | **僅自製模式** | ~270 |

合計:OpenCV 模式 + 兩階段 + 視差微調約 1600 行;再加自製模式約 1870 行(Python 行數,C++ 通常更多)。

**兩種模式其實共用大部分程式**:差別只在影像金字塔(float32 SIFT 金字塔 vs float64 正規化卷積)、
descriptor 與距離計算、以及自製模式需要反光遮罩。尺度/方向估計、搜尋、評分全部共用。

`RegionSIFTConfig` 共 84 個欄位,只有 `require_all_descriptors` 完全沒被讀取。生產設定與預設值
只差三處:`two_stage_search=True`、`search_length_px=95`、`specular_check_support=False`。
C++ 端用一個 struct 保留全部生產路徑會讀的欄位即可,Python 的 `_validate_config`(80 行)只需保留少數 assert。

### 2.4 可選(建議第二期再決定)

| 功能 | 位置 | 說明 |
|---|---|---|
| CLAHE 前處理 | MAIN:1274;camera_preprocess.py:307-310 | UI 開關,預設關;僅 OpenCV 模式的右圖。實作很小,可直接保留成設定 |
| 點擊吸附 ArUco 角點 | `snap_to_aruco_corner` MAIN:1624-1630 | 點擊在角點 15 px 內會吸附。可由 Library 提供工具函式,或由 APP 用回傳的角點自行實作 |
| 自訂平面 | `fit_plane_to_points` MAIN:669;`on_custom_plane` MAIN:8588 | 使用者點 ≥3 點擬合平面(RANSAC + SVD),只影響高度顯示 |
| 手動指定右圖匹配點 | `manual_match_pt` | 跳過搜尋直接三角化 |

---

## 3. 不打包的部分

| 類別 | 項目 | 原因 |
|---|---|---|
| UI | matplotlib/tkinter 全部、所有 `on_*` 按鈕 callback、`update_display`、各種 debug 視窗與疊圖 | 手機 UI 取代 |
| 影像擷取 | `record_video_from_camera`、`select_video_source`、錄影預覽、`estimate_aruco_pattern_distances`(僅預覽用) | 手機負責擷取 |
| 舊匹配法 | Grad-SIFT、`stereo_matching.py`(除 `score_patch_match`)、Epi-band、ECC、Precise、improved matching、Top-2 rescue | Region 模式下全被守衛擋掉(MAIN:4937-5138) |
| Reject SpecPts 相關 | combined/temporal 反光遮罩、`specular_check_support` 分支、OpenCV 模式的中心點反光拒絕(Region_SIFT_Matching.py:693-701, 746-758, 838-848, 887-890) | 使用者決定不啟用 |
| 其他反光遮罩 | V2 dichromatic、wound-adaptive、legacy 固定閾值(僅 fallback) | 預設不使用 |
| 傷口 AI | ONNX 分割模型、傷口尺寸量測 | `ENABLE_WOUND_AI=False`,不影響 Region-SIFT |
| 非預設量測模式 | `multi_dedrift`/`multi_pure`、`track_feature_and_verify`、`apply_dedrift_correction`、`multi_view_triangulation` | `MEASURE_MODE="dual_direct"` |
| 除錯與效能量測 | `region_sift_profiling/diagnostics/gt/search_debug/settings.py`、`click_latency.py`、`perf_timer.py`、`format_stage_funnel`、`timing_*`、`candidate_funnel`、`detail_profile`、`search_history`;`region_debug` 約 85 個欄位中只保留引擎內部需要的約 20 個 | 不影響數值 |
| 稽核與診斷輸出 | `configure_world_marker_sift_rt_refine` 的稽核 log、RT audit、`SAVE_RT_SIFT_DIAGNOSTICS` 寫檔 | 診斷用(只有它的 enabled 開關會影響結果,需保留其語意) |
| 死碼與無效常數 | `get_joint_relative_pose`;`MAX_EXTRA_PAIRS`、`PAIR_SCORE_BASELINE_W`、`PAIR_SCORE_MARKER_W`、`POSE_SELECT_MODE`、`ENABLE_POSE_SMOOTHING` 等 | 沒有呼叫點或 POSE 不讀取 |
| 存檔 | `save_measurement_to_txt` | 由 APP 決定如何保存;Library 回傳結果 struct |
| Python 專用加速 | numba、左圖快取的 hash 簿記 | C++ 不需要;快取可在 C++ 自行設計(不影響數值) |
| 其他 UI 功能 | Block accuracy、拖曳高度剖面、5 mm metric block 稽核、RT warp view、平面選擇按鈕 | 不在 Region-SIFT 量測主路徑 |

---

## 4. 對外介面草案

架構:**C++17 核心** → **C ABI 薄層** → **JNI** → **Kotlin 包裝類別**。只依賴 OpenCV
(core/imgproc/calib3d/features2d/objdetect/video)與一個非線性最小平方函式庫(建議 Ceres Solver)。
OpenCV 4.7 之後 ArUco 與 SIFT 都在主模組,不需要 opencv_contrib。

```cpp
namespace rsift {

struct Calibration {            // 手機相機的校正結果
    cv::Matx33d K;              // 原始內參
    cv::Vec<double, 5> dist;    // k1 k2 p1 p2 k3
    cv::Size image_size;        // 校正解析度,必須與影格解析度一致
};

class FrameSource {             // APP 實作;POSE 需要隨機存取影格
public:
    virtual ~FrameSource() = default;
    virtual int count() const = 0;
    virtual cv::Mat frame(int index) = 0;   // BGR,原始未去畸變
};

struct SessionInfo {
    bool ok;
    std::string fail_reason;
    int left_frame_index;
    int pair_count;             // 1 主 + 最多 5 次佳
    cv::Mat left_undistorted;   // 給 UI 顯示、供使用者點擊
};

enum class Descriptor { OpenCV, Masked };

struct MeasureOptions {
    Descriptor descriptor = Descriptor::OpenCV;
    bool local_plane_refine = false;   // 對應 UI「視差微調」
};

struct MeasureResult {
    bool ok;
    std::string fail_reason;
    cv::Point2f right_point;           // 主對右圖匹配點
    cv::Vec3d p3d_camera;              // 融合後,左相機座標 (mm)
    cv::Vec3d p3d_camera_best;         // 主對
    cv::Vec3d p3d_world;               // 以最小 ID ArUco 為原點
    double distance_mm;                // ‖p3d‖
    double depth_mm;                   // p3d.z
    double height_signed_mm;           // 對高度參考平面的有號距離
    int    height_plane_source;        // Shared Pattern / Marker Pose / Legacy
    double reprojection_error_px;
    double d_epi, zncc, masked_score, confidence;
    int    fused_count, fused_dropped;
    std::string method;                // 例 "Region-SIFT(21/28)+CoarseFine(expand=0)"
};

class Engine {
public:
    Engine(const Calibration&, const EngineConfig&);
    SessionInfo analyze(FrameSource&);                          // Session 層
    MeasureResult measure(float u, float v, const MeasureOptions&);  // Click 層
};

}  // namespace rsift
```

介面上要先決定的兩件事(見第 8 節):**影格如何從 APP 傳進來**、**校正參數從哪來**。

---

## 5. 給 C++ 實作者的數值陷阱

這些是 NumPy/Python 與 C++ 語意不同、會悄悄改變結果的地方(完整清單與行號見
`scratchpad/pipeline_algorithm.md` 第 3.3 節,移植時應把該節併入正式規格):

1. **四捨五入是 half-to-even。** Python `round()` 與 `np.rint` 都是;C++ 要用
   `std::nearbyint`/`std::rint`,**不能用** `std::round`/`lround`/`floor(x+0.5)`。
   實例:擴張輪點數 46×0.75=34.5,Python 得 34,`std::round` 得 35。
2. **負數取模。** Python `%` 的結果符號跟隨除數;C++ `%`/`fmod` 不同。`(peak-1) % bins`、
   `(angle - dir) % 360`、`_wrap_degrees` 都要自寫 Python 語意的取模。
3. **平手規則。** `np.argmax`/`argmin` 取第一個;多處使用穩定排序(`mergesort`/`stable`),
   C++ 用 `std::stable_sort`。候選順序由 `_union_offsets` 的 (along, across) 字典序決定,影響平手結果。
4. **float32/float64 混用。** OpenCV 模式金字塔與梯度是 float32、自製模式是 float64;
   `initial_blur` 用單精度 sqrt;frames 輸出存成 float32 後才進評分。C++ 要逐處對齊型別。
5. **累加順序。** 方向直方圖是低 bin 與高 bin 各自循序累加後再相加;自製 descriptor 的四個空間角落
   各用獨立暫存累加。
6. **自製 descriptor 的 `cell_width` 必須用呼叫端逐點傳入的 size**,不能用 context 量化後的
   `entry['size']`。這是 Python Numba 版剛踩過的 bug:真實流程的 size 是 float32,upcast 後與
   entry size 有約 1e-5 差異。
7. `warpPerspective` 傳入的是 `H_rl` 且沒有 `WARP_INVERSE_MAP`;`_centered_offsets(95, 2.0)` 會強制補入 0,
   得到 49 個偏移而不是 48 個。

效能提示(不改變數值):OpenCV 模式目前每批約 146 個候選就呼叫一次 `SIFT.compute`,
每次都在整張 warp 圖上重建內部金字塔;C++ 可合併成較少次呼叫。**合併前需驗證 descriptor 不因批次組成而改變。**

---

## 6. 驗證策略

**黃金參照**:Python 版,設定固定為 Reject SpecPts 關閉;自製模式要用修正後的版本
(見第 8 節第 7 點)。

**分層比對**,每層都用 Python 的輸出當輸入,把誤差隔離在單一層:

| 層 | 比對內容 | 建議容差(初值,需確認) |
|---|---|---|
| 去畸變 | 去畸變影像、新內參 | 影像逐像素相同;內參 < 1e-9 |
| 反光遮罩 | 空間遮罩 | 像素差異 < 0.1% |
| 位姿分析 | 選中影格、R_rel、t_rel、平面 | 影格索引相同;旋轉 < 0.01°、平移 < 0.05 mm |
| Region-SIFT | 右圖匹配點 | 見下 |
| 三角化以後 | p3d、距離、高度 | 匹配點相同時 < 0.01 mm |

**匹配點不適合用單純容差比對。** Region-SIFT 是在離散候選中挑最佳,極小的浮點差異可能讓接近平手的
兩個候選交換名次,匹配點就會跳 1 px 以上。建議判定方式:
- 大多數點擊(例如 ≥99%)匹配點必須相同(差 < 1e-3 px);
- 不同的案例逐一檢查 Python 版的 ObjRatio/分數差,只有「在 Python 裡本來就接近平手」才可接受。

**現成可轉成測試向量的資源**:`tests/test_region_sift_matching.py`、`test_region_sift_frames.py`
(含與原始逐點演算法的逐位元比對)、`test_region_sift_search.py`、`test_region_sift_scoring.py`、
`test_region_sift_refinement.py`、`test_masked_sift_descriptor.py`。

---

## 7. 分期建議

| 期 | 內容 | 產出 | 說明 |
|---|---|---|---|
| **0** | Python 端準備 | ① ~~修正 Numba descriptor bug~~(已完成);② 把散在 MAIN UI 閉包裡的量測後處理(C1、C3-C9、S5)抽成不依賴 UI 的純函式模組放進 `Algorithm/`;③ 無頭(headless)黃金資料輸出腳本:輸入影片 + 點擊座標清單,輸出每層中間結果 | ② 讓 Python 端有一份乾淨、可逐行對照的規格;③ 是後面所有驗證的基礎 |
| **1** | C++ Click 層(OpenCV 模式) | Region-SIFT 引擎 + 三角化 + 融合 + 高度 + 品質指標 | Session 資料先直接讀 Phase 0 的黃金資料,**不依賴 POSE**,可在桌機先驗證。風險最低、最快看到成果 |
| **2** | 自製模式 + 反光遮罩 + 視差微調 | masked backend、block-adaptive 空間遮罩、`local_plane_refine` | 與 Phase 3 可平行 |
| **3** | POSE 移植 | Session 層完整流程 | **先做一輪範圍界定**:追出預設設定下實際執行的子集、估行數、決定 LM 求解器(Ceres 或自寫)。這一期最大、最不確定 |
| **4** | Android 整合 | NDK/CMake、OpenCV for Android、JNI、Kotlin 包裝、`FrameSource` 實作、裝置上效能與精度測試 | 需要用**手機實際錄的影片**重新驗證精度(見第 8 節第 2 點) |

---

## 8. 風險與待決事項

1. **POSE 是最大風險。** 6929 行,含 marker map 自標定、時序 DP、多段評分與 `scipy.least_squares`
   聯合最佳化。Ceres 與 scipy 的 trf 演算法不同,收斂結果只能做到容差內一致。
   Phase 3 開始前需要一份專門的範圍界定。
   - 有人可能會提議改用 ARCore 位姿取代 POSE。**不建議當成直接替代**:那是完全不同的演算法,
     精度特性未知,量測結果會與現有版本不同,等於要重新驗證整個量測系統。
2. **錄影編碼。** 目前是 **FFV1 無損**(MAIN:1934-1938),手機一般錄 H.264/HEVC 有損壓縮。
   壓縮失真可能影響 ArUco 角點次像素精度(→ 位姿 → 深度)與 SIFT 匹配。需要評估:
   用手機擷取原始 YUV 影格,或驗證有損壓縮下精度仍可接受。
3. **手機相機校正。** 現有校正檔 `calibration_result_Zebra_1_monocular.json`(1920×1080)是目前相機的。
   換成手機相機就要重新校正,且**校正解析度必須等於影格解析度**——目前主流程沒有解析度不符的保護。
4. **影格傳入方式與記憶體。** POSE 需要隨機存取影片中的特定影格(probe、局部視窗、光流鄰格)。
   1920×1080 BGR 每格約 6 MB,不可能整段放進記憶體。選項:APP 用 MediaCodec 按需解碼(有損影片隨機
   seek 較慢)、或先寫入磁碟快取。這會決定 `FrameSource` 的實作方式。
5. **OpenCV 版本。** Python 端是 OpenCV 4.13.0。Android 端應使用相同版本(必要時自行以 NDK 編譯),
   否則 `SIFT.compute` 等結果可能不同;若無法同版本,黃金參照要用 Android 端的版本重新產生。
6. **容差數值**(第 6 節)是建議初值,需要你依量測精度需求確認。
7. **Python 端 Numba bug 已於 2026-09-30 修正。** 自製 descriptor 的 Numba 路徑曾有數值偏差
   (第 5 節第 6 點),已改為逐點 size,並經獨立驗證與 NumPy 參考路徑逐位元相同。
   **2026-09-30 之前用 Numba 版跑出的自製模式結果不能當黃金參照**,要重跑。
8. **Reject SpecPts 關閉會改變 OpenCV 模式目前的預設行為**,黃金資料要用關閉的設定重新產生,
   不能直接拿現有量測紀錄比對。

---

## 附:本規劃依據的研究報告

- 量測主流程:`scratchpad/pipeline_measure.md`
- 幾何與輸入準備:`scratchpad/pipeline_geometry.md`
- Region-SIFT 模組依賴與數值陷阱:`scratchpad/pipeline_algorithm.md`

(scratchpad 為 Claude session 暫存目錄,若需長期保存請將三份報告一併移入專案。)
