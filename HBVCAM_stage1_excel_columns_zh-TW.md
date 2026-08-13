# HBVCAM Stage 1 Excel 分頁與欄位中文說明

適用程式：`analyze_hbvcam_stage1_sift_comparison.py`

本文件對應目前程式輸出的 9 個 Excel 分頁。欄位名稱以實際 Excel 第一列為準。

## 先看這些重要規則

- `marker` 或 `marker_only`：只使用 ArUco 四角點與 IPPE 計算 RT，不使用 SIFT。
- `sift` 或 `sift_assisted`：使用 ArUco，加上 SIFT 的分支選擇或聯合 RT 精修。
- `baseline_delta_mm = algorithm_baseline_mm - json_baseline_mm`。
  - 正值：演算法算出的 baseline 比 JSON 大。
  - 負值：演算法算出的 baseline 比 JSON 小。
- `baseline_error_percent` 是有正負號的誤差百分點；`absolute_baseline_error_percent` 是其絕對值。
- 名稱含 `sift_improvement` 的欄位均為「ArUco-only 誤差 − SIFT-assisted 誤差」。正值代表 SIFT 改善，負值代表 SIFT 反而變差。
- 合格判定使用嚴格小於：`rotation_error_deg < Settings!B2`，而且 `absolute_baseline_error_percent < Settings!B3`。
- `Frame Results` 的誤差百分比以百分點儲存，例如 `5` 代表 5%。
- `Distance Pass Rate`、`Video Pass Rate` 的通過率公式以小數比例儲存，例如 `0.8` 在 Excel 顯示為 80%。
- 統計後綴：
  - `_mean`：平均值。
  - `_median`：中位數。
  - `_std`：樣本標準差。
  - `_p95`：第 95 百分位數。
  - `_count`：參與該項比較的樣本數。
  - `_frames`：幀數。
  - `_percent`：百分比或百分點，依上面規則判讀。

## 1. Settings（合格門檻設定）

修改這個分頁的 B2、B3，`Distance Pass Rate` 與 `Video Pass Rate` 的公式結果會跟著更新。

| 英文標題 | 中文意思 | 補充說明 |
|---|---|---|
| `parameter` | 參數名稱 | 設定項目的英文名稱 |
| `value` | 設定值 | 可修改的門檻數值 |
| `unit` | 單位 | `deg` 為度，`%` 為百分比 |
| `description` | 說明 | 參數用途與修改方式 |

### Settings 參數名稱

| 英文參數 | 中文意思 |
|---|---|
| `rotation_error_threshold_deg` | 旋轉誤差合格門檻，單位為度；預設 2 度 |
| `absolute_baseline_error_threshold_percent` | baseline 絕對誤差百分比合格門檻；預設 5% |

## 2. Frame Results（逐幀完整結果）

每支影片、每個 frame、每種演算法各占一列。這是最完整的原始分析結果。

| 英文標題 | 中文意思 | 單位／判讀方式 |
|---|---|---|
| `distance_cm` | 拍攝距離 | cm |
| `angle_deg` | 拍攝角度 | 度；由影片檔名解析 |
| `repeat` | 同距離下的影片編號 | 例如第 1～6 支 |
| `scene_type` | 場景類型 | `object_in_roi` 或 `pattern_only` |
| `video_file` | 影片檔名 | 不含資料夾路徑 |
| `video_path` | 影片完整路徑 | 來源檔案位置 |
| `frame_index` | 影片幀索引 | 從 0 開始 |
| `mode` | RT 計算模式 | `marker_only` 或 `sift_assisted` |
| `status` | 計算狀態 | `OK`、`QUALITY_WARNING` 或 `FAILED` |
| `failure_reason` | 失敗原因 | 成功時通常空白 |
| `rotation_error_deg` | 估計旋轉相對 JSON 外參的角度誤差 | 度，越小越好 |
| `translation_l2_error_mm` | 估計平移向量與 JSON 平移向量的 L2 距離 | mm，包含長度與方向差異 |
| `translation_direction_error_deg` | 估計平移方向與 JSON 平移方向的夾角 | 度，越小越好 |
| `algorithm_baseline_mm` | 演算法估計的 baseline | mm，即估計平移向量長度 |
| `json_baseline_mm` | JSON 外參提供的參考 baseline | mm，僅用來對答案 |
| `baseline_delta_mm` | 有正負號的 baseline 差值 | 演算法值減 JSON 值，mm |
| `absolute_baseline_delta_mm` | baseline 差值絕對值 | mm，越小越好 |
| `baseline_error_percent` | 有正負號的 baseline 誤差百分比 | `(演算法−JSON)/JSON×100`，5 代表 5% |
| `absolute_baseline_error_percent` | baseline 誤差百分比絕對值 | 百分點，越小越好；合格公式使用此欄 |
| `shared_marker_count` | 左右畫面共同偵測到的 ArUco 數量 | 個 |
| `marker_id` | 本幀主要使用的 ArUco ID | 整數 ID |
| `marker_side_left_px` | 左畫面 ArUco 的平均邊長 | pixel |
| `marker_side_right_px` | 右畫面 ArUco 的平均邊長 | pixel |
| `marker_area_left_percent` | ArUco 在左畫面所占面積比例 | 百分點 |
| `marker_area_right_percent` | ArUco 在右畫面所占面積比例 | 百分點 |
| `ippe_branch_count_left` | 左畫面 IPPE 有效姿態分支數 | 通常為 2 |
| `ippe_branch_count_right` | 右畫面 IPPE 有效姿態分支數 | 通常為 2 |
| `ippe_branch_left` | 最後採用的左畫面 IPPE 分支索引 | 從 0 開始 |
| `ippe_branch_right` | 最後採用的右畫面 IPPE 分支索引 | 從 0 開始 |
| `marker_self_reproj_left_px` | 左畫面標記姿態對自身四角點的重投影誤差 | pixel，越小越好 |
| `marker_self_reproj_right_px` | 右畫面標記姿態對自身四角點的重投影誤差 | pixel，越小越好 |
| `marker_bidirectional_rms_px` | 使用最終 RT 將標記左右互投後的雙向 RMS 重投影誤差 | pixel，越小越好 |
| `marker_bidirectional_max_px` | 標記雙向重投影誤差中的最大值 | pixel，越小越好 |
| `feature_match_count` | 通過初步配對的 SIFT 特徵數 | 個；marker-only 通常空白 |
| `feature_inlier_count` | 幾何驗證後保留的 SIFT 內點數 | 個 |
| `feature_inlier_ratio` | SIFT 內點占配對點的比例 | 0～1，越高通常越穩定 |
| `feature_median_px` | 最終 SIFT 內點極線殘差中位數 | pixel，越小越好 |
| `feature_p90_px` | 最終 SIFT 內點極線殘差第 90 百分位數 | pixel，越小越好 |
| `feature_grid_coverage` | SIFT 內點涵蓋影像網格的比例 | 0～1，越大表示分布較廣 |
| `feature_hull_coverage` | SIFT 內點凸包涵蓋影像面積的比例 | 0～1，越大表示分布較廣 |
| `feature_parallax_deg` | SIFT 特徵對應所提供的視差角 | 度 |
| `rt_sift_applied` | 最終 RT 是否採用了 SIFT 聯合精修結果 | `TRUE`／`FALSE` |
| `rt_sift_role` | 被選中的 RT 候選角色 | 例如 `final_rt` 或聯合精修候選名稱 |
| `rt_reliable` | 最終 RT 是否通過程式內部可靠性判定 | `TRUE`／`FALSE` |
| `processing_time_s` | 此列 RT 計算時間 | 秒 |

## 3. Paired Frames（同幀兩種演算法對照）

把同一影片、同一 frame 的 ArUco-only 與 SIFT-assisted 結果放在同一列，方便直接比較。

| 英文標題 | 中文意思 | 補充說明 |
|---|---|---|
| `distance_cm` | 拍攝距離 | cm |
| `repeat` | 同距離下的影片編號 | 第幾支影片 |
| `scene_type` | 場景類型 | 有物體或只有 pattern |
| `video_file` | 影片檔名 | 來源影片 |
| `frame_index` | 幀索引 | 從 0 開始 |
| `marker_status` | ArUco-only 計算狀態 | `OK`／`QUALITY_WARNING`／`FAILED` |
| `sift_status` | SIFT-assisted 計算狀態 | `OK`／`QUALITY_WARNING`／`FAILED` |
| `marker_rotation_error_deg` | ArUco-only 旋轉誤差 | 度 |
| `sift_rotation_error_deg` | SIFT-assisted 旋轉誤差 | 度 |
| `rotation_sift_improvement_deg` | SIFT 帶來的旋轉誤差改善量 | marker 誤差減 sift 誤差；正值較好 |
| `marker_abs_baseline_error_mm` | ArUco-only baseline 絕對誤差 | mm |
| `sift_abs_baseline_error_mm` | SIFT-assisted baseline 絕對誤差 | mm |
| `baseline_sift_improvement_mm` | SIFT 帶來的 baseline 絕對誤差改善量 | marker 誤差減 sift 誤差；正值較好 |
| `marker_abs_baseline_error_percent` | ArUco-only baseline 絕對誤差百分比 | 百分點 |
| `sift_abs_baseline_error_percent` | SIFT-assisted baseline 絕對誤差百分比 | 百分點 |
| `baseline_percent_sift_improvement` | SIFT 帶來的 baseline 百分比誤差改善量 | marker 誤差減 sift 誤差；正值較好 |
| `sift_feature_match_count` | SIFT 初步配對數 | 個 |
| `sift_feature_inlier_count` | SIFT 幾何內點數 | 個 |
| `sift_joint_refined` | 是否採用 SIFT 聯合 RT 精修 | `TRUE`／`FALSE` |
| `sift_rt_reliable` | SIFT-assisted RT 是否通過可靠性判定 | `TRUE`／`FALSE` |

## 4. Video Summary（各影片統計摘要）

依「距離、影片編號、場景、影片、計算模式」分組，每支影片的兩種演算法各有一列。

| 英文標題 | 中文意思 |
|---|---|
| `distance_cm` | 拍攝距離，cm |
| `repeat` | 同距離下的影片編號 |
| `scene_type` | 場景類型 |
| `video_file` | 影片檔名 |
| `mode` | RT 計算模式 |
| `total_frames` | 此分組要求分析的總幀數 |
| `solved_frames` | 成功算出 baseline 的幀數 |
| `success_rate_percent` | 成功算出 baseline 的比例，百分比 |
| `quality_ok_frames` | 狀態為 `OK` 的幀數 |
| `quality_warning_frames` | 狀態為 `QUALITY_WARNING` 的幀數 |
| `failed_frames` | 狀態為 `FAILED` 的幀數 |
| `feature_supported_frames` | 有 SIFT 配對資料的幀數 |
| `sift_joint_refined_frames` | 採用 SIFT 聯合精修的幀數 |
| `rt_reliable_frames` | RT 通過可靠性判定的幀數 |
| `rotation_error_deg_mean` | 旋轉誤差平均值，度 |
| `rotation_error_deg_median` | 旋轉誤差中位數，度 |
| `rotation_error_deg_std` | 旋轉誤差樣本標準差，度 |
| `rotation_error_deg_p95` | 旋轉誤差第 95 百分位數，度 |
| `translation_l2_error_mm_mean` | 平移向量 L2 誤差平均值，mm |
| `translation_l2_error_mm_median` | 平移向量 L2 誤差中位數，mm |
| `translation_l2_error_mm_std` | 平移向量 L2 誤差樣本標準差，mm |
| `translation_l2_error_mm_p95` | 平移向量 L2 誤差第 95 百分位數，mm |
| `translation_direction_error_deg_mean` | 平移方向誤差平均值，度 |
| `translation_direction_error_deg_median` | 平移方向誤差中位數，度 |
| `translation_direction_error_deg_std` | 平移方向誤差樣本標準差，度 |
| `translation_direction_error_deg_p95` | 平移方向誤差第 95 百分位數，度 |
| `absolute_baseline_delta_mm_mean` | baseline 絕對誤差平均值，mm |
| `absolute_baseline_delta_mm_median` | baseline 絕對誤差中位數，mm |
| `absolute_baseline_delta_mm_std` | baseline 絕對誤差樣本標準差，mm |
| `absolute_baseline_delta_mm_p95` | baseline 絕對誤差第 95 百分位數，mm |
| `absolute_baseline_error_percent_mean` | baseline 絕對誤差百分比平均值 |
| `absolute_baseline_error_percent_median` | baseline 絕對誤差百分比中位數 |
| `absolute_baseline_error_percent_std` | baseline 絕對誤差百分比樣本標準差 |
| `absolute_baseline_error_percent_p95` | baseline 絕對誤差百分比第 95 百分位數 |

## 5. Distance Summary（各距離統計摘要）

依「距離、場景、計算模式」彙整所有影片與 frame。其統計欄位和 `Video Summary` 相同，但不再細分影片編號與檔名。

| 英文標題 | 中文意思 |
|---|---|
| `distance_cm` | 拍攝距離，cm |
| `scene_type` | 場景類型 |
| `mode` | RT 計算模式 |
| `total_frames` | 此距離分組要求分析的總幀數 |
| `solved_frames` | 成功算出 baseline 的幀數 |
| `success_rate_percent` | 成功算出 baseline 的比例，百分比 |
| `quality_ok_frames` | 狀態為 `OK` 的幀數 |
| `quality_warning_frames` | 狀態為 `QUALITY_WARNING` 的幀數 |
| `failed_frames` | 狀態為 `FAILED` 的幀數 |
| `feature_supported_frames` | 有 SIFT 配對資料的幀數 |
| `sift_joint_refined_frames` | 採用 SIFT 聯合精修的幀數 |
| `rt_reliable_frames` | RT 通過可靠性判定的幀數 |
| `rotation_error_deg_mean` | 旋轉誤差平均值，度 |
| `rotation_error_deg_median` | 旋轉誤差中位數，度 |
| `rotation_error_deg_std` | 旋轉誤差樣本標準差，度 |
| `rotation_error_deg_p95` | 旋轉誤差第 95 百分位數，度 |
| `translation_l2_error_mm_mean` | 平移向量 L2 誤差平均值，mm |
| `translation_l2_error_mm_median` | 平移向量 L2 誤差中位數，mm |
| `translation_l2_error_mm_std` | 平移向量 L2 誤差樣本標準差，mm |
| `translation_l2_error_mm_p95` | 平移向量 L2 誤差第 95 百分位數，mm |
| `translation_direction_error_deg_mean` | 平移方向誤差平均值，度 |
| `translation_direction_error_deg_median` | 平移方向誤差中位數，度 |
| `translation_direction_error_deg_std` | 平移方向誤差樣本標準差，度 |
| `translation_direction_error_deg_p95` | 平移方向誤差第 95 百分位數，度 |
| `absolute_baseline_delta_mm_mean` | baseline 絕對誤差平均值，mm |
| `absolute_baseline_delta_mm_median` | baseline 絕對誤差中位數，mm |
| `absolute_baseline_delta_mm_std` | baseline 絕對誤差樣本標準差，mm |
| `absolute_baseline_delta_mm_p95` | baseline 絕對誤差第 95 百分位數，mm |
| `absolute_baseline_error_percent_mean` | baseline 絕對誤差百分比平均值 |
| `absolute_baseline_error_percent_median` | baseline 絕對誤差百分比中位數 |
| `absolute_baseline_error_percent_std` | baseline 絕對誤差百分比樣本標準差 |
| `absolute_baseline_error_percent_p95` | baseline 絕對誤差百分比第 95 百分位數 |

## 6. SIFT Effect（SIFT 效果統計）

直接比較相同 frame 下 SIFT-assisted 相對於 ArUco-only 是否改善。`ALL` 表示合併所有距離或所有場景。

| 英文標題 | 中文意思 | 補充說明 |
|---|---|---|
| `distance_cm` | 拍攝距離 | `ALL` 表示全部距離 |
| `scene_type` | 場景類型 | `ALL` 表示全部場景 |
| `paired_requested_frames` | 同時要求兩種模式計算的配對幀數 | 不代表兩種模式都成功 |
| `marker_solved_frames` | ArUco-only 成功解算幀數 | 有算出 baseline |
| `sift_solved_frames` | SIFT-assisted 成功解算幀數 | 有算出 baseline |
| `both_solved_frames` | 兩種模式都成功的幀數 | 才能直接比較誤差 |
| `marker_only_solved_frames` | 只有 ArUco-only 成功的幀數 | SIFT-assisted 失敗 |
| `sift_only_solved_frames` | 只有 SIFT-assisted 成功的幀數 | ArUco-only 失敗 |
| `neither_solved_frames` | 兩種模式都失敗的幀數 | 無可用 RT |
| `marker_success_rate_percent` | ArUco-only 解算成功率 | 百分比 |
| `sift_success_rate_percent` | SIFT-assisted 解算成功率 | 百分比 |
| `sift_feature_supported_frames` | 有 SIFT 特徵配對資料的幀數 | 只要配對數大於 0 |
| `sift_joint_refined_frames` | 採用 SIFT 聯合 RT 精修的幀數 | `rt_sift_applied=TRUE` |
| `sift_reliable_frames` | SIFT-assisted RT 通過可靠性判定的幀數 | `rt_reliable=TRUE` |
| `rotation_error_deg_paired_count` | 可比較旋轉誤差的同幀樣本數 | 兩種模式都要有誤差值 |
| `rotation_error_deg_marker_median` | ArUco-only 旋轉誤差中位數 | 度 |
| `rotation_error_deg_sift_median` | SIFT-assisted 旋轉誤差中位數 | 度 |
| `rotation_error_deg_sift_improvement_median` | SIFT 旋轉誤差改善量中位數 | 正值代表改善，度 |
| `rotation_error_deg_sift_win_rate_percent` | SIFT 旋轉誤差小於 ArUco-only 的幀比例 | 百分比 |
| `absolute_baseline_delta_mm_paired_count` | 可比較 baseline 絕對毫米誤差的同幀樣本數 | 幀數 |
| `absolute_baseline_delta_mm_marker_median` | ArUco-only baseline 絕對誤差中位數 | mm |
| `absolute_baseline_delta_mm_sift_median` | SIFT-assisted baseline 絕對誤差中位數 | mm |
| `absolute_baseline_delta_mm_sift_improvement_median` | SIFT baseline 毫米誤差改善量中位數 | 正值代表改善，mm |
| `absolute_baseline_delta_mm_sift_win_rate_percent` | SIFT baseline 毫米誤差較小的幀比例 | 百分比 |
| `absolute_baseline_error_percent_paired_count` | 可比較 baseline 絕對百分比誤差的同幀樣本數 | 幀數 |
| `absolute_baseline_error_percent_marker_median` | ArUco-only baseline 絕對誤差百分比中位數 | 百分點 |
| `absolute_baseline_error_percent_sift_median` | SIFT-assisted baseline 絕對誤差百分比中位數 | 百分點 |
| `absolute_baseline_error_percent_sift_improvement_median` | SIFT baseline 百分比誤差改善量中位數 | 正值代表改善，百分點 |
| `absolute_baseline_error_percent_sift_win_rate_percent` | SIFT baseline 百分比誤差較小的幀比例 | 百分比 |

## 7. Distance Pass Rate（各距離合格率）

依「拍攝距離＋角度」分組。帶有 `_pass_`、`_qualified_` 的欄位是 Excel 公式，修改 `Settings` 的門檻後會重新計算。

| 英文標題 | 中文意思 |
|---|---|
| `distance_cm` | 拍攝距離，cm |
| `angle_deg` | 拍攝角度，度 |
| `requested_frames` | 此距離與角度實際要求分析的不重複幀數 |
| `marker_total_frames` | ArUco-only 結果列總數 |
| `marker_rotation_pass_frames` | ArUco-only 旋轉誤差通過門檻的幀數 |
| `marker_rotation_pass_percent` | ArUco-only 旋轉誤差通過率 |
| `marker_baseline_pass_frames` | ArUco-only baseline 絕對誤差百分比通過門檻的幀數 |
| `marker_baseline_pass_percent` | ArUco-only baseline 誤差通過率 |
| `marker_qualified_frames` | ArUco-only 同時通過旋轉與 baseline 門檻的幀數 |
| `marker_qualified_percent` | ArUco-only 同時合格率 |
| `sift_total_frames` | SIFT-assisted 結果列總數 |
| `sift_rotation_pass_frames` | SIFT-assisted 旋轉誤差通過門檻的幀數 |
| `sift_rotation_pass_percent` | SIFT-assisted 旋轉誤差通過率 |
| `sift_baseline_pass_frames` | SIFT-assisted baseline 絕對誤差百分比通過門檻的幀數 |
| `sift_baseline_pass_percent` | SIFT-assisted baseline 誤差通過率 |
| `sift_qualified_frames` | SIFT-assisted 同時通過旋轉與 baseline 門檻的幀數 |
| `sift_qualified_percent` | SIFT-assisted 同時合格率 |
| `sift_joint_refined_frames` | 採用 SIFT 聯合精修的幀數 |
| `sift_joint_refined_percent` | 採用 SIFT 聯合精修的幀數占全部 SIFT 結果列的比例 |
| `sift_joint_refined_qualified_frames` | 採用 SIFT 聯合精修且同時通過兩個門檻的幀數 |
| `sift_joint_refined_qualified_percent` | 上述幀數占全部 SIFT 結果列的比例 |
| `rotation_error_threshold_deg` | 此列公式使用的旋轉誤差門檻，連到 `Settings!B2` |
| `baseline_error_threshold_percent` | 此列公式使用的 baseline 絕對誤差百分比門檻，連到 `Settings!B3` |

## 8. Video Pass Rate（各影片合格率）

每支影片獨立統計，其合格條件與 `Distance Pass Rate` 相同。

| 英文標題 | 中文意思 |
|---|---|
| `distance_cm` | 拍攝距離，cm |
| `angle_deg` | 拍攝角度，度 |
| `repeat` | 同距離下的影片編號 |
| `scene_type` | 場景類型 |
| `video_file` | 影片檔名 |
| `requested_frames` | 此影片實際要求分析的不重複幀數 |
| `marker_total_frames` | ArUco-only 結果列總數 |
| `marker_rotation_pass_frames` | ArUco-only 旋轉誤差通過門檻的幀數 |
| `marker_rotation_pass_percent` | ArUco-only 旋轉誤差通過率 |
| `marker_baseline_pass_frames` | ArUco-only baseline 絕對誤差百分比通過門檻的幀數 |
| `marker_baseline_pass_percent` | ArUco-only baseline 誤差通過率 |
| `marker_qualified_frames` | ArUco-only 同時通過旋轉與 baseline 門檻的幀數 |
| `marker_qualified_percent` | ArUco-only 同時合格率 |
| `sift_total_frames` | SIFT-assisted 結果列總數 |
| `sift_rotation_pass_frames` | SIFT-assisted 旋轉誤差通過門檻的幀數 |
| `sift_rotation_pass_percent` | SIFT-assisted 旋轉誤差通過率 |
| `sift_baseline_pass_frames` | SIFT-assisted baseline 絕對誤差百分比通過門檻的幀數 |
| `sift_baseline_pass_percent` | SIFT-assisted baseline 誤差通過率 |
| `sift_qualified_frames` | SIFT-assisted 同時通過旋轉與 baseline 門檻的幀數 |
| `sift_qualified_percent` | SIFT-assisted 同時合格率 |
| `sift_joint_refined_frames` | 採用 SIFT 聯合精修的幀數 |
| `sift_joint_refined_percent` | 採用 SIFT 聯合精修的幀數占全部 SIFT 結果列的比例 |
| `sift_joint_refined_qualified_frames` | 採用 SIFT 聯合精修且同時通過兩個門檻的幀數 |
| `sift_joint_refined_qualified_percent` | 上述幀數占全部 SIFT 結果列的比例 |
| `rotation_error_threshold_deg` | 此列公式使用的旋轉誤差門檻，連到 `Settings!B2` |
| `baseline_error_threshold_percent` | 此列公式使用的 baseline 絕對誤差百分比門檻，連到 `Settings!B3` |

## 9. Protocol（分析設定與可重現資訊）

| 英文標題 | 中文意思 |
|---|---|
| `parameter` | 分析設定或流程參數名稱 |
| `value` | 該參數在本次執行時的實際值 |

### Protocol 參數名稱

| 英文參數 | 中文意思 |
|---|---|
| `generated_at` | Excel 產生時間 |
| `source_folder` | 來源影片資料夾 |
| `calibration_file` | 使用的相機標定 JSON 檔 |
| `video_name_filter` | 用來選取影片檔名的正規表示式 |
| `frames` | 本次分析的 frame 範圍 |
| `repeat_1_to_5` | 第 1～5 支影片的場景分類規則 |
| `repeat_6` | 第 6 支影片的場景分類規則 |
| `marker_only` | ArUco-only 模式的流程說明 |
| `sift_assisted` | SIFT-assisted 模式的流程說明 |
| `JSON usage` | JSON 外參在流程中的用途；只用於算誤差，不參與選解 |
| `ROI mode` | 本次 ArUco 與 SIFT 使用的 ROI 模式 |
| `left_roi` | 左相機實際使用的 ROI 座標範圍 |
| `right_roi` | 右相機實際使用的 ROI 座標範圍 |
| `ArUco detection pixels` | ArUco 初始偵測使用哪一種影像像素 |
| `ArUco subpixel pixels` | ArUco 亞像素角點精修使用哪一種影像像素 |
| `ArUco adaptive subpixel half-window` | 自適應 `cornerSubPix` 半視窗計算規則 |
| `ArUco stability rejection` | 不同亞像素視窗結果差異過大時的拒絕門檻 |
| `rotation_error` | 旋轉誤差計算公式 |
| `baseline_error_percent` | baseline 有正負號誤差百分比公式 |
| `qualified_rotation_error_deg` | 旋轉誤差合格條件及其 Settings 儲存格 |
| `qualified_absolute_baseline_error_percent` | baseline 絕對誤差百分比合格條件及其 Settings 儲存格 |
| `ArUco diagnostic video` | 角點診斷影片的影像內容與尺寸 |
| `ArUco diagnostic padding ratio` | ArUco 周圍裁切範圍的外擴比例 |
| `ArUco diagnostic output folder` | 角點診斷影片輸出資料夾 |

## 常見文字值對照

| 英文值 | 中文意思 |
|---|---|
| `object_in_roi` | ROI／畫面內除了 ArUco pattern，還有待測物或其他可供 SIFT 使用的紋理 |
| `pattern_only` | 畫面主要只有 ArUco pattern，通常是每個距離的第 6 支影片 |
| `marker_only` | 只使用 ArUco／IPPE 的 RT 解算 |
| `sift_assisted` | 使用 SIFT 輔助分支選擇或聯合 RT 精修 |
| `OK` | 成功且通過程式內部品質檢查 |
| `QUALITY_WARNING` | 有算出結果，但部分內部品質指標警告 |
| `FAILED` | 未成功算出可用結果 |
| `TRUE` | 是／有套用／通過 |
| `FALSE` | 否／未套用／未通過 |
| `ALL` | 合併全部距離或全部場景的統計 |

