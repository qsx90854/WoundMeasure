fixed_camera_ippe_diag_v5_7
2026-08-20

目的
----
針對 4K 螢幕 Pattern 驗證修正三個問題：
1) ID2 + ID5 明明同時可見，但 marker-map support 不足時 ID5 被丟掉。
2) ArUco raw quad corner 在 LCD / moire 畫面可能有 systematic bias。
3) 原本只有固定 cornerSubPix 5x5，無法比較其他 corner refinement。

執行
----
python rt_two_video_validation_ui_v5_7_multi_marker_corner_tuning.py

請將 Algorithm/video_pose_analysis_temporal_unified_pattern_guided_local_window_validation.py
覆蓋到你專案的 Algorithm 目錄。production 的
video_pose_analysis_temporal_unified_pattern_guided_local_window.py 不需要覆蓋。

v5.7 主要修改
------------
A. Multi-marker map rescue
- 先使用 production 原本 >=3 support 規則。
- 若可見 ID 有多顆但 map 只留下部分 ID，額外掃 core frame 周圍 +/-3 frame，僅供建立 rigid marker map。
- 若掃描後仍是 3 candidate / 2 consistent 這類情況，validation 才允許 2-frame consensus；rotation / translation residual gate 仍保留。
- temporal path / pair ranking 不會因 map-rescue 把所有 rescue frame 都塞入候選。
- Result 會顯示 Marker map IDs 與 strategy。

B. Corner mode UI
- Raw detectMarkers
- SubPix 3x3
- SubPix 5x5（舊版行為）
- Contour refine
- AprilTag refine

v5.7 UI 預設：Contour refine + LCD Robust。
若要完全重現 v5.6，選 SubPix 5x5（原本） + OpenCV Default。

C. Detector preset UI
- OpenCV Default
- LCD Robust
- LCD Aggressive

LCD Robust 主要放大 adaptive threshold window，降低 4K LCD 子像素條紋 / moire
對小 window threshold 的干擾。每次分析都會把實際 DetectorParameters 寫進 log/result。

D. 顯示模式
- 原始彩色
- 偵測灰階
- Adaptive binary（診斷）
  注意：這張 binary 是用目前 preset 中間大小的 adaptive window 產生的代表性診斷圖；
  ArUco detector 內部實際會測多個 adaptive threshold windows。

Overlay
-------
黃色 x = CORNER_REFINE_NONE 的 raw detectMarkers quad corner
藍色 + = 目前 Corner mode 真正送入 pose estimator 的 selected corner
角點是在 zoom/resize 完成後才畫，因此線寬不會隨 zoom 放大。

結果頁新增
----------
Corner mode: ... | Detector preset: ...
Pose measurement: A=DUAL_PLANAR IDs=[2,5] | B=DUAL_PLANAR IDs=[2,5]
Marker map: IDs=[2,5] | strategy=...

Marker-map strategy 可能值：
- core_production
- neighbor_rescue_production
- neighbor_rescue_relaxed_2frame

建議第一輪比較
------------
1) Contour refine + LCD Robust（v5.7 預設）
2) AprilTag refine + LCD Robust
3) SubPix 5x5 + OpenCV Default（v5.6 baseline）
4) Raw detectMarkers + OpenCV Default

請比較：
- Pose measurement 是否為 DUAL_PLANAR [2,5]
- rotation change / plane-normal angle
- ||tB-tA||
- ||t_rel||
- RAW -> SELECTED corner shift
