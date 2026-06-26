# WoundDetect ArUco Triangulation

這個工具用影片中的 ArUco pattern 當作局部世界座標系，讓使用者在待測物表面點選一個候選點，輸出：

- 候選點在 ArUco marker 座標系下的 3D 座標，單位為公尺
- 候選點到 ArUco pattern 平面 `z=0` 的 signed / absolute 距離
- 候選點投影到 ArUco pattern 平面的 3D 座標
- 用於三角化的 frame pair / baseline / reprojection error

## 演算法設計

1. **相機內參**
   需要 camera matrix 與 distortion coefficients。若沒有內參，三角化仍可算出不穩定的 projective 結果，但無法得到可信的公尺尺度。因此本工具要求 calibration JSON。

2. **ArUco pose as local world**
   OpenCV 估測每個 frame 的 marker pose：

   ```text
   X_camera = R_marker_to_camera * X_marker + t_marker_to_camera
   ```

   marker 邊長由 `--marker-size-m` 指定，預設 `0.01` 公尺。ArUco pattern 平面即 marker 座標系的 `z=0`。

3. **前後 frame 的 RT 與 baseline**
   對任兩個 marker-visible frames `a` 與 `b`：

   ```text
   R_ba = R_bm * R_am.T
   t_ba = t_bm - R_ba * t_am
   C_a_marker = -R_am.T * t_am
   C_b_marker = -R_bm.T * t_bm
   baseline = ||C_b_marker - C_a_marker||
   ```

4. **時序特性**
   - 只用 marker pose 合格的 frames。
   - 使用 Lucas-Kanade optical flow 追蹤使用者點選的候選點。
   - 用 forward-backward check 排除時序追蹤漂移。
   - 選 keyframes 時偏好有足夠 baseline、triangulation angle、且時間上分散的觀測。

5. **空間特性**
   - 在候選點附近 patch 內找 Shi-Tomasi feature，若使用者點的位置不是角點，會用最接近點選位置的局部特徵輔助追蹤。
   - 三角化後用 reprojection error 做 outlier rejection。
   - 輸出 projection-to-plane：`[X, Y, 0]`，距離為 `Z` 的 signed distance 與 `abs(Z)`。

## Camera calibration JSON

範例：

```json
{
  "camera_matrix": [[1200.0, 0.0, 640.0], [0.0, 1200.0, 360.0], [0.0, 0.0, 1.0]],
  "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

## 使用方式

```powershell
pip install -r requirements.txt
python aruco_triangulate_point.py --video input.mp4 --calibration camera.json --marker-size-m 0.01 --output result.json
```

操作流程：

1. 程式會先掃描影片並偵測 ArUco pose。
2. GUI 開啟後，在待測物表面點選候選點，按 `Enter` 確認。
3. 程式會追蹤該點、挑選 keyframes、做多視角三角化，並輸出 JSON。

常用參數：

- `--aruco-dict DICT_4X4_50`：指定 ArUco dictionary。
- `--marker-id 0`：若畫面中有多個 marker，可鎖定指定 ID。
- `--start-frame` / `--end-frame`：限制使用影片片段。
- `--min-baseline-m 0.002`：最小 baseline，預設 2 mm。
- `--max-reproj-px 3.0`：outlier rejection 的重投影誤差門檻。

## 注意

- marker 必須平整貼在待測物表面，且尺寸要準確量測。
- 若相機 rolling shutter、對焦呼吸、或 marker 本身彎曲，RT 與深度會受影響。
- 候選點如果位在低紋理區，追蹤會較不穩；建議點在局部可辨識紋理附近。
