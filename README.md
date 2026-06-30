# Wound Measure / Multi-ArUco Depth Measurement

This project is an experimental stereo/video depth-measurement tool for wound or surface measurement. The current main program is:

```text
depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py
```

Run it from this project folder:

```bash
python depth_measure_multi_aruco_sbs_camera_v7_demo_zebra.py
```

The program opens a UI for selecting camera/video input, estimating relative pose from ArUco markers, matching a clicked point between left/right frames, triangulating depth, and saving measurement results.

## Main UI Buttons

### Matching Options

| Button | Function |
| --- | --- |
| 嚴格精細匹配 | Enables the precise patch/template matching fallback. |
| 梯度 SIFT 匹配 | Enables local Grad-SIFT matching around the clicked point. |
| 強制極線對齊 | Projects the matched right-image point onto the epipolar line before triangulation. |
| 啟用 ECC 精修 | Runs ECC refinement to fine tune the matched point. |
| 手動匹配模式 | Lets the user manually select the matching point on the right image. |
| 啟用 CLAHE 增強 | Applies CLAHE contrast enhancement for display and matching input. |
| 改良匹配流程 | Uses the improved matching flow with high/mid gradient reference points. |
| 顯示匹配分數 | Shows matching/quality score information in the UI/log. |
| 色彩直方圖約束 | Filters candidate matches using local color histogram similarity. |
| 啟用 RGB-SIFT | Uses RGB-SIFT descriptors instead of grayscale SIFT. |
| Opponent-SIFT | Uses opponent-color SIFT descriptors. |
| 過濾高光反光 | Filters simple highlight/specular pixels from gradient candidate selection. |
| 進階高光過濾 | Uses HSV/MSER-style specular filtering. |
| Epi-band Search | Searches/refines the match along a small epipolar band. |
| Show Spatial | Displays the spatial specular mask overlay. |
| Show Temporal | Displays the temporal instability/specular mask overlay. |
| Reject SpecPts | Rejects candidate feature points inside specular masks. |
| Point First | Switches to the point-first matching flow. |

### Right-Side Controls

| Button | Function |
| --- | --- |
| 鎖定左圖 | Locks/unlocks the left frame. |
| 鎖定右圖 | Locks/unlocks the right frame. |
| 顯示右圖 / 隱藏右圖 | Toggles right-image panel visibility. |
| 切換 HAMMING / 使用 L2 | Switches descriptor matching norm between Hamming and L2. |
| 單次計算深度 | Runs one depth measurement for the current point. |
| 連續計算: 開/關 | Toggles continuous calculation mode. |
| 顯示梯度 SIFT 連線 | Shows/hides Grad-SIFT match connection lines. |
| 自訂平面擬合 | Starts/finishes custom plane fitting from selected 3D points. |
| HighPts: On/Off | Shows/hides high-gradient reference points and successful high-gradient matches. |
| MidPts: On/Off | Shows/hides mid-gradient reference points and successful mid-gradient matches. |
| RT Diff | Displays relative pose / RT difference diagnostics. |
| 返回主選單 | Returns to the main selection menu. |

## Point Colors

| Color | Meaning |
| --- | --- |
| Light blue | High-gradient reference candidate, not accepted as a final match. |
| Dark blue | High-gradient candidate accepted as a successful match. |
| Light orange | Mid-gradient reference candidate, not accepted as a final match. |
| Dark orange | Mid-gradient candidate accepted as a successful match. |
| Red X | Clicked point on the left image. |
| Green X | Matched point on the right image. |

## Notes

- Large videos, models, virtual environments, and generated data are ignored by `.gitignore`.
- Calibration files and test data paths are currently configured directly in the Python script.
- The code is research/prototype style and contains multiple matching modes for comparison.
