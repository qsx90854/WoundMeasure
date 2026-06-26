"""
re_3D_openMVS.py
================
以 LightGlue 特徵匹配 + ArUco 絕對尺度 進行稀疏重建，
再交由 OpenMVS（DensifyPointCloud / ReconstructMesh / TextureMesh）
產生稠密點雲與網格，取代原先的 Depth Anything V2 稠密重建流程。

流程：
  Phase 1  讀取影像、undistort、LightGlue 匹配、ArUco 位姿估算、
            triangulatePoints 稀疏點雲 → 寫 COLMAP TXT 格式
  Phase 2  InterfaceCOLMAP.exe  → scene.mvs
  Phase 3  DensifyPointCloud.exe → scene_dense.mvs + 稠密 PLY
  Phase 4  ReconstructMesh.exe  → scene_mesh.ply
  Phase 5  TextureMesh.exe      （如有網格則貼紋理，可用 SKIP_TEXTURE 跳過）
  Phase 6  Open3D 讀取並顯示結果
"""

import os
import sys
import glob
import shutil
import subprocess
import configparser
import time

import cv2
import numpy as np
import onnxruntime as ort
import open3d as o3d

# ==================== 全局設定區 ====================
OPENMVS_BIN           = r"OpenMVS_Windows_x64\vc17\x64\Release"   # OpenMVS exe 目錄（相對或絕對）
IMAGE_FOLDER          = "captured_images_lego_1"                       # 輸入影像資料夾
K_INI_PATH            = "K.ini"                                    # 相機內參檔
MVS_WORKSPACE         = "mvs_workspace"                            # 工作目錄（自動建立）
ACTUAL_MARKER_SIZE_MM = 29.0                                       # ArUco 真實尺寸 (mm)
TARGET_MARKER_ID      = 12                                         # 目標 ArUco ID
TARGET_W              = 1024                                       # 統一縮放寬度
SKIP_TEXTURE          = False                                      # True 則跳過 TextureMesh
LG_ONNX_PATH          = "superpoint_lightglue_pipeline.onnx"      # LightGlue ONNX 路徑

# -------- DensifyPointCloud 品質參數 --------
DENSIFY_RESOLUTION_LEVEL = 1
# 影像縮放比例：0=全解析度(最慢/最密), 1=1/2解析度(預設), 2=1/4解析度
# 積木等小物件建TARGET_W多時建議調到 0

DENSIFY_NUMBER_VIEWS = 0
# 每個像素考慮幾個鄰近視角，0=自動（min括生效影像數, 8）
# 影像很少時可手動設為 2 或 3

DENSIFY_NUMBER_VIEWS_FUSE = 2
# 至少需要幾個視角同意才納入稠密點雲；影像數少時建議保持 2

DENSIFY_FILTER_POINT = 2.5
# 深度圖點筋選阈値（偏移倒数）；調大則保留更多點（建護 1.5~3.5）

DENSIFY_FILTER_PHOTO = 3.0
# 光度一致性阙値（像素）；調大則對綋理平坦面化容奈高（建護 2.0~5.0）

DENSIFY_ESTIMATE_COLORS = 1
# 是否估算點顏色：1=是, 0=否

DENSIFY_ESTIMATE_NORMALS = 2
# 點雲法線估算精度：0=不估算, 1=估算, 2=估算+強化（建護網格化前流程用）

DENSIFY_MIN_TRIANGULATION_ANGLE = 3.0
# 三角測距最小夹角（度）；調小可讓小殿角度的影像對也能貢獻（建護 1.0~5.0）

# -------- ReconstructMesh 品質參數 --------
MESH_QUALITY_FACTOR = 2.5
# 網格精細度係數；越高三角面越小/越密（預設 1.0，建護 1.5~3.0）

MESH_SMOOTH = 2
# 網格表面平滑迭代次數；貼饮物件用低値保留尖銃邊緣（建護 0~3）

MESH_DECIMATE = 1.0
# 網格簡化區段：1.0=不簡化保留全部網格, 0.5=保留一半

MESH_CLOSE_HOLES = 30
# 自動填補小尺寸缺口（加大可填補越大的空洞）

MESH_REMOVE_SPURIOUS = 20
# 移除懸浮小片段（面積小於此倦數的革面片段會被移除）
# ===================================================


# ─────────────────────────── 工具函式 ───────────────────────────

def load_raw_camera_params(ini_path):
    config = configparser.ConfigParser()
    config.read(ini_path)
    fx = float(config['intrinsic1']['fx'])
    fy = float(config['intrinsic1']['fy'])
    cx = float(config['intrinsic1']['cx'])
    cy = float(config['intrinsic1']['cy'])
    k1 = float(config['distortion1']['k1'])
    k2 = float(config['distortion1']['k2'])
    k3 = float(config['distortion1']['k3'])
    p1 = float(config['distortion1']['p1'])
    p2 = float(config['distortion1']['p2'])
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, dist_coeffs


def get_aruco_pose(image_gray, K, marker_size_mm, target_id):
    """回傳 (R, tvec) 或 (None, None)"""
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(image_gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image_gray, dict_4x4, parameters=cv2.aruco.DetectorParameters_create())

    if ids is None or target_id not in ids:
        return None, None

    idx = np.where(ids == target_id)[0][0]
    half = marker_size_mm / 2.0
    obj_pts = np.array([[-half, half, 0], [half, half, 0],
                         [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[idx][0], K, np.zeros(5))
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        return R.astype(np.float64), tvec.astype(np.float64)
    return None, None


def run_lightglue(lg_session, imgA_gray, imgB_gray):
    """回傳 (ptsA, ptsB)，均為 float32 shape=(N,2)，或 (None, None)"""
    t0 = imgA_gray.astype(np.float32) / 255.0
    t1 = imgB_gray.astype(np.float32) / 255.0
    inp = np.expand_dims(np.stack([t0, t1], axis=0), axis=1)
    t_s = time.perf_counter()
    kpts, matches, scores = lg_session.run(['keypoints', 'matches', 'mscores'], {"images": inp})
    print(f"  LightGlue: {(time.perf_counter()-t_s)*1000:.1f} ms, {len(matches)} matches found")

    valid = [(kpts[0, int(m[1])], kpts[1, int(m[2])]) for m, s in zip(matches, scores) if s > 0.5]
    if len(valid) < 10:
        return None, None
    return (np.array([v[0] for v in valid], dtype=np.float32),
            np.array([v[1] for v in valid], dtype=np.float32))


def rotation_to_quaternion(R):
    """旋轉矩陣 → (qw, qx, qy, qz)"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return w, x, y, z


def run_openmvs_exe(exe_name, args, cwd, env):
    """呼叫一個 OpenMVS 可執行檔，回傳 returncode"""
    exe_path = os.path.abspath(os.path.join(OPENMVS_BIN, exe_name))
    cmd = [exe_path] + args
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


# ─────────────────────────── Phase 1 ────────────────────────────

def phase1_build_colmap(image_paths, K_orig, dist_coeffs, lg_session, workspace):
    """
    undistort 影像、LightGlue 匹配、ArUco 位姿、三角測距稀疏點雲 →
    寫出 COLMAP TXT 格式 (cameras.txt / images.txt / points3D.txt)
    回傳：(成功的 image indices 列表, K_scaled, timing_dict)，或失敗時 (None, None, {})
    """
    images_dir = os.path.join(workspace, "images")
    sparse_dir = os.path.join(workspace, "sparse")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace, "debug_matches"), exist_ok=True)
    ph1_timing = {}   # 各子步驟耗時（秒）

    # 1-a  undistort & resize
    _t = time.perf_counter()
    orig_w = orig_h = TARGET_H = None
    scale = None
    processed_bgr  = []
    processed_gray = []
    saved_names    = []

    for path in image_paths:
        img_raw   = cv2.imread(path)
        if img_raw is None:
            continue
        img_undist = cv2.undistort(img_raw, K_orig, dist_coeffs)

        if orig_w is None:
            orig_h, orig_w = img_raw.shape[:2]
            scale    = TARGET_W / orig_w
            TARGET_H = int(orig_h * scale)

        img_res = cv2.resize(img_undist, (TARGET_W, TARGET_H))
        processed_bgr.append(img_res)
        processed_gray.append(cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY))

        fname = os.path.basename(path)
        cv2.imwrite(os.path.join(images_dir, fname), img_res)
        saved_names.append(fname)

    ph1_timing["undistort & resize"] = time.perf_counter() - _t

    if len(processed_bgr) < 2:
        print("❌ 有效影像不足 (至少需要 2 張)。")
        return None, None, {}

    K_scaled = K_orig.copy().astype(np.float64)
    K_scaled[0, 0] *= scale;  K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale;  K_scaled[1, 2] *= scale

    # 1-b  ArUco 位姿估算
    _t = time.perf_counter()
    poses_R = []
    poses_t = []
    poses_C = []
    valid_indices = []

    for i, gray in enumerate(processed_gray):
        R_cam, tvec = get_aruco_pose(gray, K_scaled, ACTUAL_MARKER_SIZE_MM, TARGET_MARKER_ID)
        if R_cam is None:
            print(f"  ⚠️ 影像 {i} ({saved_names[i]}) 找不到 ArUco，跳過。")
            poses_R.append(None); poses_t.append(None); poses_C.append(None)
        else:
            C_world = -R_cam.T @ tvec
            poses_R.append(R_cam)
            poses_t.append(tvec.flatten())
            poses_C.append(C_world.flatten())
            valid_indices.append(i)

    ph1_timing["ArUco 位姿估算"] = time.perf_counter() - _t

    if len(valid_indices) < 2:
        print("❌ 含 ArUco 的有效影像不足 (至少需要 2 張)。")
        return None, None, ph1_timing

    print(f"  共 {len(valid_indices)} 張影像有 ArUco 位姿。")

    # 1-c  全組合 LightGlue 匹配 + 三角測距
    import itertools
    all_points3d  = []
    all_colors    = []
    all_obs       = []
    t_lg_total    = 0.0   # LightGlue 推理累計
    t_tri_total   = 0.0   # 三角測距累計

    for ia, ib in itertools.combinations(valid_indices, 2):
        print(f"  → LightGlue: 影像 {ia} ↔ {ib}")

        _t_lg = time.perf_counter()
        ptsA, ptsB = run_lightglue(lg_session, processed_gray[ia], processed_gray[ib])
        t_lg_total += time.perf_counter() - _t_lg

        if ptsA is None:
            print(f"    ↳ 匹配點不足，跳過")
            continue

        # 繪製匹配連線（debug）
        h, w = processed_gray[ia].shape
        vis = np.zeros((h, w*2, 3), dtype=np.uint8)
        vis[:, :w]  = cv2.cvtColor(processed_gray[ia], cv2.COLOR_GRAY2BGR)
        vis[:, w:]  = cv2.cvtColor(processed_gray[ib], cv2.COLOR_GRAY2BGR)
        for pa, pb in zip(ptsA, ptsB):
            c = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0])+w, int(pb[1])), c, 1)
        cv2.imwrite(os.path.join(workspace, "debug_matches", f"pair_{ia}_{ib}.jpg"), vis)

        # 三角測距
        _t_tri = time.perf_counter()
        Ra = poses_R[ia];  ta = poses_t[ia].reshape(3, 1)
        Rb = poses_R[ib];  tb = poses_t[ib].reshape(3, 1)
        Pa = (K_scaled @ np.hstack((Ra, ta))).astype(np.float32)
        Pb = (K_scaled @ np.hstack((Rb, tb))).astype(np.float32)

        pts4d = cv2.triangulatePoints(Pa, Pb, ptsA.T, ptsB.T)
        pts3d_world = (pts4d[:3, :] / pts4d[3, :]).T

        pts3d_inA = (Ra @ pts3d_world.T + ta)
        good = pts3d_inA[2, :] > 0

        added = 0
        for k in range(len(ptsA)):
            if not good[k]:
                continue
            Xw = pts3d_world[k]
            ua = int(np.clip(ptsA[k][0], 0, TARGET_W-1))
            va = int(np.clip(ptsA[k][1], 0, TARGET_H-1))
            r, g, b = processed_bgr[ia][va, ua, 2], processed_bgr[ia][va, ua, 1], processed_bgr[ia][va, ua, 0]
            all_points3d.append(Xw.tolist())
            all_colors.append((r, g, b))
            all_obs.append([(ia, ptsA[k][0], ptsA[k][1]), (ib, ptsB[k][0], ptsB[k][1])])
            added += 1
        t_tri_total += time.perf_counter() - _t_tri
        print(f"    ↳ 加入 {added} 個三角點")

    ph1_timing["LightGlue 推理（全對）"] = t_lg_total
    ph1_timing["三角測距（全對）"]        = t_tri_total

    if len(all_points3d) == 0:
        print("❌ 三角測距失敗，無法建立稀疏點雲。")
        return None, None, ph1_timing

    print(f"  稀疏點雲：{len(all_points3d)} 個 3D 點")

    # 1-d ~ 1-g  寫 COLMAP TXT
    _t = time.perf_counter()
    colmap_id_map = {orig_idx: (col_idx+1) for col_idx, orig_idx in enumerate(valid_indices)}

    cam_w, cam_h = TARGET_W, TARGET_H
    fx, fy = K_scaled[0,0], K_scaled[1,1]
    cx, cy = K_scaled[0,2], K_scaled[1,2]
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {cam_w} {cam_h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for orig_idx in valid_indices:
            col_id = colmap_id_map[orig_idx]
            R = poses_R[orig_idx]
            t = poses_t[orig_idx]
            qw, qx, qy, qz = rotation_to_quaternion(R)
            f.write(f"{col_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {saved_names[orig_idx]}\n")
            f.write("\n")

    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for pt_id, (xyz, rgb, obs) in enumerate(zip(all_points3d, all_colors, all_obs), start=1):
            track_str = ""
            for (orig_idx, u, v) in obs:
                if orig_idx in colmap_id_map:
                    track_str += f" {colmap_id_map[orig_idx]} 0"
            r, g, b = rgb
            f.write(f"{pt_id} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                    f"{r} {g} {b} 1.0{track_str}\n")

    ph1_timing["寫 COLMAP TXT"] = time.perf_counter() - _t
    print(f"  ✅ COLMAP TXT 格式已寫出至 {sparse_dir}/")
    return valid_indices, K_scaled, ph1_timing


def _fmt(sec):
    """將秒數格式化為易讀字串"""
    if sec < 0:
        return "(略過)"
    if sec < 60:
        return f"{sec:.2f} s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m {s:.1f}s"


# ─────────────────────────── Main ────────────────────────────────

def main():
    t_total_start = time.perf_counter()
    timing = {}   # 各階段耗時（秒）

    # --- 環境準備 ---
    workspace = os.path.abspath(MVS_WORKSPACE)
    os.makedirs(workspace, exist_ok=True)

    openmvs_abs = os.path.abspath(OPENMVS_BIN)
    env = os.environ.copy()
    env["PATH"] = openmvs_abs + os.pathsep + env.get("PATH", "")

    # --- 載入影像路徑 ---
    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    )
    if len(image_paths) < 2:
        print(f"❌ 在 {IMAGE_FOLDER} 找不到足夠的圖片。"); return
    print(f"找到 {len(image_paths)} 張影像。")

    # --- 載入相機內參 ---
    K_orig, dist_coeffs = load_raw_camera_params(K_INI_PATH)

    # --- 載入 LightGlue ONNX ---
    _t = time.perf_counter()
    print("載入 LightGlue ONNX...")
    lg_session = ort.InferenceSession(LG_ONNX_PATH, providers=['CPUExecutionProvider'])
    timing["載入 LightGlue ONNX"] = time.perf_counter() - _t

    # ========================= Phase 1 =========================
    print("\n===== Phase 1: 建立 COLMAP 場景 =====")
    _t = time.perf_counter()
    result = phase1_build_colmap(image_paths, K_orig, dist_coeffs, lg_session, workspace)
    timing["Phase 1 合計"] = time.perf_counter() - _t
    valid_indices, K_scaled, ph1_timing = result
    if valid_indices is None:
        return
    # 登錄 Phase 1 子項目
    for k, v in ph1_timing.items():
        timing[f"  └ {k}"] = v

    # ========================= Phase 2 =========================
    print("\n===== Phase 2: InterfaceCOLMAP → scene.mvs =====")
    scene_mvs = os.path.join(workspace, "scene.mvs")
    _t = time.perf_counter()
    ret = run_openmvs_exe("InterfaceCOLMAP.exe", [
        "-i", workspace,
        "-o", scene_mvs,
        "--image-folder", os.path.join(workspace, "images"),
    ], cwd=workspace, env=env)
    timing["Phase 2: InterfaceCOLMAP"] = time.perf_counter() - _t
    if ret != 0 or not os.path.exists(scene_mvs):
        print("❌ InterfaceCOLMAP 執行失敗。"); return

    # ========================= Phase 3 =========================
    print("\n===== Phase 3: DensifyPointCloud =====")
    n_views = DENSIFY_NUMBER_VIEWS if DENSIFY_NUMBER_VIEWS > 0 else max(2, min(len(valid_indices), 8))
    print(f"  使用 {len(valid_indices)} 張有效影像，--number-views={n_views}")
    scene_dense_mvs = os.path.join(workspace, "scene_dense.mvs")
    _t = time.perf_counter()
    ret = run_openmvs_exe("DensifyPointCloud.exe", [
        str(MVS_WORKSPACE)+ "/scene.mvs",
        "-o", str(MVS_WORKSPACE)+"/scene_dense.mvs",
        "--resolution-level",            str(DENSIFY_RESOLUTION_LEVEL),
        "--number-views",                str(n_views),
        "--number-views-fuse",           str(DENSIFY_NUMBER_VIEWS_FUSE),
        "--filter-point",                str(DENSIFY_FILTER_POINT),
        "--filter-photo",                str(DENSIFY_FILTER_PHOTO),
        "--estimate-colors",             str(DENSIFY_ESTIMATE_COLORS),
        "--estimate-normals",            str(DENSIFY_ESTIMATE_NORMALS),
        "--min-triangulation-angle",     str(DENSIFY_MIN_TRIANGULATION_ANGLE),
    ], cwd=workspace, env=env)
    timing["Phase 3: DensifyPointCloud"] = time.perf_counter() - _t
    if ret != 0:
        print("❌ DensifyPointCloud 執行失敗。"); return

    dense_ply_candidates = glob.glob(os.path.join(workspace, "scene_dense*.ply"))
    dense_ply = dense_ply_candidates[0] if dense_ply_candidates else None
    if dense_ply:
        print(f"  稠密點雲：{dense_ply}")

    # ========================= Phase 4 =========================
    print("\n===== Phase 4: ReconstructMesh =====")
    _t = time.perf_counter()
    ret = run_openmvs_exe("ReconstructMesh.exe", [
        "scene_dense.mvs",
        "-o", "scene_mesh.mvs",
        "--quality-factor",    str(MESH_QUALITY_FACTOR),
        "--smooth",            str(MESH_SMOOTH),
        "--decimate",          str(MESH_DECIMATE),
        "--close-holes",       str(MESH_CLOSE_HOLES),
        "--remove-spurious",   str(MESH_REMOVE_SPURIOUS),
    ], cwd=workspace, env=env)
    timing["Phase 4: ReconstructMesh"] = time.perf_counter() - _t
    # ReconstructMesh 輸出檔名以 -o 的 MVS 檔名為基礎：
    #   '-o scene_mesh.mvs' → scene_mesh.ply 或 scene_mesh_raw.ply
    mesh_ply_candidates = sorted(
        glob.glob(os.path.join(workspace, "scene_mesh*.ply"))
    )
    # 優先取總變體（非 _raw），如果沒有就取第一個
    mesh_ply = next(
        (p for p in mesh_ply_candidates if "_raw" not in os.path.basename(p)),
        mesh_ply_candidates[0] if mesh_ply_candidates else None
    )
    if ret != 0 or mesh_ply is None:
        print("  ⚠️ ReconstructMesh 失敗或無輸出網格，跳過 TextureMesh。")
        mesh_ply = None
    else:
        print(f"  網格：{mesh_ply}")

    # ========================= Phase 5 =========================
    if not SKIP_TEXTURE and mesh_ply is not None:
        print("\n===== Phase 5: TextureMesh =====")
        mesh_basename = os.path.basename(mesh_ply)
        _t = time.perf_counter()
        run_openmvs_exe("TextureMesh.exe", [
            "scene_dense.mvs",
            "-m", mesh_basename,
            "-o", "scene_texture.mvs",
        ], cwd=workspace, env=env)
        timing["Phase 5: TextureMesh"] = time.perf_counter() - _t
        texture_candidates = glob.glob(os.path.join(workspace, "scene_texture*.obj"))
        if texture_candidates:
            print(f"  紋理網格：{texture_candidates[0]}")
    else:
        timing["Phase 5: TextureMesh"] = -1.0   # 標記為略過
        if SKIP_TEXTURE:
            print("\n===== Phase 5: TextureMesh（已略過）=====")

    # ========================= Phase 6 =========================
    print("\n===== Phase 6: 讀取 / 匯出結果 =====")
    _t = time.perf_counter()
    open3d_geometry = None
    open3d_title    = ""

    if dense_ply and os.path.exists(dense_ply):
        print(f"讀取稠密點雲：{dense_ply}")
        pcd = o3d.io.read_point_cloud(dense_ply)
        out_ply = os.path.join(workspace, "result_dense.ply")
        out_xyz = os.path.join(workspace, "result_dense.xyz")
        o3d.io.write_point_cloud(out_ply, pcd)
        o3d.io.write_point_cloud(out_xyz, pcd)
        print(f"✅ 匯出：{out_ply}")
        print(f"✅ 匯出：{out_xyz}")
        open3d_geometry = pcd
        open3d_title    = "OpenMVS Dense Point Cloud"
    else:
        print("⚠️ 找不到稠密點雲輸出，嘗試讀取網格...")
        if mesh_ply and os.path.exists(mesh_ply):
            mesh = o3d.io.read_triangle_mesh(mesh_ply)
            mesh.compute_vertex_normals()
            open3d_geometry = mesh
            open3d_title    = "OpenMVS Mesh"
        else:
            print("❌ 無法找到任何輸出檔案，請檢查 OpenMVS 執行日誌。")

    timing["Phase 6: 讀取/匯出 PLY"] = time.perf_counter() - _t

    # ========================= 耗時彙整（在開視窗前印出）=========================
    t_total = time.perf_counter() - t_total_start
    col = 38
    print("\n" + "=" * (col + 14))
    print(f"{'  ⏱  各階段耗時彙整（不含 3D 視窗等待）':^{col+14}}")
    print("=" * (col + 14))
    for name, sec in timing.items():
        print(f"  {name:<{col}} {_fmt(sec):>10}")
    print("-" * (col + 14))
    print(f"  {'全程合計（不含視窗）':<{col}} {_fmt(t_total):>10}")
    print("=" * (col + 14))

    # 最後才開啟 3D 視窗（阻塞，不納入計時）
    if open3d_geometry is not None:
        print("\n開啟 Open3D 視窗（關閉視窗後程式結束）...")
        o3d.visualization.draw_geometries([open3d_geometry], window_name=open3d_title)


if __name__ == "__main__":
    main()
