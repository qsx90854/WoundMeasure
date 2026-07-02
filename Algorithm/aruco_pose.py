import cv2
import numpy as np

from .camera_preprocess import preprocess_gray


def average_rotations_svd(R_list):
    if len(R_list) == 0:
        return np.eye(3, dtype=np.float32)
    M = np.zeros((3, 3), dtype=np.float64)
    for R in R_list:
        M += R.astype(np.float64)
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg.astype(np.float32)


def detect_aruco_corners_bgr_for_pose(bgr, preprocess_gray_fn=None):
    """Detect ArUco marker corners in an already processed/undistorted UI image."""
    if bgr is None or bgr.size == 0:
        return {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_fn = preprocess_gray_fn or preprocess_gray
    gray = gray_fn(gray, True)
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dict_4x4, parameters=params)
    if ids is None or len(ids) == 0:
        return {}
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.0001)
    for c in corners:
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), term)
    return {int(mid[0]): c.reshape(4, 2) for mid, c in zip(ids, corners)}


def estimate_marker_map_pose_from_corners(corners_dict, marker_map, marker_size_mm, K):
    """Estimate camera pose of the marker-map reference frame from detected marker corners."""
    if not corners_dict or not marker_map:
        return None
    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    obj_pts = []
    img_pts = []
    for mid, pts in corners_dict.items():
        if mid not in marker_map:
            continue
        R_m2ref, t_m2ref = marker_map[mid]
        pts_w = (R_m2ref @ canon.T).T + t_m2ref.T
        obj_pts.append(pts_w)
        img_pts.append(pts)
    if not obj_pts:
        return None
    obj_pts = np.vstack(obj_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)


def marker_plane_homography_from_pose(R, t, K):
    """Homography from marker-map z=0 plane coordinates to image coordinates."""
    H = K @ np.column_stack((R[:, 0], R[:, 1], t.reshape(3)))
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def compute_global_plane(imgA_gray, K_L, marker_size_mm, log_fn=print):
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dict_4x4, cv2.aruco.DetectorParameters())
        cA, idsA, _ = detector.detectMarkers(imgA_gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        cA, idsA, _ = cv2.aruco.detectMarkers(imgA_gray, dict_4x4, parameters=params)
    if idsA is None or len(idsA) < 1:
        print("⚠️ [平面擬合] 左圖未偵測到任何 ArUco 標籤。")
        return None, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for c in cA:
        cv2.cornerSubPix(imgA_gray, c, (3, 3), (-1, -1), term)

    half = marker_size_mm / 2.0
    canon = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    objA = []
    for c in cA:
        ok, rv, tv = cv2.solvePnP(canon, c[0], K_L, np.zeros(5))
        if ok:
            R, _ = cv2.Rodrigues(rv)
            objA.append((R @ canon.T).T + tv.T)
    if not objA:
        print("⚠️ [平面擬合] 所有 ArUco 標籤 PnP 失敗，無法估計平面。")
        return None, None
    objA = np.vstack(objA).astype(np.float32)
    c = np.mean(objA, axis=0)
    _, _, Vt = np.linalg.svd(objA - c)
    n = Vt[-1]
    if np.dot(n, c) > 0:
        n = -n
    log_fn(f"✅[平面擬合成功] 使用 {len(idsA)} 個標籤估計平面法線: {n.flatten()}，中心: {c.flatten()}")
    return n, c
