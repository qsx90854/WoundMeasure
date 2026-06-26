import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PoseObs:
    frame_idx: int
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    t: np.ndarray
    corners_by_id: dict
    projected_corners_by_id: dict
    marker_reprojection_errors_by_id: dict
    marker_ids: list
    reference_marker_id: int


@dataclass
class MarkerObs:
    frame_idx: int
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    t: np.ndarray
    corners: np.ndarray
    area: float


@dataclass
class BoardModel:
    reference_marker_id: int
    object_corners_by_id: dict
    marker_to_reference: dict


@dataclass
class PointObs:
    frame_idx: int
    point: np.ndarray
    pose: PoseObs


def load_calibration(path: Path, camera: str = "L"):
    data = json.loads(path.read_text(encoding="utf-8"))

    if "camera_matrix" in data:
        K_data = data["camera_matrix"]
        dist_data = data.get("dist_coeffs", [0, 0, 0, 0, 0])
    else:
        camera_key = f"intrinsic_{camera.upper()}"
        if camera_key not in data:
            available = sorted(k for k in data if k.startswith("intrinsic_"))
            raise KeyError(
                f"Calibration file must contain either 'camera_matrix' or '{camera_key}'. "
                f"Available intrinsic entries: {available}"
            )
        intrinsic = data[camera_key]
        K_data = intrinsic["matrix"]
        dist_data = intrinsic.get("distortion", [0, 0, 0, 0, 0])

    K = np.asarray(K_data, dtype=np.float64)
    dist = np.asarray(dist_data, dtype=np.float64).reshape(-1, 1)
    if K.shape != (3, 3):
        raise ValueError("Camera matrix must be 3x3")
    return K, dist


def get_aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        valid = [n for n in dir(cv2.aruco) if n.startswith("DICT_")]
        raise ValueError(f"Unknown ArUco dictionary {name}. Valid examples: {valid[:8]}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector(dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)
    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_markers(detector, gray):
    if hasattr(detector, "detectMarkers"):
        return detector.detectMarkers(gray)
    dictionary, params = detector
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def marker_object_points(marker_size_m: float):
    s = marker_size_m / 2.0
    return np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )


def estimate_marker_pose(corners, K, dist, marker_size_m):
    obj = marker_object_points(marker_size_m)
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        corners.reshape(4, 2).astype(np.float64),
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return rvec.reshape(3), tvec.reshape(3), R, tvec.reshape(3)


def scan_video_for_marker_observations(args, K, dist):
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    detector = make_detector(get_aruco_dictionary(args.aruco_dict))
    detections = {}
    frames = {}
    frame_idx = -1
    end_frame = args.end_frame if args.end_frame is not None else math.inf

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < args.start_frame:
            continue
        if frame_idx > end_frame:
            break
        if (frame_idx - args.start_frame) % args.scan_stride != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(detector, gray)
        if ids is None:
            continue

        frame_detections = {}
        for c, marker_id_arr in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id_arr)
            pose = estimate_marker_pose(c[0], K, dist, args.marker_size_m)
            if pose is None:
                continue
            area = cv2.contourArea(c[0].astype(np.float32))
            rvec, tvec, R, t = pose
            frame_detections[marker_id] = MarkerObs(
                frame_idx,
                marker_id,
                rvec,
                tvec,
                R,
                t,
                c[0].copy(),
                area,
            )

        if not frame_detections:
            continue

        detections[frame_idx] = frame_detections
        frames[frame_idx] = frame.copy()

    cap.release()
    if len(detections) < 2:
        raise RuntimeError("Need at least two frames with a detected ArUco marker.")
    return detections, frames


def project_to_rotation(R):
    u, _, vt = np.linalg.svd(R)
    R_avg = u @ vt
    if np.linalg.det(R_avg) < 0:
        u[:, -1] *= -1
        R_avg = u @ vt
    return R_avg


def average_transforms(samples):
    rotations = np.asarray([s[0] for s in samples], dtype=np.float64)
    translations = np.asarray([s[1] for s in samples], dtype=np.float64)
    R = project_to_rotation(np.mean(rotations, axis=0))
    t = np.median(translations, axis=0)
    return R, t


def transform_points(points, R, t):
    return (R @ points.T).T + t.reshape(1, 3)


def choose_reference_marker_id(args, detections):
    marker_stats = {}
    for frame_detections in detections.values():
        for marker_id, obs in frame_detections.items():
            count, area_sum = marker_stats.get(marker_id, (0, 0.0))
            marker_stats[marker_id] = (count + 1, area_sum + obs.area)

    if args.marker_id is not None:
        if args.marker_id not in marker_stats:
            available = sorted(marker_stats)
            raise RuntimeError(f"Requested marker ID {args.marker_id} was not detected. Available IDs: {available}")
        return args.marker_id

    return max(marker_stats, key=lambda marker_id: marker_stats[marker_id])


def build_board_model(args, detections):
    reference_id = choose_reference_marker_id(args, detections)
    object_corners_by_id = {reference_id: marker_object_points(args.marker_size_m)}
    marker_to_reference = {
        reference_id: (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    }

    marker_ids = sorted({marker_id for frame in detections.values() for marker_id in frame})
    for marker_id in marker_ids:
        if marker_id == reference_id:
            continue

        samples = []
        for frame_detections in detections.values():
            if reference_id not in frame_detections or marker_id not in frame_detections:
                continue
            ref = frame_detections[reference_id]
            other = frame_detections[marker_id]
            R_ref_marker = ref.R.T @ other.R
            t_ref_marker = ref.R.T @ (other.t - ref.t)
            samples.append((R_ref_marker, t_ref_marker))

        if len(samples) < args.min_board_pair_frames:
            continue

        R_ref_marker, t_ref_marker = average_transforms(samples)
        marker_to_reference[marker_id] = (R_ref_marker, t_ref_marker)
        object_corners_by_id[marker_id] = transform_points(
            marker_object_points(args.marker_size_m),
            R_ref_marker,
            t_ref_marker,
        )

    if len(object_corners_by_id) == 1:
        print(f"Using single marker board with reference marker ID {reference_id}.")
    else:
        used_ids = sorted(object_corners_by_id)
        print(f"Using auto-built ArUco board. Reference marker ID: {reference_id}. Marker IDs: {used_ids}.")

    return BoardModel(reference_id, object_corners_by_id, marker_to_reference)


def pose_from_single_marker(marker_obs, board):
    R_ref_marker, t_ref_marker = board.marker_to_reference[marker_obs.marker_id]
    R_board = marker_obs.R @ R_ref_marker.T
    t_board = marker_obs.t - R_board @ t_ref_marker
    rvec_board, _ = cv2.Rodrigues(R_board)
    return rvec_board.reshape(3), t_board.reshape(3), R_board, t_board.reshape(3)


def estimate_board_pose(frame_idx, frame_detections, board, K, dist):
    usable = [
        obs
        for marker_id, obs in frame_detections.items()
        if marker_id in board.object_corners_by_id
    ]
    if not usable:
        return None

    if len(usable) == 1:
        rvec, tvec, R, t = pose_from_single_marker(usable[0], board)
    else:
        seed = max(usable, key=lambda obs: obs.area)
        seed_rvec, seed_tvec, _, _ = pose_from_single_marker(seed, board)
        object_points = []
        image_points = []
        for obs in usable:
            object_points.append(board.object_corners_by_id[obs.marker_id])
            image_points.append(obs.corners.reshape(4, 2).astype(np.float64))

        object_points = np.vstack(object_points).astype(np.float64)
        image_points = np.vstack(image_points).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            seed_rvec.reshape(3, 1).astype(np.float64),
            seed_tvec.reshape(3, 1).astype(np.float64),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)
        t = tvec

    corners_by_id = {obs.marker_id: obs.corners for obs in usable}
    projected_corners_by_id = {}
    marker_reprojection_errors_by_id = {}
    for obs in usable:
        projected, _ = cv2.projectPoints(
            board.object_corners_by_id[obs.marker_id],
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            K,
            dist,
        )
        projected = projected.reshape(4, 2)
        projected_corners_by_id[obs.marker_id] = projected
        per_corner_errors = np.linalg.norm(projected - obs.corners.reshape(4, 2), axis=1)
        marker_reprojection_errors_by_id[obs.marker_id] = per_corner_errors.astype(np.float64)

    marker_ids = sorted(corners_by_id)
    return PoseObs(
        frame_idx,
        rvec,
        tvec,
        R,
        t,
        corners_by_id,
        projected_corners_by_id,
        marker_reprojection_errors_by_id,
        marker_ids,
        board.reference_marker_id,
    )


def scan_video_for_poses(args, K, dist):
    detections, frames = scan_video_for_marker_observations(args, K, dist)
    board = build_board_model(args, detections)

    poses = {}
    for frame_idx, frame_detections in detections.items():
        pose = estimate_board_pose(frame_idx, frame_detections, board, K, dist)
        if pose is not None:
            poses[frame_idx] = pose

    if len(poses) < 2:
        raise RuntimeError("Need at least two frames with a solved ArUco board pose.")
    return poses, frames, board


def choose_selection_frame(poses, frames):
    indices = sorted(poses)
    mid = indices[len(indices) // 2]
    return mid, frames[mid]


def select_point(frame, title="Select candidate point"):
    selected = []
    display = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(x, y)]

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        canvas = display.copy()
        if selected:
            cv2.drawMarker(canvas, selected[0], (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and selected:
            break
        if key == 27:
            raise KeyboardInterrupt("Point selection cancelled")

    cv2.destroyWindow(title)
    return np.array(selected[0], dtype=np.float32)


def select_points(frame, count, title, prompt):
    selected = []
    display = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < count:
            selected.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and selected:
            selected.pop()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        canvas = display.copy()
        for idx, point in enumerate(selected, start=1):
            cv2.drawMarker(canvas, point, (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                canvas,
                str(idx),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"{prompt} ({len(selected)}/{count})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(selected) == count:
            break
        if key == 8 and selected:
            selected.pop()
        if key == 27:
            raise KeyboardInterrupt("Point selection cancelled")

    cv2.destroyWindow(title)
    return [np.array(point, dtype=np.float32) for point in selected]


def refine_to_local_feature(gray, point, radius=25):
    x, y = point.astype(int)
    h, w = gray.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = gray[y0:y1, x0:x1]
    features = cv2.goodFeaturesToTrack(
        patch,
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=5,
    )
    if features is None:
        return point.astype(np.float32)
    pts = features.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
    distances = np.linalg.norm(pts - point.reshape(1, 2), axis=1)
    best = pts[int(np.argmin(distances))]
    return best.astype(np.float32)


def load_pose_frames(video_path, frame_indices):
    wanted = set(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in wanted:
            frames[idx] = frame
            if len(frames) == len(wanted):
                break
    cap.release()
    return frames


def track_point_across_pose_frames(args, poses, selection_idx, selected_point):
    indices = sorted(poses)
    frames = load_pose_frames(args.video, indices)
    grays = {i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in indices}
    start_gray = grays[selection_idx]
    start_point = refine_to_local_feature(start_gray, selected_point)

    tracked = {selection_idx: start_point}

    def track_direction(seq):
        prev_idx = selection_idx
        prev_pt = start_point.reshape(1, 1, 2)
        prev_gray = grays[prev_idx]
        for idx in seq:
            next_gray = grays[idx]
            next_pt, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                next_gray,
                prev_pt,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if st is None or st[0, 0] == 0:
                break
            back_pt, back_st, _ = cv2.calcOpticalFlowPyrLK(
                next_gray,
                prev_gray,
                next_pt,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            fb_err = float(np.linalg.norm(back_pt.reshape(2) - prev_pt.reshape(2)))
            if back_st is None or back_st[0, 0] == 0 or fb_err > args.max_fb_error_px:
                break
            p = next_pt.reshape(2)
            h, w = next_gray.shape[:2]
            if not (0 <= p[0] < w and 0 <= p[1] < h):
                break
            tracked[idx] = p.astype(np.float32)
            prev_idx = idx
            prev_gray = next_gray
            prev_pt = next_pt

    pos = indices.index(selection_idx)
    track_direction(indices[pos + 1 :])
    track_direction(list(reversed(indices[:pos])))

    return [PointObs(i, tracked[i], poses[i]) for i in sorted(tracked)]


def camera_center_marker(pose: PoseObs):
    return -pose.R.T @ pose.t


def triangulation_angle(pose_a: PoseObs, pose_b: PoseObs, X):
    ca = camera_center_marker(pose_a)
    cb = camera_center_marker(pose_b)
    va = ca - X
    vb = cb - X
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom <= 1e-12:
        return 0.0
    cosang = np.clip(float(np.dot(va, vb) / denom), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def projection_matrix(K, pose: PoseObs):
    Rt = np.hstack([pose.R, pose.t.reshape(3, 1)])
    return K @ Rt


def normalized_point(K, dist, point):
    pts = np.asarray(point, dtype=np.float64).reshape(1, 1, 2)
    return cv2.undistortPoints(pts, K, dist).reshape(2)


def linear_triangulate(K, dist, observations):
    A = []
    for obs in observations:
        P = np.hstack([obs.pose.R, obs.pose.t.reshape(3, 1)])
        u, v = normalized_point(K, dist, obs.point)
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        raise RuntimeError("Triangulation failed: homogeneous scale is too small.")
    return Xh[:3] / Xh[3]


def reprojection_errors(K, dist, observations, X):
    errors = []
    for obs in observations:
        uv, _ = cv2.projectPoints(
            X.reshape(1, 3),
            obs.pose.rvec.reshape(3, 1),
            obs.pose.tvec.reshape(3, 1),
            K,
            dist,
        )
        uv = uv.reshape(2)
        errors.append(float(np.linalg.norm(uv - obs.point)))
    return np.asarray(errors, dtype=np.float64)


def select_key_observations(args, K, dist, observations):
    if len(observations) < 2:
        raise RuntimeError("Need at least two tracked point observations.")

    seed = observations[len(observations) // 2]
    selected = [seed]
    seed_center = camera_center_marker(seed.pose)

    candidates = []
    for obs in observations:
        if obs.frame_idx == seed.frame_idx:
            continue
        baseline = float(np.linalg.norm(camera_center_marker(obs.pose) - seed_center))
        if baseline >= args.min_baseline_m:
            candidates.append((baseline, abs(obs.frame_idx - seed.frame_idx), obs))

    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    selected.extend([item[2] for item in candidates[: args.max_keyframes - 1]])
    selected = sorted(selected, key=lambda obs: obs.frame_idx)

    if len(selected) < 2:
        raise RuntimeError(
            f"No tracked frames meet min baseline {args.min_baseline_m} m. "
            "Try lowering --min-baseline-m or using a segment with more camera motion."
        )

    X = linear_triangulate(K, dist, selected)
    angle_max = max(triangulation_angle(a.pose, b.pose, X) for a in selected for b in selected)
    if angle_max < args.min_triangulation_angle_deg:
        raise RuntimeError(
            f"Triangulation angle is only {angle_max:.2f} deg. "
            "Use frames with larger viewpoint change or lower --min-triangulation-angle-deg."
        )

    return selected


def robust_triangulate(args, K, dist, observations):
    selected = select_key_observations(args, K, dist, observations)
    inliers = selected[:]

    for _ in range(4):
        if len(inliers) < 2:
            break
        X = linear_triangulate(K, dist, inliers)
        errs = reprojection_errors(K, dist, inliers, X)
        keep = [obs for obs, err in zip(inliers, errs) if err <= args.max_reproj_px]
        if len(keep) == len(inliers):
            break
        if len(keep) >= 2:
            inliers = keep
        else:
            break

    X = linear_triangulate(K, dist, inliers)
    errs = reprojection_errors(K, dist, inliers, X)
    return X, inliers, errs


def point_for_frame(observations, frame_idx):
    for obs in observations:
        if obs.frame_idx == frame_idx:
            return obs.point
    return None


def relative_rt(pose_a: PoseObs, pose_b: PoseObs):
    R_ba = pose_b.R @ pose_a.R.T
    t_ba = pose_b.t - R_ba @ pose_a.t
    baseline = float(np.linalg.norm(camera_center_marker(pose_b) - camera_center_marker(pose_a)))
    rvec_ba, _ = cv2.Rodrigues(R_ba)
    return rvec_ba.reshape(3), t_ba.reshape(3), baseline


def build_skin_plane(skin_points):
    p0, p1, p2 = [np.asarray(point, dtype=np.float64).reshape(3) for point in skin_points]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        raise RuntimeError("The three skin reference points are too close to collinear.")
    normal = normal / norm
    if float(np.dot(normal, np.array([0.0, 0.0, 1.0]))) < 0:
        normal = -normal
    d = -float(np.dot(normal, p0))
    return {
        "points": [p0, p1, p2],
        "normal": normal,
        "d": d,
    }


def signed_distance_to_plane(point, plane):
    return float(np.dot(plane["normal"], point) + plane["d"])


def plane_z_at_xy(x, y, plane):
    normal = plane["normal"]
    if abs(float(normal[2])) < 1e-12:
        return None
    return float(-(normal[0] * x + normal[1] * y + plane["d"]) / normal[2])


def barycentric_xy(point, triangle_points):
    p = np.asarray(point[:2], dtype=np.float64)
    tri = np.asarray([point[:2] for point in triangle_points], dtype=np.float64)
    A = np.column_stack((tri[0] - tri[2], tri[1] - tri[2]))
    b = p - tri[2]
    try:
        uv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    weights = np.array([uv[0], uv[1], 1.0 - uv.sum()], dtype=np.float64)
    return weights


def build_result(args, X, inliers, errors, selection_info=None, skin_plane_result=None):
    first = inliers[0].pose
    last = inliers[-1].pose
    rvec_rel, t_rel, baseline = relative_rt(first, last)
    marker_errors = []
    for obs in inliers:
        for per_corner_errors in obs.pose.marker_reprojection_errors_by_id.values():
            marker_errors.extend(float(err) for err in per_corner_errors)

    result = {
        "coordinate_system": "Auto-built ArUco board coordinates, meters. The reference marker plane is z=0.",
        "marker_size_m": args.marker_size_m,
        "reference_marker_id": first.reference_marker_id,
        "used_marker_ids_first_frame": first.marker_ids,
        "used_marker_ids_last_frame": last.marker_ids,
        "point_3d_marker_m": X.tolist(),
        "signed_distance_to_marker_plane_m": float(X[2]),
        "absolute_distance_to_marker_plane_m": float(abs(X[2])),
        "projection_on_marker_plane_m": [float(X[0]), float(X[1]), 0.0],
        "inlier_frame_count": len(inliers),
        "inlier_frames": [obs.frame_idx for obs in inliers],
        "marker_mean_reprojection_error_px": float(np.mean(marker_errors)) if marker_errors else None,
        "marker_max_reprojection_error_px": float(np.max(marker_errors)) if marker_errors else None,
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "front_to_back_rt": {
            "from_frame": first.frame_idx,
            "to_frame": last.frame_idx,
            "rvec_rad": rvec_rel.tolist(),
            "tvec_m": t_rel.tolist(),
            "baseline_m": baseline,
        },
        "selected_measurement_point": {
            "selection_frame_idx": selection_info["selection_frame_idx"],
            "clicked_uv_px": selection_info["measurement_clicked_uv_px"],
            "tracked_start_uv_px": selection_info["measurement_tracked_start_uv_px"],
            "point_3d_marker_m": X.tolist(),
        }
        if selection_info is not None
        else None,
        "observations": [
            {
                "frame_idx": obs.frame_idx,
                "uv_px": [float(obs.point[0]), float(obs.point[1])],
                "reprojection_error_px": float(err),
                "marker_ids": obs.pose.marker_ids,
                "marker_mean_reprojection_error_px": float(
                    np.mean([e for errs in obs.pose.marker_reprojection_errors_by_id.values() for e in errs])
                ),
            }
            for obs, err in zip(inliers, errors)
        ],
    }

    if skin_plane_result is not None:
        plane = skin_plane_result["plane"]
        signed_distance_m = signed_distance_to_plane(X, plane)
        z_at_measurement_xy = plane_z_at_xy(float(X[0]), float(X[1]), plane)
        barycentric = barycentric_xy(X, plane["points"])
        xy_distances = [
            float(np.linalg.norm((point[:2] - X[:2]) * 1000.0))
            for point in plane["points"]
        ]
        skin_selection_points = []
        if selection_info is not None:
            for idx, (clicked_uv, tracked_start_uv, point_3d) in enumerate(
                zip(
                    selection_info["skin_clicked_uv_px"],
                    selection_info["skin_tracked_start_uv_px"],
                    plane["points"],
                ),
                start=1,
            ):
                skin_selection_points.append(
                    {
                        "index": idx,
                        "selection_frame_idx": selection_info["selection_frame_idx"],
                        "clicked_uv_px": clicked_uv,
                        "tracked_start_uv_px": tracked_start_uv,
                        "point_3d_marker_m": point_3d.tolist(),
                    }
                )
        result["skin_plane"] = {
            "definition": "Plane fitted from 3 user-selected skin reference points.",
            "normal_oriented_toward_reference_marker_positive_z": plane["normal"].tolist(),
            "d": plane["d"],
            "selected_reference_points": skin_selection_points,
            "reference_points_3d_marker_m": [point.tolist() for point in plane["points"]],
            "reference_point_mean_reprojection_errors_px": skin_plane_result["mean_reprojection_errors_px"],
            "reference_point_max_reprojection_errors_px": skin_plane_result["max_reprojection_errors_px"],
            "plane_z_at_measurement_xy_m": z_at_measurement_xy,
            "plane_z_at_measurement_xy_mm": z_at_measurement_xy * 1000.0 if z_at_measurement_xy is not None else None,
            "measurement_z_minus_plane_z_at_xy_mm": (float(X[2]) - z_at_measurement_xy) * 1000.0
            if z_at_measurement_xy is not None
            else None,
            "measurement_xy_distance_to_reference_points_mm": xy_distances,
            "measurement_barycentric_xy_weights": barycentric.tolist() if barycentric is not None else None,
            "measurement_inside_reference_triangle_xy": bool(np.all(barycentric >= -1e-6))
            if barycentric is not None
            else None,
        }
        result["signed_distance_to_skin_plane_m"] = signed_distance_m
        result["signed_distance_to_skin_plane_mm"] = signed_distance_m * 1000.0
        result["absolute_distance_to_skin_plane_m"] = abs(signed_distance_m)
        result["absolute_distance_to_skin_plane_mm"] = abs(signed_distance_m) * 1000.0

    return result


def draw_debug(args, inliers, X, K, dist, skin_debug=None):
    frame_indices = {obs.frame_idx for obs in inliers}
    if skin_debug is not None:
        for item in skin_debug:
            frame_indices.update(obs.frame_idx for obs in item["inliers"])
    frames = load_pose_frames(args.video, frame_indices)
    out_dir = args.debug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wound_by_frame = {obs.frame_idx: obs for obs in inliers}
    skin_observations_by_frame = {}
    if skin_debug is not None:
        for point_idx, item in enumerate(skin_debug, start=1):
            for obs in item["inliers"]:
                skin_observations_by_frame.setdefault(obs.frame_idx, {})[point_idx] = obs

    for frame_idx in sorted(frame_indices):
        frame = frames[frame_idx]
        obs = wound_by_frame.get(frame_idx)
        if obs is not None:
            cv2.drawMarker(
                frame,
                tuple(np.round(obs.point).astype(int)),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                24,
                2,
            )
            proj, _ = cv2.projectPoints(X.reshape(1, 3), obs.pose.rvec, obs.pose.tvec, K, dist)
            uv = tuple(np.round(proj.reshape(2)).astype(int))
            cv2.circle(frame, uv, 5, (0, 0, 255), 2)
            pose = obs.pose
        else:
            pose = next(iter(skin_observations_by_frame[frame_idx].values())).pose

        if skin_debug is not None:
            for point_idx, item in enumerate(skin_debug, start=1):
                skin_obs = skin_observations_by_frame.get(frame_idx, {}).get(point_idx)
                if skin_obs is not None:
                    cv2.drawMarker(
                        frame,
                        tuple(np.round(skin_obs.point).astype(int)),
                        (255, 255, 0),
                        cv2.MARKER_CROSS,
                        20,
                        2,
                    )
                skin_proj, _ = cv2.projectPoints(
                    item["X"].reshape(1, 3),
                    pose.rvec,
                    pose.tvec,
                    K,
                    dist,
                )
                skin_uv = tuple(np.round(skin_proj.reshape(2)).astype(int))
                cv2.circle(frame, skin_uv, 4, (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    str(point_idx),
                    (skin_uv[0] + 6, skin_uv[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        for corners in pose.corners_by_id.values():
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        for corners in pose.projected_corners_by_id.values():
            cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (255, 0, 0), 1)
        cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), frame)


def parse_args():
    parser = argparse.ArgumentParser(description="Triangulate a user-selected object point using ArUco pose over time.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera", choices=("L", "R"), default="L", help="Camera intrinsics to use for stereo calibration JSON.")
    parser.add_argument("--output", type=Path, default=Path("triangulated_point.json"))
    parser.add_argument("--marker-size-m", type=float, default=0.00825)
    parser.add_argument("--marker-id", type=int, default=None, help="Optional reference marker ID. If omitted, the most visible marker is used.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--min-baseline-m", type=float, default=0.002)
    parser.add_argument("--min-triangulation-angle-deg", type=float, default=1.0)
    parser.add_argument("--max-fb-error-px", type=float, default=1.5)
    parser.add_argument("--max-reproj-px", type=float, default=3.0)
    parser.add_argument("--max-keyframes", type=int, default=12)
    parser.add_argument("--min-board-pair-frames", type=int, default=5)
    parser.add_argument("--skin-plane-points", type=int, choices=(0, 3), default=3, help="Select 3 skin points to define a local reference plane. Use 0 to disable.")
    parser.add_argument("--debug-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    K, dist = load_calibration(args.calibration, args.camera)
    poses, frames, board = scan_video_for_poses(args, K, dist)
    selection_idx, selection_frame = choose_selection_frame(poses, frames)
    print(f"Detected ArUco board pose in {len(poses)} frames.")
    print(f"Reference marker plane for Z=0: marker ID {board.reference_marker_id}.")
    print(f"Select the candidate point on frame {selection_idx}, then press Enter.")
    selected = select_point(selection_frame)
    skin_points_2d = []
    if args.skin_plane_points == 3:
        print("Select 3 skin reference points around the wound, then press Enter.")
        skin_points_2d = select_points(
            selection_frame,
            3,
            "Select 3 skin reference points",
            "Left click 3 skin reference points, right click/backspace to undo, Enter to confirm",
        )

    observations = track_point_across_pose_frames(args, poses, selection_idx, selected)
    print(f"Tracked selected point in {len(observations)} marker-visible frames.")
    X, inliers, errors = robust_triangulate(args, K, dist, observations)
    measurement_start = point_for_frame(observations, selection_idx)
    selection_info = {
        "selection_frame_idx": selection_idx,
        "measurement_clicked_uv_px": selected.tolist(),
        "measurement_tracked_start_uv_px": measurement_start.tolist() if measurement_start is not None else None,
        "skin_clicked_uv_px": [point.tolist() for point in skin_points_2d],
        "skin_tracked_start_uv_px": [],
    }

    skin_plane_result = None
    skin_debug = []
    if skin_points_2d:
        skin_points_3d = []
        skin_mean_errors = []
        skin_max_errors = []
        for idx, skin_point_2d in enumerate(skin_points_2d, start=1):
            skin_observations = track_point_across_pose_frames(args, poses, selection_idx, skin_point_2d)
            print(f"Tracked skin reference point {idx} in {len(skin_observations)} marker-visible frames.")
            skin_start = point_for_frame(skin_observations, selection_idx)
            selection_info["skin_tracked_start_uv_px"].append(
                skin_start.tolist() if skin_start is not None else None
            )
            skin_X, skin_inliers, skin_errors = robust_triangulate(args, K, dist, skin_observations)
            skin_points_3d.append(skin_X)
            skin_mean_errors.append(float(np.mean(skin_errors)))
            skin_max_errors.append(float(np.max(skin_errors)))
            skin_debug.append({"X": skin_X, "inliers": skin_inliers, "errors": skin_errors})

        skin_plane_result = {
            "plane": build_skin_plane(skin_points_3d),
            "mean_reprojection_errors_px": skin_mean_errors,
            "max_reprojection_errors_px": skin_max_errors,
        }

    result = build_result(args, X, inliers, errors, selection_info, skin_plane_result)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.debug_dir is not None:
        draw_debug(args, inliers, X, K, dist, skin_debug if skin_debug else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
