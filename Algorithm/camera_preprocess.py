import json
import os
import time

import cv2
import numpy as np


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
_CLAHE_CACHE = {}


def centered_roi_bounds(image_width, image_height, width_ratio=0.5, height_ratio=0.5):
    """Return an integer (x0, y0, x1, y1) ROI centered in one camera image."""
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image_width and image_height must be positive")
    width_ratio = float(width_ratio)
    height_ratio = float(height_ratio)
    if not (0.0 < width_ratio <= 1.0 and 0.0 < height_ratio <= 1.0):
        raise ValueError("ROI width/height ratios must be in (0, 1]")

    roi_width = max(1, min(width, int(round(width * width_ratio))))
    roi_height = max(1, min(height, int(round(height * height_ratio))))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    return x0, y0, x0 + roi_width, y0 + roi_height


def normalized_roi_bounds(
    image_width,
    image_height,
    x_ratio=0.25,
    y_ratio=0.25,
    width_ratio=0.5,
    height_ratio=0.5,
):
    """Convert a normalized top-left position and size into pixel ROI bounds."""
    width = int(image_width)
    height = int(image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image_width and image_height must be positive")

    x_ratio = float(x_ratio)
    y_ratio = float(y_ratio)
    width_ratio = float(width_ratio)
    height_ratio = float(height_ratio)
    if not (0.0 <= x_ratio < 1.0 and 0.0 <= y_ratio < 1.0):
        raise ValueError("ROI x/y ratios must be in [0, 1)")
    if not (0.0 < width_ratio <= 1.0 and 0.0 < height_ratio <= 1.0):
        raise ValueError("ROI width/height ratios must be in (0, 1]")
    tolerance = 1e-9
    if x_ratio + width_ratio > 1.0 + tolerance:
        raise ValueError("ROI x_ratio + width_ratio must not exceed 1")
    if y_ratio + height_ratio > 1.0 + tolerance:
        raise ValueError("ROI y_ratio + height_ratio must not exceed 1")

    x0 = int(round(width * x_ratio))
    y0 = int(round(height * y_ratio))
    roi_width = max(1, int(round(width * width_ratio)))
    roi_height = max(1, int(round(height * height_ratio)))
    x1 = min(width, x0 + roi_width)
    y1 = min(height, y0 + roi_height)
    return x0, y0, x1, y1


def draw_sbs_rois(
    frame,
    x_ratio=0.25,
    y_ratio=0.25,
    width_ratio=0.5,
    height_ratio=0.5,
    color=(0, 255, 255),
    thickness=2,
):
    """Draw the same configurable normalized RT ROI on both SBS halves."""
    height, full_width = frame.shape[:2]
    half_width = full_width // 2
    if half_width <= 0:
        return frame

    x0, y0, x1, y1 = normalized_roi_bounds(
        half_width,
        height,
        x_ratio,
        y_ratio,
        width_ratio,
        height_ratio,
    )
    for label, offset_x in (("L RT ROI", 0), ("R RT ROI", half_width)):
        p0 = (offset_x + x0, y0)
        p1 = (offset_x + x1 - 1, y1 - 1)
        cv2.rectangle(frame, p0, p1, color, int(thickness), cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (p0[0] + 8, max(24, p0[1] + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_sbs_independent_rois(
    frame,
    left_roi_ratio,
    right_roi_ratio,
    color=(0, 255, 255),
    thickness=2,
):
    """Draw independently configured normalized ROIs on the two SBS halves."""
    height, full_width = frame.shape[:2]
    half_width = full_width // 2
    if half_width <= 0:
        return frame
    if len(left_roi_ratio) != 4 or len(right_roi_ratio) != 4:
        raise ValueError("Each SBS ROI must contain x/y/width/height ratios")

    for label, offset_x, roi_ratio in (
        ("L RT ROI", 0, left_roi_ratio),
        ("R RT ROI", half_width, right_roi_ratio),
    ):
        x0, y0, x1, y1 = normalized_roi_bounds(
            half_width, height, *roi_ratio)
        p0 = (offset_x + x0, y0)
        p1 = (offset_x + x1 - 1, y1 - 1)
        cv2.rectangle(frame, p0, p1, color, int(thickness), cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (p0[0] + 8, max(24, p0[1] + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_sbs_center_rois(
    frame,
    width_ratio=0.5,
    height_ratio=0.5,
    color=(0, 255, 255),
    thickness=2,
):
    """Draw the same centered RT ROI on both halves of an SBS frame."""
    return draw_sbs_rois(
        frame,
        (1.0 - float(width_ratio)) * 0.5,
        (1.0 - float(height_ratio)) * 0.5,
        width_ratio,
        height_ratio,
        color,
        thickness,
    )


def fourcc_to_text(value):
    """Decode an OpenCV numeric FOURCC value into a readable four-character string."""
    try:
        code = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return "????"
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


def log_camera_stream_settings(capture, log_fn=print):
    """Print camera properties exposed through OpenCV."""
    try:
        backend_name = capture.getBackendName()
    except cv2.error:
        backend_name = "unknown"
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fourcc = fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC))
    exposure = float(capture.get(cv2.CAP_PROP_EXPOSURE))
    auto_exposure = float(capture.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    gain = float(capture.get(cv2.CAP_PROP_GAIN))
    brightness = float(capture.get(cv2.CAP_PROP_BRIGHTNESS))
    log_fn(
        f"ℹ️ [相機設定] backend={backend_name} | FOURCC={fourcc} | "
        f"{width}x{height} | FPS={fps:.2f} Hz")
    log_fn(
        f"ℹ️ [曝光設定] exposure={exposure:g} | "
        f"auto_exposure={auto_exposure:g} | gain={gain:g} | "
        f"brightness={brightness:g}")
    log_fn(
        "ℹ️ 防閃爍 50/60 Hz 不屬於 OpenCV 通用屬性；"
        "請按 P 從 DSHOW 驅動設定頁查看 Power Line Frequency/Anti-flicker。")


def show_dshow_camera_settings(capture, log_fn=print):
    """Open the native DirectShow camera property page when supported."""
    if not hasattr(cv2, "CAP_PROP_SETTINGS"):
        log_fn("⚠️ 此 OpenCV 版本沒有 CAP_PROP_SETTINGS。")
        return False
    log_fn(
        "⚙️ 正在開啟 DSHOW 相機設定頁；請尋找 "
        "Power Line Frequency / Anti-flicker，台灣通常選 60 Hz。")
    try:
        opened = bool(capture.set(cv2.CAP_PROP_SETTINGS, 1))
    except cv2.error as exc:
        log_fn(f"⚠️ 無法開啟相機設定頁: {exc}")
        return False
    if not opened:
        log_fn("⚠️ 相機驅動沒有提供可由 OpenCV 開啟的設定頁。")
        return False
    log_camera_stream_settings(capture, log_fn=log_fn)
    return True


def open_camera_with_mjpg(
    camera_index,
    backend,
    width,
    height,
    fps=30,
    buffer_size=1,
    log_fn=print,
):
    """Open a camera with MJPG/resolution/FPS negotiated in one backend call."""
    camera_index = int(camera_index)
    backend = int(backend)
    width = int(width)
    height = int(height)
    fps = float(fps)
    requested_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    open_params = [
        cv2.CAP_PROP_FOURCC,
        requested_fourcc,
        cv2.CAP_PROP_FRAME_WIDTH,
        width,
        cv2.CAP_PROP_FRAME_HEIGHT,
        height,
        cv2.CAP_PROP_FPS,
        int(round(fps)),
    ]

    started = time.perf_counter()
    capture = cv2.VideoCapture()
    try:
        opened_with_params = bool(
            capture.open(camera_index, backend, open_params))
    except cv2.error as exc:
        opened_with_params = False
        log_fn(
            "⚠️ 相機 backend 拒絕開啟參數: "
            f"{str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
    if not opened_with_params:
        capture.release()
        log_fn(
            "⚠️ 相機 backend 不接受開啟參數，退回先開裝置再設定 MJPG。")
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)

    if not capture.isOpened():
        capture.release()
        return capture

    if buffer_size is not None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))

    actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_fourcc = fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC))
    try:
        backend_name = capture.getBackendName()
    except cv2.error:
        backend_name = str(backend)
    elapsed_s = time.perf_counter() - started
    log_fn(
        f"📷 相機開啟: backend={backend_name} | FOURCC={actual_fourcc} | "
        f"{actual_width}x{actual_height}@{actual_fps:.2f} FPS | "
        f"{elapsed_s:.2f} s | "
        f"open-params={'yes' if opened_with_params else 'fallback'}")
    if actual_fourcc.upper() != "MJPG":
        log_fn(
            f"⚠️ MJPG 未成功協商，實際 FOURCC={actual_fourcc!r}；"
            "高解析度串流可能會延遲或掉幀。")
    if (actual_width, actual_height) != (width, height):
        log_fn(
            f"⚠️ 相機實際解析度 {actual_width}x{actual_height}，"
            f"與要求的 {width}x{height} 不同。")
    return capture


def get_clahe(clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_GRID_SIZE):
    key = (clip_limit, tile_size)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return _CLAHE_CACHE[key]


def preprocess_gray(gray_img, enable_clahe=True, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_GRID_SIZE):
    if enable_clahe:
        return get_clahe(clip_limit, tile_size).apply(gray_img)
    return gray_img


def load_json_camera_params(json_path):
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到相機參數檔案 {json_path}")
        return None, None, None, None, None, None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mtx_L = np.array(data["intrinsic_L"]["matrix"], dtype=np.float32)
    dist_L = np.array(data["intrinsic_L"]["distortion"], dtype=np.float32)
    mtx_R = np.array(data["intrinsic_R"]["matrix"], dtype=np.float32)
    dist_R = np.array(data["intrinsic_R"]["distortion"], dtype=np.float32)
    extrinsic = data.get("extrinsic", {})
    R_rel = np.array(extrinsic.get("R", np.eye(3)))
    t_rel = np.array(extrinsic.get("T", np.zeros(3))).reshape(3, 1)
    F_orig = np.array(extrinsic.get("F")) if "F" in extrinsic else None

    # Zebra forvideo mode keeps both sides on the same intrinsic model.
    mtx_R = mtx_L
    dist_R = dist_L

    return mtx_L, dist_L, mtx_R, dist_R, extrinsic, F_orig


def build_undistort_processor(mtx, dist, image_size, alpha=1.0):
    newK, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, alpha, image_size)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newK, image_size, cv2.CV_16SC2)

    def process_view(img, K=None, dist=None, nK=None):
        undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        return undist, 1.0

    return newK, map1, map2, process_view
