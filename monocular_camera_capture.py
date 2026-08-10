"""Preview one camera and capture images for monocular calibration.

Controls
--------
S or Space: Save the current full-resolution frame.
Q or Esc: Quit.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


WINDOW_NAME = "Monocular calibration capture"
DEFAULT_OUTPUT_DIR = "monocular_calibration_images"
DEFAULT_BOARD_COLS = 10
DEFAULT_BOARD_ROWS = 16
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30.0
DEFAULT_FOURCC = "MJPG"
DEFAULT_BACKEND = "dshow" if os.name == "nt" else "any"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a monocular camera and save calibration images."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Directory for captured PNG files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Requested frame width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Requested frame height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Requested camera FPS (default: {DEFAULT_FPS:g})",
    )
    parser.add_argument(
        "--fourcc",
        default=DEFAULT_FOURCC,
        help=f"Four-character camera format (default: {DEFAULT_FOURCC})",
    )
    parser.add_argument(
        "--backend",
        choices=("any", "msmf", "dshow"),
        default=DEFAULT_BACKEND,
        help=f"OpenCV camera backend (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        help=f"Checkerboard inner corners across columns (default: {DEFAULT_BOARD_COLS})",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        help=f"Checkerboard inner corners across rows (default: {DEFAULT_BOARD_ROWS})",
    )
    parser.add_argument(
        "--no-board-detection",
        action="store_true",
        help="Disable checkerboard detection in the preview",
    )
    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=0,
        help="Maximum preview width; 0 disables resizing (default: 0)",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=10,
        help="Run checkerboard detection every N displayed frames (default: 10)",
    )
    args = parser.parse_args()

    if args.camera_index < 0:
        parser.error("--camera-index must be zero or greater")
    if args.width < 0 or args.height < 0 or args.fps < 0:
        parser.error("--width, --height, and --fps cannot be negative")
    if args.board_cols < 2 or args.board_rows < 2:
        parser.error("--board-cols and --board-rows must both be at least 2")
    if args.preview_max_width < 0:
        parser.error("--preview-max-width cannot be negative")
    if args.detection_interval <= 0:
        parser.error("--detection-interval must be greater than 0")
    if args.fourcc and len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters")
    if args.backend in {"msmf", "dshow"} and os.name != "nt":
        parser.error(f"--backend {args.backend} is available only on Windows")
    return args


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    backend = {
        "any": cv2.CAP_ANY,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
    }[args.backend]
    capture = cv2.VideoCapture(args.camera_index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"無法開啟相機索引 {args.camera_index}；請確認相機未被其他程式占用，"
            "或改用其他 --camera-index / --backend。"
        )

    if args.fourcc:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def fourcc_to_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


class LatestFrameReader:
    """Continuously discard stale camera frames and retain only the newest one."""

    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.frame = None
        self.frame_number = 0
        self.capture_fps = 0.0
        self.failed_reads = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.first_frame_event = threading.Event()
        self.thread = threading.Thread(
            target=self._read_loop,
            name="monocular-camera-reader",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _read_loop(self) -> None:
        sample_start = time.monotonic()
        sample_frames = 0
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                with self.lock:
                    self.failed_reads += 1
                time.sleep(0.01)
                continue

            sample_frames += 1
            now = time.monotonic()
            elapsed = now - sample_start
            measured_fps = None
            if elapsed >= 1.0:
                measured_fps = sample_frames / elapsed
                sample_start = now
                sample_frames = 0

            with self.lock:
                self.frame = frame
                self.frame_number += 1
                self.failed_reads = 0
                if measured_fps is not None:
                    self.capture_fps = measured_fps
            self.first_frame_event.set()

    def latest(self, after_frame_number: int):
        with self.lock:
            is_new = self.frame is not None and self.frame_number != after_frame_number
            frame = self.frame.copy() if is_new else None
            return frame, self.frame_number, self.capture_fps, self.failed_reads

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def make_preview(
    frame,
    max_width: int,
    board_size: tuple[int, int],
    detect_board: bool,
    saved_count: int,
    capture_fps: float = 0.0,
    run_detection: bool = True,
    previous_board_found: bool | None = None,
):
    height, width = frame.shape[:2]
    scale = 1.0 if max_width == 0 else min(1.0, max_width / width)
    if scale < 1.0:
        preview = cv2.resize(
            frame,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = frame.copy()

    board_found = previous_board_found
    if detect_board and run_detection:
        detection_scale = min(1.0, 960 / preview.shape[1])
        if detection_scale < 1.0:
            detection_image = cv2.resize(
                preview,
                (
                    round(preview.shape[1] * detection_scale),
                    round(preview.shape[0] * detection_scale),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = preview
        gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        board_found, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if board_found:
            if detection_scale < 1.0:
                corners = corners / detection_scale
            cv2.drawChessboardCorners(preview, board_size, corners, board_found)

    board_text = "Board: detection disabled"
    board_color = (180, 180, 180)
    if board_found is True:
        board_text = f"Board {board_size[0]}x{board_size[1]}: FOUND"
        board_color = (0, 255, 0)
    elif board_found is False:
        board_text = f"Board {board_size[0]}x{board_size[1]}: not found"
        board_color = (0, 165, 255)

    cv2.putText(
        preview,
        "S / Space: save    Q / Esc: quit",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Saved: {saved_count}    Image: {width}x{height}    Capture FPS: {capture_fps:.1f}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        board_text,
        (20, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        board_color,
        2,
        cv2.LINE_AA,
    )
    return preview, board_found


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"[錯誤] 無法建立照片資料夾 {output_dir}：{error}")
        return 1

    try:
        capture = open_camera(args)
    except RuntimeError as error:
        print(f"[錯誤] {error}")
        return 1

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    actual_fourcc = fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC))
    try:
        actual_backend = capture.getBackendName()
    except cv2.error:
        actual_backend = args.backend

    print(
        f"相機已開啟：index={args.camera_index}, backend={actual_backend}, "
        f"FOURCC={actual_fourcc!r}, {actual_width}x{actual_height}, "
        f"FPS={actual_fps:.1f}"
    )
    print(f"照片儲存位置：{output_dir}")
    print("按 S 或空白鍵拍照；按 Q 或 Esc 離開。")

    board_size = (args.board_cols, args.board_rows)
    saved_count = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    reader = LatestFrameReader(capture)
    reader.start()
    if not reader.first_frame_event.wait(timeout=5.0):
        print("[錯誤] 開啟相機後 5 秒內仍未收到畫面。")
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()
        return 1

    last_frame_number = -1
    current_frame = None
    displayed_frames = 0
    board_found = None
    window_sized = False

    try:
        while True:
            frame, frame_number, capture_fps, failed_reads = reader.latest(
                last_frame_number
            )
            if frame is not None:
                current_frame = frame
                last_frame_number = frame_number
                displayed_frames += 1
                run_detection = (
                    displayed_frames == 1
                    or displayed_frames % args.detection_interval == 0
                )
                preview, board_found = make_preview(
                    frame,
                    args.preview_max_width,
                    board_size,
                    not args.no_board_detection,
                    saved_count,
                    capture_fps,
                    run_detection,
                    board_found,
                )
                cv2.imshow(WINDOW_NAME, preview)
                if not window_sized:
                    cv2.resizeWindow(WINDOW_NAME, preview.shape[1], preview.shape[0])
                    window_sized = True
            elif failed_reads >= 30:
                print("[錯誤] 連續 30 次無法讀取相機畫面。")
                return 1

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S"), 32):
                if current_frame is None:
                    print("[警告] 尚未取得可儲存的畫面。")
                    continue
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = output_dir / f"mono_{timestamp}.png"
                if cv2.imwrite(
                    str(image_path),
                    current_frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                ):
                    saved_count += 1
                    print(f"[已儲存 {saved_count}] {image_path}")
                else:
                    print(f"[錯誤] 無法儲存圖片：{image_path}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()

    print(f"相機已關閉，共儲存 {saved_count} 張照片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
