"""Capture still images from the HBVCAM stereo camera.

Controls
--------
S: Save the current full-resolution frame.
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


OUTPUT_DIR_NAME = "HBVCAM_4M2214HD-2-v11"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
PREVIEW_MAX_WIDTH = 900
WINDOW_NAME = "HBVCAM capture - S: save, Q/Esc: quit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the HBVCAM stereo camera and save a frame when S is pressed."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"OpenCV camera index (default: {DEFAULT_CAMERA_INDEX})",
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
        type=int,
        default=DEFAULT_FPS,
        help=f"Requested camera FPS (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--backend",
        choices=("msmf", "dshow", "any"),
        default="msmf" if os.name == "nt" else "any",
        help="OpenCV camera backend (Windows default: msmf)",
    )
    return parser.parse_args()


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    backend_name: str,
) -> cv2.VideoCapture:
    backend = {
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "any": cv2.CAP_ANY,
    }[backend_name]
    capture = cv2.VideoCapture(camera_index, backend)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"無法開啟相機索引 {camera_index}。"
            "如果相機不是索引 0，請改用 --camera-index 1（或其他索引）。"
        )

    # MJPG usually allows USB stereo cameras to provide their full resolution.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def fourcc_to_text(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))


class LatestFrameReader:
    """Continuously discard stale buffered frames and retain only the newest one."""

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
            name="camera-latest-frame-reader",
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

    def latest(self, after_frame_number: int = -1):
        with self.lock:
            has_new_frame = (
                self.frame is not None and self.frame_number != after_frame_number
            )
            frame = self.frame.copy() if has_new_frame else None
            return (
                frame,
                self.frame_number,
                self.capture_fps,
                self.failed_reads,
            )

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def make_preview(frame, capture_fps: float):
    height, width = frame.shape[:2]
    if width <= PREVIEW_MAX_WIDTH:
        preview = frame.copy()
    else:
        scale = PREVIEW_MAX_WIDTH / width
        preview = cv2.resize(
            frame,
            (PREVIEW_MAX_WIDTH, round(height * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    cv2.putText(
        preview,
        "S: save image    Q / Esc: quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Capture FPS: {capture_fps:.1f}    Saved size: {width}x{height}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def main() -> int:
    args = parse_args()
    output_dir = Path(__file__).resolve().parent / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        capture = open_camera(
            args.camera_index,
            args.width,
            args.height,
            args.fps,
            args.backend,
        )
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
        f"Camera details: backend={actual_backend}, "
        f"reported FPS={actual_fps:.1f}, FOURCC={actual_fourcc!r}"
    )
    print(f"相機已開啟：index={args.camera_index}, {actual_width}x{actual_height}")
    print(f"照片儲存位置：{output_dir}")
    print("按 S 拍照；按 Q 或 Esc 離開。")

    reader = LatestFrameReader(capture)
    reader.start()
    if not reader.first_frame_event.wait(timeout=5.0):
        print("[錯誤] 開啟相機後 5 秒內仍未收到畫面。")
        reader.stop()
        capture.release()
        return 1

    last_frame_number = -1
    current_frame = None
    try:
        while True:
            frame, frame_number, capture_fps, failed_reads = reader.latest(
                last_frame_number
            )
            if frame is not None:
                current_frame = frame
                last_frame_number = frame_number
                cv2.imshow(WINDOW_NAME, make_preview(frame, capture_fps))
            elif failed_reads >= 30:
                print("[錯誤] 連續 30 次無法讀取相機畫面，程式即將結束。")
                return 1

            key = cv2.waitKey(5) & 0xFF

            if key in (ord("s"), ord("S")):
                if current_frame is None:
                    print("[警告] 尚未取得可儲存的畫面。")
                    continue
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = output_dir / f"image_{timestamp}.png"
                saved = cv2.imwrite(
                    str(image_path),
                    current_frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )
                if saved:
                    print(f"[已儲存] {image_path}")
                else:
                    print(f"[錯誤] 無法儲存圖片：{image_path}")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        reader.stop()
        capture.release()
        cv2.destroyAllWindows()

    print("相機已關閉。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
