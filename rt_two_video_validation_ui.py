#!/usr/bin/env python3
"""Two-video, marker-only RT validation UI.

This tool deliberately lives outside the production Zebra application.  Video A
and Video B are treated as two independent temporal segments: the validation
backend selects one endpoint from each segment and estimates the B-to-A relative
pose without using SIFT.

The backend is imported lazily so the UI can still start and report a useful
error if its optional validation module is not available.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "雙影片 Marker-only RT 驗證工具"
VIDEO_FILE_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm"),
    ("All files", "*.*"),
]
JSON_FILE_TYPES = [("JSON files", "*.json"), ("All files", "*.*")]


def load_monocular_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the common calibration JSON layouts used in this workspace.

    Supported layouts include both::

        {"intrinsic_L": {"matrix": ..., "distortion": ...}}

    and::

        {"camera_matrix": ..., "dist_coeffs": ...}
    """

    calibration_path = Path(path).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    matrix: Any = None
    distortion: Any = None
    source = ""

    intrinsic_l = data.get("intrinsic_L")
    if isinstance(intrinsic_l, dict):
        matrix = intrinsic_l.get("matrix") or intrinsic_l.get("camera_matrix")
        distortion = (
            intrinsic_l.get("distortion")
            if intrinsic_l.get("distortion") is not None
            else intrinsic_l.get("dist_coeffs")
        )
        source = "intrinsic_L"

    if matrix is None:
        matrix = data.get("camera_matrix")
        distortion = (
            data.get("dist_coeffs")
            if data.get("dist_coeffs") is not None
            else data.get("distortion")
        )
        source = "camera_matrix"

    if matrix is None and isinstance(data.get("intrinsic"), dict):
        intrinsic = data["intrinsic"]
        matrix = intrinsic.get("matrix") or intrinsic.get("camera_matrix")
        distortion = (
            intrinsic.get("distortion")
            if intrinsic.get("distortion") is not None
            else intrinsic.get("dist_coeffs")
        )
        source = "intrinsic"

    if matrix is None:
        raise ValueError(
            "標定 JSON 找不到相機內參；需要 intrinsic_L.matrix 或 camera_matrix。"
        )
    if distortion is None:
        raise ValueError(
            "標定 JSON 找不到畸變係數；需要 intrinsic_L.distortion 或 dist_coeffs。"
        )

    camera_matrix = np.asarray(matrix, dtype=np.float64)
    dist_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"相機內參矩陣必須是 3x3，目前為 {camera_matrix.shape}。")
    if dist_coeffs.size < 4:
        raise ValueError(f"畸變係數至少需要 4 個，目前只有 {dist_coeffs.size} 個。")
    if not np.all(np.isfinite(camera_matrix)) or not np.all(np.isfinite(dist_coeffs)):
        raise ValueError("標定 JSON 含有 NaN 或 Infinity。")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("fx 與 fy 必須大於 0。")

    metadata = {
        "path": str(calibration_path),
        "source": source,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "distortion_count": int(dist_coeffs.size),
    }
    return camera_matrix, dist_coeffs, metadata


def _marker_colour(marker_id: Any) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR colour for one marker ID."""

    try:
        seed = int(marker_id)
    except (TypeError, ValueError):
        seed = sum(ord(ch) for ch in str(marker_id))
    palette = (
        (0, 255, 255),
        (60, 220, 60),
        (255, 180, 0),
        (255, 80, 210),
        (80, 180, 255),
        (230, 160, 70),
    )
    return palette[seed % len(palette)]


def _iter_marker_corners(corners: Any) -> Iterable[tuple[Any, np.ndarray]]:
    """Yield ``(marker_id, 4x2 corners)`` from several practical layouts."""

    if corners is None:
        return

    if isinstance(corners, dict):
        for marker_id, value in corners.items():
            if isinstance(value, dict):
                value = value.get("corners", value.get("points"))
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32).reshape(-1, 2)
            if len(array) >= 4 and np.all(np.isfinite(array[:4])):
                yield marker_id, array[:4]
        return

    # Also accept a list of {id, corners} records for exported/reloaded results.
    if isinstance(corners, (list, tuple)):
        for index, value in enumerate(corners):
            marker_id: Any = index
            points: Any = value
            if isinstance(value, dict):
                marker_id = value.get("id", value.get("marker_id", index))
                points = value.get("corners", value.get("points"))
            if points is None:
                continue
            array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if len(array) >= 4 and np.all(np.isfinite(array[:4])):
                yield marker_id, array[:4]


def draw_marker_overlay(frame: np.ndarray, corners: Any, caption: str) -> np.ndarray:
    """Draw marker outlines, IDs and corner indices on a selected frame."""

    if frame is None:
        raise ValueError("Cannot draw a null frame")
    output = np.asarray(frame).copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif output.ndim == 3 and output.shape[2] == 4:
        output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)

    marker_count = 0
    for marker_id, points_f in _iter_marker_corners(corners):
        marker_count += 1
        points = np.rint(points_f).astype(np.int32)
        colour = _marker_colour(marker_id)
        cv2.polylines(output, [points.reshape(-1, 1, 2)], True, colour, 3, cv2.LINE_AA)
        centre = np.mean(points_f, axis=0)
        for corner_index, (x, y) in enumerate(points):
            cv2.circle(output, (int(x), int(y)), 5, colour, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                str(corner_index),
                (int(x) + 6, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                colour,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            output,
            f"ID {marker_id}",
            (int(centre[0]) + 8, int(centre[1]) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            colour,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"{caption} | markers: {marker_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def _read_video_frame(video_path: str, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"無法重新開啟影片以顯示 frame：{video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise OSError(f"無法讀取 frame {frame_index}：{video_path}")
        return frame
    finally:
        capture.release()


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rotation_angle_deg(rotation: Any) -> float | None:
    try:
        matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _json_safe(value: Any, *, omit_images: bool = True) -> Any:
    """Convert backend output to a compact JSON-safe representation."""

    if isinstance(value, np.ndarray):
        if omit_images and value.ndim >= 2 and value.size > 100_000:
            return {"omitted": "image array", "shape": list(value.shape), "dtype": str(value.dtype)}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, omit_images=omit_images) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, omit_images=omit_images) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten_for_csv(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten mappings while storing arrays/lists as JSON in one CSV cell."""

    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_for_csv(item, child))
    elif isinstance(value, (list, tuple)):
        flattened[prefix] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        flattened[prefix] = value
    return flattened


def _flatten_scalars(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_scalars(item, child))
    elif isinstance(value, np.ndarray):
        if value.size <= 16:
            rows.append((prefix, np.array2string(value, precision=6, suppress_small=True)))
    elif isinstance(value, (list, tuple)):
        if len(value) <= 16 and all(not isinstance(item, (dict, list, tuple)) for item in value):
            rows.append((prefix, value))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, value))
    return rows


class TwoVideoRTValidationUI(tk.Tk):
    """Tkinter front-end for the isolated marker-only validation backend."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1460x980")
        self.minsize(1050, 760)

        self.video_a_var = tk.StringVar()
        self.video_b_var = tk.StringVar()
        self.calibration_var = tk.StringVar()
        self.marker_size_var = tk.StringVar(value="8.25")
        self.known_baseline_var = tk.StringVar(value="50.0")
        self.dx_var = tk.StringVar()
        self.dy_var = tk.StringVar()
        self.dz_var = tk.StringVar()
        self.min_baseline_var = tk.StringVar(value="0")
        self.max_baseline_var = tk.StringVar(value="220")
        self.local_window_var = tk.BooleanVar(value=True)
        self.local_window_radius_var = tk.IntVar(value=2)
        self.klt_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="請載入兩段影片與單目相機標定 JSON。")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._result: dict[str, Any] | None = None
        self._run_inputs: dict[str, Any] | None = None
        self._display_frames: tuple[np.ndarray, np.ndarray] | None = None
        self._photos: list[tk.PhotoImage | None] = [None, None]
        self._canvas_items: list[int | None] = [None, None]
        self._render_job: str | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=4)
        self.rowconfigure(4, weight=3)

        sources = ttk.LabelFrame(self, text="輸入資料", padding=8)
        sources.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        sources.columnconfigure(1, weight=1)
        self._path_row(sources, 0, "Video A（起始位置／前段）", self.video_a_var, self._choose_video_a)
        self._path_row(sources, 1, "Video B（結束位置／後段）", self.video_b_var, self._choose_video_b)
        self._path_row(sources, 2, "單目相機標定 JSON", self.calibration_var, self._choose_calibration)

        options = ttk.LabelFrame(self, text="驗證參數", padding=8)
        options.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for column in range(14):
            options.columnconfigure(column, weight=0)

        ttk.Label(options, text="Marker 邊長 (mm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.marker_size_var, width=9).grid(row=0, column=1, padx=(4, 14))
        ttk.Label(options, text="已知 baseline (mm)").grid(row=0, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.known_baseline_var, width=9).grid(row=0, column=3, padx=(4, 14))
        ttk.Label(options, text="baseline gate (mm)").grid(row=0, column=4, sticky="w")
        ttk.Entry(options, textvariable=self.min_baseline_var, width=7).grid(row=0, column=5, padx=(4, 2))
        ttk.Label(options, text="～").grid(row=0, column=6)
        ttk.Entry(options, textvariable=self.max_baseline_var, width=7).grid(row=0, column=7, padx=(2, 14))
        ttk.Checkbutton(options, text="Local window", variable=self.local_window_var).grid(row=0, column=8, padx=5)
        ttk.Label(options, text="半徑").grid(row=0, column=9)
        ttk.Spinbox(options, from_=0, to=10, textvariable=self.local_window_radius_var, width=4).grid(
            row=0, column=10, padx=(3, 10)
        )
        ttk.Checkbutton(options, text="KLT", variable=self.klt_var).grid(row=0, column=11, padx=5)

        ttk.Label(options, text="Pattern 已知物理位移 A→B (mm，可留空)").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(9, 0)
        )
        for column, label, variable in (
            (3, "dx", self.dx_var),
            (5, "dy", self.dy_var),
            (7, "dz", self.dz_var),
        ):
            ttk.Label(options, text=label).grid(row=1, column=column, sticky="e", pady=(9, 0))
            ttk.Entry(options, textvariable=variable, width=9).grid(
                row=1, column=column + 1, padx=(3, 9), pady=(9, 0)
            )
        ttk.Label(
            options,
            text=(
                "輸入是 Pattern 實體 A→B；backend 輸出 t_rel 是 B→A。"
                "空白軸視為 0。"
            ),
            foreground="#555555",
        ).grid(row=1, column=9, columnspan=5, sticky="w", pady=(9, 0))

        actions = ttk.Frame(self, padding=(8, 4))
        actions.grid(row=2, column=0, sticky="ew")
        self.run_button = ttk.Button(actions, text="開始 Marker-only 分析", command=self._start_analysis)
        self.run_button.pack(side="left")
        self.export_json_button = ttk.Button(
            actions, text="匯出 JSON", command=self._export_json, state="disabled"
        )
        self.export_json_button.pack(side="left", padx=(8, 3))
        self.export_csv_button = ttk.Button(
            actions, text="匯出 CSV", command=self._export_csv, state="disabled"
        )
        self.export_csv_button.pack(side="left", padx=3)
        ttk.Button(actions, text="清除紀錄", command=self._clear_log).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(
            actions, maximum=100.0, variable=self.progress_var, length=320, mode="determinate"
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(15, 8))
        ttk.Label(actions, textvariable=self.status_var, anchor="e").pack(side="right")

        image_frame = ttk.Frame(self)
        image_frame.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(1, weight=1)
        ttk.Label(image_frame, text="Video A 選定 frame", anchor="center").grid(row=0, column=0, sticky="ew")
        ttk.Label(image_frame, text="Video B 選定 frame", anchor="center").grid(row=0, column=1, sticky="ew")
        self.canvas_a = tk.Canvas(image_frame, background="#181818", highlightthickness=1)
        self.canvas_b = tk.Canvas(image_frame, background="#181818", highlightthickness=1)
        self.canvas_a.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.canvas_b.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        self.canvas_a.bind("<Configure>", self._schedule_render)
        self.canvas_b.bind("<Configure>", self._schedule_render)
        self._set_canvas_placeholder(self.canvas_a, "尚未分析")
        self._set_canvas_placeholder(self.canvas_b, "尚未分析")

        notebook = ttk.Notebook(self)
        notebook.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        metrics_page = ttk.Frame(notebook)
        log_page = ttk.Frame(notebook)
        notebook.add(metrics_page, text="RT 與誤差")
        notebook.add(log_page, text="執行紀錄")
        metrics_page.columnconfigure(0, weight=1)
        metrics_page.rowconfigure(0, weight=1)
        log_page.columnconfigure(0, weight=1)
        log_page.rowconfigure(0, weight=1)
        self.metrics_text = tk.Text(metrics_page, wrap="none", height=12, font=("Consolas", 10))
        self.log_text = tk.Text(log_page, wrap="word", height=12, font=("Consolas", 9))
        self._add_scrollbars(metrics_page, self.metrics_text)
        self._add_scrollbars(log_page, self.log_text)
        self.metrics_text.insert("1.0", "分析完成後會顯示 R、t、baseline、誤差與 marker diagnostics。\n")
        self.metrics_text.configure(state="disabled")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _add_scrollbars(parent: ttk.Frame, widget: tk.Text) -> None:
        vertical = ttk.Scrollbar(parent, orient="vertical", command=widget.yview)
        horizontal = ttk.Scrollbar(parent, orient="horizontal", command=widget.xview)
        widget.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        widget.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

    @staticmethod
    def _path_row(
        parent: ttk.LabelFrame,
        row: int,
        label: str,
        variable: tk.StringVar,
        command: Callable[[], None],
    ) -> None:
        ttk.Label(parent, text=label, width=28).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(parent, text="瀏覽…", command=command).grid(row=row, column=2, pady=2)

    def _choose_video_a(self) -> None:
        self._choose_file(self.video_a_var, VIDEO_FILE_TYPES, "選擇 Video A")

    def _choose_video_b(self) -> None:
        self._choose_file(self.video_b_var, VIDEO_FILE_TYPES, "選擇 Video B")

    def _choose_calibration(self) -> None:
        chosen = self._choose_file(self.calibration_var, JSON_FILE_TYPES, "選擇單目相機標定 JSON")
        if not chosen:
            return
        try:
            _matrix, _distortion, metadata = load_monocular_calibration(chosen)
            self.status_var.set(
                f"標定載入成功：fx={metadata['fx']:.2f}, fy={metadata['fy']:.2f}, {metadata['source']}"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("標定檔錯誤", str(exc), parent=self)

    @staticmethod
    def _choose_file(variable: tk.StringVar, filetypes: list[tuple[str, str]], title: str) -> str:
        initial = Path(variable.get()).parent if variable.get().strip() else Path.cwd()
        chosen = filedialog.askopenfilename(title=title, initialdir=str(initial), filetypes=filetypes)
        if chosen:
            variable.set(str(Path(chosen).resolve()))
        return chosen

    @staticmethod
    def _required_float(text: str, label: str, *, minimum: float | None = None) -> float:
        try:
            value = float(text.strip())
        except ValueError as exc:
            raise ValueError(f"{label} 必須是數字。") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} 必須是有限數值。")
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} 必須大於或等於 {minimum}。")
        return value

    def _collect_inputs(self) -> dict[str, Any]:
        video_a = Path(self.video_a_var.get().strip()).expanduser()
        video_b = Path(self.video_b_var.get().strip()).expanduser()
        calibration = Path(self.calibration_var.get().strip()).expanduser()
        for label, path in (("Video A", video_a), ("Video B", video_b), ("標定 JSON", calibration)):
            if not str(path).strip() or not path.is_file():
                raise ValueError(f"{label} 檔案不存在：{path}")

        marker_size = self._required_float(self.marker_size_var.get(), "Marker 邊長", minimum=1e-9)
        known_text = self.known_baseline_var.get().strip()
        known_baseline = (
            self._required_float(known_text, "已知 baseline", minimum=0.0) if known_text else None
        )
        min_baseline = self._required_float(self.min_baseline_var.get(), "最小 baseline", minimum=0.0)
        max_baseline = self._required_float(self.max_baseline_var.get(), "最大 baseline", minimum=0.0)
        if max_baseline <= min_baseline:
            raise ValueError("最大 baseline 必須大於最小 baseline。")

        vector_text = (self.dx_var.get().strip(), self.dy_var.get().strip(), self.dz_var.get().strip())
        known_vector = None
        if any(vector_text):
            known_vector = tuple(
                self._required_float(component, label) if component else 0.0
                for component, label in zip(vector_text, ("dx", "dy", "dz"))
            )
            if np.linalg.norm(known_vector) <= 1e-12:
                raise ValueError("已知位移向量不能是零向量；若不比較方向，請將三軸全部留空。")
            vector_norm = float(np.linalg.norm(known_vector))
            if known_baseline is not None and not math.isclose(
                known_baseline, vector_norm, rel_tol=0.005, abs_tol=0.1
            ):
                raise ValueError(
                    "已知 baseline 與位移向量長度不一致："
                    f"{known_baseline:.4f} mm vs {vector_norm:.4f} mm。"
                )

        radius = int(self.local_window_radius_var.get())
        if radius < 0:
            raise ValueError("Local window 半徑不能小於 0。")

        camera_matrix, distortion, calibration_metadata = load_monocular_calibration(calibration)
        return {
            "video_a_path": str(video_a.resolve()),
            "video_b_path": str(video_b.resolve()),
            "calibration_path": str(calibration.resolve()),
            "camera_matrix": camera_matrix,
            "distortion": distortion,
            "calibration_metadata": calibration_metadata,
            "marker_size_mm": marker_size,
            "known_translation_mm": known_baseline,
            "known_translation_vector": known_vector,
            "min_baseline_mm": min_baseline,
            "max_baseline_mm": max_baseline,
            "local_window": bool(self.local_window_var.get()),
            "local_window_radius": radius,
            "klt_enabled": bool(self.klt_var.get()),
        }

    def _start_analysis(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            inputs = self._collect_inputs()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("輸入資料錯誤", str(exc), parent=self)
            return

        self._result = None
        self._run_inputs = inputs
        self._display_frames = None
        self._photos = [None, None]
        self._canvas_items = [None, None]
        self._set_canvas_placeholder(self.canvas_a, "分析中…")
        self._set_canvas_placeholder(self.canvas_b, "分析中…")
        self.progress_var.set(0.0)
        self.status_var.set("準備分析…")
        self.run_button.configure(state="disabled")
        self.export_json_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")
        self._replace_metrics("分析進行中…\n")
        self._append_log("=" * 72)
        self._append_log(f"Video A: {inputs['video_a_path']}")
        self._append_log(f"Video B: {inputs['video_b_path']}")
        self._append_log(
            f"Marker-only | local_window={inputs['local_window']} "
            f"radius={inputs['local_window_radius']} | KLT={inputs['klt_enabled']} | SIFT=disabled"
        )

        self._worker = threading.Thread(
            target=self._analysis_worker,
            args=(inputs,),
            name="marker-only-rt-validation",
            daemon=True,
        )
        self._worker.start()
        self.after(80, self._poll_events)

    def _analysis_worker(self, inputs: dict[str, Any]) -> None:
        started = time.perf_counter()

        def progress_callback(percent: float, message: str = "") -> None:
            self._events.put(("progress", (float(percent), str(message))))

        def log_callback(message: Any) -> None:
            self._events.put(("log", str(message)))

        try:
            from Algorithm import (
                video_pose_analysis_temporal_unified_pattern_guided_local_window_validation
                as validation_backend,
            )

            result = validation_backend.analyze_two_video_segments(
                inputs["video_a_path"],
                inputs["video_b_path"],
                inputs["camera_matrix"],
                inputs["distortion"],
                marker_size_mm=inputs["marker_size_mm"],
                known_translation_mm=inputs["known_translation_mm"],
                known_translation_vector=inputs["known_translation_vector"],
                local_window=inputs["local_window"],
                local_window_radius=inputs["local_window_radius"],
                klt_enabled=inputs["klt_enabled"],
                progress_callback=progress_callback,
                log_callback=log_callback,
                min_baseline_mm=inputs["min_baseline_mm"],
                max_baseline_mm=inputs["max_baseline_mm"],
            )
            if not isinstance(result, dict):
                raise RuntimeError("Validation backend 必須回傳 dict。")
            result.setdefault("runtime_s", time.perf_counter() - started)
            self._events.put(("done", result))
        except Exception as exc:  # Worker must deliver all failures to Tk's main thread.
            self._events.put(("error", (str(exc), traceback.format_exc())))

    def _poll_events(self) -> None:
        processed_terminal_event = False
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                percent, message = payload
                # The validation backend contract defines percent on a 0..100 scale.
                percent = float(np.clip(percent, 0.0, 100.0))
                self.progress_var.set(percent)
                if message:
                    self.status_var.set(message)
            elif kind == "log":
                self._append_log(payload)
            elif kind == "done":
                processed_terminal_event = True
                self._handle_result(payload)
            elif kind == "error":
                processed_terminal_event = True
                message, trace = payload
                self._handle_error(message, trace)

        if not processed_terminal_event and self._worker is not None and self._worker.is_alive():
            self.after(80, self._poll_events)

    def _handle_result(self, result: dict[str, Any]) -> None:
        self._result = result
        self.progress_var.set(100.0)
        self.run_button.configure(state="normal")
        self.export_json_button.configure(state="normal")
        self.export_csv_button.configure(state="normal")

        sift_calls = int(result.get("sift_calls", 0) or 0)
        if sift_calls != 0:
            self.status_var.set(f"分析完成，但警告：SIFT calls = {sift_calls}")
            self._append_log(f"警告：marker-only validation 回報 SIFT calls = {sift_calls}")
        else:
            self.status_var.set("分析完成（SIFT calls = 0）")
        self._append_log("分析完成。")

        try:
            frame_a = result.get("selected_frame_a_bgr")
            frame_b = result.get("selected_frame_b_bgr")
            index_a = int(result.get("selected_frame_a_index", -1))
            index_b = int(result.get("selected_frame_b_index", -1))
            if frame_a is None:
                assert self._run_inputs is not None
                frame_a = _read_video_frame(self._run_inputs["video_a_path"], index_a)
            if frame_b is None:
                assert self._run_inputs is not None
                frame_b = _read_video_frame(self._run_inputs["video_b_path"], index_b)
            display_a = draw_marker_overlay(frame_a, result.get("corners_a"), f"Video A | frame {index_a}")
            display_b = draw_marker_overlay(frame_b, result.get("corners_b"), f"Video B | frame {index_b}")
            self._display_frames = (display_a, display_b)
            self._schedule_render()
        except Exception as exc:
            self._append_log(f"選定 frame 顯示失敗：{exc}")

        self._replace_metrics(self._format_metrics(result))

    def _handle_error(self, message: str, trace: str) -> None:
        self.progress_var.set(0.0)
        self.status_var.set("分析失敗")
        self.run_button.configure(state="normal")
        self.export_json_button.configure(state="disabled")
        self.export_csv_button.configure(state="disabled")
        self._replace_metrics(f"分析失敗\n\n{message}\n")
        self._append_log(trace.rstrip())
        messagebox.showerror("RT 驗證失敗", message, parent=self)

    def _format_metrics(self, result: dict[str, Any]) -> str:
        lines: list[str] = []
        index_a = result.get("selected_frame_a_index", "—")
        index_b = result.get("selected_frame_b_index", "—")
        lines.append("Marker-only RT validation")
        lines.append(f"Selected frame: Video A = {index_a}, Video B = {index_b}")
        lines.append(
            f"Feature mode: {result.get('feature_mode', 'marker_only')} | "
            f"SIFT calls: {result.get('sift_calls', 0)}"
        )

        rotation = result.get("R_rel")
        translation = result.get("t_rel")
        if rotation is not None:
            matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            lines.extend(("", "R_rel (B→A):", np.array2string(matrix, precision=9, suppress_small=True)))
        if translation is not None:
            vector = np.asarray(translation, dtype=np.float64).reshape(-1)
            lines.extend(("", "t_rel (B→A, mm):", np.array2string(vector, precision=9, suppress_small=True)))

        baseline = _float_or_none(result.get("baseline_mm"))
        if baseline is None and translation is not None:
            baseline = float(np.linalg.norm(np.asarray(translation, dtype=np.float64)))
        known = self._known_baseline_from_inputs()
        rotation_error = _rotation_angle_deg(rotation) if rotation is not None else None
        baseline_abs_error = abs(baseline - known) if baseline is not None and known is not None else None
        baseline_pct_error = (
            baseline_abs_error / known * 100.0
            if baseline_abs_error is not None and known is not None and known > 0
            else None
        )

        lines.append("")
        lines.append(f"Estimated baseline: {baseline:.6f} mm" if baseline is not None else "Estimated baseline: —")
        lines.append(f"Known baseline:     {known:.6f} mm" if known is not None else "Known baseline:     —")
        lines.append(
            f"Baseline abs error: {baseline_abs_error:.6f} mm"
            if baseline_abs_error is not None
            else "Baseline abs error: —"
        )
        lines.append(
            f"Baseline error:     {baseline_pct_error:.4f} %"
            if baseline_pct_error is not None
            else "Baseline error:     —"
        )
        lines.append(
            f"Rotation error vs identity: {rotation_error:.6f} deg"
            if rotation_error is not None
            else "Rotation error vs identity: —"
        )
        runtime = _float_or_none(result.get("runtime_s"))
        lines.append(f"Runtime: {runtime:.3f} s" if runtime is not None else "Runtime: —")

        ground_truth = result.get("ground_truth")
        if isinstance(ground_truth, dict) and ground_truth:
            lines.extend(("", "Ground-truth comparison reported by backend:"))
            for key, value in _flatten_scalars(ground_truth):
                lines.append(f"  {key}: {value}")

        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            lines.extend(("", "Marker / temporal diagnostics:"))
            for key, value in _flatten_scalars(diagnostics):
                lines.append(f"  {key}: {value}")

        return "\n".join(lines) + "\n"

    def _schedule_render(self, _event: tk.Event | None = None) -> None:
        if self._render_job is None:
            self._render_job = self.after_idle(self._render_selected_frames)

    def _render_selected_frames(self) -> None:
        self._render_job = None
        if self._display_frames is None:
            return
        for slot, (canvas, frame) in enumerate(
            ((self.canvas_a, self._display_frames[0]), (self.canvas_b, self._display_frames[1]))
        ):
            width = max(2, canvas.winfo_width())
            height = max(2, canvas.winfo_height())
            frame_h, frame_w = frame.shape[:2]
            scale = min(width / frame_w, height / frame_h)
            shown_w = max(1, int(round(frame_w * scale)))
            shown_h = max(1, int(round(frame_h * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            shown = cv2.resize(frame, (shown_w, shown_h), interpolation=interpolation)
            rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
            ppm = f"P6\n{shown_w} {shown_h}\n255\n".encode("ascii") + rgb.tobytes()
            photo = tk.PhotoImage(data=ppm, format="PPM")
            self._photos[slot] = photo
            if self._canvas_items[slot] is None:
                canvas.delete("all")
                self._canvas_items[slot] = canvas.create_image(
                    width // 2, height // 2, anchor="center", image=photo
                )
            else:
                canvas.coords(self._canvas_items[slot], width // 2, height // 2)
                canvas.itemconfigure(self._canvas_items[slot], image=photo)

    @staticmethod
    def _set_canvas_placeholder(canvas: tk.Canvas, message: str) -> None:
        canvas.delete("all")
        canvas.create_text(12, 12, anchor="nw", text=message, fill="#bbbbbb", font=("TkDefaultFont", 11))

    def _replace_metrics(self, text: str) -> None:
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", text)
        self.metrics_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _export_payload(self) -> dict[str, Any]:
        if self._result is None or self._run_inputs is None:
            raise RuntimeError("目前沒有可匯出的分析結果。")
        configuration = {
            key: value
            for key, value in self._run_inputs.items()
            if key not in {"camera_matrix", "distortion"}
        }
        configuration["camera_matrix"] = self._run_inputs["camera_matrix"]
        configuration["distortion"] = self._run_inputs["distortion"]

        baseline = _float_or_none(self._result.get("baseline_mm"))
        if baseline is None and self._result.get("t_rel") is not None:
            baseline = float(
                np.linalg.norm(np.asarray(self._result["t_rel"], dtype=np.float64))
            )
        known = self._known_baseline_from_inputs()
        absolute_error = (
            abs(baseline - known) if baseline is not None and known is not None else None
        )
        derived_metrics = {
            "rotation_error_vs_identity_deg": _rotation_angle_deg(
                self._result.get("R_rel")
            ),
            "estimated_baseline_mm": baseline,
            "known_baseline_mm": known,
            "baseline_absolute_error_mm": absolute_error,
            "baseline_error_percent": (
                absolute_error / known * 100.0
                if absolute_error is not None and known is not None and known > 0
                else None
            ),
        }
        return {
            "schema": "two_video_marker_only_rt_validation/v1",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "configuration": _json_safe(configuration),
            "ui_derived_metrics": _json_safe(derived_metrics),
            "result": _json_safe(self._result),
        }

    def _known_baseline_from_inputs(self) -> float | None:
        if self._run_inputs is None:
            return None
        known = self._run_inputs.get("known_translation_mm")
        if known is not None:
            return float(known)
        vector = self._run_inputs.get("known_translation_vector")
        if vector is None:
            return None
        return float(np.linalg.norm(np.asarray(vector, dtype=np.float64).reshape(3)))

    def _export_json(self) -> None:
        try:
            payload = self._export_payload()
        except RuntimeError as exc:
            messagebox.showinfo("無結果", str(exc), parent=self)
            return
        target = filedialog.asksaveasfilename(
            title="匯出 RT 驗證 JSON",
            defaultextension=".json",
            filetypes=JSON_FILE_TYPES,
            initialfile="rt_two_video_validation_result.json",
        )
        if not target:
            return
        try:
            with Path(target).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self.status_var.set(f"已匯出 JSON：{Path(target).name}")
        except OSError as exc:
            messagebox.showerror("匯出失敗", str(exc), parent=self)

    def _export_csv(self) -> None:
        try:
            payload = self._export_payload()
        except RuntimeError as exc:
            messagebox.showinfo("無結果", str(exc), parent=self)
            return
        target = filedialog.asksaveasfilename(
            title="匯出 RT 驗證 CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="rt_two_video_validation_result.csv",
        )
        if not target:
            return
        row = _flatten_for_csv(payload)
        try:
            with Path(target).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
            self.status_var.set(f"已匯出 CSV：{Path(target).name}")
        except OSError as exc:
            messagebox.showerror("匯出失敗", str(exc), parent=self)

    def _on_close(self) -> None:
        # The analysis thread is daemonized and never touches Tk directly.
        self.destroy()


def main() -> None:
    app = TwoVideoRTValidationUI()
    app.mainloop()


if __name__ == "__main__":
    main()
