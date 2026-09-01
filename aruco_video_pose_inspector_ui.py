"""Interactive ArUco video pose inspector.

This tool uses the same corner preprocessing, IPPE square pose hypotheses and
incidence-angle definition as the angle-guided frame selector in
``video_pose_analysis_temporal_unified_pattern_guided_local_window.py``.

Angle definition: the acute angle between the marker normal and the line from
the marker centre to the camera.  Zero degrees is a fronto-parallel view.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from Algorithm import camera_preprocess as camera_algo
from Algorithm import (
    video_pose_analysis_temporal_unified_pattern_guided_local_window as pose_algo,
)


APP_TITLE = "ArUco Video Pose / Angle Inspector"
DEFAULT_CALIBRATION = "calibration_result_Zebra_1_monocular.json"
DEFAULT_MARKER_SIZE_MM = 8.25
TARGET_A_DEG = 15.0
TARGET_B_DEG = 35.0


def _outlined_text(image, text, origin, scale=0.8, color=(255, 70, 0)):
    cv2.putText(image, str(text), tuple(origin), cv2.FONT_HERSHEY_SIMPLEX,
                float(scale), (255, 255, 255), 7, cv2.LINE_AA)
    cv2.putText(image, str(text), tuple(origin), cv2.FONT_HERSHEY_SIMPLEX,
                float(scale), color, 2, cv2.LINE_AA)


class ArucoPoseEstimator:
    """Thread-local detector matching the production angle scan."""

    def __init__(self, camera_matrix, distortion, marker_size_mm):
        self.camera_matrix = np.asarray(camera_matrix, np.float64).reshape(3, 3)
        self.distortion = np.asarray(distortion, np.float64).reshape(-1, 1)
        self.marker_size_mm = float(marker_size_mm)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_100)
        self.parameters = (
            cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "ArucoDetector")
            else cv2.aruco.DetectorParameters_create())
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector") else None)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def analyze(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters)
        if ids is None or len(ids) == 0:
            return {"corners": {}, "measurements": {}}
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.0001,
        )
        for corner in corners:
            try:
                cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), term)
            except cv2.error:
                pass
        corner_dict = {
            int(marker_id): np.asarray(corner, np.float32).reshape(4, 2)
            for marker_id, corner in zip(ids.reshape(-1), corners)
        }
        measurements = {}
        for marker_id in sorted(corner_dict):
            measurement = pose_algo._angle_guided_marker_measurement(
                corner_dict,
                marker_id,
                self.camera_matrix,
                self.distortion,
                self.marker_size_mm,
            )
            if measurement is not None:
                measurements[int(marker_id)] = measurement
        return {"corners": corner_dict, "measurements": measurements}


class PoseInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x850")
        self.root.minsize(980, 700)

        self.video_path = None
        self.calibration_path = None
        self.camera_matrix = None
        self.distortion = None
        self.estimator = None
        self.capture = None
        self.total_frames = 0
        self.video_fps = 25.0
        self.frame_width = 0
        self.frame_height = 0
        self.current_index = -1
        self.play_direction = 0
        self.play_after_id = None
        self.seek_after_id = None
        self.photo = None
        self.updating_slider = False
        self.current_result = None
        self.scan_results = {}
        self.scan_queue = queue.Queue()
        self.scan_generation = 0
        self.graph_bounds = None

        self.video_var = tk.StringVar(value="尚未載入影片")
        self.calibration_var = tk.StringVar(value="尚未載入標定")
        self.marker_size_var = tk.StringVar(value=str(DEFAULT_MARKER_SIZE_MM))
        self.marker_id_var = tk.StringVar(value="auto")
        self.speed_var = tk.StringVar(value="1.0")
        self.frame_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="請先載入影片")
        self.pose_var = tk.StringVar(value="Pattern: --")
        self.scan_summary_var = tk.StringVar(value="尚未掃描全片")

        self._build_ui()
        self._load_default_calibration()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll_scan_queue)

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="載入影片", command=self.open_video).grid(
            row=0, column=0, padx=3)
        ttk.Label(toolbar, textvariable=self.video_var, width=55).grid(
            row=0, column=1, padx=5, sticky="w")
        ttk.Button(toolbar, text="載入標定 JSON", command=self.open_calibration).grid(
            row=1, column=0, padx=3, pady=4)
        ttk.Label(toolbar, textvariable=self.calibration_var, width=55).grid(
            row=1, column=1, padx=5, sticky="w")

        settings = ttk.Frame(toolbar)
        settings.grid(row=0, column=2, rowspan=2, padx=10, sticky="e")
        ttk.Label(settings, text="Marker mm").grid(row=0, column=0)
        ttk.Entry(settings, textvariable=self.marker_size_var, width=8).grid(
            row=0, column=1, padx=3)
        ttk.Label(settings, text="顯示 ID").grid(row=0, column=2, padx=(8, 0))
        ttk.Entry(settings, textvariable=self.marker_id_var, width=7).grid(
            row=0, column=3, padx=3)
        ttk.Button(settings, text="套用", command=self.apply_settings).grid(
            row=0, column=4, padx=4)
        ttk.Button(settings, text="掃描全片", command=self.scan_entire_video).grid(
            row=1, column=0, columnspan=2, pady=5)
        self.export_button = ttk.Button(
            settings, text="匯出 CSV", command=self.export_scan_csv, state="disabled")
        self.export_button.grid(row=1, column=2, columnspan=2, pady=5)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8)
        left = ttk.Frame(body)
        right = ttk.Frame(body, width=300)
        body.add(left, weight=5)
        body.add(right, weight=2)

        self.video_canvas = tk.Canvas(
            left, bg="#111111", highlightthickness=1, highlightbackground="#555555")
        self.video_canvas.pack(fill="both", expand=True)
        self.video_canvas.bind("<Configure>", lambda _event: self._redraw_current())

        ttk.Label(right, text="目前姿態", font=("Microsoft JhengHei", 12, "bold")).pack(
            anchor="w", padx=8, pady=(5, 2))
        ttk.Label(
            right, textvariable=self.pose_var, justify="left", wraplength=290,
            font=("Consolas", 10)).pack(anchor="w", padx=8, pady=4)
        ttk.Separator(right).pack(fill="x", padx=8, pady=5)
        ttk.Label(right, text="全片掃描摘要", font=("Microsoft JhengHei", 12, "bold")).pack(
            anchor="w", padx=8, pady=2)
        ttk.Label(
            right, textvariable=self.scan_summary_var, justify="left",
            wraplength=290, font=("Consolas", 9)).pack(
            anchor="w", padx=8, pady=4)

        controls = ttk.Frame(self.root, padding=(8, 4))
        controls.pack(fill="x")
        ttk.Button(controls, text="⏮ -10", command=lambda: self.step_frame(-10)).pack(
            side="left", padx=2)
        ttk.Button(controls, text="◀ 倒播", command=lambda: self.start_playback(-1)).pack(
            side="left", padx=2)
        ttk.Button(controls, text="◁ 前一格", command=lambda: self.step_frame(-1)).pack(
            side="left", padx=2)
        ttk.Button(controls, text="暫停", command=self.pause).pack(side="left", padx=2)
        ttk.Button(controls, text="下一格 ▷", command=lambda: self.step_frame(1)).pack(
            side="left", padx=2)
        ttk.Button(controls, text="正播 ▶", command=lambda: self.start_playback(1)).pack(
            side="left", padx=2)
        ttk.Button(controls, text="+10 ⏭", command=lambda: self.step_frame(10)).pack(
            side="left", padx=2)
        ttk.Label(controls, text="速度").pack(side="left", padx=(12, 2))
        ttk.Combobox(
            controls, textvariable=self.speed_var, width=5, state="readonly",
            values=("0.25", "0.5", "1.0", "1.5", "2.0")).pack(side="left")
        ttk.Label(controls, textvariable=self.status_var).pack(side="right", padx=8)

        slider_row = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        slider_row.pack(fill="x")
        self.frame_slider = ttk.Scale(
            slider_row, from_=0, to=0, variable=self.frame_var,
            command=self._slider_changed)
        self.frame_slider.pack(fill="x")

        self.scan_progress = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.scan_progress.pack(fill="x", padx=8, pady=(0, 3))

        ttk.Label(
            self.root,
            text="角度定義：0°=正對 Pattern；藍線=15°，橘線=35°。點擊曲線可跳到該 frame。",
        ).pack(anchor="w", padx=9)
        self.graph_canvas = tk.Canvas(
            self.root, height=175, bg="white", highlightthickness=1,
            highlightbackground="#888888")
        self.graph_canvas.pack(fill="x", padx=8, pady=(2, 8))
        self.graph_canvas.bind("<Button-1>", self._graph_clicked)
        self.graph_canvas.bind("<Configure>", lambda _event: self._draw_graph())

        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind("<space>", lambda _event: self.pause())

    def _load_default_calibration(self):
        path = Path(DEFAULT_CALIBRATION)
        if path.exists():
            self.load_calibration(str(path.resolve()), show_error=False)

    def open_calibration(self):
        path = filedialog.askopenfilename(
            title="選擇相機標定 JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.load_calibration(path, show_error=True)

    def load_calibration(self, path, show_error=True):
        try:
            matrix, distortion, *_ = camera_algo.load_json_camera_params(path)
            if matrix is None or distortion is None:
                raise ValueError("標定檔缺少 intrinsic_L matrix/distortion")
            self.camera_matrix = np.asarray(matrix, np.float64)
            self.distortion = np.asarray(distortion, np.float64)
            self.calibration_path = str(path)
            self.calibration_var.set(os.path.basename(path))
            self._rebuild_estimator()
            self.scan_generation += 1
            self.scan_results = {}
            self.scan_summary_var.set("標定已更新；需要時請重新掃描全片")
            self.export_button.configure(state="disabled")
            self._draw_graph()
            if self.current_index >= 0:
                self.show_frame(self.current_index)
            return True
        except Exception as exc:
            if show_error:
                messagebox.showerror("標定載入失敗", str(exc))
            return False

    def _marker_size(self):
        value = float(self.marker_size_var.get())
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Marker size 必須是正數")
        return value

    def _preferred_marker_id(self):
        text = self.marker_id_var.get().strip().lower()
        if text in ("", "auto"):
            return None
        return int(text)

    def _rebuild_estimator(self):
        if self.camera_matrix is None:
            self.estimator = None
            return
        self.estimator = ArucoPoseEstimator(
            self.camera_matrix, self.distortion, self._marker_size())

    def apply_settings(self):
        try:
            self._marker_size()
            self._preferred_marker_id()
            self._rebuild_estimator()
            self.scan_results = {}
            self.scan_summary_var.set("設定已更新；需要時請重新掃描全片")
            self.export_button.configure(state="disabled")
            self._draw_graph()
            if self.current_index >= 0:
                self.show_frame(self.current_index)
        except Exception as exc:
            messagebox.showerror("設定錯誤", str(exc))

    def open_video(self):
        path = filedialog.askopenfilename(
            title="選擇影片",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")],
        )
        if not path:
            return
        self.pause()
        if self.capture is not None:
            self.capture.release()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            messagebox.showerror("影片載入失敗", path)
            return
        self.capture = capture
        self.video_path = str(path)
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.video_fps) or self.video_fps <= 0:
            self.video_fps = 25.0
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_var.set(
            f"{os.path.basename(path)} | {self.total_frames} frames | "
            f"{self.frame_width}x{self.frame_height} @ {self.video_fps:.2f}fps")
        self.frame_slider.configure(to=max(self.total_frames - 1, 0))
        self.scan_results = {}
        self.scan_summary_var.set("尚未掃描全片")
        self.export_button.configure(state="disabled")
        self.scan_generation += 1
        self._draw_graph()
        self.show_frame(0)

    def _read_frame(self, index):
        if self.capture is None:
            return None
        index = int(np.clip(index, 0, max(self.total_frames - 1, 0)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        return frame if ok else None

    def show_frame(self, index):
        if self.capture is None or self.total_frames <= 0:
            return
        index = int(np.clip(index, 0, self.total_frames - 1))
        frame = self._read_frame(index)
        if frame is None:
            self.pause()
            self.status_var.set(f"無法讀取 frame {index}")
            return
        if index in self.scan_results:
            result = self.scan_results[index]
        elif self.estimator is not None:
            result = self.estimator.analyze(frame)
        else:
            result = {"corners": {}, "measurements": {}}
        self.current_index = index
        self.current_result = result
        self.updating_slider = True
        self.frame_var.set(float(index))
        self.updating_slider = False
        self._render_frame(frame, result)
        seconds = index / max(self.video_fps, 1e-9)
        direction = {1: "正播", -1: "倒播", 0: "暫停"}[self.play_direction]
        self.status_var.set(
            f"F{index}/{self.total_frames - 1} | {seconds:.2f}s | {direction}")

    def _selected_measurement(self, result):
        measurements = result.get("measurements", {})
        try:
            preferred = self._preferred_marker_id()
        except (TypeError, ValueError):
            preferred = None
        if preferred in measurements:
            return preferred, measurements[preferred]
        if not measurements:
            return None, None
        marker_id = max(
            measurements,
            key=lambda mid: float(measurements[mid].get("area_px2", 0.0)))
        return int(marker_id), measurements[marker_id]

    def _render_frame(self, frame, result):
        overlay = frame.copy()
        selected_id, selected = self._selected_measurement(result)
        if selected is None:
            _outlined_text(overlay, "ArUco not detected", (30, 55), 1.0)
            self.pose_var.set("Pattern: 未偵測")
        else:
            summary = (
                f"ID {selected_id} | Distance {selected['range_mm']:.1f} mm "
                f"({selected['range_mm']/10.0:.1f} cm) | "
                f"Angle {selected['incidence_deg']:.2f} deg")
            _outlined_text(overlay, summary, (30, 55), 0.92)
            lines = [
                f"Frame       : {self.current_index}",
                f"Marker ID   : {selected_id}",
                f"Distance    : {selected['range_mm']:.3f} mm",
                f"Angle       : {selected['incidence_deg']:.3f} deg",
                f"Target 15 Δ : {abs(selected['incidence_deg']-15.0):.3f} deg",
                f"Target 35 Δ : {abs(selected['incidence_deg']-35.0):.3f} deg",
                f"PnP RMS     : {selected['reprojection_rms_px']:.4f} px",
                f"IPPE branch : {selected['branch']}",
            ]
            self.pose_var.set("\n".join(lines))

        for marker_id, points in result.get("corners", {}).items():
            pts = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
            color = (255, 70, 0) if marker_id == selected_id else (0, 220, 0)
            cv2.polylines(overlay, [pts], True, color, 3, cv2.LINE_AA)
            measurement = result.get("measurements", {}).get(marker_id)
            if measurement is not None:
                label = (
                    f"ID {marker_id}: {measurement['incidence_deg']:.2f} deg | "
                    f"{measurement['range_mm']:.1f} mm")
                origin = tuple(pts[0, 0].tolist())
                _outlined_text(
                    overlay, label, (origin[0], max(25, origin[1] - 12)), 0.62,
                    color=color)

        canvas_w = max(self.video_canvas.winfo_width(), 320)
        canvas_h = max(self.video_canvas.winfo_height(), 240)
        scale = min(canvas_w / overlay.shape[1], canvas_h / overlay.shape[0])
        shown_w = max(1, int(round(overlay.shape[1] * scale)))
        shown_h = max(1, int(round(overlay.shape[0] * scale)))
        shown = cv2.resize(
            overlay, (shown_w, shown_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        ppm = f"P6\n{shown_w} {shown_h}\n255\n".encode("ascii") + rgb.tobytes()
        self.photo = tk.PhotoImage(data=ppm, format="PPM")
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.photo, anchor="center")

    def _redraw_current(self):
        if self.current_index >= 0 and self.capture is not None:
            self.show_frame(self.current_index)

    def start_playback(self, direction):
        if self.capture is None:
            return
        self.play_direction = 1 if direction > 0 else -1
        if self.play_after_id is None:
            self._play_tick()

    def pause(self):
        self.play_direction = 0
        if self.play_after_id is not None:
            try:
                self.root.after_cancel(self.play_after_id)
            except tk.TclError:
                pass
            self.play_after_id = None
        if self.current_index >= 0:
            self.status_var.set(
                f"F{self.current_index}/{max(self.total_frames-1, 0)} | 暫停")

    def _play_tick(self):
        self.play_after_id = None
        if self.play_direction == 0 or self.capture is None:
            return
        next_index = self.current_index + self.play_direction
        if next_index < 0 or next_index >= self.total_frames:
            self.pause()
            return
        self.show_frame(next_index)
        speed = max(float(self.speed_var.get()), 0.05)
        delay_ms = max(1, int(round(1000.0 / (self.video_fps * speed))))
        self.play_after_id = self.root.after(delay_ms, self._play_tick)

    def step_frame(self, amount):
        if self.capture is None:
            return
        self.pause()
        self.show_frame(self.current_index + int(amount))

    def _slider_changed(self, value):
        if self.updating_slider or self.capture is None:
            return
        if self.seek_after_id is not None:
            try:
                self.root.after_cancel(self.seek_after_id)
            except tk.TclError:
                pass
        target = int(round(float(value)))
        self.seek_after_id = self.root.after(
            70, lambda: self._seek_from_slider(target))

    def _seek_from_slider(self, index):
        self.seek_after_id = None
        self.pause()
        self.show_frame(index)

    def scan_entire_video(self):
        if not self.video_path:
            messagebox.showwarning("尚未載入", "請先載入影片")
            return
        if self.camera_matrix is None:
            messagebox.showwarning("尚未載入", "請先載入標定 JSON")
            return
        try:
            marker_size = self._marker_size()
        except Exception as exc:
            messagebox.showerror("設定錯誤", str(exc))
            return
        self.scan_generation += 1
        generation = self.scan_generation
        self.scan_progress["value"] = 0
        self.scan_summary_var.set("正在掃描全片...")
        self.export_button.configure(state="disabled")
        video_path = self.video_path
        matrix = self.camera_matrix.copy()
        distortion = self.distortion.copy()

        def worker():
            capture = cv2.VideoCapture(video_path)
            estimator = ArucoPoseEstimator(
                matrix, distortion, marker_size)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            output = {}
            index = 0
            try:
                while generation == self.scan_generation:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    output[index] = estimator.analyze(frame)
                    index += 1
                    if index == 1 or index % 5 == 0 or index == total:
                        self.scan_queue.put((
                            "progress", generation,
                            100.0 * index / max(total, 1), index, total))
                if generation == self.scan_generation:
                    self.scan_queue.put(("done", generation, output))
            except Exception as exc:
                self.scan_queue.put(("error", generation, str(exc)))
            finally:
                capture.release()

        threading.Thread(target=worker, daemon=True).start()

    def _poll_scan_queue(self):
        try:
            while True:
                message = self.scan_queue.get_nowait()
                kind, generation = message[:2]
                if generation != self.scan_generation:
                    continue
                if kind == "progress":
                    _kind, _generation, percent, index, total = message
                    self.scan_progress["value"] = percent
                    self.scan_summary_var.set(f"正在掃描 {index}/{total}...")
                elif kind == "done":
                    self.scan_results = message[2]
                    self.scan_progress["value"] = 100
                    self._update_scan_summary()
                    self._draw_graph()
                    self.export_button.configure(state="normal")
                    if self.current_index >= 0:
                        frame = self._read_frame(self.current_index)
                        if frame is not None:
                            self.current_result = self.scan_results.get(
                                self.current_index, self.current_result)
                            self._render_frame(frame, self.current_result)
                elif kind == "error":
                    self.scan_summary_var.set(f"掃描失敗：{message[2]}")
                    messagebox.showerror("掃描失敗", message[2])
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(80, self._poll_scan_queue)

    def _marker_series(self):
        by_marker = {}
        for frame_index, result in sorted(self.scan_results.items()):
            for marker_id, measurement in result.get("measurements", {}).items():
                by_marker.setdefault(int(marker_id), []).append((
                    int(frame_index),
                    float(measurement["incidence_deg"]),
                    float(measurement["range_mm"]),
                    float(measurement["reprojection_rms_px"]),
                ))
        return by_marker

    def _active_graph_marker(self, series):
        try:
            preferred = self._preferred_marker_id()
        except (TypeError, ValueError):
            preferred = None
        if preferred in series:
            return preferred
        if not series:
            return None
        return max(series, key=lambda marker_id: len(series[marker_id]))

    def _update_scan_summary(self):
        series = self._marker_series()
        if not series:
            self.scan_summary_var.set("全片沒有偵測到有效 ArUco pose")
            return
        lines = []
        for marker_id in sorted(series):
            values = series[marker_id]
            angles = np.asarray([value[1] for value in values], np.float64)
            near15 = min(values, key=lambda value: abs(value[1] - TARGET_A_DEG))
            near35 = min(values, key=lambda value: abs(value[1] - TARGET_B_DEG))
            lines.extend([
                f"ID {marker_id}: {len(values)}/{self.total_frames} frames",
                f"  range {angles.min():.2f}° / med {np.median(angles):.2f}° / max {angles.max():.2f}°",
                f"  near15 F{near15[0]}={near15[1]:.2f}°",
                f"  near35 F{near35[0]}={near35[1]:.2f}°",
            ])
        self.scan_summary_var.set("\n".join(lines))

    def _draw_graph(self):
        canvas = self.graph_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 640)
        height = max(canvas.winfo_height(), 170)
        left, top, right, bottom = 52, 12, width - 18, height - 28
        self.graph_bounds = (left, top, right, bottom)
        canvas.create_rectangle(left, top, right, bottom, outline="#777777")
        series = self._marker_series()
        marker_id = self._active_graph_marker(series)
        if marker_id is None or self.total_frames <= 1:
            canvas.create_text(
                width // 2, height // 2, text="掃描全片後顯示角度曲線",
                fill="#555555")
            return
        values = series[marker_id]
        max_angle = max(45.0, max(value[1] for value in values) + 5.0)

        def x_of(frame_index):
            return left + (right - left) * frame_index / max(self.total_frames - 1, 1)

        def y_of(angle):
            return bottom - (bottom - top) * float(angle) / max_angle

        for target, color in ((15.0, "#1976D2"), (35.0, "#F57C00")):
            y = y_of(target)
            canvas.create_line(left, y, right, y, fill=color, dash=(5, 4), width=2)
            canvas.create_text(left - 5, y, text=f"{target:.0f}°", anchor="e", fill=color)
        points = []
        for frame_index, angle, _distance, _rms in values:
            points.extend((x_of(frame_index), y_of(angle)))
        if len(points) >= 4:
            canvas.create_line(*points, fill="#00897B", width=2)
        for frame_index, angle, _distance, _rms in values[::max(1, len(values)//80)]:
            x, y = x_of(frame_index), y_of(angle)
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#00695C", outline="")
        canvas.create_text(
            left + 6, top + 6, text=f"Marker ID {marker_id}", anchor="nw",
            fill="#222222", font=("Consolas", 9, "bold"))
        canvas.create_text(left, bottom + 15, text="F0", anchor="w")
        canvas.create_text(
            right, bottom + 15, text=f"F{self.total_frames-1}", anchor="e")

    def _graph_clicked(self, event):
        if not self.graph_bounds or self.total_frames <= 0:
            return
        left, _top, right, _bottom = self.graph_bounds
        ratio = np.clip((event.x - left) / max(right - left, 1), 0.0, 1.0)
        self.pause()
        self.show_frame(int(round(ratio * (self.total_frames - 1))))

    def export_scan_csv(self):
        if not self.scan_results:
            return
        initial = (
            str(Path(self.video_path).with_suffix("")) + "_aruco_pose_scan.csv"
            if self.video_path else "aruco_pose_scan.csv")
        path = filedialog.asksaveasfilename(
            title="匯出全片姿態 CSV", defaultextension=".csv",
            initialfile=os.path.basename(initial),
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow([
                "frame_index", "time_s", "marker_id", "incidence_deg",
                "distance_mm", "reprojection_rms_px", "ippe_branch",
            ])
            for frame_index, result in sorted(self.scan_results.items()):
                for marker_id, measurement in sorted(
                        result.get("measurements", {}).items()):
                    writer.writerow([
                        frame_index,
                        frame_index / max(self.video_fps, 1e-9),
                        marker_id,
                        measurement["incidence_deg"],
                        measurement["range_mm"],
                        measurement["reprojection_rms_px"],
                        measurement["branch"],
                    ])
        messagebox.showinfo("匯出完成", path)

    def _on_close(self):
        self.pause()
        self.scan_generation += 1
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    PoseInspectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
