import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

DISPLAY_MAX_WIDTH = 1400
DISPLAY_MAX_HEIGHT = 820
PARAMS_JSON_PATH = "calibration_result_Zebra_1_no_dis.json"
ACTUAL_MARKER_SIZE_MM = 8.25
START_FRAME_COUNT = 30
END_FRAME_COUNT = 30
FRAME_RANGE_MODE = "half_half"
POSE_SELECT_MODE = "reproj_min"
BASE_DIR = Path(__file__).resolve().parent
WOUND_DETECTION_DIR = BASE_DIR / "wound_detection_model"
WOUND_MODEL_PATH = WOUND_DETECTION_DIR / "model" / "assets" / "v9-t-seg_320.onnx"
_WOUND_DETECTOR = None
_WOUND_DETECTOR_ERROR_LOGGED = False
_DLL_DIR_HANDLES = []


def configure_native_dll_search_paths():
    """Help Windows find native DLLs after this project environment is moved."""
    candidate_dirs = [
        BASE_DIR / "Lib" / "site-packages" / "onnxruntime" / "capi",
        BASE_DIR / "Lib" / "site-packages" / "torch" / "lib",
        BASE_DIR / "Scripts",
    ]
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    changed_path = False
    for dll_dir in candidate_dirs:
        if not dll_dir.exists():
            continue
        dll_dir_str = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            try:
                handle = os.add_dll_directory(dll_dir_str)
                _DLL_DIR_HANDLES.append(handle)
            except OSError:
                pass
        if dll_dir_str not in path_parts:
            path_parts.insert(0, dll_dir_str)
            changed_path = True
    if changed_path:
        os.environ["PATH"] = os.pathsep.join(path_parts)


configure_native_dll_search_paths()

import cv2
import numpy as np
import onnxruntime as ort

from Algorithm.specular_detection import (
    SPEC_TEMPORAL_BRIGHT_THRESHOLD,
    SPEC_TEMPORAL_OFFSETS,
    SPEC_TEMPORAL_RESIDUAL_THRESHOLD,
    SPEC_TEMPORAL_STD_THRESHOLD,
    compute_rt_aligned_temporal_specular_mask_bgr,
    compute_specular_mask_bgr,
    compute_specular_mask_bgr_wound_adaptive,
    overlay_specular_mask_rgb,
)
from Algorithm.camera_preprocess import (
    build_undistort_processor,
    load_json_camera_params,
    preprocess_gray,
)
from Algorithm.aruco_pose import detect_aruco_corners_bgr_for_pose
from Algorithm import video_pose_analysis as video_pose_algo


def clear_failed_onnxruntime_imports():
    for name in list(sys.modules):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            sys.modules.pop(name, None)


def get_wound_detector():
    """Lazy-load the wound segmentation model used by zebra.py."""
    global _WOUND_DETECTOR, _WOUND_DETECTOR_ERROR_LOGGED
    if _WOUND_DETECTOR is not None:
        return _WOUND_DETECTOR
    if not WOUND_MODEL_PATH.exists():
        if not _WOUND_DETECTOR_ERROR_LOGGED:
            print(f"[Wound] Cannot find model: {WOUND_MODEL_PATH}")
            _WOUND_DETECTOR_ERROR_LOGGED = True
        return None

    old_cwd = Path.cwd()
    wound_dir_str = str(WOUND_DETECTION_DIR)
    try:
        configure_native_dll_search_paths()
        if wound_dir_str not in sys.path:
            sys.path.insert(0, wound_dir_str)
        os.chdir(WOUND_DETECTION_DIR)
        from wound_detector import WoundDetector

        _WOUND_DETECTOR = WoundDetector(WOUND_MODEL_PATH)
        model = _WOUND_DETECTOR.model
        print(f"[Wound] Loaded {model.model_path} input_shape={model.input_shape}")
        return _WOUND_DETECTOR
    except Exception as exc:
        clear_failed_onnxruntime_imports()
        if not _WOUND_DETECTOR_ERROR_LOGGED:
            print(f"[Wound] Failed to load wound detector: {exc}")
            print(f"[Wound] Python executable: {sys.executable}")
            print(f"[Wound] ONNX Runtime DLL dir: {BASE_DIR / 'Lib' / 'site-packages' / 'onnxruntime' / 'capi'}")
            _WOUND_DETECTOR_ERROR_LOGGED = True
        return None
    finally:
        os.chdir(old_cwd)


def predict_wound_regions_bgr(bgr):
    detector = get_wound_detector()
    if detector is None:
        return None
    try:
        return detector.predict(bgr.copy(), draw_result=False)
    except Exception as exc:
        print(f"[Wound] Inference failed: {exc}")
        return None


def prediction_to_wound_mask(prediction, image_shape):
    if not prediction:
        return None
    first = prediction[0]
    if first is None or len(first) < 4:
        return None
    h, w = image_shape[:2]
    _classes, _bboxes, scores, masks = first
    combined = np.zeros((h, w), dtype=np.uint8)
    for score, mask in zip(scores, masks):
        conf_val = float(score[0] if isinstance(score, np.ndarray) else score)
        if conf_val <= 0.01:
            continue
        mask_f = mask.astype(np.float32)
        if mask_f.shape != (h, w):
            mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
        combined[mask_f > 0.5] = 255
    return combined if np.any(combined) else None


class SpecularDetectionVideoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Specular Detection Video Viewer")
        self.root.geometry("1200x820")
        self.root.minsize(900, 620)

        self.cap = None
        self.video_path = None
        self.frame_count = 0
        self.video_fps = 30.0
        self.current_index = 0
        self.is_playing = False
        self.is_dragging_slider = False
        self.current_photo = None
        self.last_display_frame = None
        self.frame_cache = {}
        self.processed_frame_cache = {}
        self.result_cache = {}
        self.wound_prediction_cache = {}
        self.wound_mask_status_cache = {}
        self.video_data = None
        self.processed_corners_cache = {}
        self.KL = None
        self.distL = None
        self.mtxL = None
        self.process_view = None

        self.status_var = tk.StringVar(value="Open a video to start.")
        self.frame_var = tk.StringVar(value="Frame: - / -")
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        self.export_enabled = tk.BooleanVar(value=False)
        self.adaptive_spatial_enabled = tk.BooleanVar(value=False)
        self.export_fps_var = tk.StringVar(value="10")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(10, weight=1)

        ttk.Button(toolbar, text="Open Video", command=self.open_video).grid(row=0, column=0, padx=(0, 8))
        self.play_button = ttk.Button(toolbar, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_button.grid(row=0, column=1, padx=(0, 8))
        self.prev_button = ttk.Button(toolbar, text="Prev", command=self.previous_frame, state=tk.DISABLED)
        self.prev_button.grid(row=0, column=2, padx=(0, 8))
        self.next_button = ttk.Button(toolbar, text="Next", command=self.next_frame, state=tk.DISABLED)
        self.next_button.grid(row=0, column=3, padx=(0, 14))

        ttk.Checkbutton(
            toolbar,
            text="Enable Export",
            variable=self.export_enabled,
            command=self.update_export_state,
        ).grid(row=0, column=4, padx=(0, 8))
        ttk.Label(toolbar, text="Export FPS").grid(row=0, column=5, padx=(0, 4))
        self.fps_entry = ttk.Entry(toolbar, textvariable=self.export_fps_var, width=6)
        self.fps_entry.grid(row=0, column=6, padx=(0, 8))
        self.export_button = ttk.Button(toolbar, text="Save MP4", command=self.export_video, state=tk.DISABLED)
        self.export_button.grid(row=0, column=7, padx=(0, 14))

        ttk.Checkbutton(
            toolbar,
            text="Adaptive Spatial",
            variable=self.adaptive_spatial_enabled,
            command=self.on_adaptive_spatial_toggle,
        ).grid(row=0, column=8, padx=(0, 14))

        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=9, sticky="w")

        video_area = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        video_area.grid(row=1, column=0, sticky="nsew")
        video_area.columnconfigure(0, weight=1)
        video_area.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(video_area, anchor="center", background="#1f1f1f")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Configure>", lambda _event: self.redraw_last_frame())

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        self.slider = ttk.Scale(controls, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)

        info = ttk.Frame(controls)
        info.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        info.columnconfigure(2, weight=1)
        ttk.Label(info, textvariable=self.frame_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.time_var).grid(row=0, column=2, sticky="e")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Open failed", "Cannot open the selected video.")
            return

        self.release_video()
        self.cap = cap
        self.video_path = path
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        ok, first_frame = cap.read()
        if not ok:
            self.release_video()
            messagebox.showerror("Open failed", "Cannot read the first frame.")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not self.prepare_zebra_pipeline(first_frame):
            self.release_video()
            return

        self.current_index = 0
        self.frame_cache.clear()
        self.processed_frame_cache.clear()
        self.result_cache.clear()
        self.wound_prediction_cache.clear()
        self.slider.configure(to=max(self.frame_count - 1, 0))
        self.set_controls_state(tk.NORMAL)
        self.status_var.set(f"{Path(path).name} | zebra temporal ready")
        self.show_frame(0)

    def prepare_zebra_pipeline(self, first_frame):
        params = load_json_camera_params(PARAMS_JSON_PATH)
        mtxL_o, distL, _mtxR_o, _distR, _extrinsic, _F_orig = params
        if mtxL_o is None or distL is None:
            messagebox.showerror("Calibration missing", f"Cannot load calibration JSON:\n{PARAMS_JSON_PATH}")
            return False

        h_raw, w_raw = first_frame.shape[:2]
        newKL_o, _map1, _map2, process_view = build_undistort_processor(
            mtxL_o, distL, (w_raw, h_raw), alpha=1.0
        )
        self.KL = newKL_o.copy().astype(np.float64)
        self.distL = distL
        self.mtxL = mtxL_o
        self.process_view = process_view

        self.status_var.set("Analyzing video poses for zebra temporal specular detection...")
        self.root.update_idletasks()

        def progress_callback(percent, status_text):
            self.status_var.set(f"{percent:5.1f}% {status_text}")
            self.root.update_idletasks()

        self.video_data = video_pose_algo.analyze_video_frames(
            self.video_path,
            START_FRAME_COUNT,
            END_FRAME_COUNT,
            self.KL,
            self.distL,
            self.mtxL,
            ACTUAL_MARKER_SIZE_MM,
            POSE_SELECT_MODE,
            FRAME_RANGE_MODE,
            progress_callback=progress_callback,
        )
        if self.video_data is None:
            messagebox.showerror("Pose analysis failed", "Cannot build marker_map / video poses for zebra temporal detection.")
            return False
        self.build_processed_aruco_corner_cache()
        return True

    def build_processed_aruco_corner_cache(self):
        self.processed_corners_cache = {}
        if not self.video_data:
            return
        frames = self.video_data.get("all_frames") or []
        total = len(frames)
        if total == 0:
            return
        for idx, raw_frame in enumerate(frames):
            processed = self.process_view(raw_frame) if self.process_view is not None else raw_frame
            bgr = processed[0] if isinstance(processed, tuple) else processed
            self.processed_corners_cache[idx] = detect_aruco_corners_bgr_for_pose(
                bgr,
                preprocess_gray_fn=preprocess_gray,
            )
            if idx % 10 == 0 or idx == total - 1:
                self.status_var.set(f"Caching ArUco corners {idx + 1}/{total}...")
                self.root.update_idletasks()
        self.video_data["processed_corners_cache"] = self.processed_corners_cache
        self.video_data["processed_pose_cache"] = {}

    def read_frame(self, frame_index):
        if self.cap is None:
            return None
        frame_index = max(0, min(int(frame_index), max(self.frame_count - 1, 0)))
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index].copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame_cache[frame_index] = frame.copy()
        if len(self.frame_cache) > 80:
            keys = sorted(self.frame_cache.keys(), key=lambda k: abs(k - self.current_index), reverse=True)
            for key in keys[:20]:
                self.frame_cache.pop(key, None)
        return frame

    def read_processed_frame(self, frame_index):
        frame_index = max(0, min(int(frame_index), max(self.frame_count - 1, 0)))
        if frame_index in self.processed_frame_cache:
            return self.processed_frame_cache[frame_index].copy()
        frame = self.read_frame(frame_index)
        if frame is None:
            return None
        if self.process_view is not None:
            processed = self.process_view(frame)
            frame = processed[0] if isinstance(processed, tuple) else processed
        self.processed_frame_cache[frame_index] = frame.copy()
        if len(self.processed_frame_cache) > 80:
            keys = sorted(self.processed_frame_cache.keys(), key=lambda k: abs(k - self.current_index), reverse=True)
            for key in keys[:20]:
                self.processed_frame_cache.pop(key, None)
        return frame

    def get_wound_prediction(self, frame_index, frame):
        frame_index = int(frame_index)
        if frame_index in self.wound_prediction_cache:
            return self.wound_prediction_cache[frame_index]
        prediction = predict_wound_regions_bgr(frame)
        self.wound_prediction_cache[frame_index] = prediction
        if len(self.wound_prediction_cache) > 80:
            keys = sorted(self.wound_prediction_cache.keys(), key=lambda k: abs(k - self.current_index), reverse=True)
            for key in keys[:20]:
                self.wound_prediction_cache.pop(key, None)
        return prediction

    def compute_adaptive_spatial_mask(self, frame_index, frame):
        frame_index = int(frame_index)
        if not self.adaptive_spatial_enabled.get():
            self.wound_mask_status_cache[frame_index] = {
                "adaptive_enabled": False,
                "ai_prediction_used": False,
                "has_wound_mask": False,
            }
            return None
        prediction = self.get_wound_prediction(frame_index, frame)
        wound_mask = prediction_to_wound_mask(prediction, frame.shape)
        self.wound_mask_status_cache[frame_index] = {
            "adaptive_enabled": True,
            "ai_prediction_used": True,
            "has_wound_mask": wound_mask is not None,
        }
        if wound_mask is None:
            return None
        return compute_specular_mask_bgr_wound_adaptive(frame, wound_mask)

    def get_wound_mask_log_status(self, frame_index):
        frame_index = int(frame_index)
        status = self.wound_mask_status_cache.get(frame_index)
        if status is not None:
            return status
        return {
            "adaptive_enabled": bool(self.adaptive_spatial_enabled.get()),
            "ai_prediction_used": False,
            "has_wound_mask": False,
        }

    def compute_temporal_mask(self, frame_index, center_bgr, spatial_mask):
        if center_bgr is None:
            return None
        gray_center = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        samples = [gray_center]
        for offset in SPEC_TEMPORAL_OFFSETS:
            idx = int(np.clip(frame_index + offset, 0, max(self.frame_count - 1, 0)))
            if idx == frame_index:
                continue
            neighbor = self.read_frame(idx)
            if neighbor is None:
                continue
            samples.append(cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY).astype(np.float32))

        if len(samples) < 3:
            return np.zeros_like(spatial_mask) if spatial_mask is not None else np.zeros(gray_center.shape, dtype=np.uint8)

        stack = np.stack(samples, axis=0)
        temporal_std = np.std(stack, axis=0)
        temporal_median = np.median(stack, axis=0)
        residual = np.abs(gray_center - temporal_median)
        spatial_bool = (spatial_mask > 0) if spatial_mask is not None else np.zeros(gray_center.shape, dtype=bool)
        temporal_mask = (
            (gray_center >= SPEC_TEMPORAL_BRIGHT_THRESHOLD)
            & (temporal_std >= SPEC_TEMPORAL_STD_THRESHOLD)
            & ((residual >= SPEC_TEMPORAL_RESIDUAL_THRESHOLD) | spatial_bool)
        ).astype(np.uint8) * 255
        temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        temporal_mask = cv2.dilate(temporal_mask, np.ones((7, 7), np.uint8), iterations=1)
        return temporal_mask

    def process_frame(self, frame_index):
        frame_index = int(frame_index)
        if frame_index in self.result_cache:
            return self.result_cache[frame_index].copy()

        frame = self.read_processed_frame(frame_index)
        if frame is None:
            return None

        if self.video_data is not None and self.KL is not None:
            combined_mask, spatial_mask, temporal_mask = compute_rt_aligned_temporal_specular_mask_bgr(
                frame,
                frame_index,
                self.video_data,
                self.KL,
                ACTUAL_MARKER_SIZE_MM,
                self.process_view,
                return_parts=True,
                preprocess_gray_fn=preprocess_gray,
            )
        else:
            spatial_mask = compute_specular_mask_bgr(frame)
            temporal_mask = self.compute_temporal_mask(frame_index, frame, spatial_mask)
            combined_mask = spatial_mask if temporal_mask is None else cv2.bitwise_or(spatial_mask, temporal_mask)

        adaptive_spatial_mask = self.compute_adaptive_spatial_mask(frame_index, frame)
        if adaptive_spatial_mask is not None:
            spatial_mask = adaptive_spatial_mask
            combined_mask = spatial_mask if temporal_mask is None else cv2.bitwise_or(spatial_mask, temporal_mask)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        spatial_rgb = overlay_specular_mask_rgb(rgb, spatial_mask, None)
        temporal_rgb = self.overlay_temporal_mask_rgb(rgb, temporal_mask)

        spatial_panel = self.add_panel_header(cv2.cvtColor(spatial_rgb, cv2.COLOR_RGB2BGR), "Spatial Specular")
        temporal_panel = self.add_panel_header(cv2.cvtColor(temporal_rgb, cv2.COLOR_RGB2BGR), "RT Temporal Specular")
        combined = self.hconcat_with_separator(spatial_panel, temporal_panel)
        self.result_cache[frame_index] = combined.copy()
        if len(self.result_cache) > 40:
            keys = sorted(self.result_cache.keys(), key=lambda k: abs(k - self.current_index), reverse=True)
            for key in keys[:10]:
                self.result_cache.pop(key, None)
        return combined

    @staticmethod
    def overlay_temporal_mask_rgb(rgb, temporal_mask, alpha=0.55):
        if rgb is None or temporal_mask is None:
            return rgb
        out = rgb.copy()
        temporal_bool = temporal_mask > 0
        if not np.any(temporal_bool):
            return out
        dark_green = np.zeros_like(out)
        dark_green[:, :] = (35, 170, 90)
        blended = cv2.addWeighted(out, 1.0 - alpha, dark_green, alpha, 0)
        out[temporal_bool] = blended[temporal_bool]
        return out

    def on_adaptive_spatial_toggle(self):
        self.result_cache.clear()
        self.wound_mask_status_cache.clear()
        state_text = "Adaptive Spatial enabled" if self.adaptive_spatial_enabled.get() else "Adaptive Spatial disabled"
        self.status_var.set(state_text)
        if self.cap is not None:
            self.show_frame(self.current_index)

    @staticmethod
    def add_panel_header(image, title):
        header_h = 42
        panel = cv2.copyMakeBorder(image, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
        cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        return panel

    @staticmethod
    def hconcat_with_separator(left, right):
        target_h = max(left.shape[0], right.shape[0])
        panels = []
        for panel in (left, right):
            if panel.shape[0] != target_h:
                scale = target_h / panel.shape[0]
                panel = cv2.resize(panel, (max(1, int(panel.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)
            panels.append(panel)
        separator = np.full((target_h, 8, 3), 24, dtype=np.uint8)
        return cv2.hconcat([panels[0], separator, panels[1]])

    def show_frame(self, frame_index):
        if self.cap is None:
            return False
        frame_index = max(0, min(int(frame_index), max(self.frame_count - 1, 0)))
        self.current_index = frame_index
        display = self.process_frame(frame_index)
        if display is None:
            return False
        self.last_display_frame = display
        self.render_frame(display)
        self.update_info()
        if not self.is_dragging_slider:
            self.slider.set(frame_index)
        return True

    def render_frame(self, frame):
        label_w = max(self.video_label.winfo_width(), 1)
        label_h = max(self.video_label.winfo_height(), 1)
        frame_h, frame_w = frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h, DISPLAY_MAX_WIDTH / frame_w, DISPLAY_MAX_HEIGHT / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm_data = b"P6\n%d %d\n255\n" % (new_w, new_h) + rgb.tobytes()
        self.current_photo = tk.PhotoImage(data=ppm_data, format="PPM")
        self.video_label.configure(image=self.current_photo)

    def toggle_play(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        self.play_button.configure(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if not self.is_playing or self.cap is None:
            return
        start = time.perf_counter()
        ok = self.show_frame(self.current_index)
        if not ok:
            self.is_playing = False
            self.play_button.configure(text="Play")
            return

        self.current_index += 1
        if self.frame_count and self.current_index >= self.frame_count:
            self.is_playing = False
            self.play_button.configure(text="Play")
            return

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        delay_ms = max(1, int(1000 / max(self.video_fps, 1.0)) - elapsed_ms)
        self.root.after(delay_ms, self.play_loop)

    def next_frame(self):
        if self.cap is None:
            return
        self.is_playing = False
        self.play_button.configure(text="Play")
        self.show_frame(self.current_index + 1)

    def previous_frame(self):
        if self.cap is None:
            return
        self.is_playing = False
        self.play_button.configure(text="Play")
        self.show_frame(self.current_index - 1)

    def on_slider_press(self, _event):
        self.is_dragging_slider = True

    def on_slider_move(self, value):
        if self.is_dragging_slider:
            self.update_info(int(float(value)))

    def on_slider_release(self, _event):
        if self.cap is None:
            return
        self.is_dragging_slider = False
        self.is_playing = False
        self.play_button.configure(text="Play")
        self.show_frame(int(float(self.slider.get())))

    def export_video(self):
        if self.cap is None or not self.export_enabled.get():
            return
        try:
            export_fps = float(self.export_fps_var.get())
            if export_fps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid FPS", "Export FPS must be a positive number.")
            return

        default_name = f"{Path(self.video_path).stem}_specular_effects.mp4"
        output_path = filedialog.asksaveasfilename(
            title="Save processed video",
            defaultextension=".mp4",
            initialfile=default_name,
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")],
        )
        if not output_path:
            return

        log_path = Path(output_path).with_suffix(".log")
        was_playing = self.is_playing
        self.is_playing = False
        self.play_button.configure(text="Play")
        writer = None
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            log_file.write(f"video={self.video_path}\n")
            log_file.write(f"output={output_path}\n")
            log_file.write(f"export_fps={export_fps}\n")
            log_file.write(f"adaptive_spatial_enabled={bool(self.adaptive_spatial_enabled.get())}\n")
            log_file.write("frame_index,adaptive_enabled,ai_prediction_used,has_ai_wound_mask\n")
            for idx in range(self.frame_count):
                frame = self.process_frame(idx)
                if frame is None:
                    log_file.write(f"{idx},{bool(self.adaptive_spatial_enabled.get())},False,False\n")
                    continue
                mask_status = self.get_wound_mask_log_status(idx)
                log_file.write(
                    f"{idx},"
                    f"{bool(mask_status.get('adaptive_enabled'))},"
                    f"{bool(mask_status.get('ai_prediction_used'))},"
                    f"{bool(mask_status.get('has_wound_mask'))}\n"
                )
                frame = self.prepare_even_frame(frame)
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, export_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create output video.")
                writer.write(frame)
                if idx % 5 == 0 or idx == self.frame_count - 1:
                    self.status_var.set(f"Exporting {idx + 1}/{self.frame_count}...")
                    self.root.update_idletasks()
            self.status_var.set(f"Saved: {Path(output_path).name} + {log_path.name}")
            messagebox.showinfo("Export complete", f"Saved video:\n{output_path}\n\nLog:\n{log_path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            self.status_var.set("Export failed.")
        finally:
            if "log_file" in locals() and not log_file.closed:
                log_file.close()
            if writer is not None:
                writer.release()
            if was_playing:
                self.is_playing = True
                self.play_button.configure(text="Pause")
                self.play_loop()

    @staticmethod
    def prepare_even_frame(frame):
        h, w = frame.shape[:2]
        pad_bottom = h % 2
        pad_right = w % 2
        if not pad_bottom and not pad_right:
            return frame
        return cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def update_export_state(self):
        state = tk.NORMAL if self.export_enabled.get() and self.cap is not None else tk.DISABLED
        self.export_button.configure(state=state)

    def update_info(self, frame_index=None):
        idx = self.current_index if frame_index is None else frame_index
        total = self.frame_count if self.frame_count else 0
        self.frame_var.set(f"Frame: {idx + 1 if total else 0} / {total}")
        self.time_var.set(f"{self.format_time(idx / self.video_fps)} / {self.format_time(total / self.video_fps)}")

    @staticmethod
    def format_time(seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def set_controls_state(self, state):
        self.play_button.configure(state=state)
        self.prev_button.configure(state=state)
        self.next_button.configure(state=state)
        self.update_export_state()

    def redraw_last_frame(self):
        if self.last_display_frame is not None:
            self.render_frame(self.last_display_frame)

    def release_video(self):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_data = None
        self.processed_corners_cache = {}
        self.KL = None
        self.distL = None
        self.mtxL = None
        self.process_view = None
        self.frame_cache.clear()
        self.processed_frame_cache.clear()
        self.result_cache.clear()
        self.wound_prediction_cache.clear()
        self.wound_mask_status_cache.clear()
        self.set_controls_state(tk.DISABLED)

    def on_close(self):
        self.release_video()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SpecularDetectionVideoUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
