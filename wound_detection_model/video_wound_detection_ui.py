import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    cv2 = None
    np = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from wound_detector import WoundDetector
except ModuleNotFoundError as exc:
    WoundDetector = None
    IMPORT_ERROR = exc


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS = [
    ("v9-c-seg", SCRIPT_DIR / "model" / "assets" / "v9-c-seg.onnx"),
    ("v9-t-seg", None),
    ("v9-t-seg_320", SCRIPT_DIR / "model" / "assets" / "v9-t-seg_320.onnx"),
]
MODEL_RUN_ORDER = ["v9-t-seg", "v9-c-seg", "v9-t-seg_320"]
RECORDING_FPS = 5.0


class VideoWoundDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Detection Video Demo")
        self.root.geometry("1100x760")
        self.root.minsize(900, 620)

        self.detectors = {}
        self.cap = None
        self.video_path = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame_index = 0
        self.is_playing = False
        self.is_dragging_slider = False
        self.current_photo = None
        self.last_frame = None
        self.last_inference_ms = {}
        self.video_writer = None
        self.recording_path = None
        self.recording_size = None
        self.model_input_shapes = {}

        self.status_var = tk.StringVar(value="請選擇影片")
        self.frame_var = tk.StringVar(value="Frame: - / -")
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        self.inference_var = tk.StringVar(value="Inference: -")
        self.detect_enabled = tk.BooleanVar(value=True)
        self.record_enabled = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        ttk.Button(toolbar, text="開啟影片", command=self.open_video).grid(row=0, column=0, padx=(0, 8))
        self.play_button = ttk.Button(toolbar, text="播放", command=self.toggle_play, state=tk.DISABLED)
        self.play_button.grid(row=0, column=1, padx=(0, 8))
        self.prev_button = ttk.Button(toolbar, text="上一幀", command=self.previous_frame, state=tk.DISABLED)
        self.prev_button.grid(row=0, column=2, padx=(0, 8))
        self.next_button = ttk.Button(toolbar, text="下一幀", command=self.next_frame, state=tk.DISABLED)
        self.next_button.grid(row=0, column=3, padx=(0, 14))
        ttk.Checkbutton(toolbar, text="啟用偵測", variable=self.detect_enabled, command=self.refresh_current_frame).grid(
            row=0, column=4, padx=(0, 14)
        )
        ttk.Checkbutton(toolbar, text="錄影", variable=self.record_enabled, command=self.on_record_toggle).grid(
            row=0, column=5, padx=(0, 14)
        )
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=6, sticky="w")

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
        ttk.Label(info, textvariable=self.inference_var).grid(row=0, column=1, sticky="w", padx=(18, 0))
        ttk.Label(info, textvariable=self.time_var).grid(row=0, column=2, sticky="e")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="選擇影片",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("開啟失敗", "無法讀取選擇的影片。")
            return

        self.release_video()
        self.cap = cap
        self.video_path = path
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.current_frame_index = 0
        self.slider.configure(to=max(self.frame_count - 1, 0))
        self.status_var.set(Path(path).name)
        self.set_video_controls_state(tk.NORMAL)

        self.show_frame(0)

    def load_detectors(self):
        if len(self.detectors) != len(MODEL_CONFIGS):
            self.status_var.set("載入 3 個 ONNX 模型中...")
            self.root.update_idletasks()
            old_cwd = Path.cwd()
            try:
                os.chdir(SCRIPT_DIR)
                for model_name, model_path in MODEL_CONFIGS:
                    if model_name not in self.detectors:
                        if model_path is not None and not model_path.exists():
                            raise FileNotFoundError(f"找不到模型檔案：{model_path}")
                        self.detectors[model_name] = WoundDetector() if model_path is None else WoundDetector(model_path)
                        model = self.detectors[model_name].model
                        self.model_input_shapes[model_name] = tuple(model.input_shape)
                        print(
                            f"[Model] {model_name} path={model.model_path} input_shape={model.input_shape} "
                            f"strides={model.vec2box.strides} outputs={model.output_shapes}"
                        )
            finally:
                os.chdir(old_cwd)

    def toggle_play(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        self.play_button.configure(text="暫停" if self.is_playing else "播放")
        if self.is_playing:
            self.play_loop()
        else:
            self.stop_recording()

    def play_loop(self):
        if not self.is_playing or self.cap is None:
            return

        start_time = time.perf_counter()
        ok = self.show_frame(self.current_frame_index)
        if not ok:
            self.is_playing = False
            self.play_button.configure(text="播放")
            self.stop_recording()
            return

        self.current_frame_index += 1
        if self.frame_count and self.current_frame_index >= self.frame_count:
            self.is_playing = False
            self.play_button.configure(text="播放")
            self.stop_recording()
            return

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        delay_ms = max(1, int(1000 / max(self.fps, 1.0)) - elapsed_ms)
        self.root.after(delay_ms, self.play_loop)

    def show_frame(self, frame_index):
        if self.cap is None:
            return False

        frame_index = max(0, min(int(frame_index), max(self.frame_count - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return False

        self.current_frame_index = frame_index
        display_frame = self.process_frame(frame)
        self.last_frame = display_frame
        self.render_frame(display_frame)
        self.write_recording_frame(display_frame)
        self.update_info()

        if not self.is_dragging_slider:
            self.slider.set(frame_index)
        return True

    def process_frame(self, frame):
        if not self.detect_enabled.get():
            self.last_inference_ms = {}
            return self.build_comparison_frame([
                (name, frame.copy(), None, None, self.model_input_shapes.get(name))
                for name, _model_path in MODEL_CONFIGS
            ])
        try:
            self.load_detectors()
            result_by_name = {}
            self.last_inference_ms = {}
            for model_name in MODEL_RUN_ORDER:
                start_time = time.perf_counter()
                detector = self.detectors[model_name]
                result = detector.predict(frame.copy())
                inference_ms = (time.perf_counter() - start_time) * 1000
                self.last_inference_ms[model_name] = inference_ms
                detection_count = self.count_drawn_detections(detector)
                input_shape = tuple(detector.model.input_shape)
                self.model_input_shapes[model_name] = input_shape
                result_by_name[model_name] = (result, inference_ms, detection_count, input_shape)
                print(
                    f"[Inference] frame={self.current_frame_index + 1} "
                    f"model={model_name} time={inference_ms:.2f} ms "
                    f"detections={detection_count}"
                )

            total_ms = sum(self.last_inference_ms.values())
            self.status_var.set(f"{Path(self.video_path).name} | 3 models total {total_ms:.2f} ms")
            results = [
                (model_name, *result_by_name[model_name])
                for model_name, _model_path in MODEL_CONFIGS
            ]
            return self.build_comparison_frame(results)
        except Exception as exc:
            self.is_playing = False
            self.play_button.configure(text="播放")
            self.stop_recording()
            messagebox.showerror("偵測失敗", str(exc))
            return self.build_comparison_frame([
                (name, frame.copy(), None, None, self.model_input_shapes.get(name))
                for name, _model_path in MODEL_CONFIGS
            ])

    def build_comparison_frame(self, results):
        panels = []
        for model_name, image, inference_ms, detection_count, input_shape in results:
            panels.append(self.add_panel_header(image, model_name, inference_ms, detection_count, input_shape))

        target_h = max(panel.shape[0] for panel in panels)
        normalized = []
        for panel in panels:
            if panel.shape[0] != target_h:
                scale = target_h / panel.shape[0]
                new_w = max(1, int(panel.shape[1] * scale))
                panel = cv2.resize(panel, (new_w, target_h), interpolation=cv2.INTER_AREA)
            normalized.append(panel)

        separator = np.full((target_h, 6, 3), 28, dtype=normalized[0].dtype)
        comparison = normalized[0]
        for panel in normalized[1:]:
            comparison = cv2.hconcat([comparison, separator, panel])
        return comparison

    @staticmethod
    def count_detections(prediction):
        if not prediction:
            return 0
        first = prediction[0]
        if not first:
            return 0
        return len(first[0])

    @staticmethod
    def count_drawn_detections(detector):
        # WoundDetector stores the last raw prediction for UI diagnostics.
        return VideoWoundDetectionApp.count_detections(getattr(detector, "last_output", None))

    @staticmethod
    def add_panel_header(image, model_name, inference_ms, detection_count, input_shape):
        header_h = 44
        panel = cv2.copyMakeBorder(image, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
        time_text = "-" if inference_ms is None else f"{inference_ms:.2f} ms"
        count_text = "-" if detection_count is None else str(detection_count)
        size_text = "-" if input_shape is None else f"{int(input_shape[0])}x{int(input_shape[1])}"
        label = f"{model_name} | {size_text} | {time_text} | det {count_text}"
        cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        return panel

    def render_frame(self, frame):
        label_w = max(self.video_label.winfo_width(), 1)
        label_h = max(self.video_label.winfo_height(), 1)
        frame_h, frame_w = frame.shape[:2]
        scale = min(label_w / frame_w, label_h / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ppm_data = b"P6\n%d %d\n255\n" % (new_w, new_h) + rgb.tobytes()
        self.current_photo = tk.PhotoImage(data=ppm_data, format="PPM")
        self.video_label.configure(image=self.current_photo)

    def write_recording_frame(self, frame):
        if not self.is_playing or not self.record_enabled.get():
            return

        frame = self.prepare_recording_frame(frame)
        if self.video_writer is None:
            self.start_recording(frame)

        if self.video_writer is not None:
            self.video_writer.write(frame)

    @staticmethod
    def prepare_recording_frame(frame):
        h, w = frame.shape[:2]
        pad_bottom = h % 2
        pad_right = w % 2
        if not pad_bottom and not pad_right:
            return frame
        return cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def start_recording(self, frame):
        if self.video_path is None:
            return

        output_dir = SCRIPT_DIR / "recordings"
        output_dir.mkdir(exist_ok=True)
        source_name = Path(self.video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recording_path = output_dir / f"{source_name}_wound_detection_{timestamp}.mp4"

        frame_h, frame_w = frame.shape[:2]
        self.recording_size = (frame_w, frame_h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(self.recording_path), fourcc, RECORDING_FPS, self.recording_size)
        if not self.video_writer.isOpened():
            self.video_writer = None
            messagebox.showerror("錄影失敗", f"無法建立輸出影片：{self.recording_path}")
            return

        print(f"[Recording] start {self.recording_path} fps={RECORDING_FPS:.1f}")
        self.status_var.set(f"Recording: {self.recording_path.name}")

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording_size = None
            if self.recording_path is not None:
                print(f"[Recording] saved {self.recording_path}")
                self.status_var.set(f"錄影已儲存：{self.recording_path.name}")

    def on_record_toggle(self):
        if not self.record_enabled.get():
            self.stop_recording()

    def redraw_last_frame(self):
        if self.last_frame is not None:
            self.render_frame(self.last_frame)

    def refresh_current_frame(self):
        if self.cap is not None:
            self.show_frame(self.current_frame_index)

    def next_frame(self):
        if self.cap is not None:
            self.is_playing = False
            self.play_button.configure(text="播放")
            self.stop_recording()
            self.show_frame(self.current_frame_index + 1)

    def previous_frame(self):
        if self.cap is not None:
            self.is_playing = False
            self.play_button.configure(text="播放")
            self.stop_recording()
            self.show_frame(self.current_frame_index - 1)

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
        self.play_button.configure(text="播放")
        self.stop_recording()
        self.show_frame(int(float(self.slider.get())))

    def update_info(self, frame_index=None):
        idx = self.current_frame_index if frame_index is None else frame_index
        total = self.frame_count if self.frame_count else 0
        self.frame_var.set(f"Frame: {idx + 1 if total else 0} / {total}")
        self.time_var.set(f"{self.format_time(idx / self.fps)} / {self.format_time(total / self.fps)}")
        if not self.last_inference_ms:
            self.inference_var.set("Inference: -")
        else:
            parts = [f"{name}: {self.last_inference_ms[name]:.1f} ms" for name, _model_path in MODEL_CONFIGS if name in self.last_inference_ms]
            self.inference_var.set(" | ".join(parts))

    @staticmethod
    def format_time(seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def set_video_controls_state(self, state):
        self.play_button.configure(state=state)
        self.prev_button.configure(state=state)
        self.next_button.configure(state=state)

    def release_video(self):
        self.is_playing = False
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.release_video()
        self.root.destroy()


def main():
    if IMPORT_ERROR is not None:
        root = tk.Tk()
        root.withdraw()
        package_name = IMPORT_ERROR.name or str(IMPORT_ERROR)
        messagebox.showerror(
            "Missing Python package",
            "缺少 Python 套件：{}\n\n"
            "請先在目前虛擬環境安裝：\n"
            "python -m pip install -r requirements.txt\n\n"
            "或單獨安裝：\n"
            "python -m pip install {}".format(package_name, package_name),
        )
        root.destroy()
        sys.exit(1)

    root = tk.Tk()
    app = VideoWoundDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
