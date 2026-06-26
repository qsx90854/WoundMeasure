import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
except ModuleNotFoundError as exc:
    cv2 = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from wound_detector import WoundDetector
except ModuleNotFoundError as exc:
    WoundDetector = None
    IMPORT_ERROR = exc


SCRIPT_DIR = Path(__file__).resolve().parent


class VideoWoundDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Detection Video Demo")
        self.root.geometry("1100x760")
        self.root.minsize(900, 620)

        self.detector = None
        self.cap = None
        self.video_path = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame_index = 0
        self.is_playing = False
        self.is_dragging_slider = False
        self.current_photo = None
        self.last_frame = None
        self.last_inference_ms = None

        self.status_var = tk.StringVar(value="請選擇影片")
        self.frame_var = tk.StringVar(value="Frame: - / -")
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        self.inference_var = tk.StringVar(value="Inference: -")
        self.detect_enabled = tk.BooleanVar(value=True)

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
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=5, sticky="w")

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

    def load_detector(self):
        if self.detector is None:
            self.status_var.set("載入 ONNX 模型中...")
            self.root.update_idletasks()
            old_cwd = Path.cwd()
            try:
                os.chdir(SCRIPT_DIR)
                self.detector = WoundDetector()
            finally:
                os.chdir(old_cwd)

    def toggle_play(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        self.play_button.configure(text="暫停" if self.is_playing else "播放")
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if not self.is_playing or self.cap is None:
            return

        start_time = time.perf_counter()
        ok = self.show_frame(self.current_frame_index)
        if not ok:
            self.is_playing = False
            self.play_button.configure(text="播放")
            return

        self.current_frame_index += 1
        if self.frame_count and self.current_frame_index >= self.frame_count:
            self.is_playing = False
            self.play_button.configure(text="播放")
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
        self.update_info()

        if not self.is_dragging_slider:
            self.slider.set(frame_index)
        return True

    def process_frame(self, frame):
        if not self.detect_enabled.get():
            self.last_inference_ms = None
            return frame
        try:
            self.load_detector()
            start_time = time.perf_counter()
            result = self.detector.predict(frame, draw_result=True)
            self.last_inference_ms = (time.perf_counter() - start_time) * 1000
            print(f"[Inference] frame={self.current_frame_index + 1} time={self.last_inference_ms:.2f} ms")
            self.status_var.set(f"{Path(self.video_path).name} | inference {self.last_inference_ms:.2f} ms")
            return result
        except Exception as exc:
            self.is_playing = False
            self.play_button.configure(text="播放")
            messagebox.showerror("偵測失敗", str(exc))
            return frame

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
            self.show_frame(self.current_frame_index + 1)

    def previous_frame(self):
        if self.cap is not None:
            self.is_playing = False
            self.play_button.configure(text="播放")
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
        self.show_frame(int(float(self.slider.get())))

    def update_info(self, frame_index=None):
        idx = self.current_frame_index if frame_index is None else frame_index
        total = self.frame_count if self.frame_count else 0
        self.frame_var.set(f"Frame: {idx + 1 if total else 0} / {total}")
        self.time_var.set(f"{self.format_time(idx / self.fps)} / {self.format_time(total / self.fps)}")
        if self.last_inference_ms is None:
            self.inference_var.set("Inference: -")
        else:
            self.inference_var.set(f"Inference: {self.last_inference_ms:.2f} ms")

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
