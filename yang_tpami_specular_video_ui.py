import base64
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np


APP_TITLE = "Yang-Style Specular Highlight Removal Video Demo"
DISPLAY_BG = "#161616"
PANEL_BG = "#101010"
TEXT_FG = "#E8E8E8"
ACCENT = "#4FA3FF"


def rgb_to_photoimage(rgb):
    """Create a Tk PhotoImage from an RGB uint8 image without requiring Pillow."""
    if rgb is None or rgb.size == 0:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for Tk display.")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data, format="PNG")


def normalize_to_u8(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(x[finite], [1.0, 99.0])
    if hi <= lo + 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    y = (x - lo) * (255.0 / (hi - lo))
    return np.clip(y, 0, 255).astype(np.uint8)


def compute_yang_chromaticity_terms(bgr):
    bgr = bgr.astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    eps = 1e-6

    sum_rgb = np.sum(rgb, axis=2, keepdims=True)
    chroma = rgb / np.maximum(sum_rgb, eps)
    max_chroma = np.max(chroma, axis=2)
    min_rgb = np.min(rgb, axis=2, keepdims=True)

    # Pseudo specular-free image from neutral-component subtraction.
    pseudo = np.clip(rgb - min_rgb, 0.0, 1.0)
    pseudo_sum = np.sum(pseudo, axis=2, keepdims=True)
    pseudo_chroma = pseudo / np.maximum(pseudo_sum, eps)
    pseudo_max_chroma = np.max(pseudo_chroma, axis=2)

    # Difference between pseudo specular-free maximum chromaticity and the
    # original maximum chromaticity. This is the displayed right-panel cue.
    max_chroma_diff = np.clip(pseudo_max_chroma - max_chroma, 0.0, 1.0)
    return rgb, sum_rgb, min_rgb, max_chroma, pseudo_max_chroma, max_chroma_diff


def remove_highlight_single_pass(bgr, rgb, min_rgb, max_chroma_diff):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    scale = max(float(np.percentile(max_chroma_diff, 98.0)), 0.03)
    diff_strength = np.clip(max_chroma_diff / scale, 0.0, 1.0)
    bright_weight = np.clip((val - 0.35) / 0.55, 0.0, 1.0)
    saturation_weight = np.clip((0.95 - 0.35 * sat), 0.35, 1.0)

    # Soft specular confidence. The primary signal is the max-chromaticity
    # difference; brightness/saturation only regularize the estimate for video.
    soft = diff_strength * bright_weight * saturation_weight
    soft = cv2.GaussianBlur(soft.astype(np.float32), (0, 0), 1.2)
    soft = np.clip(soft, 0.0, 1.0)

    spec_amount = np.squeeze(min_rgb, axis=2) * soft * 0.92
    diffuse = np.clip(rgb - spec_amount[:, :, None], 0.0, 1.0)
    return finish_restored_image(bgr, rgb, diffuse, soft)


def remove_highlight_iterative(bgr, rgb, sum_rgb, min_rgb, max_chroma, pseudo_max_chroma, max_chroma_diff, iterations=4):
    """
    Iterative diffuse max-chromaticity refinement.

    Lambda is the estimated diffuse maximum chromaticity. Given Lambda and the
    observed maximum chromaticity sigma, the neutral specular amount follows the
    dichromatic max-chromaticity relation:

        s = sum(I) * (Lambda - sigma) / (3 * Lambda - 1)

    This is a compact real-time approximation of the refinement idea rather
    than a full paper reproduction.
    """
    eps = 1e-6
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    scale = max(float(np.percentile(max_chroma_diff, 98.0)), 0.03)
    confidence = np.clip(max_chroma_diff / scale, 0.0, 1.0)
    confidence *= np.clip((val - 0.25) / 0.65, 0.0, 1.0)
    confidence *= np.clip((1.05 - 0.30 * sat), 0.45, 1.0)
    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)

    lambda_map = np.maximum(pseudo_max_chroma, max_chroma + 1e-4).astype(np.float32)
    lambda_map = np.clip(lambda_map, 1.0 / 3.0 + 1e-4, 0.995)
    spec_amount = np.zeros(max_chroma.shape, dtype=np.float32)
    sum_scalar = np.squeeze(sum_rgb, axis=2)
    min_scalar = np.squeeze(min_rgb, axis=2)

    for _ in range(max(1, int(iterations))):
        denom = np.maximum(3.0 * lambda_map - 1.0, eps)
        spec_est = sum_scalar * (lambda_map - max_chroma) / denom
        spec_est = np.clip(spec_est, 0.0, min_scalar * 0.98)
        spec_amount = spec_amount * (1.0 - confidence) + spec_est * confidence

        diffuse = np.clip(rgb - spec_amount[:, :, None], 0.0, 1.0)
        diffuse_sum = np.sum(diffuse, axis=2, keepdims=True)
        diffuse_chroma = diffuse / np.maximum(diffuse_sum, eps)
        diffuse_max = np.max(diffuse_chroma, axis=2).astype(np.float32)
        diffuse_max = np.maximum(diffuse_max, max_chroma + 1e-4)

        # Neighbor consistency: Lambda should vary smoothly on the object, but
        # not become lower than the observed maximum chromaticity.
        smooth_lambda = cv2.bilateralFilter(diffuse_max, d=0, sigmaColor=0.035, sigmaSpace=7)
        target = np.maximum(smooth_lambda, max_chroma + 1e-4)
        lambda_map = lambda_map * (1.0 - 0.45 * confidence) + target * (0.45 * confidence)
        lambda_map = np.clip(lambda_map, 1.0 / 3.0 + 1e-4, 0.995)

    diffuse = np.clip(rgb - spec_amount[:, :, None], 0.0, 1.0)
    return finish_restored_image(bgr, rgb, diffuse, confidence)


def finish_restored_image(bgr, rgb, diffuse, soft):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    # Refill strongly saturated/white regions with surrounding colors to avoid
    # leaving gray holes after neutral subtraction.
    hard_mask = ((soft > 0.45) | ((val > 0.92) & (sat < 0.35))).astype(np.uint8) * 255
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hard_mask = cv2.dilate(hard_mask, np.ones((3, 3), np.uint8), iterations=1)
    diffuse_bgr = cv2.cvtColor((diffuse * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(diffuse_bgr, hard_mask, 3, cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    restored = rgb * (1.0 - soft[:, :, None]) + inpainted_rgb * soft[:, :, None]
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def yang_style_highlight_removal(bgr, iterative=False, iterations=4):
    """
    Real-time implementation inspired by:
    Yang et al., "Efficient and Robust Specular Highlight Removal",
    IEEE TPAMI, DOI: 10.1109/TPAMI.2014.2360402.

    Core idea used here:
    1. Use normalized RGB chromaticity and maximum chromaticity.
    2. Build a pseudo specular-free image by subtracting the neutral component
       min(R, G, B), which follows the white-specular dichromatic assumption.
    3. Use the difference between the pseudo specular-free maximum chromaticity
       and the original maximum chromaticity as the highlight cue.

    The paper contains additional optimization/propagation details. This UI keeps
    the part above real-time and exposes the key difference image for inspection.
    """
    bgr = bgr.astype(np.uint8)
    rgb, sum_rgb, min_rgb, max_chroma, pseudo_max_chroma, max_chroma_diff = compute_yang_chromaticity_terms(bgr)
    if iterative:
        restored_rgb = remove_highlight_iterative(
            bgr, rgb, sum_rgb, min_rgb, max_chroma, pseudo_max_chroma, max_chroma_diff,
            iterations=iterations
        )
    else:
        restored_rgb = remove_highlight_single_pass(bgr, rgb, min_rgb, max_chroma_diff)

    diff_u8 = normalize_to_u8(max_chroma_diff)
    diff_rgb = cv2.cvtColor(diff_u8, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), restored_rgb, diff_rgb


class SyncedImagePanel(tk.Frame):
    def __init__(self, master, title, app):
        super().__init__(master, bg=DISPLAY_BG)
        self.app = app
        self.title_label = tk.Label(self, text=title, bg=DISPLAY_BG, fg=TEXT_FG, font=("Segoe UI", 11, "bold"))
        self.title_label.pack(fill=tk.X, padx=4, pady=(2, 4))
        self.canvas = tk.Canvas(self, bg=PANEL_BG, highlightthickness=1, highlightbackground="#333333")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.photo = None
        self.image_id = None
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Configure>", lambda _event: self.app.render_all())

    def on_wheel(self, event):
        factor = 1.12 if event.delta > 0 else 1.0 / 1.12
        self.app.zoom_at(event.x, event.y, factor)

    def on_press(self, event):
        self.app.start_pan(event.x, event.y)

    def on_drag(self, event):
        self.app.pan_to(event.x, event.y)

    def draw(self, rgb, zoom, offset_x, offset_y):
        self.canvas.delete("all")
        if rgb is None:
            w = max(1, self.canvas.winfo_width())
            h = max(1, self.canvas.winfo_height())
            self.canvas.create_text(w // 2, h // 2, text="No frame", fill="#888888", font=("Segoe UI", 14))
            return

        ih, iw = rgb.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        fit = min(cw / max(1, iw), ch / max(1, ih))
        scale = max(0.03, min(24.0, fit * zoom))
        out_w = max(1, int(round(iw * scale)))
        out_h = max(1, int(round(ih * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (out_w, out_h), interpolation=interp)
        self.photo = rgb_to_photoimage(resized)
        x = int(round((cw - out_w) * 0.5 + offset_x))
        y = int(round((ch - out_h) * 0.5 + offset_y))
        self.image_id = self.canvas.create_image(x, y, image=self.photo, anchor=tk.NW)


class SpecularVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=DISPLAY_BG)
        self.cap = None
        self.video_path = None
        self.frame_count = 0
        self.fps = 30.0
        self.frame_index = 0
        self.playing = False
        self.frame_bgr = None
        self.views = [None, None, None]
        self.last_process_ms = 0.0
        self.iterative_var = tk.BooleanVar(value=False)
        self.iterations_var = tk.IntVar(value=4)
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_anchor = None
        self.after_id = None
        self._scale_updating = False

        self.build_ui()

    def build_ui(self):
        top = tk.Frame(self.root, bg=DISPLAY_BG)
        top.pack(fill=tk.X, padx=8, pady=8)

        self.load_btn = tk.Button(top, text="Load Video", command=self.load_video, bg="#2B2B2B", fg=TEXT_FG)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.play_btn = tk.Button(top, text="Play", command=self.toggle_play, bg="#2B2B2B", fg=TEXT_FG)
        self.play_btn.pack(side=tk.LEFT, padx=6)
        self.prev_btn = tk.Button(top, text="-1", command=lambda: self.goto_frame(self.frame_index - 1), bg="#2B2B2B", fg=TEXT_FG)
        self.prev_btn.pack(side=tk.LEFT, padx=3)
        self.next_btn = tk.Button(top, text="+1", command=lambda: self.goto_frame(self.frame_index + 1), bg="#2B2B2B", fg=TEXT_FG)
        self.next_btn.pack(side=tk.LEFT, padx=3)

        tk.Label(top, text="Frame", bg=DISPLAY_BG, fg=TEXT_FG).pack(side=tk.LEFT, padx=(14, 4))
        self.frame_var = tk.IntVar(value=0)
        self.frame_entry = tk.Entry(top, textvariable=self.frame_var, width=8, bg="#222222", fg=TEXT_FG, insertbackground=TEXT_FG)
        self.frame_entry.pack(side=tk.LEFT, padx=3)
        self.frame_entry.bind("<Return>", lambda _event: self.goto_frame(self.frame_var.get()))
        self.goto_btn = tk.Button(top, text="Go", command=lambda: self.goto_frame(self.frame_var.get()), bg="#2B2B2B", fg=TEXT_FG)
        self.goto_btn.pack(side=tk.LEFT, padx=3)

        self.iterative_check = tk.Checkbutton(
            top, text="Iterative", variable=self.iterative_var, command=self.reprocess_current_frame,
            bg=DISPLAY_BG, fg=TEXT_FG, selectcolor="#242424", activebackground=DISPLAY_BG,
            activeforeground=TEXT_FG
        )
        self.iterative_check.pack(side=tk.LEFT, padx=(14, 4))
        tk.Label(top, text="Iters", bg=DISPLAY_BG, fg=TEXT_FG).pack(side=tk.LEFT, padx=(4, 2))
        self.iter_spin = tk.Spinbox(
            top, from_=1, to=12, textvariable=self.iterations_var, width=4,
            command=self.reprocess_current_frame, bg="#222222", fg=TEXT_FG,
            insertbackground=TEXT_FG, buttonbackground="#303030"
        )
        self.iter_spin.pack(side=tk.LEFT, padx=(0, 6))
        self.iter_spin.bind("<Return>", lambda _event: self.reprocess_current_frame())

        self.reset_view_btn = tk.Button(top, text="Reset View", command=self.reset_view, bg="#2B2B2B", fg=TEXT_FG)
        self.reset_view_btn.pack(side=tk.LEFT, padx=(14, 6))
        self.status = tk.Label(top, text="No video loaded", bg=DISPLAY_BG, fg=ACCENT, anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.scale = tk.Scale(
            self.root, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_scale,
            bg=DISPLAY_BG, fg=TEXT_FG, troughcolor="#303030", highlightthickness=0,
            activebackground=ACCENT
        )
        self.scale.pack(fill=tk.X, padx=8, pady=(0, 6))

        panels = tk.Frame(self.root, bg=DISPLAY_BG)
        panels.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        panels.grid_columnconfigure(0, weight=1, uniform="panel")
        panels.grid_columnconfigure(1, weight=1, uniform="panel")
        panels.grid_columnconfigure(2, weight=1, uniform="panel")
        panels.grid_rowconfigure(0, weight=1)

        self.panels = [
            SyncedImagePanel(panels, "Original", self),
            SyncedImagePanel(panels, "Restored Color", self),
            SyncedImagePanel(panels, "Max-Chromaticity Difference", self),
        ]
        for i, panel in enumerate(self.panels):
            panel.grid(row=0, column=i, sticky="nsew")

        hint = tk.Label(
            self.root,
            text="Mouse wheel: zoom | Left drag: pan | Play processes frames in real time",
            bg=DISPLAY_BG, fg="#AAAAAA", anchor=tk.W
        )
        hint.pack(fill=tk.X, padx=10, pady=(0, 6))

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Load failed", "Cannot open selected video.")
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.video_path = path
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 1e-3:
            self.fps = 30.0
        self.frame_index = 0
        self.scale.configure(to=max(0, self.frame_count - 1))
        self.reset_view()
        self.goto_frame(0)

    def toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.schedule_next()

    def schedule_next(self):
        if not self.playing:
            return
        delay = max(1, int(round(1000.0 / max(1.0, self.fps))))
        self.after_id = self.root.after(delay, self.play_step)

    def play_step(self):
        if not self.playing or self.cap is None:
            return
        next_idx = self.frame_index + 1
        if self.frame_count > 0 and next_idx >= self.frame_count:
            self.playing = False
            self.play_btn.configure(text="Play")
            return
        self.goto_frame(next_idx, from_playback=True)
        self.schedule_next()

    def goto_frame(self, idx, from_playback=False):
        if self.cap is None:
            return
        idx = int(max(0, idx))
        if self.frame_count > 0:
            idx = min(idx, self.frame_count - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if not from_playback:
                messagebox.showwarning("Frame read failed", f"Cannot read frame {idx}.")
            return
        self.frame_index = idx
        self.frame_bgr = frame
        self.process_current_frame()
        self._scale_updating = True
        self.frame_var.set(idx)
        self.scale.set(idx)
        self._scale_updating = False
        self.update_status()
        self.render_all()

    def process_current_frame(self):
        if self.frame_bgr is None:
            return
        iterative = bool(self.iterative_var.get())
        iterations = int(max(1, min(12, self.iterations_var.get())))
        t0 = time.perf_counter()
        self.views = list(yang_style_highlight_removal(self.frame_bgr, iterative=iterative, iterations=iterations))
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        mode = f"iterative({iterations})" if iterative else "single-pass"
        print(f"[FrameTiming] frame={self.frame_index} mode={mode} specular_process={self.last_process_ms:.2f} ms")

    def reprocess_current_frame(self):
        if self.frame_bgr is None:
            return
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="Play")
        self.process_current_frame()
        self.update_status()
        self.render_all()

    def on_scale(self, value):
        if self._scale_updating or self.cap is None:
            return
        if self.playing:
            self.playing = False
            self.play_btn.configure(text="Play")
        self.goto_frame(int(float(value)))

    def update_status(self):
        total = self.frame_count if self.frame_count > 0 else "?"
        mode = f"iterative x{self.iterations_var.get()}" if self.iterative_var.get() else "single-pass"
        self.status.configure(
            text=f"{self.video_path} | frame {self.frame_index}/{total} | fps {self.fps:.2f} | {mode} | process {self.last_process_ms:.2f} ms | zoom {self.zoom:.2f}x"
        )

    def reset_view(self):
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.update_status()
        self.render_all()

    def start_pan(self, x, y):
        self.pan_anchor = (x, y, self.offset_x, self.offset_y)

    def pan_to(self, x, y):
        if self.pan_anchor is None:
            return
        ax, ay, ox, oy = self.pan_anchor
        self.offset_x = ox + (x - ax)
        self.offset_y = oy + (y - ay)
        self.render_all()

    def zoom_at(self, x, y, factor):
        old_zoom = self.zoom
        self.zoom = float(np.clip(self.zoom * factor, 0.05, 24.0))
        if abs(self.zoom - old_zoom) < 1e-9:
            return
        # Keep the cursor neighborhood visually stable in the panel where zoom happened.
        ratio = self.zoom / old_zoom
        self.offset_x = x - (x - self.offset_x) * ratio
        self.offset_y = y - (y - self.offset_y) * ratio
        self.update_status()
        self.render_all()

    def render_all(self):
        if not hasattr(self, "panels"):
            return
        for panel, rgb in zip(self.panels, self.views):
            panel.draw(rgb, self.zoom, self.offset_x, self.offset_y)

    def on_close(self):
        self.playing = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1500x760")
    app = SpecularVideoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
