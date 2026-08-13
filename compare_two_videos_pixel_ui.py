#!/usr/bin/env python3
"""Pixel-aligned two-video viewer with a small Tkinter UI.

Core controls
-------------
* Load A / Load B: select two videos manually.
* C: toggle A and B at the same frame index and source-pixel position.
* O: show a same-pixel alpha overlay.
* Space: pause / resume.
* Mouse wheel or +/-: zoom around the mouse cursor / canvas center.
* Left-drag: pan the zoomed frame.
* Left / Right: step one frame while paused.
* R or double-click: reset the view.

The videos must have the same resolution.  Frames are synchronized by frame
index, rather than by timestamp, so no resize or geometric transform can hide
a pixel-coordinate mismatch.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "雙影片 Pixel 對位比較器"
VIDEO_FILE_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm"),
    ("All files", "*.*"),
]
MIN_ZOOM = 0.25
MAX_ZOOM = 32.0
ZOOM_STEP = 1.25


@dataclass
class VideoTrack:
    """One OpenCV video capture plus its currently decoded frame."""

    path: Path
    capture: cv2.VideoCapture
    width: int
    height: int
    fps: float
    frame_count: int
    frame_index: int = -1
    frame: np.ndarray | None = None

    @classmethod
    def open(cls, path: str | Path) -> "VideoTrack":
        resolved = Path(path).expanduser().resolve()
        capture = cv2.VideoCapture(str(resolved))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"無法開啟影片：{resolved}")

        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError(f"讀不到影片解析度：{resolved}")
        if frame_count <= 0:
            capture.release()
            raise RuntimeError(f"讀不到影片總 frame 數：{resolved}")
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        return cls(
            path=resolved,
            capture=capture,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )

    def read(self, frame_index: int) -> np.ndarray | None:
        index = int(np.clip(frame_index, 0, self.frame_count - 1))
        if self.frame is not None and index == self.frame_index:
            return self.frame

        # Sequential playback avoids a costly random seek on every frame.
        if index != self.frame_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        self.frame_index = index
        self.frame = frame
        return frame

    def close(self) -> None:
        self.capture.release()
        self.frame = None
        self.frame_index = -1

    def description(self) -> str:
        return (
            f"{self.path.name} | {self.width}×{self.height} | "
            f"{self.fps:.3f} fps | {self.frame_count} frames"
        )


def validate_video_pair(track_a: VideoTrack, track_b: VideoTrack) -> None:
    """Reject any pair that cannot share exact source-pixel coordinates."""
    if (track_a.width, track_a.height) != (track_b.width, track_b.height):
        raise ValueError(
            "兩支影片解析度不同，無法保證相同 pixel 座標：\n"
            f"A = {track_a.width}×{track_a.height}\n"
            f"B = {track_b.width}×{track_b.height}"
        )


def compose_aligned_frame(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    mode: str,
    overlay_b_alpha: float = 0.5,
) -> np.ndarray:
    """Return A, B, or a same-coordinate alpha blend without resizing."""
    if frame_a.shape != frame_b.shape:
        raise ValueError("A/B frame shape mismatch")
    if mode == "A":
        return frame_a
    if mode == "B":
        return frame_b
    if mode == "OVERLAY":
        alpha = float(np.clip(overlay_b_alpha, 0.0, 1.0))
        return cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0.0)
    raise ValueError(f"Unknown display mode: {mode}")


def fit_scale(canvas_width: int, canvas_height: int, source_width: int, source_height: int) -> float:
    if min(canvas_width, canvas_height, source_width, source_height) <= 0:
        return 1.0
    return min(canvas_width / source_width, canvas_height / source_height)


def render_viewport(
    frame: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> np.ndarray:
    """Map a source frame into the shared canvas transform."""
    source_height, source_width = frame.shape[:2]
    base = fit_scale(canvas_width, canvas_height, source_width, source_height)
    scale = max(1e-9, base * zoom)
    transform = np.asarray(
        [
            [scale, 0.0, canvas_width / 2.0 - pan_x * scale],
            [0.0, scale, canvas_height / 2.0 - pan_y * scale],
        ],
        dtype=np.float64,
    )
    interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        frame,
        transform,
        (canvas_width, canvas_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(18, 18, 18),
    )


class PixelAlignedVideoViewer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x860")
        self.minsize(900, 620)

        self.track_a: VideoTrack | None = None
        self.track_b: VideoTrack | None = None
        self.current_index = 0
        self.mode = "A"
        self.playing = False
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self._photo: tk.PhotoImage | None = None
        self._image_item: int | None = None
        self._render_job: str | None = None
        self._play_job: str | None = None
        self._seek_job: str | None = None
        self._pending_seek_index: int | None = None
        self._updating_timeline = False
        self._resume_after_seek = False
        self._drag_start: tuple[int, int] | None = None
        self._drag_pan_start: tuple[float, float] | None = None
        self._play_origin_index = 0
        self._play_origin_time = 0.0

        self.overlay_alpha = tk.DoubleVar(value=50.0)
        self.timeline_value = tk.DoubleVar(value=0.0)
        self.mode_text = tk.StringVar(value="顯示：A")
        self.frame_text = tk.StringVar(value="Frame：—")
        self.zoom_text = tk.StringVar(value="縮放：100% fit")
        self.pixel_text = tk.StringVar(value="Pixel：將滑鼠移到畫面上查看 A/B RGB")
        self.track_a_text = tk.StringVar(value="A：尚未載入")
        self.track_b_text = tk.StringVar(value="B：尚未載入")
        self.notice_text = tk.StringVar(
            value="請載入兩支相同解析度的影片；同步方式為相同 frame index。"
        )

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after_idle(self._reset_view)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self, padding=(8, 8, 8, 4))
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="載入 A 影片", command=lambda: self._choose_video("A")).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(toolbar, text="載入 B 影片", command=lambda: self._choose_video("B")).pack(
            side="left", padx=5
        )
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.play_button = ttk.Button(toolbar, text="▶ 播放 (Space)", command=self._toggle_play)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(toolbar, text="◀ 前一幀", command=lambda: self._step_frame(-1)).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="後一幀 ▶", command=lambda: self._step_frame(1)).pack(
            side="left", padx=3
        )
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="A", width=4, command=lambda: self._set_mode("A")).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="B", width=4, command=lambda: self._set_mode("B")).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="50% 疊圖 (O)", command=lambda: self._set_mode("OVERLAY")).pack(
            side="left", padx=4
        )
        ttk.Label(toolbar, textvariable=self.mode_text).pack(side="left", padx=8)
        ttk.Button(toolbar, text="重設視角 (R)", command=self._reset_view).pack(
            side="right", padx=2
        )

        info = ttk.Frame(self, padding=(8, 0, 8, 5))
        info.grid(row=1, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)
        ttk.Label(info, text="A", width=2).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.track_a_text).grid(row=0, column=1, sticky="w")
        ttk.Label(info, text="B", width=2).grid(row=1, column=0, sticky="w")
        ttk.Label(info, textvariable=self.track_b_text).grid(row=1, column=1, sticky="w")
        ttk.Label(info, textvariable=self.notice_text, foreground="#A15C00").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        canvas_frame = ttk.Frame(self, padding=(8, 0, 8, 0))
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            background="#121212",
            highlightthickness=1,
            highlightbackground="#666666",
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._request_render())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, ZOOM_STEP))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, 1.0 / ZOOM_STEP))
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._drag_pan)
        self.canvas.bind("<ButtonRelease-1>", self._end_pan)
        self.canvas.bind("<Double-Button-1>", lambda _event: self._reset_view())
        self.canvas.bind("<Motion>", self._update_pixel_readout)
        self.canvas.bind("<Leave>", lambda _event: self.pixel_text.set("Pixel：—"))

        controls = ttk.Frame(self, padding=(8, 5, 8, 8))
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, textvariable=self.frame_text, width=20).grid(
            row=0, column=0, sticky="w"
        )
        self.timeline = ttk.Scale(
            controls,
            from_=0,
            to=1,
            orient="horizontal",
            variable=self.timeline_value,
            command=self._schedule_seek,
        )
        self.timeline.grid(row=0, column=1, sticky="ew", padx=8)
        self.timeline.bind("<ButtonPress-1>", self._begin_seek)
        self.timeline.bind("<ButtonRelease-1>", self._end_seek)
        ttk.Label(controls, textvariable=self.zoom_text, width=20).grid(
            row=0, column=2, sticky="e"
        )

        ttk.Label(controls, text="疊圖 B 比例").grid(row=1, column=0, sticky="w", pady=(5, 0))
        alpha_scale = ttk.Scale(
            controls,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.overlay_alpha,
            command=self._on_alpha_changed,
        )
        alpha_scale.grid(row=1, column=1, sticky="ew", padx=8, pady=(5, 0))
        ttk.Label(controls, textvariable=self.pixel_text).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(5, 0)
        )
        ttk.Label(
            controls,
            text="快捷鍵：C 切換 A/B｜O 疊圖｜Space 播放/暫停｜滾輪縮放｜左鍵拖曳｜←/→ 單幀",
            foreground="#555555",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _bind_shortcuts(self) -> None:
        self.bind_all("<KeyPress>", self._on_key_press)

    def _on_key_press(self, event: tk.Event) -> str | None:
        key = str(event.keysym).lower()
        char = str(event.char).lower()
        if char == "c":
            self._toggle_a_b()
        elif char == "o":
            self._set_mode("OVERLAY")
        elif char == "a":
            self._set_mode("A")
        elif char == "b":
            self._set_mode("B")
        elif char == "r":
            self._reset_view()
        elif key == "space":
            self._toggle_play()
            return "break"
        elif key in ("plus", "equal", "kp_add"):
            self._zoom_at_canvas_center(ZOOM_STEP)
        elif key in ("minus", "underscore", "kp_subtract"):
            self._zoom_at_canvas_center(1.0 / ZOOM_STEP)
        elif key == "left":
            self._step_frame(-1)
            return "break"
        elif key == "right":
            self._step_frame(1)
            return "break"
        return None

    def _choose_video(self, slot: str) -> None:
        initial_dir = None
        current = self.track_a if slot == "A" else self.track_b
        if current is not None:
            initial_dir = str(current.path.parent)
        path = filedialog.askopenfilename(
            title=f"選擇 {slot} 影片",
            initialdir=initial_dir,
            filetypes=VIDEO_FILE_TYPES,
        )
        if path:
            self._load_video(slot, path)

    def _load_video(self, slot: str, path: str | Path) -> None:
        self._pause()
        try:
            candidate = VideoTrack.open(path)
            other = self.track_b if slot == "A" else self.track_a
            if other is not None:
                validate_video_pair(candidate, other)
        except Exception as exc:
            if "candidate" in locals():
                candidate.close()
            messagebox.showerror("影片載入失敗", str(exc), parent=self)
            return

        old = self.track_a if slot == "A" else self.track_b
        if old is not None:
            old.close()
        if slot == "A":
            self.track_a = candidate
            self.track_a_text.set(f"A：{candidate.description()}")
        else:
            self.track_b = candidate
            self.track_b_text.set(f"B：{candidate.description()}")

        if self._pair_ready():
            assert self.track_a is not None and self.track_b is not None
            self.timeline.configure(to=max(1, self._common_frame_count() - 1))
            fps_note = ""
            if abs(self.track_a.fps - self.track_b.fps) > 0.01:
                fps_note = (
                    f"；注意 A/B fps 不同 ({self.track_a.fps:.3f}/{self.track_b.fps:.3f})，"
                    "仍以相同 frame index 同步"
                )
            self.notice_text.set(
                f"Pixel 對位有效：{self.track_a.width}×{self.track_a.height}；"
                f"共同比較 {self._common_frame_count()} frames{fps_note}"
            )
            self.mode = "A"
            self._load_common_frame(0)
            self._reset_view()
        else:
            candidate.read(0)
            self.current_index = 0
            self.mode = slot
            self.timeline.configure(to=max(1, candidate.frame_count - 1))
            self._update_labels()
            self._reset_view()

    def _pair_ready(self) -> bool:
        return self.track_a is not None and self.track_b is not None

    def _common_frame_count(self) -> int:
        if self._pair_ready():
            assert self.track_a is not None and self.track_b is not None
            return min(self.track_a.frame_count, self.track_b.frame_count)
        track = self.track_a or self.track_b
        return track.frame_count if track is not None else 0

    def _playback_fps(self) -> float:
        track = self.track_a or self.track_b
        return track.fps if track is not None else 30.0

    def _load_common_frame(self, index: int) -> bool:
        total = self._common_frame_count()
        if total <= 0:
            return False
        target = int(np.clip(index, 0, total - 1))
        tracks = [track for track in (self.track_a, self.track_b) if track is not None]
        frames = [track.read(target) for track in tracks]
        if any(frame is None for frame in frames):
            self._pause()
            self.notice_text.set(f"Frame {target} 解碼失敗，播放已暫停。")
            return False
        self.current_index = target
        self._update_labels()
        self._request_render()
        return True

    def _toggle_play(self) -> None:
        if self._common_frame_count() <= 0:
            messagebox.showinfo("尚未載入影片", "請先載入影片。", parent=self)
            return
        if self.playing:
            self._pause()
            return
        if self.current_index >= self._common_frame_count() - 1:
            self._load_common_frame(0)
        self.playing = True
        self.play_button.configure(text="⏸ 暫停 (Space)")
        self._play_origin_index = self.current_index
        self._play_origin_time = time.perf_counter()
        self._schedule_play_tick()

    def _pause(self) -> None:
        self.playing = False
        self.play_button.configure(text="▶ 播放 (Space)")
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None

    def _schedule_play_tick(self) -> None:
        if self.playing and self._play_job is None:
            self._play_job = self.after(5, self._play_tick)

    def _play_tick(self) -> None:
        self._play_job = None
        if not self.playing:
            return
        elapsed = time.perf_counter() - self._play_origin_time
        target = self._play_origin_index + int(elapsed * self._playback_fps())
        end = self._common_frame_count() - 1
        if target >= end:
            self._load_common_frame(end)
            self._pause()
            return
        if target > self.current_index:
            self._load_common_frame(target)
        self._schedule_play_tick()

    def _step_frame(self, delta: int) -> None:
        if self._common_frame_count() <= 0:
            return
        self._pause()
        self._load_common_frame(self.current_index + int(delta))

    def _begin_seek(self, _event: tk.Event) -> None:
        self._resume_after_seek = self.playing
        self._pause()

    def _schedule_seek(self, value: str) -> None:
        if self._updating_timeline:
            return
        self._pending_seek_index = int(round(float(value)))
        if self._seek_job is not None:
            self.after_cancel(self._seek_job)
        self._seek_job = self.after(35, self._apply_pending_seek)

    def _apply_pending_seek(self) -> None:
        self._seek_job = None
        if self._pending_seek_index is not None:
            self._load_common_frame(self._pending_seek_index)
            self._pending_seek_index = None

    def _end_seek(self, _event: tk.Event) -> None:
        if self._seek_job is not None:
            self.after_cancel(self._seek_job)
            self._seek_job = None
        self._pending_seek_index = int(round(self.timeline_value.get()))
        self._apply_pending_seek()
        if self._resume_after_seek:
            self._resume_after_seek = False
            self._toggle_play()

    def _set_mode(self, mode: str) -> None:
        if mode == "A" and self.track_a is None:
            return
        if mode == "B" and self.track_b is None:
            return
        if mode == "OVERLAY" and not self._pair_ready():
            messagebox.showinfo("需要兩支影片", "疊圖模式需要先載入 A 與 B。", parent=self)
            return
        self.mode = mode
        self._update_labels()
        self._request_render()

    def _toggle_a_b(self) -> None:
        if not self._pair_ready():
            return
        self._set_mode("B" if self.mode == "A" else "A")

    def _on_alpha_changed(self, _value: str) -> None:
        if self.mode == "OVERLAY":
            self._update_labels()
            self._request_render()

    def _active_source_frame(self) -> np.ndarray | None:
        frame_a = self.track_a.frame if self.track_a is not None else None
        frame_b = self.track_b.frame if self.track_b is not None else None
        if self.mode == "A":
            return frame_a
        if self.mode == "B":
            return frame_b
        if self.mode == "OVERLAY" and frame_a is not None and frame_b is not None:
            return compose_aligned_frame(
                frame_a,
                frame_b,
                "OVERLAY",
                self.overlay_alpha.get() / 100.0,
            )
        return frame_a if frame_a is not None else frame_b

    def _source_size(self) -> tuple[int, int] | None:
        track = self.track_a or self.track_b
        return (track.width, track.height) if track is not None else None

    def _effective_scale(self) -> float:
        size = self._source_size()
        if size is None:
            return 1.0
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        return fit_scale(canvas_width, canvas_height, size[0], size[1]) * self.zoom

    def _canvas_to_source(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        scale = max(self._effective_scale(), 1e-9)
        return (
            self.pan_x + (canvas_x - self.canvas.winfo_width() / 2.0) / scale,
            self.pan_y + (canvas_y - self.canvas.winfo_height() / 2.0) / scale,
        )

    def _clamp_pan(self) -> None:
        size = self._source_size()
        if size is None:
            return
        source_width, source_height = size
        scale = max(self._effective_scale(), 1e-9)
        half_visible_width = self.canvas.winfo_width() / (2.0 * scale)
        half_visible_height = self.canvas.winfo_height() / (2.0 * scale)
        if half_visible_width >= source_width / 2.0:
            self.pan_x = source_width / 2.0
        else:
            self.pan_x = float(
                np.clip(self.pan_x, half_visible_width, source_width - half_visible_width)
            )
        if half_visible_height >= source_height / 2.0:
            self.pan_y = source_height / 2.0
        else:
            self.pan_y = float(
                np.clip(self.pan_y, half_visible_height, source_height - half_visible_height)
            )

    def _reset_view(self) -> None:
        self.zoom = 1.0
        size = self._source_size()
        if size is not None:
            self.pan_x = size[0] / 2.0
            self.pan_y = size[1] / 2.0
        self._update_labels()
        self._request_render()

    def _zoom_at_canvas_center(self, factor: float) -> None:
        self._zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, factor)

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        factor = ZOOM_STEP if event.delta > 0 else 1.0 / ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        if self._source_size() is None:
            return
        before_x, before_y = self._canvas_to_source(canvas_x, canvas_y)
        new_zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        if math.isclose(new_zoom, self.zoom):
            return
        self.zoom = new_zoom
        new_scale = max(self._effective_scale(), 1e-9)
        self.pan_x = before_x - (canvas_x - self.canvas.winfo_width() / 2.0) / new_scale
        self.pan_y = before_y - (canvas_y - self.canvas.winfo_height() / 2.0) / new_scale
        self._clamp_pan()
        self._update_labels()
        self._request_render()

    def _start_pan(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_pan_start = (self.pan_x, self.pan_y)
        self.canvas.configure(cursor="fleur")

    def _drag_pan(self, event: tk.Event) -> None:
        if self._drag_start is None or self._drag_pan_start is None:
            return
        scale = max(self._effective_scale(), 1e-9)
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self.pan_x = self._drag_pan_start[0] - dx / scale
        self.pan_y = self._drag_pan_start[1] - dy / scale
        self._clamp_pan()
        self._request_render()

    def _end_pan(self, _event: tk.Event) -> None:
        self._drag_start = None
        self._drag_pan_start = None
        self.canvas.configure(cursor="crosshair")

    def _update_pixel_readout(self, event: tk.Event) -> None:
        size = self._source_size()
        if size is None:
            return
        source_x, source_y = self._canvas_to_source(event.x, event.y)
        x, y = int(math.floor(source_x)), int(math.floor(source_y))
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            self.pixel_text.set("Pixel：畫面外")
            return

        def rgb_at(track: VideoTrack | None) -> tuple[int, int, int] | None:
            if track is None or track.frame is None:
                return None
            b, g, r = (int(value) for value in track.frame[y, x])
            return r, g, b

        rgb_a = rgb_at(self.track_a)
        rgb_b = rgb_at(self.track_b)
        if rgb_a is not None and rgb_b is not None:
            delta = tuple(abs(a - b) for a, b in zip(rgb_a, rgb_b))
            self.pixel_text.set(
                f"Pixel ({x}, {y})｜A RGB={rgb_a}｜B RGB={rgb_b}｜|Δ|={delta}"
            )
        else:
            value = rgb_a if rgb_a is not None else rgb_b
            slot = "A" if rgb_a is not None else "B"
            self.pixel_text.set(f"Pixel ({x}, {y})｜{slot} RGB={value}")

    def _request_render(self) -> None:
        if self._render_job is None:
            self._render_job = self.after_idle(self._render)

    def _render(self) -> None:
        self._render_job = None
        frame = self._active_source_frame()
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if frame is None or canvas_width < 2 or canvas_height < 2:
            self.canvas.delete("all")
            self._image_item = None
            return
        self._clamp_pan()
        viewport = render_viewport(
            frame,
            canvas_width,
            canvas_height,
            self.zoom,
            self.pan_x,
            self.pan_y,
        )
        rgb = cv2.cvtColor(viewport, cv2.COLOR_BGR2RGB)
        ppm = f"P6\n{canvas_width} {canvas_height}\n255\n".encode("ascii") + rgb.tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        if self._image_item is None:
            self._image_item = self.canvas.create_image(
                0, 0, anchor="nw", image=self._photo
            )
        else:
            self.canvas.itemconfigure(self._image_item, image=self._photo)

    def _update_labels(self) -> None:
        display_name = {
            "A": "A",
            "B": "B",
            "OVERLAY": f"A/B 疊圖（B {self.overlay_alpha.get():.0f}%）",
        }[self.mode]
        self.mode_text.set(f"顯示：{display_name}")
        total = self._common_frame_count()
        if total > 0:
            fps = self._playback_fps()
            seconds = self.current_index / fps
            self.frame_text.set(
                f"Frame：{self.current_index}/{total - 1}  ({seconds:.3f}s)"
            )
            self._updating_timeline = True
            self.timeline_value.set(self.current_index)
            self._updating_timeline = False
        else:
            self.frame_text.set("Frame：—")
        self.zoom_text.set(f"縮放：{self.zoom * 100:.0f}% fit")

    def _on_close(self) -> None:
        self._pause()
        if self._seek_job is not None:
            self.after_cancel(self._seek_job)
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        for track in (self.track_a, self.track_b):
            if track is not None:
                track.close()
        self.destroy()


def run_self_test() -> None:
    frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_b = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert compose_aligned_frame(frame_a, frame_b, "A") is frame_a
    assert compose_aligned_frame(frame_a, frame_b, "B") is frame_b
    overlay = compose_aligned_frame(frame_a, frame_b, "OVERLAY", 0.25)
    assert overlay.shape == frame_a.shape
    assert np.all(overlay == 25)
    viewport = render_viewport(frame_b, 8, 8, 1.0, 4.0, 4.0)
    assert viewport.shape == frame_b.shape
    assert np.array_equal(viewport, frame_b)
    print("Self-test passed: A/B switching, aligned overlay, and viewport transform")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run non-GUI alignment tests and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    try:
        app = PixelAlignedVideoViewer()
        app.mainloop()
    except tk.TclError as exc:
        print(f"GUI 啟動失敗：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
