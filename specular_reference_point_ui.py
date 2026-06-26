# -*- coding: utf-8 -*-
"""
Interactive video frame picker for specular-mask-aware reference-point selection.

Workflow:
1. Load a video.
2. Pick one frame as the left image and another as the right image.
3. Compute likely specular highlight masks.
4. Toggle mask overlay on/off.
5. Click the left image to show robust local reference points around the click.

This is intentionally a geometry-ready prototype: RT/baseline epipolar search can be
plugged in after the left-side reliable reference points are selected.
"""

from __future__ import annotations

import math
import os
import threading
import tkinter as tk
import base64
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


BG = "#111418"
PANEL = "#1b2027"
PANEL_2 = "#242a33"
TEXT = "#e7edf5"
MUTED = "#9aa7b6"
ACCENT = "#2f80ed"
LEFT_POINT = "#2fd17c"
RIGHT_POINT = "#36d6ff"
CLICK_POINT = "#ffcc00"
MATCH_POINT = "#ff7a45"
MASK_RED = (255, 54, 54)


@dataclass
class DisplayState:
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    width: int = 1
    height: int = 1


@dataclass
class PickedFrame:
    index: int
    bgr: np.ndarray
    mask: np.ndarray | None = None
    stable_bad_mask: np.ndarray | None = None


class SpecularReferencePointUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Specular-mask reference point picker")
        self.root.geometry("1480x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=BG)

        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_index = 0
        self.current_bgr: np.ndarray | None = None
        self.frame_cache: dict[int, np.ndarray] = {}
        self.cache_lock = threading.Lock()

        self.left: PickedFrame | None = None
        self.right: PickedFrame | None = None
        self.left_display = DisplayState()
        self.right_display = DisplayState()
        self.left_photo: tk.PhotoImage | None = None
        self.right_photo: tk.PhotoImage | None = None
        self.preview_photo: tk.PhotoImage | None = None
        self.left_zoom = 1.0
        self.right_zoom = 1.0
        self.left_pan = [0.0, 0.0]
        self.right_pan = [0.0, 0.0]
        self.drag_start: tuple[bool, int, int, float, float] | None = None

        self.show_mask = tk.BooleanVar(value=True)
        self.use_temporal_mask = tk.BooleanVar(value=True)
        self.v_threshold = tk.IntVar(value=210)
        self.s_threshold = tk.IntVar(value=80)
        self.rgb_high_threshold = tk.IntVar(value=235)
        self.whiteness_threshold = tk.IntVar(value=36)
        self.mask_dilate = tk.IntVar(value=5)
        self.search_radius = tk.IntVar(value=65)
        self.max_points = tk.IntVar(value=80)
        self.min_distance = tk.IntVar(value=7)

        self.left_clicked_xy: tuple[int, int] | None = None
        self.right_clicked_xy: tuple[int, int] | None = None
        self.left_reference_points: list[tuple[int, int, float]] = []
        self.right_reference_points: list[tuple[int, int, float]] = []
        self.matched_pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        self.predicted_right_xy: tuple[float, float] | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT, fieldbackground=PANEL_2)
        style.configure("TButton", background=PANEL_2, foreground=TEXT, padding=7)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.configure("TScale", background=PANEL)
        style.configure("TLabelframe", background=PANEL, foreground=TEXT)
        style.configure("TLabelframe.Label", background=PANEL, foreground=TEXT)

        top = tk.Frame(self.root, bg=PANEL, height=54)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)

        ttk.Button(top, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=(12, 8), pady=10)
        ttk.Button(top, text="Set Current as Left", command=self.set_left).pack(side=tk.LEFT, padx=4, pady=10)
        ttk.Button(top, text="Set Current as Right", command=self.set_right).pack(side=tk.LEFT, padx=4, pady=10)
        ttk.Button(top, text="Recompute Masks", command=self.recompute_masks).pack(side=tk.LEFT, padx=4, pady=10)
        ttk.Button(top, text="Match Candidates", command=self.match_candidates).pack(side=tk.LEFT, padx=4, pady=10)
        ttk.Button(top, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=4, pady=10)

        self.mask_check = ttk.Checkbutton(
            top,
            text="Show specular mask",
            variable=self.show_mask,
            command=self.redraw_selected_frames,
        )
        self.mask_check.pack(side=tk.LEFT, padx=(18, 4), pady=10)

        self.temporal_check = ttk.Checkbutton(
            top,
            text="Use temporal instability",
            variable=self.use_temporal_mask,
            command=self.on_temporal_toggle,
        )
        self.temporal_check.pack(side=tk.LEFT, padx=8, pady=10)

        self.status = tk.Label(top, text="Load a video to begin.", bg=PANEL, fg=MUTED, anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=16)

        body = tk.Frame(self.root, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_box = ttk.LabelFrame(body, text="Left frame: click to select a measurement point")
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.left_canvas = tk.Canvas(left_box, bg="#0a0d10", highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.left_canvas.bind("<Button-1>", lambda event: self.on_image_click(event, True))
        self.left_canvas.bind("<Configure>", lambda _e: self.redraw_selected_frames())
        self.bind_zoom_pan(self.left_canvas, True)

        right_box = ttk.LabelFrame(body, text="Right frame: click to select a search center")
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.right_canvas = tk.Canvas(right_box, bg="#0a0d10", highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.right_canvas.bind("<Button-1>", lambda event: self.on_image_click(event, False))
        self.right_canvas.bind("<Configure>", lambda _e: self.redraw_selected_frames())
        self.bind_zoom_pan(self.right_canvas, False)

        bottom = tk.Frame(self.root, bg=PANEL)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        preview_row = tk.Frame(bottom, bg=PANEL)
        preview_row.pack(fill=tk.X, padx=12, pady=(10, 0))
        preview_box = ttk.LabelFrame(preview_row, text="Current video frame")
        preview_box.pack(side=tk.LEFT, padx=(0, 12))
        self.preview_canvas = tk.Canvas(preview_box, width=260, height=145, bg="#0a0d10", highlightthickness=0)
        self.preview_canvas.pack(padx=6, pady=6)
        self.preview_canvas.bind("<Configure>", lambda _e: self.draw_current_preview())

        hint = tk.Label(
            preview_row,
            text="Click both selected images to create local candidates. Match Candidates estimates the left start point on the right image.",
            bg=PANEL,
            fg=MUTED,
            anchor="w",
            justify=tk.LEFT,
        )
        hint.pack(side=tk.LEFT, fill=tk.X, expand=True)

        nav = tk.Frame(bottom, bg=PANEL)
        nav.pack(fill=tk.X, padx=12, pady=(8, 6))
        ttk.Button(nav, text="-10", command=lambda: self.step_frame(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="-1", command=lambda: self.step_frame(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="+1", command=lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="+10", command=lambda: self.step_frame(10)).pack(side=tk.LEFT, padx=2)
        self.frame_label = tk.Label(nav, text="Frame: - / -", bg=PANEL, fg=TEXT, width=22)
        self.frame_label.pack(side=tk.LEFT, padx=12)
        self.frame_slider = ttk.Scale(nav, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        params = tk.Frame(bottom, bg=PANEL)
        params.pack(fill=tk.X, padx=12, pady=(0, 10))
        self._add_scale(params, "V bright", self.v_threshold, 120, 255)
        self._add_scale(params, "S low", self.s_threshold, 10, 180)
        self._add_scale(params, "RGB high", self.rgb_high_threshold, 180, 255)
        self._add_scale(params, "Whiteness", self.whiteness_threshold, 5, 100)
        self._add_scale(params, "Dilate", self.mask_dilate, 0, 15)
        self._add_scale(params, "Search radius", self.search_radius, 15, 180)
        self._add_scale(params, "Max pts", self.max_points, 10, 250)
        self._add_scale(params, "Min dist", self.min_distance, 3, 20)

    def _add_scale(
        self,
        parent: tk.Frame,
        label: str,
        variable: tk.IntVar,
        low: int,
        high: int,
    ) -> None:
        group = tk.Frame(parent, bg=PANEL)
        group.pack(side=tk.LEFT, padx=8)
        tk.Label(group, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        row = tk.Frame(group, bg=PANEL)
        row.pack()
        value_label = tk.Label(row, text=str(variable.get()), bg=PANEL, fg=TEXT, width=4)
        value_label.pack(side=tk.RIGHT)

        def changed(value: str) -> None:
            variable.set(int(float(value)))
            value_label.configure(text=str(variable.get()))
            self.recompute_masks()

        scale = ttk.Scale(row, from_=low, to=high, orient=tk.HORIZONTAL, command=changed, length=110)
        scale.set(variable.get())
        scale.pack(side=tk.LEFT)

    def load_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Cannot open video", f"Failed to open:\n{path}")
            return

        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.video_path = path
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_index = 0
        self.frame_cache.clear()
        self.left = None
        self.right = None
        self.clear_measurement_state()

        self.frame_slider.configure(to=max(0, self.frame_count - 1))
        self.frame_slider.set(0)
        self.current_bgr = self.get_frame(0)
        self.status.configure(
            text=f"{os.path.basename(path)} | {self.frame_count} frames | {self.fps:.2f} fps"
        )
        self.update_frame_label()
        self.draw_current_preview()
        self.redraw_selected_frames()

    def get_frame(self, index: int) -> np.ndarray | None:
        if self.cap is None or self.frame_count <= 0:
            return None
        index = int(np.clip(index, 0, self.frame_count - 1))
        with self.cache_lock:
            cached = self.frame_cache.get(index)
            if cached is not None:
                return cached.copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None

        with self.cache_lock:
            if len(self.frame_cache) > 80:
                for key in sorted(self.frame_cache.keys())[:30]:
                    self.frame_cache.pop(key, None)
            self.frame_cache[index] = frame.copy()
        return frame

    def on_slider(self, value: str) -> None:
        if self.cap is None:
            return
        idx = int(float(value))
        if idx == self.current_index:
            return
        self.current_index = idx
        self.current_bgr = self.get_frame(idx)
        self.update_frame_label()
        self.draw_current_preview()

    def update_frame_label(self) -> None:
        self.frame_label.configure(text=f"Frame: {self.current_index} / {max(0, self.frame_count - 1)}")

    def step_frame(self, delta: int) -> None:
        if self.cap is None:
            return
        idx = int(np.clip(self.current_index + delta, 0, self.frame_count - 1))
        self.frame_slider.set(idx)
        self.current_index = idx
        self.current_bgr = self.get_frame(idx)
        self.update_frame_label()
        self.draw_current_preview()

    def set_left(self) -> None:
        frame = self.get_frame(self.current_index)
        if frame is None:
            return
        self.left = PickedFrame(self.current_index, frame)
        self.left.mask = self.compute_specular_mask(frame)
        self.left.stable_bad_mask = self.compute_temporal_instability_mask(self.current_index)
        self.left_zoom = 1.0
        self.left_pan = [0.0, 0.0]
        self.left_clicked_xy = None
        self.left_reference_points.clear()
        self.clear_match_state()
        self.redraw_selected_frames()

    def set_right(self) -> None:
        frame = self.get_frame(self.current_index)
        if frame is None:
            return
        self.right = PickedFrame(self.current_index, frame)
        self.right.mask = self.compute_specular_mask(frame)
        self.right.stable_bad_mask = self.compute_temporal_instability_mask(self.current_index)
        self.right_zoom = 1.0
        self.right_pan = [0.0, 0.0]
        self.right_clicked_xy = None
        self.right_reference_points.clear()
        self.clear_match_state()
        self.redraw_selected_frames()

    def recompute_masks(self) -> None:
        if self.left is not None:
            self.left.mask = self.compute_specular_mask(self.left.bgr)
            self.left.stable_bad_mask = self.compute_temporal_instability_mask(self.left.index)
        if self.right is not None:
            self.right.mask = self.compute_specular_mask(self.right.bgr)
            self.right.stable_bad_mask = self.compute_temporal_instability_mask(self.right.index)
        if self.left is not None and self.left_clicked_xy is not None:
            self.left_reference_points = self.find_reference_points(
                self.left.bgr,
                self.combined_bad_mask(self.left),
                self.left_clicked_xy,
            )
        if self.right is not None and self.right_clicked_xy is not None:
            self.right_reference_points = self.find_reference_points(
                self.right.bgr,
                self.combined_bad_mask(self.right),
                self.right_clicked_xy,
            )
        self.clear_match_state()
        self.redraw_selected_frames()

    def clear_measurement_state(self) -> None:
        self.left_clicked_xy = None
        self.right_clicked_xy = None
        self.left_reference_points.clear()
        self.right_reference_points.clear()
        self.clear_match_state()

    def clear_match_state(self) -> None:
        self.matched_pairs.clear()
        self.predicted_right_xy = None

    def on_temporal_toggle(self) -> None:
        self.recompute_masks()

    def compute_specular_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _h, s, v = cv2.split(hsv)
        b, g, r = cv2.split(bgr)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        whiteness = max_rgb.astype(np.int16) - min_rgb.astype(np.int16)

        bright_low_sat = (v >= self.v_threshold.get()) & (s <= self.s_threshold.get())
        near_white = (max_rgb >= self.rgb_high_threshold.get()) & (
            whiteness <= self.whiteness_threshold.get()
        )

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 9)
        local_hot = gray.astype(np.int16) - blur.astype(np.int16)
        adaptive_hot = (gray >= max(160, self.v_threshold.get() - 20)) & (local_hot >= 18)

        mask = (bright_low_sat | near_white | adaptive_hot).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        dilate = self.mask_dilate.get()
        if dilate > 0:
            k = 2 * dilate + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
        return mask

    def compute_temporal_instability_mask(self, center_index: int) -> np.ndarray | None:
        if self.cap is None or self.frame_count <= 0 or not self.use_temporal_mask.get():
            return None

        offsets = [-6, -3, 0, 3, 6]
        frames: list[np.ndarray] = []
        for offset in offsets:
            frame = self.get_frame(int(np.clip(center_index + offset, 0, self.frame_count - 1)))
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))

        if len(frames) < 3:
            return None

        stack = np.stack(frames, axis=0)
        std = np.std(stack, axis=0)
        mean = np.mean(stack, axis=0)
        unstable = ((std > 18.0) & (mean > 150.0)).astype(np.uint8) * 255
        unstable = cv2.morphologyEx(unstable, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        unstable = cv2.dilate(unstable, np.ones((7, 7), np.uint8), iterations=1)
        return unstable

    def combined_bad_mask(self, picked: PickedFrame | None) -> np.ndarray | None:
        if picked is None:
            return None
        masks = [m for m in [picked.mask, picked.stable_bad_mask] if m is not None]
        if not masks:
            return None
        out = masks[0].copy()
        for mask in masks[1:]:
            out = cv2.bitwise_or(out, mask)
        return out

    def find_reference_points(
        self,
        bgr: np.ndarray,
        specular_mask: np.ndarray | None,
        click_xy: tuple[int, int],
    ) -> list[tuple[int, int, float]]:
        x0, y0 = click_xy
        h, w = bgr.shape[:2]
        radius = self.search_radius.get()
        x1, x2 = max(0, x0 - radius), min(w - 1, x0 + radius)
        y1, y2 = max(0, y0 - radius), min(h - 1, y0 + radius)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        yy, xx = np.mgrid[0:h, 0:w]
        circle = ((xx - x0) ** 2 + (yy - y0) ** 2) <= radius * radius
        roi = np.zeros((h, w), dtype=bool)
        roi[y1 : y2 + 1, x1 : x2 + 1] = True
        valid = roi & circle

        if specular_mask is not None:
            valid &= specular_mask == 0

        border = 8
        valid[:border, :] = False
        valid[-border:, :] = False
        valid[:, :border] = False
        valid[:, -border:] = False

        grad_valid = np.where(valid, grad, 0.0)
        if float(grad_valid.max()) <= 0.0:
            return []

        threshold = max(12.0, float(np.percentile(grad_valid[valid], 82)))
        candidates = np.argwhere(grad_valid >= threshold)
        scores = grad_valid[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(scores)[::-1]

        selected: list[tuple[int, int, float]] = []
        min_dist_sq = self.min_distance.get() ** 2
        for idx in order:
            y, x = int(candidates[idx, 0]), int(candidates[idx, 1])
            score = float(scores[idx])
            if any((x - px) ** 2 + (y - py) ** 2 < min_dist_sq for px, py, _ in selected):
                continue
            selected.append((x, y, score))
            if len(selected) >= self.max_points.get():
                break
        return selected

    def on_image_click(self, event: tk.Event, is_left: bool) -> None:
        picked = self.left if is_left else self.right
        display = self.left_display if is_left else self.right_display
        side_name = "left" if is_left else "right"
        if picked is None:
            return
        xy = self.canvas_to_image(event.x, event.y, display, picked.bgr.shape)
        if xy is None:
            return
        bad_mask = self.combined_bad_mask(picked)
        if bad_mask is not None and bad_mask[xy[1], xy[0]] > 0:
            self.status.configure(text=f"Clicked {side_name} point is inside/near a specular or unstable region.")
        else:
            self.status.configure(text=f"Clicked {side_name} point: x={xy[0]}, y={xy[1]}")

        points = self.find_reference_points(picked.bgr, bad_mask, xy)
        if is_left:
            self.left_clicked_xy = xy
            self.left_reference_points = points
        else:
            self.right_clicked_xy = xy
            self.right_reference_points = points
        self.clear_match_state()
        self.redraw_selected_frames()

    def match_candidates(self) -> None:
        if self.left is None or self.right is None:
            messagebox.showinfo("Need frames", "Please select both left and right frames first.")
            return
        if self.left_clicked_xy is None or self.right_clicked_xy is None:
            messagebox.showinfo("Need clicks", "Please click one start point on both left and right images.")
            return
        if not self.left_reference_points or not self.right_reference_points:
            messagebox.showinfo("Need candidates", "No candidate points are available to match.")
            return

        left_desc, left_pts = self.build_patch_descriptors(self.left.bgr, self.left_reference_points)
        right_desc, right_pts = self.build_patch_descriptors(self.right.bgr, self.right_reference_points)
        if len(left_pts) < 2 or len(right_pts) < 2:
            messagebox.showinfo("Too few candidates", "Need at least two usable patch descriptors on each side.")
            return

        sim = left_desc @ right_desc.T
        left_best = np.argmax(sim, axis=1)
        right_best = np.argmax(sim, axis=0)

        pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
        for li, ri in enumerate(left_best):
            score = float(sim[li, ri])
            if right_best[ri] == li and score >= 0.35:
                pairs.append((left_pts[li], right_pts[ri], score))

        if len(pairs) < 2:
            messagebox.showinfo("Weak match", "Candidate matching found too few mutual patch matches.")
            self.matched_pairs = pairs
            self.predicted_right_xy = None
            self.redraw_selected_frames()
            return

        self.matched_pairs = pairs
        self.predicted_right_xy = self.estimate_right_start_from_matches(pairs)
        if self.predicted_right_xy is None:
            self.status.configure(text=f"Matched {len(pairs)} candidate pairs, but could not estimate a point.")
        else:
            x, y = self.predicted_right_xy
            self.status.configure(
                text=f"Matched {len(pairs)} candidate pairs. Predicted right point: x={x:.1f}, y={y:.1f}"
            )
        self.redraw_selected_frames()

    def build_patch_descriptors(
        self,
        bgr: np.ndarray,
        points: list[tuple[int, int, float]],
        patch_radius: int = 9,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        h, w = gray.shape[:2]

        descriptors: list[np.ndarray] = []
        usable_points: list[tuple[int, int]] = []
        for x, y, _score in points:
            if (
                x - patch_radius < 0
                or y - patch_radius < 0
                or x + patch_radius >= w
                or y + patch_radius >= h
            ):
                continue
            intensity_patch = gray[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
            grad_patch = grad[y - patch_radius : y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
            descriptor = np.concatenate(
                [
                    intensity_patch.astype(np.float32).reshape(-1),
                    grad_patch.astype(np.float32).reshape(-1),
                ]
            )
            descriptor -= float(descriptor.mean())
            norm = float(np.linalg.norm(descriptor))
            if norm < 1e-5:
                continue
            descriptors.append(descriptor / norm)
            usable_points.append((x, y))

        if not descriptors:
            return np.empty((0, 0), dtype=np.float32), []
        return np.stack(descriptors).astype(np.float32), usable_points

    def estimate_right_start_from_matches(
        self,
        pairs: list[tuple[tuple[int, int], tuple[int, int], float]],
    ) -> tuple[float, float] | None:
        if self.left_clicked_xy is None:
            return None

        src = np.array([left for left, _right, _score in pairs], dtype=np.float32)
        dst = np.array([right for _left, right, _score in pairs], dtype=np.float32)
        start = np.array([[self.left_clicked_xy]], dtype=np.float32)

        if len(pairs) >= 3:
            affine, inliers = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=6.0,
                maxIters=1500,
                confidence=0.98,
            )
            if affine is not None and inliers is not None and int(inliers.sum()) >= 2:
                pred = cv2.transform(start, affine)[0, 0]
                return float(pred[0]), float(pred[1])

        shift = np.median(dst - src, axis=0)
        pred = np.array(self.left_clicked_xy, dtype=np.float32) + shift
        return float(pred[0]), float(pred[1])

    def canvas_to_image(
        self,
        cx: int,
        cy: int,
        display: DisplayState,
        image_shape: tuple[int, int, int],
    ) -> tuple[int, int] | None:
        h, w = image_shape[:2]
        ix = int(round((cx - display.offset_x) / display.scale))
        iy = int(round((cy - display.offset_y) / display.scale))
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    def image_to_canvas(self, x: int, y: int, display: DisplayState) -> tuple[int, int]:
        return (
            int(round(display.offset_x + x * display.scale)),
            int(round(display.offset_y + y * display.scale)),
        )

    def redraw_selected_frames(self) -> None:
        self.draw_picked(self.left_canvas, self.left, True)
        self.draw_picked(self.right_canvas, self.right, False)

    def bind_zoom_pan(self, canvas: tk.Canvas, is_left: bool) -> None:
        canvas.bind("<MouseWheel>", lambda event: self.on_image_wheel(event, is_left))
        canvas.bind("<Button-4>", lambda event: self.on_image_wheel(event, is_left))
        canvas.bind("<Button-5>", lambda event: self.on_image_wheel(event, is_left))
        canvas.bind("<ButtonPress-2>", lambda event: self.start_pan(event, is_left))
        canvas.bind("<B2-Motion>", lambda event: self.drag_pan(event, is_left))
        canvas.bind("<ButtonPress-3>", lambda event: self.start_pan(event, is_left))
        canvas.bind("<B3-Motion>", lambda event: self.drag_pan(event, is_left))

    def reset_zoom(self) -> None:
        self.left_zoom = 1.0
        self.right_zoom = 1.0
        self.left_pan = [0.0, 0.0]
        self.right_pan = [0.0, 0.0]
        self.redraw_selected_frames()

    def get_view_params(self, is_left: bool) -> tuple[float, list[float]]:
        if is_left:
            return self.left_zoom, self.left_pan
        return self.right_zoom, self.right_pan

    def set_view_params(self, is_left: bool, zoom: float, pan: list[float]) -> None:
        if is_left:
            self.left_zoom = zoom
            self.left_pan = pan
        else:
            self.right_zoom = zoom
            self.right_pan = pan

    def on_image_wheel(self, event: tk.Event, is_left: bool) -> str:
        picked = self.left if is_left else self.right
        if picked is None:
            return "break"

        canvas = self.left_canvas if is_left else self.right_canvas
        cw = max(1, canvas.winfo_width())
        ch = max(1, canvas.winfo_height())
        ih, iw = picked.bgr.shape[:2]
        base_scale = min(cw / iw, ch / ih)
        old_zoom, _old_pan = self.get_view_params(is_left)
        old_display = self.left_display if is_left else self.right_display

        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.18
        else:
            factor = 1.0 / 1.18
        new_zoom = float(np.clip(old_zoom * factor, 1.0, 12.0))
        if abs(new_zoom - old_zoom) < 1e-6:
            return "break"

        ix = (event.x - old_display.offset_x) / max(old_display.scale, 1e-6)
        iy = (event.y - old_display.offset_y) / max(old_display.scale, 1e-6)
        new_scale = base_scale * new_zoom
        centered_x = (cw - iw * new_scale) / 2.0
        centered_y = (ch - ih * new_scale) / 2.0
        new_offset_x = event.x - ix * new_scale
        new_offset_y = event.y - iy * new_scale
        new_pan = [new_offset_x - centered_x, new_offset_y - centered_y]

        self.set_view_params(is_left, new_zoom, new_pan)
        self.redraw_selected_frames()
        return "break"

    def start_pan(self, event: tk.Event, is_left: bool) -> str:
        _zoom, pan = self.get_view_params(is_left)
        self.drag_start = (is_left, event.x, event.y, pan[0], pan[1])
        return "break"

    def drag_pan(self, event: tk.Event, is_left: bool) -> str:
        if self.drag_start is None or self.drag_start[0] != is_left:
            return "break"
        _side, start_x, start_y, pan_x, pan_y = self.drag_start
        zoom, _pan = self.get_view_params(is_left)
        self.set_view_params(is_left, zoom, [pan_x + event.x - start_x, pan_y + event.y - start_y])
        self.redraw_selected_frames()
        return "break"

    def draw_current_preview(self) -> None:
        self.preview_canvas.delete("all")
        if self.current_bgr is None:
            self.preview_canvas.create_text(
                max(1, self.preview_canvas.winfo_width() // 2),
                max(1, self.preview_canvas.winfo_height() // 2),
                text="No video",
                fill=MUTED,
                font=("Segoe UI", 11),
            )
            return

        rgb = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2RGB)
        ch = max(1, self.preview_canvas.winfo_height())
        cw = max(1, self.preview_canvas.winfo_width())
        ih, iw = rgb.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        self.preview_photo = self.rgb_to_photoimage(resized)
        self.preview_canvas.create_image(ox, oy, anchor=tk.NW, image=self.preview_photo)
        self.preview_canvas.create_text(
            ox + 8,
            oy + 8,
            anchor=tk.NW,
            text=f"frame {self.current_index}",
            fill=TEXT,
            font=("Segoe UI", 10, "bold"),
        )

    def draw_picked(self, canvas: tk.Canvas, picked: PickedFrame | None, is_left: bool) -> None:
        canvas.delete("all")
        if picked is None:
            canvas.create_text(
                max(1, canvas.winfo_width() // 2),
                max(1, canvas.winfo_height() // 2),
                text="No frame selected",
                fill=MUTED,
                font=("Segoe UI", 16),
            )
            return

        bgr = picked.bgr.copy()
        if self.show_mask.get():
            bad_mask = self.combined_bad_mask(picked)
            if bad_mask is not None:
                overlay = np.zeros_like(bgr)
                overlay[:, :] = MASK_RED
                mask_bool = bad_mask > 0
                bgr[mask_bool] = cv2.addWeighted(bgr, 0.45, overlay, 0.55, 0)[mask_bool]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ch = max(1, canvas.winfo_height())
        cw = max(1, canvas.winfo_width())
        ih, iw = rgb.shape[:2]
        base_scale = min(cw / iw, ch / ih)
        zoom, pan = self.get_view_params(is_left)
        scale = base_scale * zoom
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        ox = int(round((cw - nw) / 2.0 + pan[0]))
        oy = int(round((ch - nh) / 2.0 + pan[1]))

        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        photo = self.rgb_to_photoimage(resized)
        canvas.create_image(ox, oy, anchor=tk.NW, image=photo)

        display = DisplayState(scale=scale, offset_x=ox, offset_y=oy, width=nw, height=nh)
        if is_left:
            self.left_photo = photo
            self.left_display = display
            self.draw_annotations(canvas, True)
        else:
            self.right_photo = photo
            self.right_display = display
            self.draw_annotations(canvas, False)

        canvas.create_text(
            ox + 10,
            oy + 10,
            anchor=tk.NW,
            text=f"frame {picked.index}",
            fill=TEXT,
            font=("Segoe UI", 12, "bold"),
        )

    def draw_annotations(self, canvas: tk.Canvas, is_left: bool) -> None:
        picked = self.left if is_left else self.right
        display = self.left_display if is_left else self.right_display
        clicked = self.left_clicked_xy if is_left else self.right_clicked_xy
        points = self.left_reference_points if is_left else self.right_reference_points
        point_color = LEFT_POINT if is_left else RIGHT_POINT
        if picked is None:
            return

        if clicked is not None:
            x, y = clicked
            cx, cy = self.image_to_canvas(x, y, display)
            rr = max(8, int(self.search_radius.get() * display.scale))
            canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline="#6ab0ff", width=2)
            canvas.create_line(cx - 9, cy, cx + 9, cy, fill=CLICK_POINT, width=2)
            canvas.create_line(cx, cy - 9, cx, cy + 9, fill=CLICK_POINT, width=2)

        for x, y, score in points:
            cx, cy = self.image_to_canvas(x, y, display)
            r = max(2, min(5, int(2 + math.log1p(score) * 0.4)))
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=point_color, outline="")

        for left_pt, right_pt, _score in self.matched_pairs:
            x, y = left_pt if is_left else right_pt
            cx, cy = self.image_to_canvas(x, y, display)
            canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, outline=MATCH_POINT, width=2)

        if not is_left and self.predicted_right_xy is not None:
            px, py = self.predicted_right_xy
            cx, cy = self.image_to_canvas(int(round(px)), int(round(py)), display)
            canvas.create_oval(cx - 11, cy - 11, cx + 11, cy + 11, outline="#ff2d55", width=3)
            canvas.create_line(cx - 14, cy, cx + 14, cy, fill="#ff2d55", width=3)
            canvas.create_line(cx, cy - 14, cx, cy + 14, fill="#ff2d55", width=3)
            canvas.create_text(
                cx + 14,
                cy - 18,
                anchor=tk.NW,
                text="pred",
                fill="#ff2d55",
                font=("Segoe UI", 10, "bold"),
            )

        if points:
            canvas.create_text(
                display.offset_x + 10,
                display.offset_y + 34,
                anchor=tk.NW,
                text=f"reference points: {len(points)}",
                fill=point_color,
                font=("Segoe UI", 11, "bold"),
            )

        if self.matched_pairs:
            canvas.create_text(
                display.offset_x + 10,
                display.offset_y + 56,
                anchor=tk.NW,
                text=f"matched pairs: {len(self.matched_pairs)}",
                fill=MATCH_POINT,
                font=("Segoe UI", 11, "bold"),
            )

    @staticmethod
    def rgb_to_photoimage(rgb: np.ndarray) -> tk.PhotoImage:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to encode frame for Tk display.")
        return tk.PhotoImage(data=base64.b64encode(encoded.tobytes()), format="PNG")

    def on_close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = SpecularReferencePointUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
