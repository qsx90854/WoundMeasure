#!/usr/bin/env python3
"""用圖形介面從影片選取指定影格並存成 JPG。

需求：
    pip install opencv-python

使用方式：
    python video_frame_jpg_exporter_ui.py
    python video_frame_jpg_exporter_ui.py "D:\\videos\\sample.mp4"
"""

from __future__ import annotations

import argparse
import base64
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2


class VideoFrameExporterApp:
    PREVIEW_MAX_WIDTH = 1280
    PREVIEW_MAX_HEIGHT = 720
    SEEK_DEBOUNCE_MS = 80

    def __init__(self, root: tk.Tk, initial_video: str | None = None) -> None:
        self.root = root
        self.root.title("影片影格 JPG 匯出工具")
        self.root.geometry("1180x820")
        self.root.minsize(760, 580)

        self.capture: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.current_frame = None
        self.current_index = -1
        self.total_frames = 0
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.preview_photo: tk.PhotoImage | None = None
        self.pending_seek: str | None = None
        self.updating_slider = False

        self.frame_var = tk.IntVar(value=0)
        self.frame_entry_var = tk.StringVar(value="0")
        self.quality_var = tk.IntVar(value=95)
        self.info_var = tk.StringVar(value="請先開啟影片")
        self.position_var = tk.StringVar(value="Frame: —    Time: —")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if initial_video:
            self.root.after(50, lambda: self.open_video(initial_video))

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="開啟影片", command=self.choose_video).pack(side=tk.LEFT)
        self.save_button = ttk.Button(
            toolbar, text="將目前影格存成 JPG", command=self.save_current_frame,
            state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(8, 16))

        ttk.Label(toolbar, text="JPG 品質：").pack(side=tk.LEFT)
        quality = ttk.Spinbox(
            toolbar, from_=1, to=100, textvariable=self.quality_var,
            width=5, justify=tk.CENTER)
        quality.pack(side=tk.LEFT)
        ttk.Label(toolbar, text="（1–100）").pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(
            self.root, textvariable=self.info_var, anchor=tk.W,
            padding=(10, 0, 10, 6)).pack(fill=tk.X)

        preview_frame = ttk.Frame(self.root, padding=(10, 0))
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_label = tk.Label(
            preview_frame, text="尚未載入影片", bg="#15191f", fg="#d9e2ef",
            font=("Microsoft JhengHei UI", 14), anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_label.bind("<Configure>", self._on_preview_resize)

        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        self.slider = ttk.Scale(
            controls, from_=0, to=0, orient=tk.HORIZONTAL,
            variable=self.frame_var, command=self._on_slider_changed,
            state=tk.DISABLED)
        self.slider.pack(fill=tk.X, pady=(0, 8))
        self.slider.bind("<ButtonRelease-1>", self._on_slider_released)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X)
        self.back10_button = ttk.Button(
            row, text="◀◀ -10", command=lambda: self.step_frame(-10), state=tk.DISABLED)
        self.back10_button.pack(side=tk.LEFT)
        self.back_button = ttk.Button(
            row, text="◀ 上一幀", command=lambda: self.step_frame(-1), state=tk.DISABLED)
        self.back_button.pack(side=tk.LEFT, padx=(6, 0))
        self.next_button = ttk.Button(
            row, text="下一幀 ▶", command=lambda: self.step_frame(1), state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=(6, 0))
        self.next10_button = ttk.Button(
            row, text="+10 ▶▶", command=lambda: self.step_frame(10), state=tk.DISABLED)
        self.next10_button.pack(side=tk.LEFT, padx=(6, 16))

        ttk.Label(row, text="跳至幀號：").pack(side=tk.LEFT)
        self.frame_entry = ttk.Entry(
            row, textvariable=self.frame_entry_var, width=11, justify=tk.RIGHT,
            state=tk.DISABLED)
        self.frame_entry.pack(side=tk.LEFT)
        self.frame_entry.bind("<Return>", lambda _event: self.go_to_entered_frame())
        self.go_button = ttk.Button(
            row, text="前往", command=self.go_to_entered_frame, state=tk.DISABLED)
        self.go_button.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(row, textvariable=self.position_var).pack(side=tk.RIGHT)

        status = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN,
            anchor=tk.W, padding=(8, 4))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind("<Control-s>", lambda _event: self.save_current_frame())
        self.root.bind("<Control-o>", lambda _event: self.choose_video())

    def choose_video(self) -> None:
        filename = filedialog.askopenfilename(
            title="選擇影片",
            filetypes=[
                ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm"),
                ("所有檔案", "*.*"),
            ],
        )
        if filename:
            self.open_video(filename)

    def open_video(self, filename: str) -> None:
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            messagebox.showerror("開啟失敗", f"找不到影片：\n{path}")
            return

        new_capture = cv2.VideoCapture(str(path))
        if not new_capture.isOpened():
            new_capture.release()
            messagebox.showerror(
                "開啟失敗",
                "OpenCV 無法開啟此影片。可能是 codec 不支援或檔案已損壞。\n\n"
                f"{path}",
            )
            return

        total = int(round(new_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(new_capture.get(cv2.CAP_PROP_FPS))
        width = int(round(new_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(new_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if total <= 0:
            new_capture.release()
            messagebox.showerror("開啟失敗", "無法取得影片總影格數。")
            return

        if self.capture is not None:
            self.capture.release()
        self.capture = new_capture
        self.video_path = path
        self.total_frames = total
        self.fps = fps if math.isfinite(fps) and fps > 0 else 0.0
        self.frame_width = width
        self.frame_height = height
        self.current_frame = None
        self.current_index = -1

        duration = self._format_time((total - 1) / self.fps) if self.fps else "未知"
        fps_text = f"{self.fps:.3f}" if self.fps else "未知"
        self.info_var.set(
            f"{path.name}    {width}×{height}    FPS: {fps_text}    "
            f"總幀數: {total:,}    時長: {duration}"
        )
        self.slider.configure(to=max(0, total - 1), state=tk.NORMAL)
        self.frame_entry.configure(state=tk.NORMAL)
        self.go_button.configure(state=tk.NORMAL)
        for button in (
            self.save_button, self.back10_button, self.back_button,
            self.next_button, self.next10_button,
        ):
            button.configure(state=tk.NORMAL)
        self.show_frame(0)

    def _on_slider_changed(self, value: str) -> None:
        if self.updating_slider or self.capture is None:
            return
        target = self._clamp_index(int(round(float(value))))
        self.frame_entry_var.set(str(target))
        self._set_position_text(target)
        if self.pending_seek is not None:
            self.root.after_cancel(self.pending_seek)
        self.pending_seek = self.root.after(
            self.SEEK_DEBOUNCE_MS, lambda: self.show_frame(target))

    def _on_slider_released(self, _event: tk.Event) -> None:
        if self.capture is None:
            return
        if self.pending_seek is not None:
            self.root.after_cancel(self.pending_seek)
            self.pending_seek = None
        self.show_frame(self._clamp_index(int(round(self.frame_var.get()))))

    def go_to_entered_frame(self) -> None:
        if self.capture is None:
            return
        try:
            target = int(self.frame_entry_var.get().strip())
        except ValueError:
            messagebox.showwarning("幀號錯誤", "請輸入整數幀號。")
            self.frame_entry_var.set(str(max(0, self.current_index)))
            return
        self.show_frame(self._clamp_index(target))

    def step_frame(self, delta: int) -> None:
        if self.capture is None:
            return
        base = self.current_index if self.current_index >= 0 else 0
        self.show_frame(self._clamp_index(base + delta))

    def show_frame(self, index: int) -> None:
        if self.capture is None:
            return
        index = self._clamp_index(index)
        self.pending_seek = None
        self.status_var.set(f"正在讀取 Frame {index:,}...")
        self.root.update_idletasks()

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.status_var.set(f"讀取 Frame {index:,} 失敗")
            messagebox.showerror("讀取失敗", f"無法讀取 Frame {index:,}。")
            return

        # CAP_PROP_POS_FRAMES points to the next frame after read().  Some codecs
        # seek to a nearby keyframe internally, but read() returns the requested
        # frame after OpenCV decodes forward.
        self.current_frame = frame
        self.current_index = index
        self.frame_entry_var.set(str(index))
        self._set_slider(index)
        self._set_position_text(index)
        self._render_preview()
        self.status_var.set(f"已載入 Frame {index:,}（輸出將使用原始解析度）")

    def _render_preview(self) -> None:
        if self.current_frame is None:
            return
        available_w = max(320, self.preview_label.winfo_width() - 12)
        available_h = max(240, self.preview_label.winfo_height() - 12)
        max_w = min(available_w, self.PREVIEW_MAX_WIDTH)
        max_h = min(available_h, self.PREVIEW_MAX_HEIGHT)
        height, width = self.current_frame.shape[:2]
        scale = min(max_w / width, max_h / height, 1.0)
        if scale < 1.0:
            preview = cv2.resize(
                self.current_frame,
                (max(1, int(round(width * scale))),
                 max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            preview = self.current_frame

        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            self.status_var.set("預覽影像編碼失敗")
            return
        png_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        self.preview_photo = tk.PhotoImage(data=png_base64, format="png")
        self.preview_label.configure(image=self.preview_photo, text="")

    def _on_preview_resize(self, _event: tk.Event) -> None:
        if self.current_frame is None:
            return
        # Avoid repeatedly encoding while the window is actively resizing.
        resize_job = getattr(self, "_resize_job", None)
        if resize_job is not None:
            self.root.after_cancel(resize_job)
        self._resize_job = self.root.after(120, self._render_preview)

    def save_current_frame(self) -> None:
        if self.current_frame is None or self.video_path is None:
            messagebox.showinfo("尚無影格", "請先開啟影片並選擇影格。")
            return
        try:
            quality = max(1, min(100, int(self.quality_var.get())))
        except (ValueError, tk.TclError):
            quality = 95
            self.quality_var.set(quality)

        default_name = f"{self.video_path.stem}_frame_{self.current_index:06d}.jpg"
        filename = filedialog.asksaveasfilename(
            title="儲存目前影格",
            initialdir=str(self.video_path.parent),
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG 圖片", "*.jpg *.jpeg"), ("所有檔案", "*.*")],
        )
        if not filename:
            return

        target = Path(filename)
        if target.suffix.lower() not in {".jpg", ".jpeg"}:
            target = target.with_suffix(".jpg")
        ok, encoded = cv2.imencode(
            ".jpg", self.current_frame,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if not ok:
            messagebox.showerror("儲存失敗", "OpenCV 無法將影格編碼為 JPG。")
            return
        try:
            # tofile supports Unicode Windows paths more reliably than imwrite.
            encoded.tofile(str(target))
        except OSError as exc:
            messagebox.showerror("儲存失敗", f"無法寫入檔案：\n{exc}")
            return

        self.status_var.set(
            f"已儲存 Frame {self.current_index:,}：{target} "
            f"({self.current_frame.shape[1]}×{self.current_frame.shape[0]}, quality={quality})"
        )
        messagebox.showinfo(
            "儲存完成",
            f"Frame {self.current_index:,} 已存成 JPG：\n\n{target}\n\n"
            f"尺寸：{self.current_frame.shape[1]}×{self.current_frame.shape[0]}\n"
            f"JPG 品質：{quality}",
        )

    def _set_slider(self, index: int) -> None:
        self.updating_slider = True
        try:
            self.frame_var.set(index)
        finally:
            self.updating_slider = False

    def _set_position_text(self, index: int) -> None:
        time_text = self._format_time(index / self.fps) if self.fps else "未知"
        self.position_var.set(
            f"Frame: {index:,} / {max(0, self.total_frames - 1):,}    Time: {time_text}"
        )

    def _clamp_index(self, index: int) -> int:
        return max(0, min(max(0, self.total_frames - 1), index))

    @staticmethod
    def _format_time(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "未知"
        milliseconds = int(round(seconds * 1000))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def close(self) -> None:
        if self.pending_seek is not None:
            self.root.after_cancel(self.pending_seek)
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="從影片選取指定影格並存成 JPG")
    parser.add_argument("video", nargs="?", help="啟動時直接開啟的影片路徑")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass
    VideoFrameExporterApp(root, args.video)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
