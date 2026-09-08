"""A small Tkinter application for resizing a single image.

The application uses OpenCV for image I/O so it can run in this workspace
without requiring Pillow.  Unicode file paths are supported on Windows.
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
PREVIEW_WIDTH = 760
PREVIEW_HEIGHT = 428
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image while supporting paths containing non-ASCII characters."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"無法讀取檔案：{exc}") from exc

    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("不支援此圖片格式，或檔案已損壞。")
    return image


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to the exact requested dimensions using a suitable filter."""
    if width <= 0 or height <= 0:
        raise ValueError("寬度與高度必須是大於 0 的整數。")

    source_height, source_width = image.shape[:2]
    shrinking = width < source_width or height < source_height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _prepare_for_format(image: np.ndarray, suffix: str) -> np.ndarray:
    """Convert unsupported alpha-channel images before JPEG output."""
    if suffix not in {".jpg", ".jpeg"} or image.ndim != 3 or image.shape[2] != 4:
        return image

    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    color = image[:, :, :3].astype(np.float32)
    white = np.full_like(color, 255.0)
    return np.clip(color * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image while supporting paths containing non-ASCII characters."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("請使用 JPG、PNG、BMP、TIFF 或 WebP 副檔名。")

    output = _prepare_for_format(image, suffix)
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]

    ok, encoded = cv2.imencode(suffix, output, params)
    if not ok:
        raise ValueError("圖片編碼失敗。")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise ValueError(f"無法儲存檔案：{exc}") from exc


def make_preview_ppm(image: np.ndarray, max_width: int, max_height: int) -> bytes:
    """Return a base64 PPM preview that Tk can display without Pillow."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    preview_width = max(1, round(width * scale))
    preview_height = max(1, round(height * scale))
    preview = cv2.resize(
        image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST,
    )

    if preview.ndim == 2:
        rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    elif preview.shape[2] == 4:
        # Composite transparent pixels over a light checkerboard.
        yy, xx = np.indices((preview_height, preview_width))
        checker = np.where(((xx // 12 + yy // 12) % 2)[..., None] == 0, 238, 210)
        checker = np.repeat(checker, 3, axis=2).astype(np.float32)
        alpha = preview[:, :, 3:4].astype(np.float32) / 255.0
        bgr = preview[:, :, :3].astype(np.float32) * alpha + checker * (1.0 - alpha)
        rgb = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    header = f"P6\n{preview_width} {preview_height}\n255\n".encode("ascii")
    return base64.b64encode(header + np.ascontiguousarray(rgb).tobytes())


class ImageResizeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.image: np.ndarray | None = None
        self.source_path: Path | None = None
        self.preview_photo: tk.PhotoImage | None = None

        self.width_var = tk.StringVar(value=str(DEFAULT_WIDTH))
        self.height_var = tk.StringVar(value=str(DEFAULT_HEIGHT))
        self.path_var = tk.StringVar(value="尚未選取圖片")
        self.info_var = tk.StringVar(value="請先載入一張圖片")
        self.target_var = tk.StringVar(value=f"輸出尺寸：{DEFAULT_WIDTH} × {DEFAULT_HEIGHT} px")
        self.status_var = tk.StringVar(value="就緒")

        self._configure_window()
        self._build_ui()
        self._bind_shortcuts()

    def _configure_window(self) -> None:
        self.root.title("圖片尺寸調整工具")
        self.root.geometry("900x690")
        self.root.minsize(760, 620)
        try:
            self.root.tk.call("tk", "scaling", 1.2)
        except tk.TclError:
            pass

        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 18, "bold"))
        style.configure("Hint.TLabel", foreground="#5f6368")
        style.configure("Action.TButton", font=("Microsoft JhengHei UI", 11, "bold"))

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="圖片尺寸調整工具", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="載入圖片後，輸出為指定的精確尺寸",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(header, text="載入圖片…", command=self.open_image).grid(
            row=0, column=1, rowspan=2, padx=(16, 0), ipadx=8, ipady=5
        )

        file_frame = ttk.LabelFrame(container, text="來源圖片", padding=(12, 9))
        file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        file_frame.columnconfigure(0, weight=1)
        ttk.Label(file_frame, textvariable=self.path_var, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(file_frame, textvariable=self.info_var, style="Hint.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )

        preview_frame = ttk.LabelFrame(container, text="圖片預覽", padding=8)
        preview_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = tk.Label(
            preview_frame,
            text="尚未載入圖片",
            bg="#202124",
            fg="#d0d0d0",
            font=("Microsoft JhengHei UI", 12),
            compound="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(container, text="輸出設定", padding=12)
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="寬度").grid(row=0, column=0, sticky="w")
        width_entry = ttk.Entry(controls, textvariable=self.width_var, width=10)
        width_entry.grid(row=0, column=1, padx=(7, 16), sticky="w")
        ttk.Label(controls, text="高度").grid(row=0, column=2, sticky="w")
        height_entry = ttk.Entry(controls, textvariable=self.height_var, width=10)
        height_entry.grid(row=0, column=3, padx=(7, 12), sticky="w")
        ttk.Label(controls, text="px", style="Hint.TLabel").grid(row=0, column=4, sticky="w")

        self.save_button = ttk.Button(
            controls,
            text="Resize 並另存…",
            command=self.resize_and_save,
            state="disabled",
            style="Action.TButton",
        )
        self.save_button.grid(row=0, column=5, rowspan=2, padx=(16, 0), ipadx=10, ipady=5)

        ttk.Label(controls, textvariable=self.target_var, style="Hint.TLabel").grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(9, 0)
        )
        ttk.Label(
            controls,
            text="注意：原圖比例不同時，圖片會被拉伸以精確符合指定尺寸。",
            style="Hint.TLabel",
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(7, 0))

        status = ttk.Label(container, textvariable=self.status_var, anchor="w")
        status.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        width_entry.bind("<KeyRelease>", self._update_target_text)
        height_entry.bind("<KeyRelease>", self._update_target_text)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self.open_image())
        self.root.bind("<Control-s>", lambda _event: self.resize_and_save())

    def _update_target_text(self, _event: tk.Event | None = None) -> None:
        width = self.width_var.get().strip() or "?"
        height = self.height_var.get().strip() or "?"
        self.target_var.set(f"輸出尺寸：{width} × {height} px")

    def _get_dimensions(self) -> tuple[int, int]:
        try:
            width = int(self.width_var.get().strip())
            height = int(self.height_var.get().strip())
        except ValueError as exc:
            raise ValueError("寬度與高度必須是整數。") from exc
        if width <= 0 or height <= 0:
            raise ValueError("寬度與高度必須大於 0。")
        return width, height

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="選取圖片",
            filetypes=[
                ("圖片檔案", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("所有檔案", "*.*"),
            ],
        )
        if not filename:
            return

        path = Path(filename)
        try:
            image = read_image(path)
            ppm = make_preview_ppm(image, PREVIEW_WIDTH, PREVIEW_HEIGHT)
            photo = tk.PhotoImage(data=ppm, format="PPM")
        except (ValueError, OSError, cv2.error, tk.TclError) as exc:
            messagebox.showerror("載入失敗", str(exc), parent=self.root)
            self.status_var.set("載入失敗")
            return

        self.image = image
        self.source_path = path
        self.preview_photo = photo
        self.preview_label.configure(image=photo, text="")
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        self.path_var.set(str(path))
        self.info_var.set(f"原始尺寸：{width} × {height} px　｜　色彩通道：{channels}")
        self.save_button.configure(state="normal")
        self.status_var.set("圖片載入完成")

    def resize_and_save(self) -> None:
        if self.image is None or self.source_path is None:
            return

        try:
            width, height = self._get_dimensions()
        except ValueError as exc:
            messagebox.showwarning("尺寸錯誤", str(exc), parent=self.root)
            return

        default_suffix = self.source_path.suffix.lower()
        if default_suffix not in SUPPORTED_SUFFIXES:
            default_suffix = ".png"
        default_name = f"{self.source_path.stem}_{width}x{height}{default_suffix}"
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="儲存縮放後圖片",
            initialdir=str(self.source_path.parent),
            initialfile=default_name,
            defaultextension=default_suffix,
            filetypes=[
                ("PNG 圖片", "*.png"),
                ("JPEG 圖片", "*.jpg *.jpeg"),
                ("WebP 圖片", "*.webp"),
                ("BMP 圖片", "*.bmp"),
                ("TIFF 圖片", "*.tif *.tiff"),
            ],
        )
        if not filename:
            return

        output_path = Path(filename)
        self.status_var.set("正在調整尺寸並儲存…")
        self.root.update_idletasks()
        try:
            resized = resize_image(self.image, width, height)
            write_image(output_path, resized)
        except (ValueError, OSError, MemoryError, cv2.error) as exc:
            messagebox.showerror("儲存失敗", str(exc), parent=self.root)
            self.status_var.set("儲存失敗")
            return

        self.status_var.set(f"已儲存：{output_path}")
        messagebox.showinfo(
            "完成",
            f"圖片已調整為 {width} × {height} px。\n\n{output_path}",
            parent=self.root,
        )


def self_test() -> None:
    """Exercise image resizing and Unicode-path I/O without opening the GUI."""
    sample = np.zeros((24, 32, 3), dtype=np.uint8)
    sample[:, :, 1] = 180
    resized = resize_image(sample, 80, 45)
    assert resized.shape == (45, 80, 3)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "測試圖片.png"
        write_image(output, resized)
        loaded = read_image(output)
        assert loaded.shape == (45, 80, 3)
    print("Self-test passed: resize and Unicode-path image I/O are working.")


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return
    root = tk.Tk()
    ImageResizeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
