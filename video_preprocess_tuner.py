#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Pre-processing Tuner Tool
===============================
This tool provides an interactive GUI to load a video, seek frames, 
and configure image pre-processing stages. Users can adjust parameters and 
reorder the pre-processing pipeline dynamically to preview the combined effect.
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Check dependencies and guide user if missing
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
except ImportError:
    print("Missing dependency. Please install Pillow to enable GUI support:")
    print("pip install pillow opencv-python numpy")
    sys.exit(1)

class PreprocessingTunerApp:
    def __init__(self, root, initial_video=None):
        self.root = root
        self.root.title("互動式影像前處理調試工具 (Video Pre-processing Tuner)")
        self.root.geometry("1280, 800")
        self.root.minsize(1024, 720)

        # Video State Variables
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.current_frame_idx = 0
        self.raw_frame = None
        self.processed_frame = None

        # Preprocessing Pipeline Configuration
        # Each dict describes a step in the pipeline
        self.pipeline_steps = [
            {"id": "clahe", "name": "1. 局部直方圖等化 (CLAHE)", "enabled": False, 
             "params": {"clip_limit": 2.0, "grid_size": 8}},
            {"id": "denoise", "name": "2. 濾波降噪 (Denoise)", "enabled": False, 
             "params": {"method": "Gaussian", "ksize": 5, "sigma": 1.5}},
            {"id": "sharpen", "name": "3. 影像銳化 (Sharpen)", "enabled": False, 
             "params": {"strength": 1.0}},
            {"id": "bright_contrast", "name": "4. 亮度與對比度 (BC)", "enabled": False, 
             "params": {"brightness": 0, "contrast": 1.0}},
            {"id": "gamma", "name": "5. 伽馬校正 (Gamma)", "enabled": False, 
             "params": {"gamma": 1.0}},
        ]

        self.setup_ui()
        
        # Load initial video if provided
        if initial_video and os.path.exists(initial_video):
            self.load_video(initial_video)

    def setup_ui(self):
        # 建立主排版
        # 頂部：影片載入區域
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="影片路徑:").pack(side=tk.LEFT, padx=5)
        self.video_path_var = tk.StringVar()
        self.video_entry = ttk.Entry(top_frame, textvariable=self.video_path_var, width=80)
        self.video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(top_frame, text="瀏覽...", command=self.browse_video).pack(side=tk.LEFT, padx=5)

        # 中間主體：左側 Pipeline 控制面板，右側影像預覽區
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左側控制區
        left_frame = ttk.Frame(main_paned, padding="5")
        main_paned.add(left_frame, weight=1)

        # 右側影像區
        right_frame = ttk.Frame(main_paned, padding="5")
        main_paned.add(right_frame, weight=3)

        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)

        # 底部：進度滑桿控制區
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.setup_bottom_panel(bottom_frame)

    def setup_left_panel(self, parent):
        # 標題
        ttk.Label(parent, text="前處理 Pipeline (可調整執行順序)", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=5)

        # 順序排版清單與控制按鈕
        list_btn_frame = ttk.Frame(parent)
        list_btn_frame.pack(fill=tk.X, pady=5)

        # 建立 Listbox 用來表示與調整順序
        self.pipeline_listbox = tk.Listbox(list_btn_frame, height=6, font=("Helvetica", 10))
        self.pipeline_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pipeline_listbox.bind("<<ListboxSelect>>", self.on_step_select)

        # 清單滾動條
        scrollbar = ttk.Scrollbar(list_btn_frame, orient=tk.VERTICAL, command=self.pipeline_listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.pipeline_listbox.config(yscrollcommand=scrollbar.set)

        # 上移 / 下移 / 啟用 按鈕
        btn_frame = ttk.Frame(list_btn_frame)
        btn_frame.pack(side=tk.RIGHT, padx=5, fill=tk.Y)

        self.btn_up = ttk.Button(btn_frame, text="▲ 上移", command=self.move_step_up, width=8)
        self.btn_up.pack(fill=tk.X, pady=2)
        
        self.btn_down = ttk.Button(btn_frame, text="▼ 下移", command=self.move_step_down, width=8)
        self.btn_down.pack(fill=tk.X, pady=2)

        self.enable_var = tk.BooleanVar()
        self.chk_enable = ttk.Checkbutton(btn_frame, text="啟用", variable=self.enable_var, command=self.toggle_step)
        self.chk_enable.pack(anchor=tk.W, pady=5)

        # 重新整理 Listbox 顯示
        self.update_listbox()

        # 分割線
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 參數設置區域 (動態顯示選中步驟的參數)
        self.param_label = ttk.Label(parent, text="參數設定:", font=("Helvetica", 11, "bold"))
        self.param_label.pack(anchor=tk.W, pady=5)

        self.param_container = ttk.LabelFrame(parent, text="選擇一個步驟以配置參數", padding="10")
        self.param_container.pack(fill=tk.BOTH, expand=True, pady=5)

    def setup_right_panel(self, parent):
        # 影像預覽區標題與重置按鈕
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(title_frame, text="影像即時預覽", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)
        ttk.Button(title_frame, text="重置所有前處理", command=self.reset_all_processing).pack(side=tk.RIGHT)

        # 影像顯示 Canvas
        self.canvas = tk.Canvas(parent, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def setup_bottom_panel(self, parent):
        # 播放進度控制條
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, pady=2)

        self.time_label = ttk.Label(slider_frame, text="0 / 0 影格")
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # 影片進度滑桿
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.frame_slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)

        # 微調按鈕
        control_btns = ttk.Frame(parent)
        control_btns.pack(fill=tk.X, pady=5)

        ttk.Button(control_btns, text="⏪ -10 幀", command=lambda: self.seek_frame(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_btns, text="◀ -1 幀", command=lambda: self.seek_frame(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_btns, text="1 幀 ▶", command=lambda: self.seek_frame(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_btns, text="10 幀 ⏩", command=lambda: self.seek_frame(10)).pack(side=tk.LEFT, padx=2)

        # 當前狀態資訊顯示
        self.info_label = ttk.Label(parent, text="尚未載入影片。", font=("Consolas", 9))
        self.info_label.pack(side=tk.LEFT, pady=5)

    # ================== 影片載入與 seek 處理 ==================
    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.load_video(file_path)

    def load_video(self, file_path):
        if self.cap:
            self.cap.release()
            
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", f"無法開啟影片檔案: {file_path}")
            self.info_label.config(text="讀取影片失敗。")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 更新滑桿範圍與底部狀態
        self.frame_slider.config(to=self.total_frames - 1)
        self.frame_slider.set(0)
        self.current_frame_idx = 0
        
        self.info_label.config(text=f"格式: {self.width}x{self.height} | FPS: {self.fps:.2f} | 總影格數: {self.total_frames}")
        self.read_and_process_frame()

    def seek_frame(self, offset):
        if not self.cap:
            return
        target = self.current_frame_idx + offset
        target = max(0, min(self.total_frames - 1, target))
        self.frame_slider.set(target)

    def on_slider_move(self, val):
        if not self.cap:
            return
        idx = int(float(val))
        if idx != self.current_frame_idx:
            self.current_frame_idx = idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.read_and_process_frame()

    def read_and_process_frame(self):
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            # 讀取失敗可能是解碼器問題，或者指針需要重置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                return

        self.raw_frame = frame
        self.time_label.config(text=f"{self.current_frame_idx + 1} / {self.total_frames} 影格")
        
        # 回退一格以免下次 read() 自動遞增造成 seek 與 slider 錯位
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        self.update_pipeline_and_render()

    # ================== Pipeline 控制邏輯 ==================
    def update_listbox(self):
        self.pipeline_listbox.delete(0, tk.END)
        for i, step in enumerate(self.pipeline_steps):
            status = "[v]" if step["enabled"] else "[ ]"
            self.pipeline_listbox.insert(tk.END, f"{status} {step['name']}")
        
        # 保持之前的選擇項目
        self.btn_up.config(state=tk.NORMAL)
        self.btn_down.config(state=tk.NORMAL)

    def on_step_select(self, event):
        selection = self.pipeline_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        step = self.pipeline_steps[idx]
        self.enable_var.set(step["enabled"])
        
        # 顯示該步驟對應的參數面板
        self.param_label.config(text=f"參數設定: {step['name']}")
        self.show_parameters_panel(step)

    def toggle_step(self):
        selection = self.pipeline_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.pipeline_steps[idx]["enabled"] = self.enable_var.get()
        self.update_listbox()
        self.pipeline_listbox.select_set(idx)
        self.update_pipeline_and_render()

    def move_step_up(self):
        selection = self.pipeline_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        idx = selection[0]
        # 交換
        self.pipeline_steps[idx], self.pipeline_steps[idx-1] = self.pipeline_steps[idx-1], self.pipeline_steps[idx]
        self.update_listbox()
        self.pipeline_listbox.select_set(idx-1)
        self.on_step_select(None)
        self.update_pipeline_and_render()

    def move_step_down(self):
        selection = self.pipeline_listbox.curselection()
        if not selection or selection[0] == len(self.pipeline_steps) - 1:
            return
        idx = selection[0]
        # 交換
        self.pipeline_steps[idx], self.pipeline_steps[idx+1] = self.pipeline_steps[idx+1], self.pipeline_steps[idx]
        self.update_listbox()
        self.pipeline_listbox.select_set(idx+1)
        self.on_step_select(None)
        self.update_pipeline_and_render()

    def reset_all_processing(self):
        for step in self.pipeline_steps:
            step["enabled"] = False
        self.update_listbox()
        self.update_pipeline_and_render()
        # 重新選擇當前項目以更新 checkbox 狀態
        selection = self.pipeline_listbox.curselection()
        if selection:
            self.enable_var.set(False)

    # ================== 參數面板生成與回調 ==================
    def show_parameters_panel(self, step):
        # 清除舊的參數元件
        for widget in self.param_container.winfo_children():
            widget.destroy()

        step_id = step["id"]
        params = step["params"]

        if step_id == "clahe":
            # Clip Limit 滑桿
            ttk.Label(self.param_container, text="Clip Limit (對比度限制):").pack(anchor=tk.W)
            clip_val_lbl = ttk.Label(self.param_container, text=f"{params['clip_limit']:.1f}")
            
            def update_clip(val):
                v = round(float(val), 1)
                params["clip_limit"] = v
                clip_val_lbl.config(text=f"{v:.1f}")
                self.update_pipeline_and_render()

            s_clip = ttk.Scale(self.param_container, from_=0.1, to=10.0, value=params["clip_limit"], command=update_clip)
            s_clip.pack(fill=tk.X, pady=2)
            clip_val_lbl.pack(anchor=tk.E)

            # Grid Size 滑桿
            ttk.Label(self.param_container, text="Tile Grid Size (網格大小):").pack(anchor=tk.W)
            grid_val_lbl = ttk.Label(self.param_container, text=f"{params['grid_size']}x{params['grid_size']}")
            
            def update_grid(val):
                v = int(float(val))
                params["grid_size"] = v
                grid_val_lbl.config(text=f"{v}x{v}")
                self.update_pipeline_and_render()

            s_grid = ttk.Scale(self.param_container, from_=2, to=32, value=params["grid_size"], command=update_grid)
            s_grid.pack(fill=tk.X, pady=2)
            grid_val_lbl.pack(anchor=tk.E)

        elif step_id == "denoise":
            # Denoise Method
            ttk.Label(self.param_container, text="降噪方法:").pack(anchor=tk.W)
            method_var = tk.StringVar(value=params["method"])
            methods_cb = ttk.Combobox(self.param_container, textvariable=method_var, values=["Gaussian", "Median", "Bilateral"], state="readonly")
            methods_cb.pack(fill=tk.X, pady=5)
            
            # Kernel Size
            ttk.Label(self.param_container, text="核心大小 (Kernel Size - 奇數):").pack(anchor=tk.W)
            k_val_lbl = ttk.Label(self.param_container, text=f"{params['ksize']}")
            
            # Sigma Parameter
            sig_lbl_title = ttk.Label(self.param_container, text="Sigma (強度):")
            sig_lbl_title.pack(anchor=tk.W)
            sig_val_lbl = ttk.Label(self.param_container, text=f"{params['sigma']:.1f}")
            
            def on_method_change(event):
                params["method"] = method_var.get()
                self.update_pipeline_and_render()
            methods_cb.bind("<<ComboboxSelected>>", on_method_change)

            def update_ksize(val):
                v = int(float(val))
                if v % 2 == 0: v += 1 # 強制奇數
                params["ksize"] = v
                k_val_lbl.config(text=f"{v}")
                self.update_pipeline_and_render()

            s_k = ttk.Scale(self.param_container, from_=3, to=31, value=params["ksize"], command=update_ksize)
            s_k.pack(fill=tk.X, pady=2)
            k_val_lbl.pack(anchor=tk.E)

            def update_sigma(val):
                v = round(float(val), 1)
                params["sigma"] = v
                sig_val_lbl.config(text=f"{v:.1f}")
                self.update_pipeline_and_render()

            s_sig = ttk.Scale(self.param_container, from_=0.5, to=10.0, value=params["sigma"], command=update_sigma)
            s_sig.pack(fill=tk.X, pady=2)
            sig_val_lbl.pack(anchor=tk.E)

        elif step_id == "sharpen":
            # Sharpen Strength
            ttk.Label(self.param_container, text="銳化強度 (Strength):").pack(anchor=tk.W)
            str_val_lbl = ttk.Label(self.param_container, text=f"{params['strength']:.1f}")
            
            def update_strength(val):
                v = round(float(val), 1)
                params["strength"] = v
                str_val_lbl.config(text=f"{v:.1f}")
                self.update_pipeline_and_render()

            s_str = ttk.Scale(self.param_container, from_=0.1, to=5.0, value=params["strength"], command=update_strength)
            s_str.pack(fill=tk.X, pady=2)
            str_val_lbl.pack(anchor=tk.E)

        elif step_id == "bright_contrast":
            # Brightness
            ttk.Label(self.param_container, text="亮度 (Brightness):").pack(anchor=tk.W)
            bright_lbl = ttk.Label(self.param_container, text=f"{int(params['brightness'])}")
            
            def update_bright(val):
                v = int(float(val))
                params["brightness"] = v
                bright_lbl.config(text=f"{v}")
                self.update_pipeline_and_render()

            s_br = ttk.Scale(self.param_container, from_=-100, to=100, value=params["brightness"], command=update_bright)
            s_br.pack(fill=tk.X, pady=2)
            bright_lbl.pack(anchor=tk.E)

            # Contrast
            ttk.Label(self.param_container, text="對比度 (Contrast):").pack(anchor=tk.W)
            contrast_lbl = ttk.Label(self.param_container, text=f"{params['contrast']:.2f}")
            
            def update_contrast(val):
                v = round(float(val), 2)
                params["contrast"] = v
                contrast_lbl.config(text=f"{v:.2f}")
                self.update_pipeline_and_render()

            s_ct = ttk.Scale(self.param_container, from_=0.1, to=3.0, value=params["contrast"], command=update_contrast)
            s_ct.pack(fill=tk.X, pady=2)
            contrast_lbl.pack(anchor=tk.E)

        elif step_id == "gamma":
            # Gamma
            ttk.Label(self.param_container, text="Gamma 值:").pack(anchor=tk.W)
            gamma_lbl = ttk.Label(self.param_container, text=f"{params['gamma']:.2f}")
            
            def update_gamma_val(val):
                v = round(float(val), 2)
                params["gamma"] = v
                gamma_lbl.config(text=f"{v:.2f}")
                self.update_pipeline_and_render()

            s_gm = ttk.Scale(self.param_container, from_=0.1, to=5.0, value=params["gamma"], command=update_gamma_val)
            s_gm.pack(fill=tk.X, pady=2)
            gamma_lbl.pack(anchor=tk.E)

    # ================== 核心影像運算 Pipeline ==================
    def update_pipeline_and_render(self):
        if self.raw_frame is None:
            return

        # 複製原始影格作為 Pipeline 輸入
        img = self.raw_frame.copy()

        # 循序套用啟用的處理
        for step in self.pipeline_steps:
            if not step["enabled"]:
                continue
            
            step_id = step["id"]
            params = step["params"]

            if step_id == "clahe":
                img = self.apply_clahe(img, params["clip_limit"], params["grid_size"])
            elif step_id == "denoise":
                img = self.apply_denoise(img, params["method"], params["ksize"], params["sigma"])
            elif step_id == "sharpen":
                img = self.apply_sharpen(img, params["strength"])
            elif step_id == "bright_contrast":
                img = self.apply_brightness_contrast(img, params["brightness"], params["contrast"])
            elif step_id == "gamma":
                img = self.apply_gamma(img, params["gamma"])

        self.processed_frame = img
        self.render_image_to_canvas()

    def apply_clahe(self, img, clip_limit, grid_size):
        # 支援彩色 YUV/LAB 空間等化與單通道灰階
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            return clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_denoise(self, img, method, ksize, sigma):
        if ksize % 2 == 0:
            ksize += 1
        if method == "Gaussian":
            return cv2.GaussianBlur(img, (ksize, ksize), sigma)
        elif method == "Median":
            return cv2.medianBlur(img, ksize)
        elif method == "Bilateral":
            # 雙邊濾波在 OpenCV 中只支援 8-bit
            # d: 鄰域直徑，-1 表示由 sigmaSpace 計算
            return cv2.bilateralFilter(img, d=-1, sigmaColor=sigma*20.0, sigmaSpace=sigma)
        return img

    def apply_sharpen(self, img, strength):
        # 使用卷積核心進行邊緣銳化
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + strength, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    def apply_brightness_contrast(self, img, brightness, contrast):
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    def apply_gamma(self, img, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    # ================== 預覽渲染與縮放 ==================
    def render_image_to_canvas(self):
        if self.processed_frame is None:
            return

        # 獲取畫布的寬高
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        # 初次啟動時，若 canvas 寬高尚未繪製成功 (預設為 1)，先用合適尺寸
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w = 800
            canvas_h = 500

        # 將 OpenCV 的 BGR 轉為 RGB
        rgb_img = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
        
        # 計算等比例縮放以適應 Canvas
        img_h, img_w = rgb_img.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        
        # 限制不放大影像，只縮小
        if scale < 1.0:
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            rgb_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_w, new_h = img_w, img_h

        # 轉換為 PhotoImage
        pil_img = Image.fromarray(rgb_img)
        self.photo_img = ImageTk.PhotoImage(image=pil_img)

        # 清除並重繪 Canvas
        self.canvas.delete("all")
        # 置中繪製
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=self.photo_img)

    def on_canvas_resize(self, event):
        # 視窗大小變更時，重新計算縮放並渲染
        self.render_image_to_canvas()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive video pre-processing tuning tool.")
    parser.add_argument("--video", type=str, default=None, help="Path to initial video file")
    args = parser.parse_args()

    root = tk.Tk()
    
    # 使用 Windows 平滑主題 (若可用)
    style = ttk.Style()
    if 'winnative' in style.theme_names():
        style.theme_use('vista')

    app = PreprocessingTunerApp(root, initial_video=args.video)
    
    # 啟動時先做一次重繪以適應視窗
    root.update()
    app.render_image_to_canvas()
    
    root.mainloop()
