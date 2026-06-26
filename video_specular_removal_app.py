# -*- coding: utf-8 -*-
"""
影片時空反光排除與還原應用程式 (video_specular_removal_app.py)

本程式實作了基於雙色反射模型 (Dichromatic Reflection Model) 色度分離、
多影格 (3或5幀) 稠密光流對齊與時序一致性中位數融合、以及高斯權重羽化融合之進階去反光演算法。
提供 Tkinter Dark Mode 互動介面，可載入影片、滑動檢視特定影格、動態調校參數。
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# 設定 UI 樣式常量
BG_COLOR = "#121212"
CARD_COLOR = "#1E1E1E"
TEXT_COLOR = "#E0E0E0"
ACCENT_COLOR = "#007ACC"
HIGHLIGHT_COLOR = "#FF3B30"

class SpecularRemovalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("影片時空反光檢測與還原分析工具")
        self.root.geometry("1280x768")
        self.root.configure(bg=BG_COLOR)
        
        # 影片控制相關變數
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.frame_width = 0
        self.frame_height = 0
        self.is_playing = False
        
        # 影像快取 (快取灰階與BGR，用以快速讀取鄰近影格)
        self.frame_cache_bgr = {}  # idx -> frame_bgr
        self.frame_cache_gray = {} # idx -> frame_gray
        self.cache_lock = threading.Lock()
        
        # 演算法控制變數
        self.v_thresh = tk.IntVar(value=190)
        self.s_thresh = tk.IntVar(value=60)
        self.window_size = tk.StringVar(value="5") # 3 或 5
        self.show_mask = tk.BooleanVar(value=False)
        self.smooth_sigma = tk.DoubleVar(value=5.0)
        
        # 建立 UI 介面
        self.create_widgets()
        
        # 啟動時序載入背景執行緒
        self.play_thread = None
        
        # 綁定視窗關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 1. 頂部控制欄
        top_frame = tk.Frame(self.root, bg=CARD_COLOR, height=50)
        top_frame.pack(fill=tk.X, side=tk.TOP)
        top_frame.pack_propagate(False)
        
        load_btn = tk.Button(top_frame, text="載入影片", command=self.load_video, 
                             bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Microsoft JhengHei", 10, "bold"),
                             activebackground="#005A9E", activeforeground=TEXT_COLOR, bd=0, padx=15)
        load_btn.pack(side=tk.LEFT, padx=15, pady=10)
        
        self.status_lbl = tk.Label(top_frame, text="請先載入影片以開始分析...", 
                                   fg="#888888", bg=CARD_COLOR, font=("Microsoft JhengHei", 10))
        self.status_lbl.pack(side=tk.LEFT, padx=10)
        
        # 2. 主視圖區域 (左右雙側比對)
        view_frame = tk.Frame(self.root, bg=BG_COLOR)
        view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側原始圖卡
        self.left_card = tk.LabelFrame(view_frame, text=" 原始影像 (Original Frame) ", 
                                       fg=TEXT_COLOR, bg=CARD_COLOR, font=("Microsoft JhengHei", 10, "bold"), bd=1)
        self.left_card.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5, pady=5)
        
        self.left_canvas = tk.Canvas(self.left_card, bg="#101010", bd=0, highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右側還原圖卡
        self.right_card = tk.LabelFrame(view_frame, text=" 時空還原影像 (Specular-Free Spatiotemporal Restored) ", 
                                        fg=TEXT_COLOR, bg=CARD_COLOR, font=("Microsoft JhengHei", 10, "bold"), bd=1)
        self.right_card.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=5, pady=5)
        
        self.right_canvas = tk.Canvas(self.right_card, bg="#101010", bd=0, highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 3. 右側控制面板 (參數調校)
        control_panel = tk.Frame(self.root, bg=CARD_COLOR, width=250)
        # 我們把控制面板放到最右邊，這裡為了簡化排版，我們將其以右側欄位放入，但也可以塞入底部的左右排版
        
        # 4. 底部播放控制欄
        bottom_frame = tk.Frame(self.root, bg=CARD_COLOR, height=130)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_frame.pack_propagate(False)
        
        # 播放/暫停, 上一幀, 下一幀
        btn_frame = tk.Frame(bottom_frame, bg=CARD_COLOR)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.play_btn = tk.Button(btn_frame, text="▶ 播放", command=self.toggle_play,
                                  bg="#333333", fg=TEXT_COLOR, font=("Microsoft JhengHei", 9, "bold"),
                                  activebackground="#444444", activeforeground=TEXT_COLOR, bd=0, width=8)
        self.play_btn.pack(side=tk.LEFT, padx=15)
        
        prev_btn = tk.Button(btn_frame, text="◀ 上一影格", command=self.prev_frame,
                             bg="#252525", fg=TEXT_COLOR, font=("Microsoft JhengHei", 9),
                             activebackground="#333333", activeforeground=TEXT_COLOR, bd=0, width=10)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        next_btn = tk.Button(btn_frame, text="下一影格 ▶", command=self.next_frame,
                             bg="#252525", fg=TEXT_COLOR, font=("Microsoft JhengHei", 9),
                             activebackground="#333333", activeforeground=TEXT_COLOR, bd=0, width=10)
        next_btn.pack(side=tk.LEFT, padx=5)
        
        self.mask_chk = tk.Checkbutton(btn_frame, text="標示反光遮罩", variable=self.show_mask, command=self.update_current_view,
                                       bg=CARD_COLOR, fg=TEXT_COLOR, selectcolor=CARD_COLOR,
                                       activebackground=CARD_COLOR, activeforeground=TEXT_COLOR,
                                       font=("Microsoft JhengHei", 9))
        self.mask_chk.pack(side=tk.LEFT, padx=20)
        
        # 影格滑軌
        slider_frame = tk.Frame(bottom_frame, bg=CARD_COLOR)
        slider_frame.pack(fill=tk.X, padx=15, pady=2)
        
        self.time_lbl = tk.Label(slider_frame, text="0 / 0 幀", fg=TEXT_COLOR, bg=CARD_COLOR, font=("Courier New", 9))
        self.time_lbl.pack(side=tk.RIGHT, padx=10)
        
        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        
        # 5. 參數微調滑桿 (放於底部控制欄的最下排)
        param_frame = tk.Frame(bottom_frame, bg=CARD_COLOR)
        param_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # V 閾值
        v_lbl = tk.Label(param_frame, text="V 亮度門檻:", fg=TEXT_COLOR, bg=CARD_COLOR, font=("Microsoft JhengHei", 9))
        v_lbl.pack(side=tk.LEFT, padx=5)
        self.v_scale = ttk.Scale(param_frame, from_=100, to=255, value=190, orient=tk.HORIZONTAL, command=self.on_param_change, length=120)
        self.v_scale.pack(side=tk.LEFT, padx=5)
        self.v_val_lbl = tk.Label(param_frame, text="190", fg=ACCENT_COLOR, bg=CARD_COLOR, font=("Courier New", 9, "bold"))
        self.v_val_lbl.pack(side=tk.LEFT, padx=5)
        
        # S 閾值
        s_lbl = tk.Label(param_frame, text="S 飽和上限:", fg=TEXT_COLOR, bg=CARD_COLOR, font=("Microsoft JhengHei", 9))
        s_lbl.pack(side=tk.LEFT, padx=20)
        self.s_scale = ttk.Scale(param_frame, from_=10, to=150, value=60, orient=tk.HORIZONTAL, command=self.on_param_change, length=120)
        self.s_scale.pack(side=tk.LEFT, padx=5)
        self.s_val_lbl = tk.Label(param_frame, text="60", fg=ACCENT_COLOR, bg=CARD_COLOR, font=("Courier New", 9, "bold"))
        self.s_val_lbl.pack(side=tk.LEFT, padx=5)
        
        # 時序窗口大小
        w_lbl = tk.Label(param_frame, text="時序窗口:", fg=TEXT_COLOR, bg=CARD_COLOR, font=("Microsoft JhengHei", 9))
        w_lbl.pack(side=tk.LEFT, padx=20)
        self.w_combo = ttk.Combobox(param_frame, textvariable=self.window_size, values=["3", "5"], width=5, state="readonly")
        self.w_combo.pack(side=tk.LEFT, padx=5)
        self.w_combo.bind("<<ComboboxSelected>>", lambda e: self.update_current_view())
        
        # 羽化 Sigma
        sigma_lbl = tk.Label(param_frame, text="羽化平滑:", fg=TEXT_COLOR, bg=CARD_COLOR, font=("Microsoft JhengHei", 9))
        sigma_lbl.pack(side=tk.LEFT, padx=20)
        self.sigma_scale = ttk.Scale(param_frame, from_=1.0, to=15.0, value=5.0, orient=tk.HORIZONTAL, command=self.on_param_change, length=100)
        self.sigma_scale.pack(side=tk.LEFT, padx=5)
        self.sigma_val_lbl = tk.Label(param_frame, text="5.0", fg=ACCENT_COLOR, bg=CARD_COLOR, font=("Courier New", 9, "bold"))
        self.sigma_val_lbl.pack(side=tk.LEFT, padx=5)

        # 設定 ttk 滑軌樣式為深色系
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TScale", background=CARD_COLOR, troughcolor="#2A2A2A", borderwidth=0)
        style.configure("TCombobox", fieldbackground="#2A2A2A", background="#333333", foreground=TEXT_COLOR)

    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv"), ("所有檔案", "*.*")]
        )
        if not file_path:
            return
            
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 重設滑軌範圍與標籤
        self.slider.configure(to=self.total_frames - 1)
        self.slider.set(0)
        
        # 清除快取
        with self.cache_lock:
            self.frame_cache_bgr.clear()
            self.frame_cache_gray.clear()
            
        self.current_frame_idx = 0
        self.status_lbl.configure(text=f"已載入影片: {os.path.basename(file_path)} ({self.frame_width}x{self.frame_height}, {self.total_frames}幀)")
        self.time_lbl.configure(text=f"0 / {self.total_frames - 1} 幀")
        
        # 讀取並顯示第一幀
        self.update_current_view()
        
    def get_frame(self, idx):
        """從快取或影片檔取得指定影格的 BGR 與灰階圖"""
        if idx < 0 or idx >= self.total_frames:
            return None, None
            
        with self.cache_lock:
            if idx in self.frame_cache_bgr:
                return self.frame_cache_bgr[idx], self.frame_cache_gray[idx]
                
        # 若未命中快取，則讀取影片檔
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        # 限制記憶體快取大小，最多存放 30 幀 (配合滑動視窗需求)
        with self.cache_lock:
            if len(self.frame_cache_bgr) > 30:
                # 刪除距離最遠的快取
                keys = list(self.frame_cache_bgr.keys())
                furthest_key = max(keys, key=lambda k: abs(k - self.current_frame_idx))
                del self.frame_cache_bgr[furthest_key]
                del self.frame_cache_gray[furthest_key]
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_cache_bgr[idx] = frame
            self.frame_cache_gray[idx] = gray
            
        return frame, gray

    def run_specular_removal_pipeline(self, t_idx):
        """
        核心時空反光排除重建演算法 Pipeline
        """
        frame_t, gray_t = self.get_frame(t_idx)
        if frame_t is None:
            return None, None
            
        v_th = self.v_thresh.get()
        s_th = self.s_thresh.get()
        w_size = int(self.window_size.get())
        sigma = self.smooth_sigma.get()
        
        # ------------------ 階段 1: 雙色反射色度分離與遮罩生成 ------------------
        hsv_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2HSV)
        H_t, S_t, V_t = cv2.split(hsv_t)
        
        # 局部自適應均值與標準差檢測
        mean_v = cv2.blur(V_t, (15, 15))
        mean_sq_v = cv2.blur(V_t.astype(np.float32)**2, (15, 15))
        std_v = cv2.sqrt(np.clip(mean_sq_v - mean_v.astype(np.float32)**2, 0, None))
        
        # 高光像素判定 (低飽和度 + 絕對高亮度 + 局部突出)
        # 我們將 std_v 限制下限以防平滑區除以 0
        mask_t = (V_t > v_th) & (S_t < s_th) & (V_t > (mean_v + 1.0 * std_v))
        mask_t_u8 = mask_t.astype(np.uint8)
        
        # 利用 Morphological 進行微小連通域濾除與適應性膨脹
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_t_u8)
        refined_mask = np.zeros_like(mask_t_u8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 2:  # 濾除細小雜訊
                # 根據斑點大小動態設定膨脹核心半徑
                k_size = 3 if area < 10 else (5 if area < 50 else 7)
                comp = (labels == i).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                dilated = cv2.dilate(comp, kernel)
                refined_mask = cv2.bitwise_or(refined_mask, dilated)
                
        # 空間域 Specular-free 漫反射基底估算 (作為降級備用)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        local_min_v = cv2.erode(V_t, kernel_erode)
        V_sf = V_t.copy()
        V_sf[refined_mask == 1] = local_min_v[refined_mask == 1]
        
        hsv_sf = cv2.merge([H_t, S_t, V_sf])
        frame_sf = cv2.cvtColor(hsv_sf, cv2.COLOR_HSV2BGR)
        
        # ------------------ 階段 2: 5 幀時序光流對齊與中位數融合 ------------------
        # 決定時域窗口的影格索引
        offsets = [-1, 1] if w_size == 3 else [-2, -1, 1, 2]
        warped_frames = []
        warped_masks = []
        
        for offset in offsets:
            k_idx = t_idx + offset
            frame_k, gray_k = self.get_frame(k_idx)
            if frame_k is None:
                continue
                
            # 計算 gray_t 到 gray_k 的 Farneback 稠密光流
            # 由於微距影片通常位移較小，設定 pyramid 縮放與適當的 window
            flow = cv2.calcOpticalFlowFarneback(
                gray_t, gray_k, None, 
                pyr_scale=0.5, levels=3, winsize=11, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # 建立映射矩陣進行 Warp
            h, w = gray_t.shape
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (map_x + flow[..., 0]).astype(np.float32)
            map_y = (map_y + flow[..., 1]).astype(np.float32)
            
            frame_k_warped = cv2.remap(frame_k, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # 對齊鄰近影格的高光遮罩
            # 重新為相鄰影格計算遮罩
            hsv_k = cv2.cvtColor(frame_k, cv2.COLOR_BGR2HSV)
            S_k, V_k = hsv_k[:, :, 1], hsv_k[:, :, 2]
            mean_v_k = cv2.blur(V_k, (15, 15))
            mean_sq_v_k = cv2.blur(V_k.astype(np.float32)**2, (15, 15))
            std_v_k = cv2.sqrt(np.clip(mean_sq_v_k - mean_v_k.astype(np.float32)**2, 0, None))
            mask_k = (V_k > v_th) & (S_k < s_th) & (V_k > (mean_v_k + 1.0 * std_v_k))
            
            mask_k_warped = cv2.remap(mask_k.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
            
            warped_frames.append(frame_k_warped)
            warped_masks.append(mask_k_warped)
            
        # 開始時序像素融合
        temporal_restored = frame_t.copy()
        
        if len(warped_frames) > 0:
            # 將所有 warping 影格疊加成一個 numpy 陣列
            stacked_frames = np.stack(warped_frames, axis=0) # Shape: (N, H, W, 3)
            stacked_masks = np.stack(warped_masks, axis=0)   # Shape: (N, H, W)
            
            # 找出需要還原的像素點坐標
            y_coords, x_coords = np.where(refined_mask == 1)
            
            for y, x in zip(y_coords, x_coords):
                # 在此空間位置，提取出所有相鄰影格對齊後的健康像素（即對齊遮罩為 0 的影格）
                valid_pixel_indices = np.where(stacked_masks[:, y, x] == 0)[0]
                
                if len(valid_pixel_indices) > 0:
                    # 計算這些健康像素在 RGB 三通道上的中位數 (Median)
                    valid_pixels = stacked_frames[valid_pixel_indices, y, x, :]
                    temporal_restored[y, x, :] = np.median(valid_pixels, axis=0).astype(np.uint8)
                else:
                    # 若所有對齊影格在該點也都是高光（例如靜止無運動），則採用空間雙色分離基底
                    temporal_restored[y, x, :] = frame_sf[y, x, :]
                    
        # ------------------ 階段 3: 梯度羽化泊松融合 (Poisson-like Blending) ------------------
        # 對高光遮罩進行高斯模糊以建立平滑的羽化權重 (Alpha mask)
        kernel_size = int(round(sigma * 3)) | 1  # 確保為奇數
        alpha = cv2.GaussianBlur(refined_mask.astype(np.float32), (kernel_size, kernel_size), sigma)
        alpha = np.expand_dims(alpha, axis=2) # Shape: (H, W, 1)
        
        # 最終影像線性融合
        frame_restored = (alpha * temporal_restored + (1.0 - alpha) * frame_t).astype(np.uint8)
        
        return frame_restored, refined_mask

    def update_current_view(self):
        """讀取當前影格，運行演算法，並將結果呈現在畫面上"""
        if self.cap is None:
            return
            
        frame_t, _ = self.get_frame(self.current_frame_idx)
        if frame_t is None:
            return
            
        t0 = time.perf_counter()
        # 執行去反光演算法
        frame_restored, mask = self.run_specular_removal_pipeline(self.current_frame_idx)
        t1 = time.perf_counter()
        
        if frame_restored is None:
            return
            
        # 更新狀態欄耗時
        elapsed_ms = (t1 - t0) * 1000.0
        self.status_lbl.configure(text=f"影格 {self.current_frame_idx} / {self.total_frames - 1} | 演算法處理耗時: {elapsed_ms:.1f} ms")
        
        # 處理左圖顯示 (原圖，可視需要疊加紅色反光遮罩)
        disp_left = frame_t.copy()
        if self.show_mask.get() and mask is not None:
            # 將遮罩區域染色為紅色高亮顯示
            disp_left[mask == 1] = [0, 0, 255]
            
        # 將 BGR 轉換為 RGB 以利 PIL 顯示
        disp_left_rgb = cv2.cvtColor(disp_left, cv2.COLOR_BGR2RGB)
        disp_right_rgb = cv2.cvtColor(frame_restored, cv2.COLOR_BGR2RGB)
        
        self.show_on_canvas(self.left_canvas, disp_left_rgb)
        self.show_on_canvas(self.right_canvas, disp_right_rgb)
        
        # 更新時間標籤與滑軌 (阻斷滑軌連動以防無效重入)
        self.time_lbl.configure(text=f"{self.current_frame_idx} / {self.total_frames - 1} 幀")
        self.slider.set(self.current_frame_idx)

    def show_on_canvas(self, canvas, img_rgb):
        """將 RGB 圖像縮放並繪製到 Tkinter Canvas 上"""
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        # 初始載入時可能尺寸為 1
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w = 600
            canvas_h = 450
            
        img_h, img_w, _ = img_rgb.shape
        
        # 計算保持長寬比的縮放比例
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        if new_w <= 0 or new_h <= 0:
            return
            
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(image=img_resized)
        
        # 擦除舊內容並置中繪製
        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=img_tk, anchor=tk.CENTER)
        
        # 防止垃圾回收機制銷毀圖片物件
        canvas.image = img_tk

    def on_slider_move(self, value):
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self.current_frame_idx = idx
            self.update_current_view()
            
    def on_param_change(self, event=None):
        # 更新數值標籤
        self.v_val_lbl.configure(text=str(int(self.v_scale.get())))
        self.s_val_lbl.configure(text=str(int(self.s_scale.get())))
        self.sigma_val_lbl.configure(text=f"{self.sigma_scale.get():.1f}")
        
        # 即時重繪畫面
        self.update_current_view()
        
    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_current_view()
            
    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.update_current_view()
            
    def toggle_play(self):
        if self.cap is None:
            return
            
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.configure(text="⏸ 暫停", bg="#D83B01")
            self.play_next_frame_loop()
        else:
            self.play_btn.configure(text="▶ 播放", bg="#333333")
            
    def play_next_frame_loop(self):
        if not self.is_playing:
            return
            
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.update_current_view()
            
            # 根據影片 FPS 來排程下一影格
            delay_ms = int(1000 / self.fps)
            # 演算法計算可能耗時，減去一部分以維持正常播放速率
            self.root.after(max(1, delay_ms - 20), self.play_next_frame_loop)
        else:
            self.is_playing = False
            self.play_btn.configure(text="▶ 播放", bg="#333333")

    def on_closing(self):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    
    # 全局字型與深色主題微調
    style = ttk.Style()
    style.configure(".", font=("Microsoft JhengHei", 9))
    
    app = SpecularRemovalApp(root)
    
    # 第一次顯示時等 UI 大小確認後再更新畫面
    root.update()
    root.mainloop()
