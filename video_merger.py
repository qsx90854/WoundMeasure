import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np


class VideoMergerApp:

    def __init__(self, root):
        self.root = root
        self.root.title("影片拼接工具 (Video Merger)")
        self.root.geometry("660x490")
        self.root.resizable(False, False)

        # 質感配色與字型設定
        self.font_title = ("Microsoft JhengHei", 12, "bold")
        self.font_normal = ("Microsoft JhengHei", 10)
        self.bg_color = "#F5F5F7"
        self.btn_color = "#007ACC"

        self.root.configure(bg=self.bg_color)

        # 影片路徑與設定變數
        self.video1_path = tk.StringVar()
        self.video2_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.merge_mode = tk.StringVar(value="concat")  # concat, sbs, stacked

        # 影格範圍變數 (手動輸入從第幾幀接到第幾幀)
        self.v1_start = tk.StringVar(value="0")
        self.v1_end = tk.StringVar(value="0")
        self.v2_start = tk.StringVar(value="0")
        self.v2_end = tk.StringVar(value="0")

        self.is_processing = False

        # 綁定路徑改變監聽器，自動讀取並顯示影片總幀數
        self.video1_path.trace_add("write", self.update_video1_info)
        self.video2_path.trace_add("write", self.update_video2_info)

        # 綁定模式改變監聽器，動態啟用/停用範圍輸入
        self.merge_mode.trace_add("write", self.on_mode_changed)

        self.create_widgets()

    def create_widgets(self):
        # 標題
        lbl_title = tk.Label(
            self.root,
            text="🎥 影片拼接與局部裁剪工具",
            font=("Microsoft JhengHei", 14, "bold"),
            bg=self.bg_color,
            fg="#333333",
        )
        lbl_title.pack(pady=10)

        # 主要輸入區域框架
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.X, padx=20)

        # 影片 1
        tk.Label(
            main_frame, text="影片 1 來源:", font=self.font_normal, bg=self.bg_color
        ).grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(
            main_frame,
            textvariable=self.video1_path,
            width=52,
            font=self.font_normal,
        ).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(
            main_frame,
            text="瀏覽...",
            command=self.browse_video1,
            font=self.font_normal,
            bg="#E1E1E1",
        ).grid(row=0, column=2, padx=5, pady=5)

        # 影片 2
        tk.Label(
            main_frame, text="影片 2 來源:", font=self.font_normal, bg=self.bg_color
        ).grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(
            main_frame,
            textvariable=self.video2_path,
            width=52,
            font=self.font_normal,
        ).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(
            main_frame,
            text="瀏覽...",
            command=self.browse_video2,
            font=self.font_normal,
            bg="#E1E1E1",
        ).grid(row=1, column=2, padx=5, pady=5)

        # 影片 3 (輸出)
        tk.Label(
            main_frame, text="儲存為影片 3:", font=self.font_normal, bg=self.bg_color
        ).grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(
            main_frame,
            textvariable=self.output_path,
            width=52,
            font=self.font_normal,
        ).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(
            main_frame,
            text="選擇路徑...",
            command=self.browse_output,
            font=self.font_normal,
            bg="#E1E1E1",
        ).grid(row=2, column=2, padx=5, pady=5)

        # 拼接模式選擇
        mode_frame = tk.LabelFrame(
            self.root, text=" 拼接模式設定 ", font=self.font_normal, bg=self.bg_color
        )
        mode_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Radiobutton(
            mode_frame,
            text="前後相接 (Concatenate)",
            variable=self.merge_mode,
            value="concat",
            font=self.font_normal,
            bg=self.bg_color,
        ).pack(side=tk.LEFT, padx=15, pady=5)
        tk.Radiobutton(
            mode_frame,
            text="左右拼接 (Side-by-Side)",
            variable=self.merge_mode,
            value="sbs",
            font=self.font_normal,
            bg=self.bg_color,
        ).pack(side=tk.LEFT, padx=15, pady=5)
        tk.Radiobutton(
            mode_frame,
            text="上下拼接 (Stacked)",
            variable=self.merge_mode,
            value="stacked",
            font=self.font_normal,
            bg=self.bg_color,
        ).pack(side=tk.LEFT, padx=15, pady=5)

        # 前後相接影格範圍設定
        self.range_frame = tk.LabelFrame(
            self.root,
            text=" 前後相接擷取影格設定 (僅 Concatenate 模式啟用) ",
            font=self.font_normal,
            bg=self.bg_color,
        )
        self.range_frame.pack(fill=tk.X, padx=20, pady=5)

        # 影片 1 範圍
        tk.Label(
            self.range_frame, text="影片 1 範圍：", font=self.font_normal, bg=self.bg_color
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.lbl_v1_info = tk.Label(
            self.range_frame,
            text="影片 1 總影格數: 未載入",
            font=self.font_normal,
            bg=self.bg_color,
            fg="#666666",
        )
        self.lbl_v1_info.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        tk.Label(
            self.range_frame, text="從第", font=self.font_normal, bg=self.bg_color
        ).grid(row=0, column=2, padx=2, pady=5)
        self.entry_v1_start = tk.Entry(
            self.range_frame,
            textvariable=self.v1_start,
            width=6,
            font=self.font_normal,
            justify="center",
        )
        self.entry_v1_start.grid(row=0, column=3, padx=2, pady=5)

        tk.Label(
            self.range_frame, text="幀 到", font=self.font_normal, bg=self.bg_color
        ).grid(row=0, column=4, padx=2, pady=5)
        self.entry_v1_end = tk.Entry(
            self.range_frame,
            textvariable=self.v1_end,
            width=6,
            font=self.font_normal,
            justify="center",
        )
        self.entry_v1_end.grid(row=0, column=5, padx=2, pady=5)
        tk.Label(
            self.range_frame, text="幀", font=self.font_normal, bg=self.bg_color
        ).grid(row=0, column=6, padx=2, pady=5)

        # 影片 2 範圍
        tk.Label(
            self.range_frame, text="影片 2 範圍：", font=self.font_normal, bg=self.bg_color
        ).grid(row=1, column=0, sticky="w", padx=10, pady=5)

        self.lbl_v2_info = tk.Label(
            self.range_frame,
            text="影片 2 總影格數: 未載入",
            font=self.font_normal,
            bg=self.bg_color,
            fg="#666666",
        )
        self.lbl_v2_info.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        tk.Label(
            self.range_frame, text="從第", font=self.font_normal, bg=self.bg_color
        ).grid(row=1, column=2, padx=2, pady=5)
        self.entry_v2_start = tk.Entry(
            self.range_frame,
            textvariable=self.v2_start,
            width=6,
            font=self.font_normal,
            justify="center",
        )
        self.entry_v2_start.grid(row=1, column=3, padx=2, pady=5)

        tk.Label(
            self.range_frame, text="幀 到", font=self.font_normal, bg=self.bg_color
        ).grid(row=1, column=4, padx=2, pady=5)
        self.entry_v2_end = tk.Entry(
            self.range_frame,
            textvariable=self.v2_end,
            width=6,
            font=self.font_normal,
            justify="center",
        )
        self.entry_v2_end.grid(row=1, column=5, padx=2, pady=5)
        tk.Label(
            self.range_frame, text="幀", font=self.font_normal, bg=self.bg_color
        ).grid(row=1, column=6, padx=2, pady=5)

        # 進度條與狀態
        self.progress_frame = tk.Frame(self.root, bg=self.bg_color)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=5)

        self.progress_bar = ttk.Progressbar(
            self.progress_frame, orient="horizontal", mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.lbl_status = tk.Label(
            self.progress_frame,
            text="就緒。請選取輸入與輸出路徑，設定影格範圍，點擊「開始拼接」。",
            font=self.font_normal,
            bg=self.bg_color,
            fg="#555555",
        )
        self.lbl_status.pack()

        # 執行按鈕
        self.btn_run = tk.Button(
            self.root,
            text="開始拼接影片",
            font=self.font_title,
            bg=self.btn_color,
            fg="white",
            command=self.start_merging_thread,
            height=1,
            width=20,
        )
        self.btn_run.pack(pady=10)

        # 初始狀態觸發
        self.on_mode_changed()

    def update_video1_info(self, *args):
        path = self.video1_path.get().strip()
        if not path or not os.path.exists(path):
            self.lbl_v1_info.config(text="影片 1 總影格數: 未載入")
            return

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.lbl_v1_info.config(text=f"影片 1 總影格數: {total_frames} 幀")
            self.v1_start.set("0")
            self.v1_end.set(str(total_frames))
        cap.release()

    def update_video2_info(self, *args):
        path = self.video2_path.get().strip()
        if not path or not os.path.exists(path):
            self.lbl_v2_info.config(text="影片 2 總影格數: 未載入")
            return

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.lbl_v2_info.config(text=f"影片 2 總影格數: {total_frames} 幀")
            self.v2_start.set("0")
            self.v2_end.set(str(total_frames))
        cap.release()

    def on_mode_changed(self, *args):
        mode = self.merge_mode.get()
        if mode == "concat":
            self.enable_range_settings(True)
        else:
            self.enable_range_settings(False)

    def enable_range_settings(self, enable):
        state = tk.NORMAL if enable else tk.DISABLED
        self.entry_v1_start.config(state=state)
        self.entry_v1_end.config(state=state)
        self.entry_v2_start.config(state=state)
        self.entry_v2_end.config(state=state)

    def browse_video1(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv"), ("所有檔案", "*.*")]
        )
        if file_path:
            self.video1_path.set(file_path)

    def browse_video2(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv"), ("所有檔案", "*.*")]
        )
        if file_path:
            self.video2_path.set(file_path)

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 影片", "*.mp4"), ("AVI 影片", "*.avi"), ("所有檔案", "*.*")],
        )
        if file_path:
            self.output_path.set(file_path)

    def start_merging_thread(self):
        if self.is_processing:
            return

        v1 = self.video1_path.get().strip()
        v2 = self.video2_path.get().strip()
        out = self.output_path.get().strip()

        if not v1 or not v2 or not out:
            messagebox.showwarning(
                "欄位未填寫", "請選取影片 1、影片 2 以及輸出影片 3 的路徑！"
            )
            return

        if not os.path.exists(v1):
            messagebox.showerror("錯誤", f"影片 1 路徑不存在：\n{v1}")
            return

        if not os.path.exists(v2):
            messagebox.showerror("錯誤", f"影片 2 路徑不存在：\n{v2}")
            return

        # 停用 UI 按鈕，避免重入
        self.is_processing = True
        self.btn_run.config(state=tk.DISABLED, bg="#CCCCCC")
        self.progress_bar["value"] = 0

        # 開啟背景執行緒進行處理
        t = threading.Thread(
            target=self.process_merging, args=(v1, v2, out), daemon=True
        )
        t.start()

    def process_merging(self, v1_path, v2_path, out_path):
        cap1 = cv2.VideoCapture(v1_path)
        cap2 = cv2.VideoCapture(v2_path)

        if not cap1.isOpened() or not cap2.isOpened():
            self.update_ui_on_error("無法開啟輸入的影片檔案，請檢查格式是否支援。")
            cap1.release()
            cap2.release()
            return

        # 讀取影片屬性
        w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

        mode = self.merge_mode.get()

        # 決定輸出影片的解析度與 FPS
        out_w, out_h = w1, h1
        out_fps = fps1 if fps1 > 0 else 30.0

        if mode == "concat":
            # 取得並驗證前後相接區間
            try:
                s1 = int(self.v1_start.get())
                e1 = int(self.v1_end.get())
                s2 = int(self.v2_start.get())
                e2 = int(self.v2_end.get())
            except ValueError:
                self.update_ui_on_error("前後相接影格範圍必須輸入有效的整數！")
                cap1.release()
                cap2.release()
                return

            # 安全範圍修正
            s1 = max(0, min(s1, total_frames1 - 1))
            e1 = max(s1 + 1, min(e1, total_frames1))
            s2 = max(0, min(s2, total_frames2 - 1))
            e2 = max(s2 + 1, min(e2, total_frames2))

            out_w, out_h = w1, h1
            total_target_frames = (e1 - s1) + (e2 - s2)
        elif mode == "sbs":
            # 左右拼接，高度對齊影片 1，寬度為 影片 1 的寬度 + 影片 2 的縮放寬度
            scale2_w = int(w2 * (h1 / h2)) if h2 > 0 else w2
            out_w = w1 + scale2_w
            out_h = h1
            total_target_frames = min(total_frames1, total_frames2)
        elif mode == "stacked":
            # 上下拼接，寬度對齊影片 1，高度為 影片 1 的高度 + 影片 2 的縮放高度
            scale2_h = int(h2 * (w1 / w2)) if w2 > 0 else h2
            out_w = w1
            out_h = h1 + scale2_h
            total_target_frames = min(total_frames1, total_frames2)
        else:
            self.update_ui_on_error("未知的拼接模式。")
            cap1.release()
            cap2.release()
            return

        # 建立影片寫入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

        if not writer.isOpened():
            self.update_ui_on_error("無法建立輸出影片，請檢查是否有寫入權限。")
            cap1.release()
            cap2.release()
            return

        processed_frames = 0

        try:
            if mode == "concat":
                # 前後相接模式：寫入影片 1 區間 [s1, e1)
                self.lbl_status.config(text=f"正在定位影片 1 到第 {s1} 幀...")
                cap1.set(cv2.CAP_PROP_POS_FRAMES, s1)
                for f_idx in range(s1, e1):
                    ret, frame = cap1.read()
                    if not ret:
                        break
                    if frame.shape[1] != out_w or frame.shape[0] != out_h:
                        frame = cv2.resize(frame, (out_w, out_h))
                    writer.write(frame)
                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        pct = int((processed_frames / total_target_frames) * 100)
                        self.progress_bar["value"] = pct
                        self.lbl_status.config(
                            text=f"影片 1 寫入中：{processed_frames}/{total_target_frames} 幀 ({pct}%)"
                        )

                # 寫入影片 2 區間 [s2, e2)
                self.lbl_status.config(text=f"正在定位影片 2 到第 {s2} 幀...")
                cap2.set(cv2.CAP_PROP_POS_FRAMES, s2)
                for f_idx in range(s2, e2):
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    if frame.shape[1] != out_w or frame.shape[0] != out_h:
                        frame = cv2.resize(frame, (out_w, out_h))
                    writer.write(frame)
                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        pct = int((processed_frames / total_target_frames) * 100)
                        self.progress_bar["value"] = pct
                        self.lbl_status.config(
                            text=f"影片 2 寫入中：{processed_frames}/{total_target_frames} 幀 ({pct}%)"
                        )

            elif mode == "sbs":
                # 左右拼接模式：同步讀取兩影片，橫向拼接
                self.lbl_status.config(text="正在進行左右拼接...")
                scale2_w = out_w - w1
                for _ in range(total_target_frames):
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    if not ret1 or not ret2:
                        break

                    # 對齊高度
                    if frame1.shape[1] != w1 or frame1.shape[0] != h1:
                        frame1 = cv2.resize(frame1, (w1, h1))
                    if frame2.shape[1] != scale2_w or frame2.shape[0] != h1:
                        frame2 = cv2.resize(frame2, (scale2_w, h1))

                    merged_frame = np.hstack((frame1, frame2))
                    writer.write(merged_frame)
                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        pct = int((processed_frames / total_target_frames) * 100)
                        self.progress_bar["value"] = pct
                        self.lbl_status.config(
                            text=f"左右拼接中：{processed_frames}/{total_target_frames} 幀 ({pct}%)"
                        )

            elif mode == "stacked":
                # 上下拼接模式：同步讀取兩影片，縱向拼接
                self.lbl_status.config(text="正在進行上下拼接...")
                scale2_h = out_h - h1
                for _ in range(total_target_frames):
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    if not ret1 or not ret2:
                        break

                    # 對齊寬度
                    if frame1.shape[1] != w1 or frame1.shape[0] != h1:
                        frame1 = cv2.resize(frame1, (w1, h1))
                    if frame2.shape[1] != w1 or frame2.shape[0] != scale2_h:
                        frame2 = cv2.resize(frame2, (w1, scale2_h))

                    merged_frame = np.vstack((frame1, frame2))
                    writer.write(merged_frame)
                    processed_frames += 1

                    if processed_frames % 5 == 0:
                        pct = int((processed_frames / total_target_frames) * 100)
                        self.progress_bar["value"] = pct
                        self.lbl_status.config(
                            text=f"上下拼接中：{processed_frames}/{total_target_frames} 幀 ({pct}%)"
                        )

            writer.release()
            cap1.release()
            cap2.release()

            # 完成通知
            self.progress_bar["value"] = 100
            self.lbl_status.config(text="🎉 拼接與裁剪成功！影片已儲存。")
            messagebox.showinfo(
                "完成", f"影片拼接完成！\n已成功儲存至：\n{out_path}"
            )

        except Exception as e:
            writer.release()
            cap1.release()
            cap2.release()
            self.update_ui_on_error(f"拼接過程中發生錯誤：{str(e)}")

        finally:
            self.is_processing = False
            self.btn_run.config(state=tk.NORMAL, bg=self.btn_color)

    def update_ui_on_error(self, err_msg):
        self.lbl_status.config(text="❌ 發生錯誤。")
        messagebox.showerror("拼接失敗", err_msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoMergerApp(root)
    root.mainloop()
