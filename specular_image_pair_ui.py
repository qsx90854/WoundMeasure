"""Two-image spatial specular viewer. Run: .\Scripts\python.exe specular_image_pair_ui.py

Shares detectors/settings with GradSIFTDebug. Reads its startup config via AST
without importing the video application. No temporal, warp or descriptor margins.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Algorithm.specular_detection import (
    BlockSpatialSpecularConfig, compute_specular_mask_bgr_block_adaptive,
    compute_specular_mask_bgr)
from Algorithm.specular_model_v2 import SpecularModelV2Config, detect_specular_model_v2
from Algorithm.specular_settings import show_specular_settings
from Algorithm.specular_model_v2_settings import show_specular_model_v2_settings

MAIN_FILE = Path(__file__).with_name('depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py')


def read_main_defaults(path=MAIN_FILE):
    """Read only whitelisted dataclass constructors and literal arguments."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    wanted = {'SPATIAL_BLOCK_CONFIG': BlockSpatialSpecularConfig,
              'SPECULAR_MODEL_V2_CONFIG': SpecularModelV2Config}
    found = {}
    enabled = False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == 'SPECULAR_MODEL_V2_ENABLED_DEFAULT':
                enabled = bool(ast.literal_eval(node.value))
            if target.id in wanted:
                call = node.value
                cls = wanted[target.id]
                if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                        or call.func.id != cls.__name__ or call.args
                        or any(k.arg is None for k in call.keywords)):
                    raise ValueError(f'Cannot safely read {target.id} from {path}')
                found[target.id] = cls(**{k.arg: ast.literal_eval(k.value) for k in call.keywords})
    if set(found) != set(wanted):
        raise ValueError('Main UI spatial detector configurations not found')
    return found['SPATIAL_BLOCK_CONFIG'], found['SPECULAR_MODEL_V2_CONFIG'], enabled


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'無法讀取圖片：{path}')
    return image


def detect_pair(images, mode, legacy, v2):
    result = []
    for image in images:
        if image is None:
            result.append(None)
            continue
        started = time.perf_counter()
        if mode == 'V2':
            mask = detect_specular_model_v2(image, v2)
        elif mode == 'Fixed':
            mask = compute_specular_mask_bgr(image)
        else:
            mask = compute_specular_mask_bgr_block_adaptive(image, legacy)
        result.append((mask, (time.perf_counter()-started)*1000))
    return result


class SpecularPairUI:
    def __init__(self, root):
        self.root = root
        self.legacy, self.v2, enabled = read_main_defaults()
        root.title('雙圖空間反光偵測 — Region-SIFT mask')
        root.geometry('1400x850')
        self.images = [None, None]
        self.paths = [None, None]
        self.results = [None, None]
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.closed = False
        self.drag = None
        self.mode = tk.StringVar(value='V2' if enabled else 'Legacy')
        self.display = tk.StringVar(value='疊圖')
        self.alpha = tk.DoubleVar(value=.55)
        self.status = tk.StringVar(value='載入左右圖片；滾輪縮放、左鍵拖曳平移，兩側同步。')
        bar = ttk.Frame(root, padding=6)
        bar.pack(fill='x')
        for label, command in [('載入左圖', lambda: self.load(0)), ('載入右圖', lambda: self.load(1)),
                               ('重算兩圖', self.recompute), ('反光參數', self.settings),
                               ('適合視窗', self.home), ('匯出遮罩', self.export)]:
            ttk.Button(bar, text=label, command=command).pack(side='left', padx=3)
        ttk.Label(bar, text='模式').pack(side='left', padx=(12,3))
        mode = ttk.Combobox(bar, textvariable=self.mode, values=('Legacy','V2','Fixed'), width=8, state='readonly')
        mode.pack(side='left'); mode.bind('<<ComboboxSelected>>', lambda e: self.recompute())
        view = ttk.Combobox(bar, textvariable=self.display, values=('疊圖','原圖','遮罩'), width=7, state='readonly')
        view.pack(side='left', padx=8); view.bind('<<ComboboxSelected>>', lambda e: self.draw())
        ttk.Label(bar, text='疊圖透明度').pack(side='left')
        ttk.Scale(bar, from_=0, to=1, variable=self.alpha, command=lambda v: self.draw()).pack(side='left')
        ttk.Label(root, text='深藍＝空間反光排除區；V2 uncertain 不列入。僅處理載入原圖，不執行去畸變、時間反光或 warp。').pack(anchor='w', padx=10)
        self.figure = Figure(figsize=(12,7), facecolor='#202020')
        left = self.figure.add_subplot(121)
        right = self.figure.add_subplot(122, sharex=left, sharey=left)
        self.axes = [left,right]
        self.artists = []
        for ax, label in zip(self.axes, ('Left','Right')):
            ax.set_facecolor('#111111'); ax.tick_params(colors='white')
            ax.set_title(label, color='white')
            self.artists.append(ax.imshow(np.zeros((1,1,3), np.uint8), interpolation='nearest', origin='upper'))
            ax.set_aspect('equal', adjustable='box')
        self.figure.subplots_adjust(left=.04, right=.98, bottom=.06, top=.93, wspace=.12)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        for event, callback in [('scroll_event', self.scroll), ('button_press_event', self.press),
                                ('motion_notify_event', self.motion), ('button_release_event', self.release)]:
            self.canvas.mpl_connect(event, callback)
        ttk.Label(root, textvariable=self.status, padding=6).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Escape>', lambda e: self.release(None))
        self.home()

    def load(self, side):
        if self.future is not None:
            return
        path = filedialog.askopenfilename(parent=self.root, title='載入左圖' if side == 0 else '載入右圖',
            filetypes=[('Images','*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp'),('All files','*.*')])
        if not path:
            return
        try:
            image = read_image(path)
        except Exception as exc:
            messagebox.showerror('讀取失敗',str(exc),parent=self.root); return
        self.images[side], self.paths[side], self.results[side] = image, path, None
        self.draw(); self.home(); self.recompute()

    def settings(self):
        if self.future is not None:
            return
        if self.mode.get() == 'Fixed':
            messagebox.showinfo('Fixed','Fixed 使用共用模組的固定門檻；可切換 Legacy 或 V2 調整參數。',parent=self.root)
            return
        mode = self.mode.get()
        def apply(config):
            if mode == 'V2': self.v2 = config
            else: self.legacy = config
            self.mode.set(mode)
            self.recompute()
        if mode == 'V2': show_specular_model_v2_settings(self.root,self.v2,apply)
        else: show_specular_settings(self.root,self.legacy,apply)

    def recompute(self):
        if self.future is not None:
            self.pending = True
            return
        if all(image is None for image in self.images):
            return
        self.pending = False
        self.status.set(f'{self.mode.get()} 偵測中…（縮放與平移仍可操作）')
        self.future = self.worker.submit(detect_pair, tuple(self.images), self.mode.get(), self.legacy, self.v2)
        self.root.after(40, self.poll)

    def poll(self):
        if self.closed: return
        if not self.future.done():
            self.root.after(40,self.poll); return
        future, self.future = self.future, None
        try:
            self.results = future.result()
            parts = []
            for label, item in zip(('左','右'),self.results):
                if item is not None:
                    mask, ms = item
                    parts.append(f'{label}：{np.count_nonzero(mask)}/{mask.size} px ({100*np.count_nonzero(mask)/mask.size:.2f}%)，{ms:.1f} ms')
            self.status.set(' | '.join(parts))
            self.draw()
        except Exception as exc:
            self.results = [None,None]
            self.draw()
            messagebox.showerror('偵測失敗',str(exc),parent=self.root)
            self.status.set('偵測失敗，請檢查圖片或參數。')
        if self.pending: self.recompute()

    def draw(self):
        if not hasattr(self,'artists'): return
        for i, (image, result, ax, artist) in enumerate(zip(self.images,self.results,self.axes,self.artists)):
            if image is None: continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if result is not None:
                mask = result[0]
                if self.display.get() == '疊圖':
                    color = np.empty_like(rgb)
                    color[:] = (0, 0, 160)
                    selected = mask > 0
                    blend = cv2.addWeighted(rgb, 1-self.alpha.get(), color, self.alpha.get(), 0)
                    rgb[selected] = blend[selected]
                elif self.display.get() == '遮罩': rgb = np.repeat(mask[...,None],3,axis=2)
            artist.set_data(rgb)
            h,w = image.shape[:2]
            # set_extent updates shared limits unless autoscale is disabled.
            artist.set_extent((-.5,w-.5,h-.5,-.5))
            ax.set_title(f'{"Left" if i == 0 else "Right"} | {Path(self.paths[i]).name} | {w} x {h}', color='white')
        self.canvas.draw_idle()

    def home(self):
        sizes = [image.shape[:2] for image in self.images if image is not None]
        h = max((s[0] for s in sizes),default=600); w = max((s[1] for s in sizes),default=800)
        self.axes[0].set_xlim(-.5,w-.5); self.axes[0].set_ylim(h-.5,-.5)
        for ax in self.axes: ax.set_autoscale_on(False)
        self.canvas.draw_idle()

    def scroll(self,event):
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None: return
        scale = 1.2 ** (-event.step)
        xlim,ylim = event.inaxes.get_xlim(),event.inaxes.get_ylim()
        if not 1 <= abs(xlim[1]-xlim[0])*scale <= 1e7: return
        event.inaxes.set_xlim([event.xdata+(v-event.xdata)*scale for v in xlim])
        event.inaxes.set_ylim([event.ydata+(v-event.ydata)*scale for v in ylim])
        self.canvas.draw_idle()

    def press(self,event):
        if event.button != 1 or event.inaxes not in self.axes: return
        if event.dblclick: self.home(); return
        ax = event.inaxes
        self.drag = (ax, ax.transData.frozen().inverted(), (event.x,event.y), ax.get_xlim(), ax.get_ylim())

    def motion(self,event):
        if self.drag is None or event.x is None or event.y is None: return
        ax, inverse, start, xlim, ylim = self.drag
        dx,dy = inverse.transform((event.x,event.y))-inverse.transform(start)
        ax.set_xlim(np.asarray(xlim)-dx); ax.set_ylim(np.asarray(ylim)-dy)
        self.canvas.draw_idle()

    def release(self,event): self.drag = None

    def export(self):
        if self.future is not None or not any(r is not None for r in self.results): return
        folder = filedialog.askdirectory(parent=self.root,title='選擇遮罩輸出資料夾')
        if not folder: return
        try:
            stamp = time.time_ns()
            for label,result in zip(('left','right'),self.results):
                if result is None: continue
                ok,buffer = cv2.imencode('.png',result[0])
                if not ok: raise ValueError('PNG encoding failed')
                with (Path(folder)/f'{label}_spatial_mask_{stamp}.png').open('xb') as stream:
                    stream.write(buffer.tobytes())
            self.status.set(f'已匯出原尺寸二值遮罩：{folder}')
        except Exception as exc: messagebox.showerror('匯出失敗',str(exc),parent=self.root)

    def close(self):
        self.closed = True
        self.worker.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    try: SpecularPairUI(root)
    except Exception as exc:
        messagebox.showerror('啟動失敗',str(exc),parent=root)
        root.destroy()
        raise
    root.mainloop()


if __name__ == '__main__': main()
