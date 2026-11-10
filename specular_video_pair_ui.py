"""Standalone dual-video spatial-mask UI; shares the image viewer and detectors.
Run from this project: Scripts/python.exe specular_video_pair_ui.py
Playback is processing-paced (no dropped analysis frames), without audio.
"""
from pathlib import Path
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from specular_image_pair_ui import SpecularPairUI, detect_pair


DARK_BLUE_BGR = (160, 0, 0)


def normalize_roi(start, end, size):
    """Clip to shared pixel bounds, then shrink right/bottom to even dimensions."""
    w,h = size
    if w < 2 or h < 2: return None
    x0 = int(np.clip(np.floor(min(start[0],end[0])+.5),0,w))
    y0 = int(np.clip(np.floor(min(start[1],end[1])+.5),0,h))
    x1 = int(np.clip(np.ceil(max(start[0],end[0])+.5),0,w))
    y1 = int(np.clip(np.ceil(max(start[1],end[1])+.5),0,h))
    x1 -= (x1-x0)%2; y1 -= (y1-y0)%2
    return (x0,y0,x1,y1) if x1-x0 >= 2 and y1-y0 >= 2 else None


def render_video_frame(image, mask, display, alpha):
    """One shared renderer for preview and exported frames (uint8 BGR)."""
    if mask is None or display == '原圖': return image.copy()
    if display == '遮罩': return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = image.copy()
    selected = mask > 0
    color = np.empty_like(image); color[:] = DARK_BLUE_BGR
    blend = cv2.addWeighted(image, 1-float(alpha), color, float(alpha), 0)
    out[selected] = blend[selected]
    return out


def export_videos(paths, output_dir, extension, mode, legacy, v2, display, alpha,
                  cancel, progress, roi=None):
    """Export independent full-length videos using private decoders; no audio.

    All files stay .partial until both videos finish. Cleanup only files created
    by this job; the caller supplies a new, exclusive output directory.
    """
    output_dir = Path(output_dir)
    codec = 'mp4v' if extension == 'mp4' else 'MJPG'
    pending = []
    completed = False
    sources = []
    writer = None
    try:
        for side, path in enumerate(paths):
            if path is not None: sources.append((side, VideoSource(path)))
        total = sum(source.count for _,source in sources)
        done = 0
        for side,source in sources:
            label = 'left' if side == 0 else 'right'
            target = output_dir / f'{label}_{display}.{extension}'
            partial = output_dir / f'{label}.partial.{extension}'
            if target.exists() or partial.exists(): raise FileExistsError(str(target))
            pending.append((partial,target))
            h,w = source.image.shape[:2]
            if roi is not None:
                x0,y0,x1,y1 = roi
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                    raise ValueError(f'ROI 超出 {label} 影片範圍')
                w,h = x1-x0,y1-y0
            if w % 2 or h % 2:
                raise ValueError('影片尺寸含奇數邊長；為避免編碼器靜默裁切，請先轉成偶數尺寸。')
            writer = cv2.VideoWriter(str(partial),cv2.VideoWriter_fourcc(*codec),source.fps,(w,h))
            if not writer.isOpened(): raise ValueError(f'無法建立 {extension.upper()} 編碼器，請改用另一種格式。')
            for index in range(source.count):
                if cancel.is_set(): return dict(cancelled=True, files=[])
                image,_ = source.read_at(index/source.fps)
                mask = None if display == '原圖' else detect_pair([image],mode,legacy,v2)[0][0]
                rendered = render_video_frame(image,mask,display,alpha)
                if roi is not None: rendered = rendered[y0:y1,x0:x1]
                writer.write(np.ascontiguousarray(rendered))
                done += 1
                if done == total or done % 5 == 0 or index == 0:
                    progress.put((done,total,label,index+1,source.count))
            writer.release(); writer = None
            # Catch silent writer failures before publishing any completed video.
            check = cv2.VideoCapture(str(partial))
            try:
                ok,_ = check.read()
                if not ok or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != source.count:
                    raise ValueError(f'匯出驗證失敗：{partial.name}')
            finally: check.release()
        if cancel.is_set(): return dict(cancelled=True, files=[])
        for partial,target in pending: partial.rename(target)
        completed = True
        return dict(cancelled=False,files=[str(target) for _,target in pending])
    finally:
        if writer is not None: writer.release()
        for _,source in sources: source.close()
        if not completed:
            for partial,_ in pending:
                if partial.exists(): partial.unlink()


class VideoSource:
    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        try:
            if not self.cap.isOpened(): raise ValueError(f'無法開啟影片：{path}')
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError('影片缺少有效 FPS／總幀數，無法建立同步時間軸。')
            self.count = int(count)
            ok, image = self.cap.read()
            if not ok or image is None: raise ValueError('無法解碼影片第一幀。')
            self.index, self.image = 0, image
        except Exception:
            self.cap.release()
            raise

    @property
    def last_time(self): return (self.count-1)/self.fps

    def read_at(self, seconds):
        index = min(self.count-1, max(0, int(math.floor(seconds*self.fps+.5))))
        if index != self.index:
            if index != self.index+1:
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES,index):
                    raise ValueError(f'無法跳至 frame {index}: {self.path}')
            ok, image = self.cap.read()
            if not ok or image is None: raise ValueError(f'解碼失敗 frame {index}: {self.path}')
            self.index, self.image = index, image
        return self.image, index

    def close(self): self.cap.release()


def process_time(sources, seconds, mode, legacy, v2):
    images, indices = [], []
    for source in sources:
        image, index = (None,None) if source is None else source.read_at(seconds)
        images.append(image); indices.append(index)
    return images, indices, detect_pair(images,mode,legacy,v2)


class SpecularVideoPairUI(SpecularPairUI):
    def __init__(self, root):
        self.roi = None
        self.roi_selecting = False
        self.roi_drag = None
        self.roi_patches = []
        self.sources = [None,None]
        self.indices = [None,None]
        self.position = 0.
        self.requested = 0.
        self.playing = False
        self.queued = False
        self.timer = None
        self.slider_timer = None
        self.setting_slider = False
        self.reset_view = False
        self.export_future = None
        self.export_cancel = threading.Event()
        self.export_progress = queue.SimpleQueue()
        super().__init__(root)
        root.title('雙影片空間反光偵測 — 同步時間／縮放／平移')
        # Reuse image-viewer widgets and all detector/settings implementations.
        bar = root.winfo_children()[0]
        for widget in bar.winfo_children():
            if isinstance(widget,ttk.Button):
                label = widget.cget('text')
                if label == '載入左圖': widget.configure(text='載入左影片')
                elif label == '載入右圖': widget.configure(text='載入右影片')
                elif label == '重算兩圖': widget.configure(text='重算目前幀')
        for widget in root.winfo_children():
            if isinstance(widget,ttk.Label) and '淺藍' in str(widget.cget('text')):
                widget.configure(text='深藍＝空間反光排除區；V2 uncertain 不列入。不執行去畸變、時間反光或 warp。')
        controls = ttk.Frame(root,padding=6)
        controls.pack(fill='x',before=self.canvas.get_tk_widget())
        self.play_button = ttk.Button(controls,text='播放',command=self.toggle_play)
        self.play_button.pack(side='left',padx=3)
        ttk.Button(controls,text='上一幀',command=lambda:self.step(-1)).pack(side='left',padx=3)
        ttk.Button(controls,text='下一幀',command=lambda:self.step(1)).pack(side='left',padx=3)
        ttk.Label(controls,text='跳至秒數').pack(side='left',padx=(12,3))
        self.time_entry = ttk.Entry(controls,width=10)
        self.time_entry.insert(0,'0'); self.time_entry.pack(side='left')
        self.time_entry.bind('<Return>',lambda e:self.jump())
        ttk.Button(controls,text='跳轉',command=self.jump).pack(side='left',padx=3)
        self.clock_label = ttk.Label(controls,text='0.000 / 0.000 s')
        self.clock_label.pack(side='right')
        export_bar = ttk.Frame(root,padding=5)
        export_bar.pack(fill='x',before=self.canvas.get_tk_widget())
        self.export_format = tk.StringVar(value='mp4')
        ttk.Combobox(export_bar,textvariable=self.export_format,values=('mp4','avi'),
                     state='readonly',width=5).pack(side='left',padx=3)
        self.export_button = ttk.Button(export_bar,text='匯出左右影片',command=self.export_movies)
        self.export_button.pack(side='left',padx=3)
        ttk.Button(export_bar,text='取消匯出',command=self.export_cancel.set).pack(side='left',padx=3)
        self.export_message = tk.StringVar(value='完整左右影片分開匯出；保留各自 FPS／尺寸，使用目前顯示模式與透明度，無音訊。')
        ttk.Label(export_bar,textvariable=self.export_message).pack(side='left',padx=8)
        roi_bar = ttk.Frame(root,padding=5)
        roi_bar.pack(fill='x',before=self.canvas.get_tk_widget())
        self.roi_button = ttk.Button(roi_bar,text='框選錄影 ROI',command=self.toggle_roi)
        self.roi_button.pack(side='left',padx=3)
        ttk.Button(roi_bar,text='清除 ROI／匯出全圖',command=self.clear_roi).pack(side='left',padx=3)
        self.roi_message = tk.StringVar(value='匯出全圖；框選時兩邊同像素座標同步，ROI 寬高自動對齊偶數。')
        ttk.Label(roi_bar,textvariable=self.roi_message).pack(side='left',padx=8)
        from matplotlib.patches import Rectangle
        for ax in self.axes:
            patch = Rectangle((0,0),0,0,fill=False,edgecolor='#FFFF00',linewidth=1.6,zorder=20,visible=False)
            ax.add_patch(patch); self.roi_patches.append(patch)
        root.bind('<Escape>',lambda e:self.cancel_roi())
        self.slider = ttk.Scale(root,from_=0,to=1,command=self.slider_seek)
        self.slider.pack(fill='x',padx=12,before=self.canvas.get_tk_widget())
        ttk.Label(root,text='依 FPS 換算時間同步（不同起拍時間需先對齊）；逐格以左片 FPS 為準。播放逐步完成偵測，較慢時不保證即時速度。',padding=5).pack(fill='x',before=self.canvas.get_tk_widget())
        self.status.set('載入一或兩支影片。滾輪／拖曳任一圖，另一圖同步；雙擊恢復全圖。')

    def common_size(self):
        shapes = [im.shape[:2] for im in self.images if im is not None]
        return (min(s[1] for s in shapes),min(s[0] for s in shapes)) if shapes else (0,0)

    def update_roi_overlay(self, roi):
        for patch in self.roi_patches:
            patch.set_visible(roi is not None)
            if roi is not None:
                x0,y0,x1,y1 = roi
                patch.set_bounds(x0-.5,y0-.5,x1-x0,y1-y0)
        if hasattr(self,'canvas'): self.canvas.draw_idle()

    def toggle_roi(self):
        if self.roi_selecting: self.cancel_roi(); return
        if not any(im is not None for im in self.images): return
        self.pause(); self.drag = None
        self.roi_selecting = True
        self.roi_button.configure(text='取消框選 (Esc)')
        self.roi_message.set('請在任一圖按住左鍵拖出 ROI；兩側同步，限共同有效畫面；放開後恢復平移。')

    def cancel_roi(self):
        self.roi_selecting = False; self.roi_drag = None; self.drag = None
        if hasattr(self,'roi_button'): self.roi_button.configure(text='框選錄影 ROI')
        self.update_roi_overlay(self.roi)
        if hasattr(self,'roi_message'):
            if self.roi is None: self.roi_message.set('匯出全圖；框選時兩邊同像素座標同步，ROI 寬高自動對齊偶數。')
            else:
                x0,y0,x1,y1 = self.roi
                self.roi_message.set(f'匯出 ROI：x={x0}, y={y0}, 寬={x1-x0}, 高={y1-y0} px（左右相同；全圖偵測後裁切）')

    def clear_roi(self):
        self.roi = None
        self.cancel_roi()

    def press(self,event):
        if not self.roi_selecting: return super().press(event)
        if event.button != 1 or event.inaxes not in self.axes or event.xdata is None: return
        side = self.axes.index(event.inaxes)
        if self.images[side] is None: return
        self.roi_drag = (event.inaxes.transData.frozen().inverted(),(event.xdata,event.ydata))

    def motion(self,event):
        if self.roi_drag is None:
            if not self.roi_selecting: super().motion(event)
            return
        if event.x is None or event.y is None: return
        inverse,start = self.roi_drag
        end = inverse.transform((event.x,event.y))
        self.update_roi_overlay(normalize_roi(start,end,self.common_size()))

    def release(self,event):
        if self.roi_drag is None: return super().release(event)
        inverse,start = self.roi_drag
        if event is not None and event.x is not None and event.y is not None:
            selected = normalize_roi(start,inverse.transform((event.x,event.y)),self.common_size())
            if selected is not None: self.roi = selected
        self.cancel_roi()

    def scroll(self,event):
        if self.roi_drag is None: super().scroll(event)

    def draw(self):
        if not hasattr(self,'artists'): return
        for i,(image,result,ax,artist) in enumerate(zip(self.images,self.results,self.axes,self.artists)):
            if image is None: continue
            mask = None if result is None else result[0]
            frame = render_video_frame(image,mask,self.display.get(),self.alpha.get())
            artist.set_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            h,w = image.shape[:2]
            artist.set_extent((-.5,w-.5,h-.5,-.5))
            ax.set_title(f'{"Left" if i == 0 else "Right"} | {Path(self.paths[i]).name} | {w} x {h}',color='white')
        self.canvas.draw_idle()

    def export_movies(self):
        if self.export_future is not None: return
        if not any(path is not None for path in self.paths):
            messagebox.showinfo('匯出影片','請先載入影片。',parent=self.root); return
        self.pause()
        folder = filedialog.askdirectory(parent=self.root,title='選擇匯出影片資料夾（自動建立獨立子資料夾）')
        if not folder: return
        try:
            destination = Path(folder)/f'specular_export_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000000:09d}'
            destination.mkdir(exist_ok=False)
        except Exception as exc:
            messagebox.showerror('無法建立輸出目錄',str(exc),parent=self.root); return
        self.export_cancel.clear()
        self.export_progress = queue.SimpleQueue()
        self.export_button.configure(state='disabled')
        self.export_message.set(f'匯出中：{self.mode.get()}／{self.display.get()}（參數及 ROI 已固定）')
        self.export_future = self.worker.submit(export_videos,tuple(self.paths),destination,
            self.export_format.get(),self.mode.get(),self.legacy,self.v2,
            self.display.get(),self.alpha.get(),self.export_cancel,self.export_progress,self.roi)
        self.root.after(100,self.poll_export)

    def poll_export(self):
        if self.closed: return
        latest = None
        while True:
            try: latest = self.export_progress.get_nowait()
            except queue.Empty: break
        if latest is not None:
            done,total,side,index,count = latest
            self.export_message.set(f'匯出 {100*done/total:.1f}% | {side} {index}/{count} | {done}/{total} frames')
        if not self.export_future.done(): self.root.after(100,self.poll_export); return
        future,self.export_future = self.export_future,None
        self.export_button.configure(state='normal')
        try:
            result = future.result()
            if result['cancelled']: self.export_message.set('已取消匯出，未完成檔案已移除。')
            else:
                self.export_message.set('匯出完成：'+str(Path(result['files'][0]).parent))
                messagebox.showinfo('匯出完成','\n'.join(result['files']),parent=self.root)
        except Exception as exc:
            self.export_message.set('匯出失敗，未完成檔案已移除。')
            messagebox.showerror('匯出失敗',str(exc),parent=self.root)

    def end_time(self):
        return min((s.last_time for s in self.sources if s is not None),default=0.)

    def master_fps(self):
        return next((s.fps for s in self.sources if s is not None),30.)

    def pause(self):
        self.playing = False
        if hasattr(self,'play_button'): self.play_button.configure(text='播放')
        if self.timer is not None:
            self.root.after_cancel(self.timer); self.timer = None

    def load(self,side):
        self.pause()
        if self.future is not None:
            self.status.set('目前幀處理中，完成後再載入影片。'); return
        path = filedialog.askopenfilename(parent=self.root,title='載入影片',
            filetypes=[('Video','*.mp4 *.avi *.mov *.mkv *.m4v *.wmv'),('All files','*.*')])
        if not path: return
        try: source = VideoSource(path)
        except Exception as exc:
            messagebox.showerror('影片載入失敗',str(exc),parent=self.root); return
        old = self.sources[side]
        if old is not None: old.close()
        self.sources[side] = source
        self.paths[side] = path
        self.images[side] = source.image
        self.clear_roi()
        self.results[side] = None
        self.reset_view = True
        self.slider.configure(to=max(self.end_time(),.001))
        self.seek(0.)

    def seek(self,seconds):
        if not math.isfinite(seconds): raise ValueError('秒數必須為有限數值')
        self.requested = float(np.clip(seconds,0,self.end_time()))
        self.recompute()

    def slider_seek(self,value):
        if self.setting_slider: return
        self.pause()
        if self.slider_timer is not None: self.root.after_cancel(self.slider_timer)
        self.slider_timer = self.root.after(100,lambda:self.finish_slider(float(value)))

    def finish_slider(self,value):
        self.slider_timer = None
        self.seek(value)

    def jump(self):
        self.pause()
        try: self.seek(float(self.time_entry.get()))
        except ValueError as exc: messagebox.showerror('時間格式錯誤',str(exc),parent=self.root)

    def step(self,direction):
        self.pause()
        self.seek(self.requested + direction/self.master_fps())

    def toggle_play(self):
        if self.playing: self.pause(); return
        if not any(s is not None for s in self.sources): return
        self.playing = True; self.play_button.configure(text='暫停')
        if self.position >= self.end_time()-1e-8: self.seek(0.)
        elif self.future is None: self.advance()

    def advance(self):
        self.timer = None
        if not self.playing or self.closed: return
        if self.position >= self.end_time()-1e-8:
            self.pause(); return
        self.seek(min(self.position+1/self.master_fps(),self.end_time()))

    def recompute(self):
        if not any(s is not None for s in self.sources): return
        if self.future is not None:
            self.queued = True; return
        self.queued = False
        self.job_time = self.requested
        self.job_started = time.perf_counter()
        self.status.set(f'{self.mode.get()} 處理 {self.job_time:.3f} s…')
        self.future = self.worker.submit(process_time,tuple(self.sources),self.job_time,
                                         self.mode.get(),self.legacy,self.v2)
        self.root.after(25,self.poll)

    def poll(self):
        if self.closed: return
        if not self.future.done(): self.root.after(25,self.poll); return
        future, self.future = self.future,None
        try:
            images,indices,results = future.result()
        except Exception as exc:
            self.queued = False; self.pause()
            self.status.set('處理失敗；保留上一組完整畫面。')
            messagebox.showerror('影片處理失敗',str(exc),parent=self.root); return
        if self.queued:
            self.recompute(); return  # Do not show a stale seek/mode result.
        self.images,self.indices,self.results = images,indices,results
        self.position = self.job_time
        self.setting_slider = True
        self.slider.set(self.position)
        self.setting_slider = False
        self.clock_label.configure(text=f'{self.position:.3f} / {self.end_time():.3f} s')
        self.time_entry.delete(0,'end'); self.time_entry.insert(0,f'{self.position:.6f}')
        self.draw()
        if self.reset_view: self.home(); self.reset_view = False
        parts = []
        for label,source,index,result in zip(('左','右'),self.sources,self.indices,self.results):
            if source is not None:
                mask,ms = result
                parts.append(f'{label} F{index}/{source.count-1} ({index/source.fps:.3f}s, {source.fps:g} FPS) 反光 {100*np.count_nonzero(mask)/mask.size:.2f}% / {ms:.1f}ms')
        self.status.set(' | '.join(parts))
        if self.playing:
            if self.position >= self.end_time()-1e-8: self.pause()
            else:
                remaining=1/self.master_fps()-(time.perf_counter()-self.job_started)
                self.timer=self.root.after(max(1,int(remaining*1000)),self.advance)

    def settings(self):
        self.pause()
        super().settings()

    def close(self):
        self.export_cancel.set()
        self.pause()
        self.closed = True
        if self.slider_timer is not None: self.root.after_cancel(self.slider_timer)
        # Release captures on the same worker, after any in-flight read completes.
        sources=tuple(self.sources)
        self.worker.submit(lambda:[s.close() for s in sources if s is not None])
        self.worker.shutdown(wait=False)
        self.root.destroy()


def main():
    root=tk.Tk()
    try: SpecularVideoPairUI(root)
    except Exception as exc:
        messagebox.showerror('啟動失敗',str(exc),parent=root)
        root.destroy(); raise
    root.mainloop()


if __name__ == '__main__': main()
