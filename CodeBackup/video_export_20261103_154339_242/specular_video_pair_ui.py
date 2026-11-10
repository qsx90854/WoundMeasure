"""Standalone dual-video spatial-mask UI; shares the image viewer and detectors.
Run from this project: Scripts/python.exe specular_video_pair_ui.py
Playback is processing-paced (no dropped analysis frames), without audio.
"""
from pathlib import Path
import math
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from specular_image_pair_ui import SpecularPairUI, detect_pair


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
        self.slider = ttk.Scale(root,from_=0,to=1,command=self.slider_seek)
        self.slider.pack(fill='x',padx=12,before=self.canvas.get_tk_widget())
        ttk.Label(root,text='依 FPS 換算時間同步（不同起拍時間需先對齊）；逐格以左片 FPS 為準。播放逐步完成偵測，較慢時不保證即時速度。',padding=5).pack(fill='x',before=self.canvas.get_tk_widget())
        self.status.set('載入一或兩支影片。滾輪／拖曳任一圖，另一圖同步；雙擊恢復全圖。')

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
