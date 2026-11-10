"""Select two video frames for GradSIFTDebug's left/right image input.

Run: Scripts/python.exe video_image_pair_exporter.py [video_path]
Only previews are resized. Exports preserve the original decoded BGR pixels.
"""
from datetime import datetime
from pathlib import Path
import argparse
import base64
import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2


def read_frame(capture, index, count):
    if not 0 <= index < count:
        raise ValueError(f'幀號必須介於 0～{count-1}')
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
        raise ValueError(f'無法定位影格 {index}')
    ok, frame = capture.read()
    if not ok or frame is None:
        raise ValueError(f'影格 {index} 解碼失敗')
    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if not math.isfinite(position) or abs(position-(index+1)) > .5:
        raise ValueError(f'影片解碼器回報的位置不符：要求 {index}，讀取後位置 {position}')
    return frame


def export_pair(video_path, left, right, output_parent, fps, *, sources=None):
    """left/right=(zero-based index, original decoded frame); never overwrite."""
    if left is None or right is None:
        raise ValueError('請先選定左圖與右圖')
    sources = sources or {key: (video_path, fps) for key in ('left', 'right')}
    same_video = Path(sources['left'][0]).resolve() == Path(sources['right'][0]).resolve()
    if same_video and left[0] == right[0]:
        raise ValueError('左右圖不可選同一影格；三角測距需要不同視角')
    if left[1].shape != right[1].shape:
        raise ValueError('左右影格尺寸不同')
    folder = Path(output_parent)/('image_pair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    written = []
    try:
        metadata = dict(frame_index_base=0,
                        pixels='original decoded BGR; no resize, undistortion or overlays', images={})
        for label, (index, frame) in [('left', left), ('right', right)]:
            source_path, source_fps = sources[label]
            path = folder/f'{label}_F{index:06d}.png'
            ok, data = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise ValueError('PNG 編碼失敗')
            written.append(path)
            with path.open('xb') as stream:
                stream.write(data.tobytes())
            metadata['images'][label] = dict(file=path.name, frame_index=int(index),
                source_video=str(Path(source_path).resolve()),
                fps=source_fps if math.isfinite(source_fps) and source_fps > 0 else None,
                nominal_time_s=index/source_fps if math.isfinite(source_fps) and source_fps > 0 else None,
                width=int(frame.shape[1]), height=int(frame.shape[0]))
        manifest = folder/'source.json'
        written.append(manifest)
        with manifest.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
    except Exception:
        # Only remove files created by this export, in its exclusive directory.
        for path in written:
            path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    return folder


class ImagePairExporter:
    def __init__(self, root, video=None):
        self.root = root
        root.title('影片左右圖選取・無損 PNG')
        root.geometry('1220x760')
        root.minsize(850, 580)
        self.capture = None
        self.video = None
        self.count = 0
        self.fps = 0.
        self.current = None
        self.selected = {'left': None, 'right': None}
        self.sources = {'left': None, 'right': None}
        self.pending = None
        self.resize_job = None
        self.syncing = False
        self.photos = {}
        self.info = tk.StringVar(value='請載入影片。幀號從 0 開始。')
        self.status = tk.StringVar(value='可從不同影片選左／右圖；切換影片會保留已選圖片。')
        self.entry = tk.StringVar(value='0')
        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='載入／切換影片', command=self.open_video).pack(side='left')
        for delta in (-10, -1, 1, 10):
            ttk.Button(toolbar, text=f'{delta:+d} 幀', command=lambda d=delta: self.step(d)).pack(side='left', padx=3)
        ttk.Label(toolbar, text='幀號：').pack(side='left', padx=(10, 0))
        entry = ttk.Entry(toolbar, textvariable=self.entry, width=8)
        entry.pack(side='left')
        entry.bind('<Return>', lambda event: self.goto())
        ttk.Button(toolbar, text='跳至', command=self.goto).pack(side='left', padx=4)
        ttk.Label(root, textvariable=self.info, padding=8).pack(fill='x')
        self.slider = tk.Scale(root, from_=0, to=0, orient='horizontal', showvalue=False,
                               command=self.slide, state='disabled')
        self.slider.pack(fill='x', padx=10)
        content = ttk.Frame(root, padding=8)
        content.pack(fill='both', expand=True)
        content.columnconfigure(0, weight=2, uniform='preview')
        content.columnconfigure(1, weight=1, uniform='preview')
        content.rowconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        self.labels = {}
        for key, title, row, column, rowspan in [('current', '目前影格', 0, 0, 2),
                ('left', '左圖：尚未選擇', 0, 1, 1), ('right', '右圖：尚未選擇', 1, 1, 1)]:
            box = ttk.LabelFrame(content, text=title)
            box.grid(row=row, column=column, rowspan=rowspan, sticky='nsew', padx=4, pady=4)
            # Fixed grid request prevents PhotoImage sizes driving window growth.
            box.grid_propagate(False)
            box.columnconfigure(0, weight=1)
            box.rowconfigure(0, weight=1)
            label = tk.Label(box, bg='#222222', fg='white', text='尚無影像')
            label.grid(row=0, column=0, sticky='nsew')
            self.labels[key] = (box, label)
            label.bind('<Configure>', self.resize)
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill='x')
        for key, name in [('left', '左圖'), ('right', '右圖')]:
            ttk.Button(controls, text=f'目前影格 → {name}', command=lambda k=key: self.choose(k)).pack(side='left', padx=5)
        ttk.Button(controls, text='左右互換', command=self.swap).pack(side='left', padx=5)
        self.save_button = ttk.Button(controls, text='輸出左右 PNG…', command=self.save, state='disabled')
        self.save_button.pack(side='right', padx=5)
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1100).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW', self.close)
        if video:
            root.after(50, lambda: self.open_video(video))

    def cancel_seek(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None

    def open_video(self, path=None):
        path = path or filedialog.askopenfilename(parent=self.root, title='載入影片',
            filetypes=[('影片', '*.mp4 *.avi *.mkv *.mov'), ('所有檔案', '*.*')])
        if not path:
            return
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count <= 0:
                raise ValueError('無法開啟影片或取得影格數')
            frame = read_frame(cap, 0, count)
        except (ValueError, cv2.error) as exc:
            cap.release()
            messagebox.showerror('載入失敗', str(exc), parent=self.root)
            return
        self.cancel_seek()
        if self.capture is not None:
            self.capture.release()
        self.capture, self.video = cap, Path(path).resolve()
        self.count, self.fps = count, float(cap.get(cv2.CAP_PROP_FPS))
        self.slider.configure(to=count-1, state='normal')
        self.set_current(0, frame)
        self.refresh_selected()
        self.status.set('已切換影片，保留先前選取；可將目前影片的影格設為左圖或右圖。')

    def set_current(self, index, frame):
        self.current = (index, frame)
        self.syncing = True
        self.slider.set(index)
        self.syncing = False
        self.entry.set(str(index))
        self.info.set(f'{self.video.name}  |  {frame.shape[1]}×{frame.shape[0]}  |  F{index} / {self.count-1}  |  FPS={self.fps:.3f}')
        self.labels['current'][0].configure(text=f'目前影格 F{index}')
        self.render('current', frame)

    def slide(self, value):
        if self.syncing or self.capture is None:
            return
        index = int(float(value))
        if self.current is not None and self.current[0] == index:
            return
        self.cancel_seek()
        self.pending = self.root.after(100, lambda: self.show(index))

    def show(self, index):
        self.cancel_seek()
        if self.capture is None:
            return
        try:
            frame = read_frame(self.capture, index, self.count)
            self.set_current(index, frame)
            self.status.set(f'已讀取 F{index}；可設為左圖或右圖。')
        except (ValueError, cv2.error) as exc:
            self.current = None
            self.render('current', None)
            self.status.set(str(exc))

    def goto(self):
        try:
            index = int(self.entry.get())
        except ValueError:
            self.status.set('請輸入整數幀號')
            return
        self.show(index)

    def step(self, delta):
        if self.capture is not None:
            self.show(max(0, min(self.count-1, int(self.slider.get())+delta)))

    def choose(self, key):
        # Flush slider debounce so a click cannot capture the previously shown frame.
        if self.pending is not None:
            self.show(int(self.slider.get()))
        if self.current is None:
            return
        self.selected[key] = (self.current[0], self.current[1].copy())
        self.sources[key] = (self.video, self.fps)
        self.refresh_selected()

    def swap(self):
        self.selected['left'], self.selected['right'] = self.selected['right'], self.selected['left']
        self.sources['left'], self.sources['right'] = self.sources['right'], self.sources['left']
        self.refresh_selected()

    def refresh_selected(self):
        for key, title in [('left', '左圖'), ('right', '右圖')]:
            item = self.selected[key]
            self.labels[key][0].configure(text=f'{title}：{Path(self.sources[key][0]).name} · F{item[0]}' if item is not None else f'{title}：尚未選擇')
            self.render(key, item[1] if item is not None else None)
        left, right = self.selected['left'], self.selected['right']
        both = left is not None and right is not None
        same_frame = both and self.sources['left'][0] == self.sources['right'][0] and left[0] == right[0]
        valid = both and not same_frame and left[1].shape == right[1].shape
        self.save_button.configure(state='normal' if valid else 'disabled')
        if both and not valid:
            self.status.set('左右圖是同一影片的同一影格，請選擇不同視角。' if same_frame else '左右圖尺寸不同，無法輸出給目前的量測程式。')

    def render(self, key, frame):
        label = self.labels[key][1]
        if frame is None:
            label.configure(image='', text='尚無影像')
            self.photos.pop(key, None)
            return
        h, w = frame.shape[:2]
        scale = min(max(1, label.winfo_width()-8)/w, max(1, label.winfo_height()-8)/h, 1.)
        preview = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_LINEAR)
        ok, data = cv2.imencode('.png', preview)
        if ok:
            self.photos[key] = tk.PhotoImage(data=base64.b64encode(data).decode('ascii'), format='png')
            label.configure(image=self.photos[key], text='')

    def resize(self, event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(120, self.redraw)

    def redraw(self):
        self.resize_job = None
        self.render('current', self.current[1] if self.current is not None else None)
        self.refresh_selected()

    def save(self):
        parent = filedialog.askdirectory(parent=self.root, title='選擇輸出位置（自動建立獨立資料夾）',
                                         initialdir=str(self.video.parent))
        if not parent:
            return
        try:
            folder = export_pair(self.video, self.selected['left'], self.selected['right'], parent, self.fps,
                                 sources=self.sources)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror('輸出失敗', str(exc), parent=self.root)
            return
        self.status.set(f'已輸出原始解析度 PNG：{folder}')
        messagebox.showinfo('輸出完成', f'{folder}\n\n主程式依序載入 left_*.png、right_*.png。', parent=self.root)

    def close(self):
        self.cancel_seek()
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


class VideoPane(ttk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, padding=8)
        self.cap = None
        self.path = None
        self.frame = None
        self.index = 0
        self.count = 0
        self.fps = 0.
        self.zoom = 1.
        self.pan = [0., 0.]
        self.sync = False
        bar = ttk.Frame(self); bar.pack(fill='x')
        ttk.Button(bar, text=f'載入{title}影片', command=self.load).pack(side='left')
        ttk.Button(bar, text='重設視野', command=self.reset_view).pack(side='right')
        self.caption = tk.StringVar(value='尚未載入')
        ttk.Label(self, textvariable=self.caption, wraplength=500).pack(fill='x')
        self.canvas = tk.Canvas(self, bg='#222222', width=480, height=420, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.draw())
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<ButtonPress-1>', self.start_drag)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<Double-Button-1>', lambda e: self.reset_view())
        self.slider = tk.Scale(self, from_=0, to=0, orient='horizontal', command=self.slide)
        self.slider.pack(fill='x')
        controls = ttk.Frame(self); controls.pack(fill='x')
        for delta in (-10,-1,1,10):
            ttk.Button(controls, text=f'{delta:+d}', command=lambda d=delta:self.show(max(0,min(self.count-1,self.index+d)))).pack(side='left')
        self.number = tk.StringVar(value='0')
        entry=ttk.Entry(controls,textvariable=self.number,width=8);entry.pack(side='left')
        entry.bind('<Return>',lambda e:self.goto())
        ttk.Button(controls,text='跳至幀號',command=self.goto).pack(side='left')

    def load(self):
        path=filedialog.askopenfilename(title=self['text']+'影片',filetypes=[('影片','*.mp4 *.avi *.mkv *.mov'),('所有檔案','*.*')])
        if not path:return
        cap=cv2.VideoCapture(path)
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try: frame=read_frame(cap,0,count)
        except (ValueError,cv2.error) as exc:
            cap.release();messagebox.showerror('載入失敗',str(exc));return
        if self.cap is not None:self.cap.release()
        self.cap,self.path,self.count,self.fps=cap,Path(path).resolve(),count,float(cap.get(cv2.CAP_PROP_FPS))
        self.slider.configure(to=count-1)
        self.zoom=1.;self.pan=[0.,0.]
        self.show(0)

    def slide(self,value):
        if not self.sync:self.show(int(float(value)))

    def goto(self):
        try:self.show(int(self.number.get()))
        except ValueError:messagebox.showerror('幀號','請輸入整數')

    def show(self,index):
        if self.cap is None:return
        try:frame=read_frame(self.cap,index,self.count)
        except (ValueError,cv2.error) as exc:
            messagebox.showerror('讀取失敗',str(exc));return
        self.frame,self.index=frame,index
        self.sync=True;self.slider.set(index);self.sync=False
        self.number.set(str(index))
        self.caption.set(f'{self.path.name} | F{index}/{self.count-1} | {frame.shape[1]}×{frame.shape[0]}')
        self.draw()

    def reset_view(self):
        self.zoom=1.;self.pan=[0.,0.];self.draw()

    def wheel(self,event):
        if self.frame is None:return
        old=self.zoom;self.zoom=max(1.,min(20.,old*(1.2 if event.delta>0 else 1/1.2)))
        ratio=self.zoom/old
        for i,value in enumerate((event.x-self.canvas.winfo_width()/2,event.y-self.canvas.winfo_height()/2)):
            self.pan[i]=value-(value-self.pan[i])*ratio
        self.draw()

    def start_drag(self,event):self.last=(event.x,event.y)

    def drag(self,event):
        self.pan[0]+=event.x-self.last[0];self.pan[1]+=event.y-self.last[1]
        self.last=(event.x,event.y);self.draw()

    def draw(self):
        if self.frame is None:return
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        ih,iw=self.frame.shape[:2];scale=min(w/iw,h/ih)*self.zoom
        import numpy as np
        transform=np.array([[scale,0,(w-iw*scale)/2+self.pan[0]],[0,scale,(h-ih*scale)/2+self.pan[1]]],dtype=float)
        preview=cv2.warpAffine(self.frame,transform,(w,h),flags=cv2.INTER_LINEAR,borderValue=(34,34,34))
        data=cv2.imencode('.png',preview)[1]
        self.photo=tk.PhotoImage(data=base64.b64encode(data).decode('ascii'),format='png')
        self.canvas.delete('all');self.canvas.create_image(0,0,anchor='nw',image=self.photo)


def dual_main():
    root=tk.Tk();root.title('左右影片選幀・原始 PNG 輸出');root.geometry('1280x760')
    root.minsize(1000,600)
    body=ttk.Frame(root);body.pack(fill='both',expand=True)
    body.columnconfigure(0,weight=1,uniform='panes');body.columnconfigure(1,weight=1,uniform='panes');body.rowconfigure(0,weight=1)
    left,right=VideoPane(body,'左圖'),VideoPane(body,'右圖')
    left.grid(row=0,column=0,sticky='nsew');right.grid(row=0,column=1,sticky='nsew')
    def save():
        if left.frame is None or right.frame is None:
            messagebox.showwarning('尚未選定','請分別載入左右影片並選幀');return
        parent=filedialog.askdirectory(title='選擇輸出位置')
        if not parent:return
        try:
            folder=export_pair(left.path,(left.index,left.frame),(right.index,right.frame),parent,left.fps,
                sources={'left':(left.path,left.fps),'right':(right.path,right.fps)})
            messagebox.showinfo('輸出完成',str(folder))
        except (ValueError,OSError,cv2.error) as exc:messagebox.showerror('輸出失敗',str(exc))
    ttk.Label(root,text='滾輪：游標位置縮放　左鍵拖曳：平移　雙擊：重設視野　輸出完整原圖，不受預覽縮放影響').pack(pady=5)
    ttk.Button(root,text='輸出目前左／右影格為 PNG',command=save).pack(pady=8)
    def close():
        for pane in (left,right):
            if pane.cap is not None:pane.cap.release()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW',close);root.mainloop()


if __name__ == '__main__':
    dual_main()


def legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', nargs='?')
    args = parser.parse_args()
    root = tk.Tk()
    ImagePairExporter(root, args.video)
    root.mainloop()
