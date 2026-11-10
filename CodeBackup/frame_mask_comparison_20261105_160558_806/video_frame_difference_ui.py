"""Single-video six-panel frame difference / spatial-specular inspector.
Run: Scripts/python.exe video_frame_difference_ui.py
Frame indices are zero-based. Difference is abs(gray1-gray2) > threshold.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from specular_image_pair_ui import SpecularPairUI, read_main_defaults, detect_pair
from specular_video_pair_ui import VideoSource, render_video_frame
from Algorithm.specular_settings import show_specular_settings
from Algorithm.specular_model_v2_settings import show_specular_model_v2_settings


def roi_slice(shape,roi):
    h,w=shape[:2]
    x0,y0,x1,y1=(0,0,w,h) if roi is None else roi
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('ROI 超出影格範圍或沒有面積')
    return (slice(y0,y1),slice(x0,x1)),(x1-x0)*(y1-y0)


def frame_difference(first,second,threshold,roi=None):
    if first.shape != second.shape: raise ValueError('兩影格尺寸不同')
    if not 0 <= threshold <= 255: raise ValueError('門檻必須介於 0～255')
    region,_=roi_slice(first.shape,roi)
    difference=np.zeros(first.shape[:2],np.uint8)
    local=cv2.absdiff(cv2.cvtColor(first[region],cv2.COLOR_BGR2GRAY),cv2.cvtColor(second[region],cv2.COLOR_BGR2GRAY))
    difference[region]=local
    mask=np.zeros_like(difference);mask[region]=(local > threshold).astype(np.uint8)*255
    return difference,mask


def read_index(source,index):
    if not 0 <= index < source.count: raise ValueError(f'index 必須介於 0～{source.count-1}')
    return source.read_at(index/source.fps)[0].copy()


def analyze_pair(source,indices,mode,legacy,v2):
    images=[read_index(source,i) for i in indices]
    return images,detect_pair(images,mode,legacy,v2)


class FrameDifferenceUI(SpecularPairUI):
    # Reuse linked navigation only; do not initialize the two-image layout.
    def __init__(self,root):
        self.root=root;root.title('單影片：雙影格差異與空間反光偵測');root.geometry('1500x950')
        self.legacy,self.v2,enabled=read_main_defaults()
        self.source=None;self.images=[None,None,None];self.pair_results=None;self.pair_indices=None
        self.current_index=0;self.desired_current=0
        self.worker=ThreadPoolExecutor(max_workers=1);self.future=None;self.pending_pair=None;self.pending_current=None
        self.closed=False;self.playing=False;self.play_timer=None;self.slider_timer=None;self.setting_slider=False
        self.roi=None;self.roi_selecting=False;self.roi_drag=None;self.roi_patches=[]
        self.drag=None;self.pair_revision=0;self.active_pair_revision=0;self.job=None
        self.mode=tk.StringVar(value='V2' if enabled else 'Legacy');self.alpha=tk.DoubleVar(value=.55)
        self.display=tk.StringVar(value='疊圖');self.threshold=tk.StringVar(value='30')
        self.index1=tk.StringVar(value='0');self.index2=tk.StringVar(value='1');self.current=tk.StringVar(value='0')
        self.status=tk.StringVar(value='請載入影片；index 從 0 開始。滾輪縮放、左鍵拖曳，六個面板同步。')
        top=ttk.Frame(root,padding=5);top.pack(fill='x')
        for label,fn in [('載入影片',self.load),('播放／暫停',self.toggle_play),('上一幀',lambda:self.step(-1)),
                         ('下一幀',lambda:self.step(1)),('適合視窗',self.home)]:
            ttk.Button(top,text=label,command=fn).pack(side='left',padx=3)
        ttk.Label(top,text='目前 index').pack(side='left',padx=5)
        entry=ttk.Entry(top,textvariable=self.current,width=8);entry.pack(side='left');entry.bind('<Return>',lambda e:self.jump())
        ttk.Button(top,text='跳轉',command=self.jump).pack(side='left',padx=3)
        self.video_info=ttk.Label(top,text='');self.video_info.pack(side='left',padx=10)
        pair=ttk.Frame(root,padding=5);pair.pack(fill='x')
        for label,var,slot in [('index1',self.index1,1),('index2',self.index2,2)]:
            ttk.Label(pair,text=label).pack(side='left',padx=4)
            entry=ttk.Entry(pair,textvariable=var,width=8);entry.pack(side='left');entry.bind('<Return>',lambda e:self.apply_pair())
            ttk.Button(pair,text='取目前幀',command=lambda v=var:self.capture_current(v)).pack(side='left',padx=3)
        ttk.Button(pair,text='分析兩幀',command=self.apply_pair).pack(side='left',padx=5)
        ttk.Label(pair,text='灰階絕對差 >').pack(side='left',padx=5)
        entry=ttk.Entry(pair,textvariable=self.threshold,width=6);entry.pack(side='left');entry.bind('<Return>',lambda e:self.draw())
        ttk.Button(pair,text='更新門檻',command=self.draw).pack(side='left',padx=3)
        ttk.Label(pair,text='反光模式').pack(side='left',padx=5)
        combo=ttk.Combobox(pair,textvariable=self.mode,values=('Legacy','V2','Fixed'),width=8,state='readonly')
        combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.recompute())
        ttk.Button(pair,text='反光參數',command=self.settings).pack(side='left',padx=3)
        combo=ttk.Combobox(pair,textvariable=self.display,values=('疊圖','遮罩'),width=6,state='readonly')
        combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.draw())
        ttk.Scale(pair,from_=0,to=1,variable=self.alpha,command=lambda v:self.draw()).pack(side='left',padx=5)
        roi_bar=ttk.Frame(root,padding=5);roi_bar.pack(fill='x')
        self.roi_button=ttk.Button(roi_bar,text='框選 ROI',command=self.toggle_roi);self.roi_button.pack(side='left',padx=3)
        ttk.Button(roi_bar,text='清除 ROI／全圖',command=self.clear_roi).pack(side='left',padx=3)
        self.roi_text=tk.StringVar(value='分析全圖；可在任一面板框選，同步套用六個面板。')
        ttk.Label(roi_bar,textvariable=self.roi_text).pack(side='left',padx=8)
        self.slider=ttk.Scale(root,from_=0,to=1,command=self.slider_seek);self.slider.pack(fill='x',padx=12)
        ttk.Label(root,text='差異以原始灰階直接相減，不做影像對齊；白色＝差異大於門檻。深藍＝反光排除區。選定兩幀不會隨播放改變。').pack(anchor='w',padx=10)
        self.figure=Figure(figsize=(14,8),facecolor='#202020');self.axes=[];self.artists=[]
        titles=['Current frame','index1: original','index2: original','Absolute grayscale difference mask','index1: specular','index2: specular']
        for i,title in enumerate(titles):
            ax=self.figure.add_subplot(2,3,i+1,**(dict(sharex=self.axes[0],sharey=self.axes[0]) if self.axes else {}))
            ax.set_facecolor('#111111');ax.tick_params(colors='white');ax.set_title(title,color='white',fontsize=10)
            ax.set_aspect('equal',adjustable='box');self.axes.append(ax)
            self.artists.append(ax.imshow(np.zeros((1,1,3),np.uint8),origin='upper',interpolation='nearest'))
        from matplotlib.patches import Rectangle
        for ax in self.axes:
            box=Rectangle((0,0),0,0,fill=False,edgecolor='yellow',linewidth=1.5,zorder=20,visible=False)
            ax.add_patch(box);self.roi_patches.append(box)
        self.figure.subplots_adjust(left=.035,right=.98,bottom=.04,top=.94,wspace=.12,hspace=.22)
        self.canvas=FigureCanvasTkAgg(self.figure,master=root);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        for event,callback in [('scroll_event',self.scroll),('button_press_event',self.press),('motion_notify_event',self.motion),('button_release_event',self.release)]:self.canvas.mpl_connect(event,callback)
        ttk.Label(root,textvariable=self.status,padding=5).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close);root.bind('<Escape>',lambda e:self.cancel_roi())
        self.home()

    def selected_roi(self,start,end):
        image=next((im for im in self.images if im is not None),None)
        if image is None:return None
        h,w=image.shape[:2]
        x0=int(np.clip(np.floor(np.round(min(start[0],end[0])+.5,9)),0,w))
        y0=int(np.clip(np.floor(np.round(min(start[1],end[1])+.5,9)),0,h))
        x1=int(np.clip(np.ceil(np.round(max(start[0],end[0])+.5,9)),0,w))
        y1=int(np.clip(np.ceil(np.round(max(start[1],end[1])+.5,9)),0,h))
        return (x0,y0,x1,y1) if x1>x0 and y1>y0 else None

    def show_roi(self,roi):
        for box in self.roi_patches:
            box.set_visible(roi is not None)
            if roi is not None:
                x0,y0,x1,y1=roi;box.set_bounds(x0-.5,y0-.5,x1-x0,y1-y0)
        self.canvas.draw_idle()

    def toggle_roi(self):
        if self.roi_selecting:self.cancel_roi();return
        if all(im is None for im in self.images):return
        self.pause();self.drag=None;self.roi_selecting=True
        self.roi_button.configure(text='取消框選 (Esc)')
        self.roi_text.set('在任一已有影像的面板按住左鍵框選；放開即套用，Esc 取消。')

    def cancel_roi(self):
        self.roi_selecting=False;self.roi_drag=None;self.drag=None
        self.roi_button.configure(text='框選 ROI');self.show_roi(self.roi)
        if self.roi is None:self.roi_text.set('分析全圖；可在任一面板框選，同步套用六個面板。')
        else:
            x0,y0,x1,y1=self.roi
            self.roi_text.set(f'ROI x={x0}, y={y0}, 寬={x1-x0}, 高={y1-y0}；百分比分母={(x1-x0)*(y1-y0)} pixels。反光全圖偵測，僅顯示／統計 ROI。')

    def clear_roi(self):
        self.roi=None;self.cancel_roi();self.draw()

    def press(self,event):
        if not self.roi_selecting:return super().press(event)
        if event.button!=1 or event.inaxes not in self.axes or event.xdata is None:return
        panel=self.axes.index(event.inaxes)
        if (panel==0 and self.images[0] is None) or (panel>0 and self.pair_results is None):return
        self.roi_drag=(event.inaxes.transData.frozen().inverted(),(event.xdata,event.ydata))

    def motion(self,event):
        if self.roi_drag is None:
            if not self.roi_selecting:super().motion(event)
            return
        if event.x is None or event.y is None:return
        inverse,start=self.roi_drag;self.show_roi(self.selected_roi(start,inverse.transform((event.x,event.y))))

    def release(self,event):
        if self.roi_drag is None:return super().release(event)
        inverse,start=self.roi_drag
        if event is not None and event.x is not None and event.y is not None:
            end=inverse.transform((event.x,event.y))
            if np.linalg.norm(np.asarray(end)-start)>=1.:
                roi=self.selected_roi(start,end)
                if roi is not None:self.roi=roi
        self.cancel_roi();self.draw()

    def scroll(self,event):
        if self.roi_drag is None:super().scroll(event)

    def pause(self):
        self.playing=False
        if self.play_timer is not None:self.root.after_cancel(self.play_timer);self.play_timer=None

    def load(self):
        self.pause()
        if self.future is not None:self.status.set('處理中，請完成後再載入影片。');return
        path=filedialog.askopenfilename(parent=self.root,filetypes=[('Video','*.mp4 *.avi *.mov *.mkv *.m4v *.wmv'),('All','*.*')])
        if not path:return
        try:source=VideoSource(path)
        except Exception as exc:messagebox.showerror('載入失敗',str(exc),parent=self.root);return
        if self.slider_timer is not None:self.root.after_cancel(self.slider_timer);self.slider_timer=None
        if self.source is not None:self.source.close()
        self.source=source;self.pending_pair=None;self.pending_current=None
        self.images=[source.image.copy(),None,None];self.pair_results=None;self.pair_indices=None
        self.roi=None;self.cancel_roi()
        self.current_index=self.desired_current=0;self.current.set('0');self.index1.set('0');self.index2.set(str(min(1,source.count-1)))
        self.slider.configure(to=max(1,source.count-1));self.setting_slider=True;self.slider.set(0);self.setting_slider=False
        self.video_info.configure(text=f'{Path(path).name} | {source.count} frames | {source.fps:g} FPS')
        for artist in self.artists[1:]:artist.set_data(np.zeros((1,1,3),np.uint8))
        self.draw();self.home();self.apply_pair()

    def valid_index(self,text):
        index=int(text)
        if self.source is None:raise ValueError('請先載入影片')
        if not 0<=index<self.source.count:raise ValueError(f'index 必須介於 0～{self.source.count-1}')
        return index

    def capture_current(self,var):var.set(str(self.current_index));self.apply_pair()

    def apply_pair(self):
        self.pause()
        try:indices=(self.valid_index(self.index1.get()),self.valid_index(self.index2.get()))
        except ValueError as exc:messagebox.showerror('Index 錯誤',str(exc),parent=self.root);return
        self.pair_revision+=1
        self.pending_pair=(indices,self.mode.get(),self.legacy,self.v2,self.pair_revision)
        self.dispatch()

    def recompute(self):
        if self.source is not None:self.apply_pair()

    def settings(self):
        self.pause()
        mode=self.mode.get()
        if mode=='Fixed':messagebox.showinfo('Fixed','Fixed 使用固定門檻；Legacy／V2 可編輯參數。',parent=self.root);return
        def apply(config):
            if mode=='V2':self.v2=config
            else:self.legacy=config
            self.mode.set(mode);self.recompute()
        (show_specular_model_v2_settings if mode=='V2' else show_specular_settings)(self.root,self.v2 if mode=='V2' else self.legacy,apply)

    def seek(self,index):
        if self.source is None:return
        self.desired_current=int(np.clip(index,0,self.source.count-1));self.pending_current=self.desired_current;self.dispatch()

    def jump(self):
        self.pause()
        try:self.seek(self.valid_index(self.current.get()))
        except ValueError as exc:messagebox.showerror('Index 錯誤',str(exc),parent=self.root)

    def step(self,delta):self.pause();self.seek(self.desired_current+delta)

    def slider_seek(self,value):
        if self.setting_slider:return
        self.pause()
        if self.slider_timer is not None:self.root.after_cancel(self.slider_timer)
        self.slider_timer=self.root.after(80,lambda:self.finish_slider(int(float(value)+.5)))

    def finish_slider(self,index):self.slider_timer=None;self.seek(index)

    def toggle_play(self):
        if self.playing:self.pause();return
        if self.source is None:return
        self.playing=True
        self.seek(0 if self.current_index>=self.source.count-1 else self.current_index+1)

    def advance(self):
        self.play_timer=None
        if self.playing:self.seek(self.current_index+1)

    def dispatch(self):
        if self.future is not None or self.closed:return
        if self.pending_pair is not None:
            indices,mode,legacy,v2,revision=self.pending_pair;self.pending_pair=None
            self.job=('pair',indices,revision,mode);self.status.set(f'分析 F{indices[0]} / F{indices[1]}，{mode}…')
            self.future=self.worker.submit(analyze_pair,self.source,indices,mode,legacy,v2)
        elif self.pending_current is not None:
            index=self.pending_current;self.pending_current=None;self.job=('current',index)
            self.future=self.worker.submit(read_index,self.source,index)
        else:return
        self.started=time.perf_counter();self.root.after(20,self.poll)

    def poll(self):
        if self.closed:return
        if not self.future.done():self.root.after(20,self.poll);return
        future,self.future=self.future,None;job=self.job
        try:result=future.result()
        except Exception as exc:
            self.pause();messagebox.showerror('處理失敗',str(exc),parent=self.root);self.status.set('處理失敗，保留上次結果。');self.dispatch();return
        if job[0]=='pair' and job[2]==self.pair_revision:
            self.images[1:]=result[0];self.pair_results=result[1];self.pair_indices=job[1];self.pair_mode=job[3]
            self.status.set(f'已分析 F{job[1][0]} / F{job[1][1]} | {job[3]} | 左／右偵測 {result[1][0][1]:.1f}/{result[1][1][1]:.1f} ms')
        elif job[0]=='current' and job[1]==self.desired_current:
            self.images[0]=result;self.current_index=job[1];self.current.set(str(job[1]))
            self.setting_slider=True;self.slider.set(job[1]);self.setting_slider=False
        self.draw();self.dispatch()
        if self.playing and self.future is None:
            if self.current_index>=self.source.count-1:self.pause()
            else:self.play_timer=self.root.after(max(1,int((1/self.source.fps-(time.perf_counter()-self.started))*1000)),self.advance)

    def draw(self):
        if not hasattr(self,'artists'):return
        frames=[self.images[0],self.images[1],self.images[2],None,None,None]
        titles=[f'Current F{self.current_index}','index1: original','index2: original','abs(gray1-gray2) > threshold','index1: specular','index2: specular']
        if self.pair_results is not None:
            try:threshold=float(self.threshold.get());difference,mask=frame_difference(self.images[1],self.images[2],threshold,self.roi)
            except ValueError:
                self.status.set('門檻請輸入 0～255 的有效數字。');return
            frames[3]=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            region,area=roi_slice(mask.shape,self.roi)
            scope='ROI' if self.roi is not None else 'full'
            titles[3]=f'abs(gray1-gray2) > {threshold:g} | {scope}: {np.count_nonzero(mask)}/{area} ({100*np.count_nonzero(mask)/area:.2f}%)'
            for i in range(2):
                shown_mask=np.zeros_like(self.pair_results[i][0])
                shown_mask[region]=self.pair_results[i][0][region]
                count=np.count_nonzero(shown_mask)
                frames[4+i]=render_video_frame(self.images[i+1],shown_mask,self.display.get(),self.alpha.get())
                titles[i+1]=f'index{i+1}: F{self.pair_indices[i]}'
                titles[i+4]=f'F{self.pair_indices[i]} | {self.pair_mode} {scope}: {count}/{area} ({100*count/area:.2f}%)'
        for ax,artist,image,title in zip(self.axes,self.artists,frames,titles):
            if image is None:continue
            artist.set_data(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));h,w=image.shape[:2];artist.set_extent((-.5,w-.5,h-.5,-.5));ax.set_title(title,color='white',fontsize=10)
        self.canvas.draw_idle()

    def close(self):
        self.pause();self.closed=True
        if self.slider_timer is not None:self.root.after_cancel(self.slider_timer)
        if self.source is not None:self.worker.submit(self.source.close)
        self.worker.shutdown(wait=False);self.root.destroy()


def main():
    root=tk.Tk()
    try:FrameDifferenceUI(root)
    except Exception as exc:messagebox.showerror('啟動失敗',str(exc),parent=root);root.destroy();raise
    root.mainloop()


if __name__=='__main__':main()
