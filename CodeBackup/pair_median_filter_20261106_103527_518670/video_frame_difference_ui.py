"""Single-video seven-panel frame difference / spatial-specular inspector.
Run: Scripts/python.exe video_frame_difference_ui.py
Frame indices are zero-based. Optional robust ROI calibration and positive-only difference.
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
from Algorithm.frame_specular_candidates import DEFAULTS as CHANGE_DEFAULTS, specular_change_candidates
from Algorithm.specular_detection import BlockSpatialSpecularConfig
from Algorithm.specular_settings import show_specular_settings
from Algorithm.specular_model_v2_settings import show_specular_model_v2_settings


# Balanced Legacy preset from the 2026-11-05 F250/F550 experiment.
# Local to this viewer; does not depend on experiment files at runtime.
BALANCED_LEGACY_CONFIG = BlockSpatialSpecularConfig(
    block_width_px=64,
    block_height_px=64,
    v_percentile=90.0,
    rgb_percentile=90.0,
    local_hot_percentile=80.0,
    v_min=140.0,
    v_max=245.0,
    rgb_min=120.0,
    rgb_max=250.0,
    local_hot_min=15.0,
    local_hot_max=45.0,
    gray_min=80.0,
    gray_below_v=50.0,
    s_max=120,
    whiteness_max=45,
    background_sigma=5.0,
    enable_prominence_gate=True,
    prominence_min=20.0,
    prominence_mad_multiplier=2.0,
    enable_strong_highlight_exception=True,
    strong_v_min=250.0,
    strong_s_max=30.0,
    strong_whiteness_max=20.0,
    interpolate_thresholds=True,
    open_kernel_px=1,
    close_kernel_px=1,
    dilate_px=0,
)


def roi_slice(shape,roi):
    h,w=shape[:2]
    x0,y0,x1,y1=(0,0,w,h) if roi is None else roi
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('ROI 超出影格範圍或沒有面積')
    return (slice(y0,y1),slice(x0,x1)),(x1-x0)*(y1-y0)


def robust_linear_fit(reference,target):
    """Fit target = a*reference+b with Huber IRLS; ignore clipped pixels."""
    x=reference.ravel().astype(np.float64);y=target.ravel().astype(np.float64)
    keep=(x>0)&(x<255)&(y>0)&(y<255)
    x=x[keep];y=y[keep]
    if x.size<32:raise ValueError('校正 ROI 至少需要 32 個未飽和的對應像素')
    # Bound fitting cost for large ROIs, keeping deterministic spatial coverage.
    if x.size>20000:
        sample=np.linspace(0,x.size-1,20000,dtype=int);x=x[sample];y=y[sample]
    if np.std(x)<2:raise ValueError('校正 ROI 亮度變化不足，請包含更多紋理或灰階變化')
    design=np.column_stack((x,np.ones_like(x)))
    coef=np.linalg.lstsq(design,y,rcond=None)[0]
    for _ in range(30):
        residual=y-design@coef
        sigma=max(0.5,1.4826*np.median(np.abs(residual-np.median(residual))))
        weights=np.minimum(1.,1.345*sigma/np.maximum(np.abs(residual),1e-12))
        if np.sqrt(np.average((x-np.average(x,weights=weights))**2,weights=weights))<2:
            raise ValueError('校正有效像素的亮度變化不足，請重新選取校正 ROI')
        root=np.sqrt(weights)
        updated=np.linalg.lstsq(design*root[:,None],y*root,rcond=None)[0]
        if np.max(np.abs(updated-coef))<1e-6:
            coef=updated;break
        coef=updated
    a,b=map(float,coef)
    if not np.isfinite(coef).all() or a<=0:
        raise ValueError('校正無有效正向亮度關係，請檢查對齊與校正 ROI')
    residual=y-design@coef
    return a,b,dict(n=int(x.size),mae=float(np.median(np.abs(residual))))


def calibrate_reference(first,second,calibration_roi,mode):
    """Predict frame2's non-specular gray intensity from frame1 (no clipping)."""
    if first.shape!=second.shape:raise ValueError('兩影格尺寸不同')
    if calibration_roi is None:raise ValueError('請先框選兩幀皆無反光的校正 ROI（青色虛線）')
    region,_=roi_slice(first.shape,calibration_roi)
    if mode=='Gray':
        source=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)
        target=cv2.cvtColor(second,cv2.COLOR_BGR2GRAY)
        a,b,info=robust_linear_fit(source[region],target[region])
        return source.astype(np.float32)*a+b,[('Gray',a,b,info)]
    if mode!='BGR':raise ValueError('未知校正模式')
    predicted=np.empty(first.shape,np.float32);details=[]
    for c,name in enumerate('BGR'):
        a,b,info=robust_linear_fit(first[region][:,:,c],second[region][:,:,c])
        predicted[:,:,c]=first[:,:,c].astype(np.float32)*a+b
        details.append((name,a,b,info))
    return cv2.cvtColor(predicted,cv2.COLOR_BGR2GRAY),details


def frame_difference(first,second,threshold,roi=None,normalize=False,positive_only=False,predicted_gray=None):
    if first.shape != second.shape: raise ValueError('兩影格尺寸不同')
    if not 0 <= threshold <= 255: raise ValueError('門檻必須介於 0～255')
    region,_=roi_slice(first.shape,roi)
    grays=[cv2.cvtColor(image[region],cv2.COLOR_BGR2GRAY) for image in (first,second)]
    if predicted_gray is not None:
        if normalize:raise ValueError('線性校正與 min-max normalize 不可同時使用')
        if predicted_gray.shape!=first.shape[:2]:raise ValueError('校正影像尺寸不同')
        grays[0]=predicted_gray[region]
    if normalize:
        # Independent ROI min-max scaling; constant patches map to zero.
        # Keep float precision until thresholding (no uint8 rounding).
        for i,gray in enumerate(grays):
            low,high=float(gray.min()),float(gray.max())
            grays[i]=(gray.astype(np.float32)-low)*(255.0/(high-low)) if high>low else np.zeros(gray.shape,np.float32)
    use_float=normalize or positive_only or predicted_gray is not None
    difference=np.zeros(first.shape[:2],np.float32 if use_float else np.uint8)
    if use_float:
        residual=grays[1].astype(np.float32)-grays[0].astype(np.float32)
        local=np.maximum(residual,0) if positive_only else np.abs(residual)
    else:local=cv2.absdiff(*grays)
    difference[region]=local
    mask=np.zeros(first.shape[:2],np.uint8);mask[region]=(local > threshold).astype(np.uint8)*255
    return difference,mask


def compare_masks(ground_truth,prediction,roi=None,show=(True,True,True)):
    """Return BGR TP/FN/FP pixels; visibility never changes comparison counts."""
    if ground_truth.shape != prediction.shape or ground_truth.ndim != 2:
        raise ValueError('GT 與反光 mask 必須為同尺寸二維影像')
    region,area=roi_slice(ground_truth.shape,roi)
    gt=ground_truth[region]>0;pred=prediction[region]>0
    tp=gt & pred;fn=gt & ~pred;fp=~gt & pred
    out=np.zeros((*ground_truth.shape,3),np.uint8)
    patch=out[region]
    for enabled,selected,color in zip(show,(tp,fn,fp),((0,255,0),(160,0,0),(0,0,255))):
        if enabled:patch[selected]=color
    counts=dict(tp=int(tp.sum()),fn=int(fn.sum()),fp=int(fp.sum()),tn=int((~gt & ~pred).sum()),area=area)
    return out,counts


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
        _,self.v2,_=read_main_defaults()
        self.legacy=BALANCED_LEGACY_CONFIG
        self.source=None;self.images=[None,None,None];self.pair_results=None;self.pair_indices=None
        self.current_index=0;self.desired_current=0
        self.worker=ThreadPoolExecutor(max_workers=1);self.future=None;self.pending_pair=None;self.pending_current=None
        self.closed=False;self.playing=False;self.play_timer=None;self.slider_timer=None;self.setting_slider=False
        self.calibration_roi=None;self.calibration_patches=[];self.roi_kind='analysis';self.calibration_cache=None
        self.roi=None;self.roi_selecting=False;self.roi_drag=None;self.roi_patches=[]
        self.drag=None;self.pair_revision=0;self.active_pair_revision=0;self.job=None
        self.mode=tk.StringVar(value='Legacy');self.alpha=tk.DoubleVar(value=.55)
        self.display=tk.StringVar(value='疊圖');self.threshold=tk.StringVar(value='30')
        self.show_tp=tk.BooleanVar(value=True);self.show_fn=tk.BooleanVar(value=True);self.show_fp=tk.BooleanVar(value=True)
        self.show_crosshair=tk.BooleanVar(value=False)
        self.normalize_gray=tk.BooleanVar(value=False)
        self.gt_mode=tk.StringVar(value='相減');self.use_whitening=tk.BooleanVar(value=True);self.use_white_brightening=tk.BooleanVar(value=True)
        self.change_values={key:tk.StringVar(value=str(value)) for key,value in CHANGE_DEFAULTS.items()};self.change_cache=None
        self.calibration_mode=tk.StringVar(value='Off');self.positive_only=tk.BooleanVar(value=False)
        self.calibration_text=tk.StringVar(value='校正尚未啟用；先選兩幀皆無明顯反光且有灰階變化的區域。')
        self.index1=tk.StringVar(value='0');self.index2=tk.StringVar(value='1');self.current=tk.StringVar(value='0')
        self.status=tk.StringVar(value='請載入影片；index 從 0 開始。滾輪縮放、左鍵拖曳，七個面板同步。')
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
        ttk.Label(pair,text='差異門檻 >').pack(side='left',padx=5)
        self.threshold_entry=ttk.Entry(pair,textvariable=self.threshold,width=6);self.threshold_entry.pack(side='left');self.threshold_entry.bind('<Return>',lambda e:self.draw())
        self.threshold_button=ttk.Button(pair,text='更新門檻',command=self.draw);self.threshold_button.pack(side='left',padx=3)
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
        ttk.Checkbutton(roi_bar,text='同步十字游標',variable=self.show_crosshair,command=self.hide_crosshair).pack(side='left',padx=8)
        self.roi_text=tk.StringVar(value='分析全圖；可在任一面板框選，同步套用七個面板。')
        ttk.Label(roi_bar,textvariable=self.roi_text).pack(side='left',padx=8)
        roi_values_bar=ttk.Frame(root,padding=5);roi_values_bar.pack(fill='x')
        self.roi_values=[tk.StringVar(value='') for _ in range(4)]
        for label,var in zip(('x 起點','y 起點','x 終點','y 終點'),self.roi_values):
            ttk.Label(roi_values_bar,text=label).pack(side='left',padx=4)
            entry=ttk.Entry(roi_values_bar,textvariable=var,width=7);entry.pack(side='left')
            entry.bind('<Return>',lambda e:self.apply_roi_values())
        ttk.Button(roi_values_bar,text='套用分析 ROI',command=self.apply_roi_values).pack(side='left',padx=6)
        ttk.Button(roi_values_bar,text='複製 ROI 座標',command=self.copy_roi_values).pack(side='left',padx=4)
        ttk.Label(roi_values_bar,text='原圖 pixel 座標；含起點、不含終點。框選後自動更新。').pack(side='left',padx=6)
        norm_bar=ttk.Frame(root,padding=5);norm_bar.pack(fill='x')
        self.normalize_button=ttk.Checkbutton(norm_bar,text='ROI 灰階 normalize 後相減',variable=self.normalize_gray,command=self.draw)
        self.normalize_button.pack(side='left',padx=4)
        ttk.Label(norm_bar,text='兩幀各自 min-max 到 0～255；無 ROI 使用全圖；灰階為常數時設為 0。僅影響差異與 GT 比較。').pack(side='left',padx=8)
        calibration_bar=ttk.Frame(root,padding=5);calibration_bar.pack(fill='x')
        self.calibration_button=ttk.Button(calibration_bar,text='框選校正 ROI',command=lambda:self.toggle_roi('calibration'))
        self.calibration_button.pack(side='left',padx=3)
        ttk.Button(calibration_bar,text='清除校正 ROI',command=self.clear_calibration_roi).pack(side='left',padx=3)
        ttk.Label(calibration_bar,text='Robust 線性校正').pack(side='left',padx=5)
        combo=ttk.Combobox(calibration_bar,textvariable=self.calibration_mode,values=('Off','Gray','BGR'),width=7,state='readonly')
        self.calibration_combo=combo
        combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.calibration_changed())
        self.positive_button=ttk.Checkbutton(calibration_bar,text='單向差異：只找 index2 變亮',variable=self.positive_only,command=self.draw)
        self.positive_button.pack(side='left',padx=8)
        change_bar=ttk.Frame(root,padding=5);change_bar.pack(fill='x')
        ttk.Label(change_bar,text='GT 候選模式').pack(side='left',padx=4)
        combo=ttk.Combobox(change_bar,textvariable=self.gt_mode,values=('相減','變白／局部增亮'),width=18,state='readonly')
        combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.calibration_changed())
        ttk.Checkbutton(change_bar,text='有色變白',variable=self.use_whitening,command=self.draw).pack(side='left',padx=6)
        ttk.Checkbutton(change_bar,text='原本偏白：額外增亮',variable=self.use_white_brightening,command=self.draw).pack(side='left',padx=6)
        ttk.Label(change_bar,text='新模式使用原圖；局部中位數扣除背景變化；不使用相減門檻／線性校正。').pack(side='left',padx=6)
        labels={'window':'視窗 px','extra_y':'額外亮度 >','saturation_drop':'S 下降 >','extra_s':'額外 S 下降 >',
                'target_v':'V2 ≥','target_s':'S2 ≤','color_s':'有色 S1 ≥','white_s':'偏白 S1 ≤','white_v':'偏白 V1 ≥'}
        for offset in (0,5):
            row=ttk.Frame(root,padding=2);row.pack(fill='x')
            for key in list(CHANGE_DEFAULTS)[offset:offset+5]:
                ttk.Label(row,text=labels[key]).pack(side='left',padx=4)
                entry=ttk.Entry(row,textvariable=self.change_values[key],width=6);entry.pack(side='left')
                entry.bind('<Return>',lambda e:self.draw())
            if offset==5:ttk.Button(row,text='更新候選門檻',command=self.draw).pack(side='left',padx=6)
        ttk.Label(root,textvariable=self.calibration_text,wraplength=1450).pack(anchor='w',padx=10)
        compare_bar=ttk.Frame(root,padding=5);compare_bar.pack(fill='x')
        ttk.Label(compare_bar,text='比較圖（差異 mask 作為 GT，index2 反光 mask 作為預測）：').pack(side='left',padx=4)
        for label,var in [('綠色：都有偵測 TP',self.show_tp),('深藍：漏偵測 FN',self.show_fn),('紅色：多偵測 FP',self.show_fp)]:
            ttk.Checkbutton(compare_bar,text=label,variable=var,command=self.draw).pack(side='left',padx=5)
        ttk.Label(compare_bar,text='未勾選與兩者皆無的像素顯示黑色；統計不受勾選影響。').pack(side='left',padx=5)
        self.slider=ttk.Scale(root,from_=0,to=1,command=self.slider_seek);self.slider.pack(fill='x',padx=12)
        ttk.Label(root,text='差異可選絕對差／單向變亮；校正只影響差異與 GT，不做影像對齊；白色＝差異大於門檻。深藍＝反光排除區。選定兩幀不會隨播放改變。').pack(anchor='w',padx=10)
        self.figure=Figure(figsize=(14,8),facecolor='#202020');self.axes=[];self.artists=[]
        titles=['Current frame','index1: original','index2: original','Absolute grayscale difference mask','index1: specular','index2: specular','Difference GT vs index2: TP / FN / FP']
        for i,title in enumerate(titles):
            ax=self.figure.add_subplot(2,4,(1,2,3,5,6,7,4)[i],**(dict(sharex=self.axes[0],sharey=self.axes[0]) if self.axes else {}))
            ax.set_facecolor('#111111');ax.tick_params(colors='white');ax.set_title(title,color='white',fontsize=10)
            ax.set_aspect('equal',adjustable='box');self.axes.append(ax)
            self.artists.append(ax.imshow(np.zeros((1,1,3),np.uint8),origin='upper',interpolation='nearest'))
        from matplotlib import patheffects
        self.crosshairs=[]
        for ax in self.axes:
            marker,=ax.plot([],[],marker='+',markersize=10,markeredgewidth=1.3,
                           color='white',linestyle='None',visible=False,zorder=30)
            marker.set_path_effects([patheffects.Stroke(linewidth=3,foreground='black'),patheffects.Normal()])
            self.crosshairs.append(marker)
        from matplotlib.patches import Rectangle
        for ax in self.axes:
            box=Rectangle((0,0),0,0,fill=False,edgecolor='yellow',linewidth=1.5,zorder=20,visible=False)
            ax.add_patch(box);self.roi_patches.append(box)
            calibration_box=Rectangle((0,0),0,0,fill=False,edgecolor='cyan',linestyle='--',linewidth=1.5,zorder=21,visible=False)
            ax.add_patch(calibration_box);self.calibration_patches.append(calibration_box)
        self.figure.subplots_adjust(left=.035,right=.98,bottom=.04,top=.94,wspace=.12,hspace=.22)
        self.canvas=FigureCanvasTkAgg(self.figure,master=root);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        for event,callback in [('scroll_event',self.scroll),('button_press_event',self.press),('motion_notify_event',self.motion),('button_release_event',self.release)]:self.canvas.mpl_connect(event,callback)
        self.canvas.mpl_connect('figure_leave_event',self.hide_crosshair)
        self.canvas.mpl_connect('axes_leave_event',self.hide_crosshair)
        ttk.Label(root,textvariable=self.status,padding=5).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close);root.bind('<Escape>',lambda e:self.cancel_roi())
        self.home()

    def hide_crosshair(self,event=None):
        changed=any(marker.get_visible() for marker in self.crosshairs)
        for marker in self.crosshairs:marker.set_visible(False)
        if changed:self.canvas.draw_idle()

    def update_crosshair(self,event):
        if not self.show_crosshair.get() or event.inaxes not in self.axes or event.x is None or event.y is None:
            self.hide_crosshair();return
        panel=self.axes.index(event.inaxes)
        if (panel==0 and self.images[0] is None) or (panel>0 and self.pair_results is None):
            self.hide_crosshair();return
        # Transform after panning so the marker stays under the pointer.
        x,y=event.inaxes.transData.inverted().transform((event.x,event.y))
        image=self.images[0] if panel==0 else self.images[1]
        h,w=image.shape[:2]
        if not (-.5<=x<w-.5 and -.5<=y<h-.5):
            self.hide_crosshair();return
        for i,marker in enumerate(self.crosshairs):
            available=self.images[0] is not None if i==0 else self.pair_results is not None
            marker.set_data([x],[y]);marker.set_visible(available)
        self.canvas.draw_idle()

    def selected_roi(self,start,end):
        image=next((im for im in self.images if im is not None),None)
        if image is None:return None
        h,w=image.shape[:2]
        x0=int(np.clip(np.floor(np.round(min(start[0],end[0])+.5,9)),0,w))
        y0=int(np.clip(np.floor(np.round(min(start[1],end[1])+.5,9)),0,h))
        x1=int(np.clip(np.ceil(np.round(max(start[0],end[0])+.5,9)),0,w))
        y1=int(np.clip(np.ceil(np.round(max(start[1],end[1])+.5,9)),0,h))
        return (x0,y0,x1,y1) if x1>x0 and y1>y0 else None

    def show_roi(self,roi,kind='analysis'):
        for box in (self.calibration_patches if kind=='calibration' else self.roi_patches):
            box.set_visible(roi is not None)
            if roi is not None:
                x0,y0,x1,y1=roi;box.set_bounds(x0-.5,y0-.5,x1-x0,y1-y0)
        self.canvas.draw_idle()

    def toggle_roi(self,kind='analysis'):
        if self.roi_selecting and self.roi_kind==kind:self.cancel_roi();return
        self.cancel_roi()
        if all(im is None for im in self.images):return
        self.pause();self.drag=None;self.roi_selecting=True;self.roi_kind=kind
        (self.calibration_button if kind=='calibration' else self.roi_button).configure(text='取消框選 (Esc)')
        self.roi_text.set('在任一已有影像的面板按住左鍵框選；放開即套用，Esc 取消。')

    def cancel_roi(self):
        self.roi_selecting=False;self.roi_drag=None;self.drag=None
        self.roi_button.configure(text='框選 ROI');self.show_roi(self.roi)
        self.calibration_button.configure(text='框選校正 ROI');self.show_roi(self.calibration_roi,'calibration')
        image=next((im for im in self.images if im is not None),None)
        bounds=self.roi if self.roi is not None else ((0,0,image.shape[1],image.shape[0]) if image is not None else ('',)*4)
        for var,value in zip(self.roi_values,bounds):var.set(str(value))
        if self.roi is None:self.roi_text.set('分析全圖；可在任一面板框選，同步套用七個面板。')
        else:
            x0,y0,x1,y1=self.roi
            self.roi_text.set(f'ROI x=[{x0}, {x1}), y=[{y0}, {y1})；寬={x1-x0}, 高={y1-y0}；面積={(x1-x0)*(y1-y0)} pixels。')

    def apply_roi_values(self):
        image=next((im for im in self.images if im is not None),None)
        if image is None:self.status.set('請先載入影片再設定 ROI。');return
        try:
            roi=tuple(int(var.get()) for var in self.roi_values)
            roi_slice(image.shape,roi)
        except ValueError:
            self.status.set(f'ROI 請輸入整數且滿足 0 <= x起點 < x終點 <= {image.shape[1]}、0 <= y起點 < y終點 <= {image.shape[0]}；未變更 ROI。');return
        self.pause();self.roi=roi;self.cancel_roi();self.draw()
        self.status.set(f'已套用分析 ROI：x=[{roi[0]}, {roi[2]}), y=[{roi[1]}, {roi[3]})')

    def copy_roi_values(self):
        image=next((im for im in self.images if im is not None),None)
        if image is None:self.status.set('請先載入影片。');return
        x0,y0,x1,y1=self.roi if self.roi is not None else (0,0,image.shape[1],image.shape[0])
        text=f'x=[{x0}, {x1}), y=[{y0}, {y1}); ROI=(x0,y0,x1,y1)=({x0},{y0},{x1},{y1})'
        self.root.clipboard_clear();self.root.clipboard_append(text)
        self.status.set('已複製目前生效的 ROI 座標。')

    def clear_calibration_roi(self):
        self.calibration_roi=None;self.calibration_cache=None;self.cancel_roi();self.draw()

    def calibration_changed(self):
        candidate_mode=self.gt_mode.get()!='相減'
        enabled=candidate_mode or self.calibration_mode.get()!='Off'
        if enabled:self.normalize_gray.set(False)
        self.calibration_combo.configure(state='disabled' if candidate_mode else 'readonly')
        for widget in (self.threshold_entry,self.threshold_button,self.positive_button):
            widget.configure(state='disabled' if candidate_mode else 'normal')
        self.normalize_button.configure(state='disabled' if enabled else 'normal')
        self.draw()

    def change_candidate_mask(self):
        options={key:(int(var.get()) if key=='window' else float(var.get())) for key,var in self.change_values.items()}
        key=(id(self.images[1]),id(self.images[2]),self.roi,self.use_whitening.get(),self.use_white_brightening.get(),tuple(options.items()))
        if self.change_cache is None or self.change_cache[0]!=key:
            mask,info=specular_change_candidates(self.images[1],self.images[2],self.roi,
                whitening=self.use_whitening.get(),white_brightening=self.use_white_brightening.get(),**options)
            self.change_cache=(key,mask,info)
        _,mask,info=self.change_cache
        self.calibration_text.set(f'候選：有色變白={info["whitening"]}；偏白額外增亮={info["white_brightening"]}；重疊={info["overlap"]}；聯集={info["total"]} pixels。S/V 為 0～255；只使用 ROI 內的背景樣本。')
        return mask

    def calibrated_reference(self):
        mode=self.calibration_mode.get()
        if mode=='Off':
            self.calibration_text.set('校正 OFF；黃色框＝分析 ROI，青色虛線＝校正 ROI。')
            return None
        key=(id(self.images[1]),id(self.images[2]),self.calibration_roi,mode)
        if self.calibration_cache is None or self.calibration_cache[0]!=key:
            predicted,details=calibrate_reference(self.images[1],self.images[2],self.calibration_roi,mode)
            self.calibration_cache=(key,predicted,details)
        _,predicted,details=self.calibration_cache
        info=' | '.join(f'{name}: a={a:.3f}, b={b:+.2f}, median|res|={d["mae"]:.2f}, n={d["n"]}' for name,a,b,d in details)
        self.calibration_text.set(f'校正 ROI {self.calibration_roi}；預測 index2 = a × index1 + b；{info}')
        return predicted

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
            self.update_crosshair(event)
            return
        if event.x is None or event.y is None:return
        inverse,start=self.roi_drag;self.show_roi(self.selected_roi(start,inverse.transform((event.x,event.y))),self.roi_kind)
        self.update_crosshair(event)

    def release(self,event):
        if self.roi_drag is None:return super().release(event)
        inverse,start=self.roi_drag
        if event is not None and event.x is not None and event.y is not None:
            end=inverse.transform((event.x,event.y))
            if np.linalg.norm(np.asarray(end)-start)>=1.:
                roi=self.selected_roi(start,end)
                if roi is not None:
                    if self.roi_kind=='calibration':self.calibration_roi=roi;self.calibration_cache=None
                    else:self.roi=roi
        self.cancel_roi();self.draw()

    def scroll(self,event):
        if self.roi_drag is None:
            super().scroll(event)
            self.update_crosshair(event)

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
        self.hide_crosshair();self.change_cache=None
        self.roi=None;self.calibration_roi=None;self.calibration_cache=None;self.cancel_roi()
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
            self.calibration_cache=None;self.change_cache=None
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
        frames=[self.images[0],self.images[1],self.images[2],None,None,None,None]
        titles=[f'Current F{self.current_index}','index1: original','index2: original','abs(gray1-gray2) > threshold','index1: specular','index2: specular','Difference GT vs index2: TP / FN / FP']
        if self.pair_results is not None:
            error=None
            try:
                threshold=0
                if self.gt_mode.get()!='相減':mask=self.change_candidate_mask()
                else:
                    threshold=float(self.threshold.get())
                    predicted=self.calibrated_reference()
                    difference,mask=frame_difference(self.images[1],self.images[2],threshold,self.roi,self.normalize_gray.get(),self.positive_only.get(),predicted)
            except ValueError as exc:
                error=str(exc);self.calibration_text.set(f'差異／GT 尚未產生：{error}')
                mask=np.zeros(self.images[1].shape[:2],np.uint8)
                threshold=0
            frames[3]=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            region,area=roi_slice(mask.shape,self.roi)
            scope='ROI' if self.roi is not None else 'full'
            diff_mode=self.calibration_mode.get() if self.calibration_mode.get()!='Off' else ('normalized' if self.normalize_gray.get() else 'raw')
            direction='positive' if self.positive_only.get() else 'absolute'
            diff_mode=f'{diff_mode} {direction}'
            titles[3]=f'{diff_mode} diff > {threshold:g} | {scope}: {np.count_nonzero(mask)}/{area} ({100*np.count_nonzero(mask)/area:.2f}%)'
            if self.gt_mode.get()!='相減':
                diff_mode='whitening / extra brightening'
                titles[3]=f'Whitening / extra brightening\n{scope}: {np.count_nonzero(mask)}/{area} ({100*np.count_nonzero(mask)/area:.2f}%)'
            for i in range(2):
                shown_mask=np.zeros_like(self.pair_results[i][0])
                shown_mask[region]=self.pair_results[i][0][region]
                count=np.count_nonzero(shown_mask)
                frames[4+i]=render_video_frame(self.images[i+1],shown_mask,self.display.get(),self.alpha.get())
                titles[i+1]=f'index{i+1}: F{self.pair_indices[i]}'
                titles[i+4]=f'F{self.pair_indices[i]} | {self.pair_mode} {scope}: {count}/{area} ({100*count/area:.2f}%)'
            frames[6],counts=compare_masks(mask,self.pair_results[1][0],self.roi,
                (self.show_tp.get(),self.show_fn.get(),self.show_fp.get()))
            titles[6]=(f'GT: {diff_mode} difference | Prediction: F{self.pair_indices[1]}\n'
                       f'TP={counts["tp"]} FN={counts["fn"]} FP={counts["fp"]} | {scope}={counts["area"]} px')
            if error is not None:
                # Do not display stale or seemingly valid GT when calibration failed.
                frames[6]=np.zeros_like(frames[3]);titles[3]='Difference unavailable - check calibration / threshold'
                titles[6]='GT comparison unavailable'
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
