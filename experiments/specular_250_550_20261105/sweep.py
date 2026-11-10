import sys,json,csv,time
from pathlib import Path
from dataclasses import asdict,replace
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from specular_image_pair_ui import read_main_defaults
from Algorithm.specular_detection import compute_specular_mask_bgr_block_adaptive
from video_frame_difference_ui import frame_difference,compare_masks
OUT=Path(__file__).resolve().parent
VIDEO=r'D:\Lightglue\2026_1103_ZebraCamera_02video\1105\GigaImage_FailedVideo_High_20261105_150713.mp4'
ROI=(650,50,1300,930)
cap=cv2.VideoCapture(VIDEO);frames=[]
for idx in (250,550):
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
    if not ok:raise RuntimeError(f'cannot read {idx}')
    frames.append(frame)
cap.release()
x0,y0,x1,y1=ROI;region=np.s_[y0:y1,x0:x1]
_,gt=frame_difference(*frames,35,ROI,True,True)
cv2.imwrite(str(OUT/'ground_truth.png'),gt)
for i,f in zip((250,550),frames):cv2.imwrite(str(OUT/f'frame_{i}_roi.png'),f[region])
base=read_main_defaults()[0];rows=[];seen=set()
def evaluate(name,config):
    key=json.dumps(asdict(config),sort_keys=True)
    if key in seen:return
    seen.add(key)
    start=time.perf_counter();mask=compute_specular_mask_bgr_block_adaptive(frames[1],config)
    _,c=compare_masks(gt,mask,ROI)
    tp,fn,fp=c['tp'],c['fn'],c['fp'];precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    row=dict(name=name,**c,precision=precision,recall=recall,f1=2*tp/max(1,2*tp+fp+fn),iou=tp/max(1,tp+fp+fn),ms=(time.perf_counter()-start)*1000,config=asdict(config))
    rows.append(row)
    if len(rows)%10==0:print('evaluated',len(rows),flush=True)
    return row

evaluate('baseline',base)
variations={
'local_hot_min':[10,20,25], 'local_hot_percentile':[70,80,85],
'gray_below_v':[20,30,50], 'gray_min':[80,100,140],
's_max':[70,90,120], 'whiteness_max':[45,60,90],
'v_percentile':[70,80,85], 'rgb_percentile':[70,80,85],
'prominence_min':[3,5], 'prominence_mad_multiplier':[1.,1.5],
'enable_prominence_gate':[False], 'dilate_px':[1,2], 'close_kernel_px':[1,3,7],
'background_sigma':[5.,15.], 'v_min':[80.,100.], 'rgb_min':[80.,100.]
}
for key,values in variations.items():
    for val in values:evaluate(f'{key}={val}',replace(base,**{key:val}))
(OUT/'stage1.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT/'conditions.json').write_text(json.dumps(dict(video=VIDEO,indices=[250,550],roi=ROI,roi_end_exclusive=True,threshold=35,normalize=True,positive_only=True,calibration='Off',baseline=asdict(base),opencv=cv2.__version__,numpy=np.__version__),indent=2),encoding='utf-8')
print('BASELINE',json.dumps({k:v for k,v in rows[0].items() if k!='config'}))
for r in sorted(rows,key=lambda r:r['f1'],reverse=True)[:12]:print(json.dumps({k:v for k,v in r.items() if k!='config'}))
