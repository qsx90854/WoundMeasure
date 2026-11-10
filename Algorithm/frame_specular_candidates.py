"""Two-frame specular candidates; raw uint8 BGR, no detector-mask dependency."""
import cv2
import numpy as np

DEFAULTS = dict(window=65, extra_y=15., saturation_drop=20., extra_s=10.,
                target_v=140., target_s=80., color_s=40., white_s=80., white_v=100.)


def local_median_background(values, window):
    """Overlapping clipped windows sampled on a half-window grid, interpolated.

    Only values inside the analysis ROI contribute. A majority of ordinary
    pixels per window is required; broad highlights may enter this background.
    This is a smooth background estimate, not an exact per-pixel median filter.
    """
    h,w=values.shape;radius=window//2;step=max(1,radius)
    xs=np.unique(np.r_[np.arange(0,w,step),w-1]);ys=np.unique(np.r_[np.arange(0,h,step),h-1])
    grid=np.empty((len(ys),len(xs)),np.float32)
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            grid[j,i]=np.median(values[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)])
    mx,my=np.meshgrid(np.interp(np.arange(w),xs,np.arange(len(xs))).astype(np.float32),
                      np.interp(np.arange(h),ys,np.arange(len(ys))).astype(np.float32))
    return cv2.remap(grid,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def specular_change_candidates(first,second,roi=None,*,whitening=True,white_brightening=True,**options):
    if first.shape!=second.shape or first.ndim!=3 or first.shape[2]!=3:
        raise ValueError('兩影格必須為相同尺寸的 BGR 影像')
    if first.dtype!=np.uint8 or second.dtype!=np.uint8:raise ValueError('需要原始 uint8 影像')
    unknown=set(options)-set(DEFAULTS)
    if unknown:raise ValueError(f'未知參數：{unknown}')
    p={**DEFAULTS,**options}
    window=p['window']
    if not isinstance(window,(int,np.integer)) or not 3<=window<=511 or window%2!=1:
        raise ValueError('局部視窗須為 3～511 的奇數')
    for key,value in p.items():
        if key!='window' and (not np.isfinite(value) or not 0<=value<=255):raise ValueError(f'{key} 須為 0～255')
    h,w=first.shape[:2];x0,y0,x1,y1=(0,0,w,h) if roi is None else roi
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('ROI 範圍無效')
    region=np.s_[y0:y1,x0:x1];a=first[region];b=second[region]
    hsv1=cv2.cvtColor(a,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv2=cv2.cvtColor(b,cv2.COLOR_BGR2HSV).astype(np.float32)
    s1,v1=hsv1[:,:,1],hsv1[:,:,2];s2,v2=hsv2[:,:,1],hsv2[:,:,2]
    dy=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32)-cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32)
    ds=s1-s2
    by=local_median_background(dy,window);bs=local_median_background(ds,window)
    ey=dy-by;es=ds-bs
    bright=(dy>0)&(ey>p['extra_y'])&(v2>=p['target_v'])
    # Overlapping source-color ranges avoid a gap between the two branches.
    colored=(s1>=p['color_s'])&(s2<=p['target_s'])&(ds>p['saturation_drop'])&(es>p['extra_s'])&bright
    white=(s1<=p['white_s'])&(v1>=p['white_v'])&bright
    union=(colored if whitening else np.zeros_like(colored)) | (white if white_brightening else np.zeros_like(white))
    mask=np.zeros((h,w),np.uint8);mask[region]=union.astype(np.uint8)*255
    return mask,dict(whitening=int(colored.sum()) if whitening else 0,
                     white_brightening=int(white.sum()) if white_brightening else 0,
                     overlap=int((colored&white).sum()) if whitening and white_brightening else 0,
                     total=int(union.sum()),background_y=by,extra_y=ey,background_s=bs,extra_s=es)
