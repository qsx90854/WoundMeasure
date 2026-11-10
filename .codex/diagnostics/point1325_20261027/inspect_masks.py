import diagnose as d
from unittest.mock import patch
from Algorithm import masked_sift_descriptor as masked
from Algorithm import Region_SIFT_Matching as matcher
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np,cv2=d.np,d.cv2
s=d.session(); c=s['cands'][0]
lm,rm=d.core.ensure_spatial_masks(s,c)
lg=cv2.cvtColor(s['left_bgr'],cv2.COLOR_BGR2GRAY)
rg=c['gray']
offsets=np.array([(25,a) for a in (-2,-1,0,1,2)]+[(-21,-2),(-29,0)],float)
records=[]
for cell,ppc in ((10,3),(40,7)):
    for maskmode in ('actual','left_only','right_only','empty'):
        cfg=d.replace(d.core.REGION_SIFT_CONFIG,use_masked_sift=True,two_stage_search=False,
            confidence_mode='legacy',max_group_score=None,max_objective_score_ratio=None,
            cell_width_px=cell,cell_height_px=cell,points_per_cell=ppc)
        captures=[]; anchors=[]
        orig=masked.compare_descriptors; sel=matcher.select_region_points
        def compare(left,right,config,**kw):
            result=orig(left,right,config,**kw)
            captures.append((left,right,result))
            return result
        def select(*args,**kw):
            result=sel(*args,**kw); anchors.append(result); return result
        with patch.object(masked,'compare_descriptors',compare),patch.object(matcher,'select_region_points',select):
            r=matcher.run_region_sift_matching(lg,rg,(1325.,291.),c,s['KL'],config=cfg,
                left_spec_mask=lm if maskmode in ('actual','left_only') else np.zeros_like(lm),
                right_spec_mask=rm if maskmode in ('actual','right_only') else np.zeros_like(rm),
                _offsets=offsets)
        rows=[]
        if captures:
            left,right,(dist,valid,common,details)=captures[0]
            gate=masked.group_validity(valid,anchors[0][1],cfg,return_details=True)
            for i,off in enumerate(offsets):
                rows.append(dict(offset=off,valid=int(valid[i].sum()),required=gate['required'],
                    deficient=int(gate['deficient_cell_count'][i]),groupOK=bool(gate['valid'][i]),
                    leftValid=int(left['valid_points'].sum()),rightValid=int(right['valid_points'][i].sum()),
                    commonFail=int((common[i]<cfg.masked_min_common_fraction).sum()),
                    meanCommon=float(common[i].mean()),
                    leftCoverageMean=float(left['valid_fraction'].mean()) if 'valid_fraction' in left else None,
                    medianL2=float(np.median(dist[i]))))
            (d.OUT/f'packets_{cell}_{maskmode}.pkl').write_bytes(d.pickle.dumps(dict(captures=captures,anchors=anchors,result=r)))
        item=dict(cell=cell,mask=maskmode,rows=rows,reason=r.get('reject_reason'))
        records.append(item)
        print(d.json.dumps(d.clean(item)),flush=True)
(d.OUT/'mask_point_counts.json').write_text(d.json.dumps(d.clean(records),indent=2),encoding='utf8')

base=d.pickle.loads((d.OUT/'shadow_opencv_actual.pkl').read_bytes())['region_debug']
H=base['H_rl']; warped=cv2.warpPerspective(c['bgr'],H,(lg.shape[1],lg.shape[0]))
wm=cv2.warpPerspective(cv2.dilate(rm,np.ones((3,3),np.uint8)),H,(lg.shape[1],lg.shape[0]),flags=cv2.INTER_NEAREST)
pc=np.array([1325.,291.]); rc=base['best_center_warp']
fig,axs=plt.subplots(2,3,figsize=(12,8))
for row,(img,mask,center,title) in enumerate(((s['left_bgr'],lm,pc,'Left'),(warped,wm,rc,'Right warped at OpenCV match'))):
    x,y=np.rint(center).astype(int); box=(slice(y-65,y+66),slice(x-65,x+66))
    rgb=cv2.cvtColor(img[box],cv2.COLOR_BGR2RGB); ms=mask[box]>0
    for col in range(3):
        axs[row,col].imshow(rgb if col<2 else ms,cmap='gray',vmin=0,vmax=1)
        if col==1:
            overlay=np.zeros((*ms.shape,4));overlay[ms]=[0,1,1,.6];axs[row,col].imshow(overlay)
        axs[row,col].plot(65,65,'r+',ms=10)
        axs[row,col].set_title(title+' / '+('raw','mask overlay','mask')[col]);axs[row,col].axis('off')
    print(title,'mask 30x30',float((mask[y-15:y+15,x-15:x+15]>0).mean()),'mask 131x131',float(ms.mean()),flush=True)
fig.tight_layout();fig.savefig(d.OUT/'mask_comparison.png',dpi=130)
