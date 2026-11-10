from pathlib import Path
exec(compile((Path(__file__).with_name('sweep.py')).read_text(encoding='utf-8').split("evaluate('baseline',base)")[0],str(Path(__file__).with_name('sweep.py')),'exec'))
from Algorithm.specular_detection import BlockSpatialSpecularConfig
rows=json.loads((OUT/'stage2.json').read_text());seen={json.dumps(r['config'],sort_keys=True) for r in rows}
best=max(rows,key=lambda r:r['f1']);anchor=BlockSpatialSpecularConfig(**best['config'])
for key,vals in {'background_sigma':[3.,7.,9.], 'prominence_min':[10.,12.,15.,18.,22.,25.], 'prominence_mad_multiplier':[1.,1.5,2.5,3.], 'close_kernel_px':[3,5], 'dilate_px':[1], 's_max':[70,90,150], 'whiteness_max':[30,60,90], 'local_hot_min':[5.,10.,25.,35.], 'gray_min':[80.,120.], 'v_min':[100.,120.,160.], 'v_percentile':[80.,95.], 'rgb_percentile':[80.,95.]}.items():
    for val in vals:evaluate(f'fine_{key}={val}',replace(anchor,**{key:val}))
# Select best balanced F1 and higher-recall option that still improves on baseline FP.
balanced=max(rows,key=lambda r:r['f1'])
high_recall=max((r for r in rows if r['fp']<=rows[0]['fp'] and r['precision']>=.55),key=lambda r:r['recall'])
chosen={'baseline':rows[0],'balanced':balanced,'higher_recall':high_recall}
(OUT/'all_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(OUT/'recommendations.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
fields=['name','tp','fn','fp','precision','recall','f1','iou']+list(asdict(base))
with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in sorted(rows,key=lambda r:r['f1'],reverse=True):w.writerow({**{k:r[k] for k in fields if k in r},**r['config']})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(13,11))
for ax,im,title in zip(axes[0],[frames[0][region],frames[1][region],cv2.cvtColor(gt[region],cv2.COLOR_GRAY2BGR)],['F250 reference','F550 target','GT: normalized positive difference > 35']):
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(title);ax.axis('off')
for ax,(name,r) in zip(axes[1],chosen.items()):
    mask=compute_specular_mask_bgr_block_adaptive(frames[1],BlockSpatialSpecularConfig(**r['config']))
    comparison,counts=compare_masks(gt,mask,ROI)
    assert all(counts[k]==r[k] for k in ('tp','fn','fp'))
    cv2.imwrite(str(OUT/f'{name}_mask.png'),mask);cv2.imwrite(str(OUT/f'{name}_comparison_roi.png'),comparison[region])
    ax.imshow(cv2.cvtColor(comparison[region],cv2.COLOR_BGR2RGB));ax.set_title(f'{name}: TP {r["tp"]}, FN {r["fn"]}, FP {r["fp"]}\nP {r["precision"]:.1%} / R {r["recall"]:.1%} / F1 {r["f1"]:.1%}');ax.axis('off')
fig.suptitle('Green = TP | dark blue = FN | red = FP; evaluation only within ROI',fontsize=14)
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=140);plt.close(fig)
print('TOTAL',len(rows))
print(json.dumps(chosen,indent=2))
