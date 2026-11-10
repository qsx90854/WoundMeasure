"""Offline A/B replay; no writes to original video/measurement records."""
from pathlib import Path
import sys
import pickle
import json
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import cv2
# Stabilize SIMD gradient ties during offline A/B selection; production unchanged.
cv2.setNumThreads(1)
cv2.setUseOptimized(False)
import matplotlib
matplotlib.use('Agg')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.codex/diagnostics/point1325_20261027'))
import diagnose as d
from Algorithm.Region_SIFT_Matching import run_region_sift_matching
from Algorithm.region_sift_center_refinement import attach_center_heights,summary_for_log,CenterRefinementViewer

out=Path(__file__).resolve().parent
s=d.session(); c=s['cands'][0]; p=(1325.,291.)
lg=cv2.cvtColor(s['left_bgr'],cv2.COLOR_BGR2GRAY)
lm,rm=d.core.ensure_spatial_masks(s,c)
results=[]
for label,masked,empty in [('opencv',False,False),('custom_actual_mask',True,False),('custom_empty_mask_control',True,True)]:
    masks=(np.zeros_like(lm),np.zeros_like(rm)) if empty else (lm,rm)
    pair=[]
    for enabled in (False,True):
        cfg=replace(d.core.REGION_SIFT_CONFIG,use_masked_sift=masked,center_refine_enabled=enabled)
        r=run_region_sift_matching(lg,c['gray'],p,c,s['KL'],config=cfg,reject_specular=False,
                                  left_spec_mask=masks[0],right_spec_mask=masks[1])
        n,cent,_=s['height_plane']
        tri=lambda q:d.core.triangulate_point_3d(p,q,s['KL'],c['K_R'],c['R_rel'],c['t_rel'],F=c['F'])
        attach_center_heights(r,tri,n,cent,max_depth=d.core.MAX_DEPTH_MM)
        height=float(np.dot(n,tri(r['m_pt'])-cent)) if r['m_pt'] is not None else None
        item=dict(label=label,enabled=enabled,height_mm=height,
                  right=None if r['m_pt'] is None else r['m_pt'].tolist(),rejection=r.get('reject_reason'),
                  center=summary_for_log(r.get('center_refinement')))
        results.append(item);pair.append(r)
        print(json.dumps(item,ensure_ascii=False),flush=True)
        if enabled:
            viewer=CenterRefinementViewer()
            with patch('matplotlib.figure.Figure.show'):
                viewer.show(dict(region_debug=r.get('region_debug')))
            if viewer.figure is not None:
                viewer.figure.savefig(out/(label+'.png'),dpi=120)
            viewer.close()
    for key in ('candidate_group_scores','candidate_objective_scores','candidate_centers_warp'):
        np.testing.assert_array_equal(pair[0]['region_debug'][key],pair[1]['region_debug'][key])
(out/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
print('PASS: On/Off preserves every original candidate score and coordinate')
