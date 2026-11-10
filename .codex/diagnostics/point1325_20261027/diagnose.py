"""Isolated read-only algorithm replay; writes only beside this script."""
import ast
import json
import pickle
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import regionsift_lite_core as core
from Algorithm.Region_SIFT_Matching import run_region_sift_matching

OUT = Path(__file__).resolve().parent
MAIN = ROOT / 'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py'
tree = ast.parse(MAIN.read_text(encoding='utf-8-sig'))
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name.isupper() and hasattr(core, name) and name != 'BASE_DIR':
            try:
                value = eval(compile(ast.Expression(node.value), str(MAIN), 'eval'), vars(core))
            except (NameError, AttributeError):
                continue
            setattr(core, name, value)

def session():
    path = OUT / 'session.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = core.build_session(ROOT / 'test_video_Zebra/video_20261006_105420_207407_lossless.avi',
                                analysis_copy_dir=OUT / 'video_copy')
    path.write_bytes(pickle.dumps(result))
    return result

def clean(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, dict): return {k: clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    return v

def run(s, label, config, mask_mode='actual'):
    c = s['cands'][0]
    lg = cv2.cvtColor(s['left_bgr'], cv2.COLOR_BGR2GRAY)
    rg = c['gray']
    lm, rm = core.ensure_spatial_masks(s,c)
    if mask_mode == 'empty': lm,rm = np.zeros_like(lm),np.zeros_like(rm)
    result = run_region_sift_matching(lg,rg,(1325.,291.),c,s['KL'],config=config,
                    reject_specular=False,left_spec_mask=lm,right_spec_mask=rm)
    (OUT / (label+'.pkl')).write_bytes(pickle.dumps(result))
    d = result.get('region_debug') or {}
    p = d.get('best_center_right')
    height = None
    if p is not None:
        xyz = core.triangulate_point_3d((1325.,291.),p,s['KL'],c['K_R'],c['R_rel'],c['t_rel'],F=c['F'])
        n,cent,_ = s['height_plane']
        height = float(np.dot(n,xyz-cent))
    summary = dict(label=label,accepted=result.get('m_pt'),reject=result.get('reject_reason'),
        best=p,height_of_best_even_if_rejected=height,
        scores={k:d.get(k) for k in ('group_score','objective_score','objective_score_ratio',
            'best_along_offset_px','best_across_offset_px','point_count','valid_candidate_count')},
        history=[{k:h.get(k) for k in ('round','stage','reason','candidate_funnel','valley')} for h in result.get('search_history',[])])
    (OUT / (label+'.json')).write_text(json.dumps(clean(summary),indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(clean(summary),ensure_ascii=False),flush=True)
    return result

if __name__ == '__main__':
    s = session()
    c = s['cands'][0]
    print('SESSION',s['left_frame_idx'],c['right_frame_idx'],c['R_rel'],c['t_rel'],s['height_plane'],flush=True)
    if len(sys.argv)>1 and sys.argv[1]=='setup': sys.exit()
    for mode in ('shadow','active'):
        for masked,mask_mode in ((False,'actual'),(True,'empty'),(True,'actual')):
            label=f'{mode}_{"custom" if masked else "opencv"}_{mask_mode}'
            run(s,label,replace(core.REGION_SIFT_CONFIG,confidence_mode=mode,use_masked_sift=masked),mask_mode)
