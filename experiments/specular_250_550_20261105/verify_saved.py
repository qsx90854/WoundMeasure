import sys,json
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Algorithm.specular_detection import BlockSpatialSpecularConfig,compute_specular_mask_bgr_block_adaptive
from video_frame_difference_ui import frame_difference,compare_masks
p=Path(__file__).resolve().parent;c=json.loads((p/'conditions.json').read_text());configs=json.loads((p/'recommendations.json').read_text())
cap=cv2.VideoCapture(c['video']);images=[]
for index in c['indices']:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read()
    if not ok:raise RuntimeError(f'Failed reading frame {index}')
    images.append(im)
cap.release()
_,gt=frame_difference(*images,c['threshold'],tuple(c['roi']),c['normalize'],c['positive_only'])
for name,r in configs.items():
    mask=compute_specular_mask_bgr_block_adaptive(images[1],BlockSpatialSpecularConfig(**r['config']))
    _,counts=compare_masks(gt,mask,tuple(c['roi']))
    assert all(counts[k]==r[k] for k in ('tp','fn','fp','tn','area')),(name,counts)
    print(name,counts)
print('PASS: saved recommendations reproduce exact TP/FN/FP')
