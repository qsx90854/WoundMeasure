from pathlib import Path
exec(compile((Path(__file__).with_name('sweep.py')).read_text(encoding='utf-8').split("evaluate('baseline',base)")[0],str(Path(__file__).with_name('sweep.py')),'exec'))
rows=json.loads((OUT/'stage1.json').read_text());seen={json.dumps(r['config'],sort_keys=True) for r in rows}
from itertools import product
from Algorithm.specular_detection import BlockSpatialSpecularConfig
for white,sat,close,gate in product((60,90),(70,120),(1,3),(10,20,30)):
    evaluate(f'combo_{len(rows)}',replace(base,whiteness_max=white,s_max=sat,close_kernel_px=close,prominence_min=gate,gray_below_v=30))
rng=np.random.default_rng(20261105)
for i in range(90):
    evaluate(f'random_{i}',replace(base,
        whiteness_max=int(rng.choice([45,60,90,120])),s_max=int(rng.choice([50,90,120,150])),
        close_kernel_px=int(rng.choice([1,3])),prominence_min=float(rng.choice([10,15,20,30])),
        prominence_mad_multiplier=float(rng.choice([1.5,2,3])),
        gray_below_v=float(rng.choice([20,30,50])),gray_min=float(rng.choice([100,120,140])),
        local_hot_min=float(rng.choice([15,25,35])),local_hot_percentile=80.,
        v_min=float(rng.choice([120,140,160])),rgb_min=float(rng.choice([120,140,160])),
        v_percentile=float(rng.choice([85,90,95])),rgb_percentile=float(rng.choice([85,90,95])),
        background_sigma=float(rng.choice([5,9]))))
(OUT/'stage2.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
for r in sorted(rows,key=lambda r:r['f1'],reverse=True)[:10]:print(json.dumps(r),flush=True)
