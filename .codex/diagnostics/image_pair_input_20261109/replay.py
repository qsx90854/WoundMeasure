"""Offline fixed raw pair integration test; never launches main UI."""
import contextlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
out = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('measurement_main', ROOT/'depth_measure_multi_aruco_sbs_camera_v7_demo_zebra_GradSIFTDebug.py')
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
main.RECORD_SAVE_DIR = str(out)
session = pickle.loads((ROOT/'.codex/diagnostics/point1325_20261027/session.pkl').read_bytes())
cap = cv2.VideoCapture(str(ROOT/'test_video_Zebra/video_20261006_105420_207407_lossless.avi'))
paths = []
for label, index in [('left', session['left_frame_idx']), ('right', session['cands'][0]['right_frame_idx'])]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok, label
    path = out/(label+'.png')
    cv2.imencode('.png', frame)[1].tofile(path)
    paths.append(path)
cap.release()
left, right = main.load_image_pair(*paths)
K, d, *_ = main.camera_algo.load_json_camera_params(main.PARAMS_JSON_PATH)
newK, _, _, process = main.camera_algo.build_undistort_processor(K, d, left.shape[1::-1], alpha=1.)
manifest = main.create_pair_session(*paths, left.shape, out, main.PARAMS_JSON_PATH)
with (out/'analysis.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
    result = main.analyze_video_frames(manifest, 1, 1, newK, d, K, main.ACTUAL_MARKER_SIZE_MM,
                                      frames_override=[right, left])
assert result is not None, 'See analysis.log for pose rejection'
assert (result['idx_A'], result['idx_B']) == (0, 1)
np.testing.assert_array_equal(result['frame_A'], right)
np.testing.assert_array_equal(result['frame_B'], left)
print('PASS fixed image pair: left/B=1, right/A=0, baseline=', result['baseline'])
print('Output keys:', sorted(result.keys()))
