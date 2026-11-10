"""Raw still-image input for the existing B/left -> A/right measurement pipeline."""
from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np


def load_image_pair(left_path, right_path):
    """Read original pixels, including Unicode paths; never resize or auto-rotate."""
    paths = [Path(left_path).resolve(), Path(right_path).resolve()]
    if paths[0] == paths[1]:
        raise ValueError('Please select two different images (left and right viewpoints).')
    frames = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f'Cannot decode image: {path}')
        frames.append(image)
    if frames[0].shape != frames[1].shape:
        raise ValueError('Left/right image dimensions must match. No automatic resizing is applied.')
    return frames[0], frames[1]


def create_pair_session(left_path, right_path, image_shape, output_root, calibration_path):
    """Unique output prefix compatible with existing .txt/CSV/XLSX exporters."""
    folder = Path(output_root).resolve() / ('image_pair_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    manifest = folder / 'image_pair.json'
    manifest.write_text(json.dumps(dict(
        source_type='image_pair', left_image=str(Path(left_path).resolve()),
        right_image=str(Path(right_path).resolve()), width=int(image_shape[1]), height=int(image_shape[0]),
        calibration=str(Path(calibration_path).resolve()),
        frame_mapping={'0': 'right/A', '1': 'left/B'}, input_pixels='raw, not undistorted',
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    return str(manifest)
