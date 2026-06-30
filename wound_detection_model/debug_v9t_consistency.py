import os
import sys
from pathlib import Path

import cv2
import numpy as np

from wound_detector import WoundDetector


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model" / "assets"


def summarize(name, output):
    if not output:
        print(f"{name}: detections=0 max_score=-")
        return

    classes, bboxes, scores, masks = output[0]
    max_score = float(np.max(scores)) if len(scores) else 0.0
    print(f"{name}: detections={len(scores)} max_score={max_score:.4f}")


def predict_raw(detector, image):
    return detector.predict(image.copy(), draw_result=False)


def main():
    os.chdir(SCRIPT_DIR)

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "data" / "3.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print(f"image={image_path}")

    v9t_single = WoundDetector()
    summarize("v9-t single", predict_raw(v9t_single, image))

    v9c = WoundDetector(MODEL_DIR / "v9-c-seg.onnx")
    v9t_after_c = WoundDetector()
    v9t_320 = WoundDetector(MODEL_DIR / "v9-t-seg_320.onnx")

    summarize("v9-c", predict_raw(v9c, image))
    summarize("v9-t after loading three", predict_raw(v9t_after_c, image))
    summarize("v9-t-320", predict_raw(v9t_320, image))


if __name__ == "__main__":
    main()
