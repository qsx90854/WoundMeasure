import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from wound_detector import WoundDetector


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS = [
    ("v9-c-seg", SCRIPT_DIR / "model" / "assets" / "v9-c-seg.onnx"),
    ("v9-t-seg", None),
    ("v9-t-seg_320", SCRIPT_DIR / "model" / "assets" / "v9-t-seg_320.onnx"),
]
MODEL_RUN_ORDER = ["v9-t-seg", "v9-c-seg", "v9-t-seg_320"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
OUTPUT_FPS = 5.0


def load_detectors():
    old_cwd = Path.cwd()
    detectors = {}
    try:
        os.chdir(SCRIPT_DIR)
        for model_name, model_path in MODEL_CONFIGS:
            if model_path is not None and not model_path.exists():
                raise FileNotFoundError(f"Cannot find model: {model_path}")
            detectors[model_name] = WoundDetector() if model_path is None else WoundDetector(model_path)
            model = detectors[model_name].model
            print(f"[Model] {model_name} path={model.model_path} input_shape={model.input_shape}")
    finally:
        os.chdir(old_cwd)
    return detectors


def count_detections(prediction):
    if not prediction:
        return 0
    first = prediction[0]
    if not first:
        return 0
    return len(first[0])


def add_panel_header(image, model_name, input_shape, inference_ms, detection_count):
    header_h = 44
    panel = cv2.copyMakeBorder(image, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(32, 32, 32))
    size_text = f"{int(input_shape[0])}x{int(input_shape[1])}"
    label = f"{model_name} | {size_text} | {inference_ms:.2f} ms | det {detection_count}"
    cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


def build_comparison_frame(results):
    panels = [add_panel_header(*result) for result in results]
    target_h = max(panel.shape[0] for panel in panels)

    normalized = []
    for panel in panels:
        if panel.shape[0] != target_h:
            scale = target_h / panel.shape[0]
            new_w = max(1, int(panel.shape[1] * scale))
            panel = cv2.resize(panel, (new_w, target_h), interpolation=cv2.INTER_AREA)
        normalized.append(panel)

    separator = np.full((target_h, 6, 3), 28, dtype=normalized[0].dtype)
    comparison = normalized[0]
    for panel in normalized[1:]:
        comparison = cv2.hconcat([comparison, separator, panel])
    return comparison


def prepare_output_frame(frame):
    h, w = frame.shape[:2]
    pad_bottom = h % 2
    pad_right = w % 2
    if not pad_bottom and not pad_right:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def process_frame(frame, detectors, frame_index):
    result_by_name = {}
    for model_name in MODEL_RUN_ORDER:
        detector = detectors[model_name]
        start_time = time.perf_counter()
        image = detector.predict(frame.copy())
        inference_ms = (time.perf_counter() - start_time) * 1000
        detection_count = count_detections(detector.last_output)
        input_shape = tuple(detector.model.input_shape)
        result_by_name[model_name] = (image, input_shape, inference_ms, detection_count)
        print(
            f"[Inference] frame={frame_index + 1} model={model_name} "
            f"time={inference_ms:.2f} ms detections={detection_count}"
        )

    ordered_results = []
    for model_name, _model_path in MODEL_CONFIGS:
        image, input_shape, inference_ms, detection_count = result_by_name[model_name]
        ordered_results.append((image, model_name, input_shape, inference_ms, detection_count))
    return build_comparison_frame(ordered_results)


def process_video(video_path, output_dir, detectors):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Skip] Cannot open video: {video_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    output_path = output_dir / f"{video_path.stem}_wound_detection.mp4"
    writer = None
    processed = 0

    print(f"[Video] {video_path} frames={frame_count} output={output_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output_frame = prepare_output_frame(process_frame(frame, detectors, processed))
            if writer is None:
                h, w = output_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")

            writer.write(output_frame)
            processed += 1
            if frame_count:
                print(f"[Progress] {video_path.name}: {processed}/{frame_count}")
            else:
                print(f"[Progress] {video_path.name}: {processed}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"[Done] {video_path.name}: saved {output_path} frames={processed} fps={OUTPUT_FPS:.1f}")
    return True


def iter_videos(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Batch process wound detection videos.")
    parser.add_argument("input_dir", type=Path, help="Folder containing videos.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for output videos. Defaults to <input_dir>\\wound_detection_outputs.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_dir}")

    output_dir = (args.output_dir or (input_dir / "wound_detection_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(iter_videos(input_dir))
    if not videos:
        print(f"No video files found in: {input_dir}")
        return

    detectors = load_detectors()
    for video_path in videos:
        process_video(video_path, output_dir, detectors)

    print(f"All done. Output folder: {output_dir}")


if __name__ == "__main__":
    main()
