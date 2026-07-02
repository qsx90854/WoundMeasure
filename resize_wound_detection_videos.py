from pathlib import Path
import argparse

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
DEFAULT_INPUT_DIR = Path("test_video_Zebra") / "wound_detection_outputs"
SCALE_DIVISOR = 3.5


def even_size(value):
    value = max(2, int(round(value)))
    return value - (value % 2)


def resize_video(input_path, output_path, divisor=SCALE_DIVISOR):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[Skip] Cannot open: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_w = even_size(src_w / divisor)
    dst_h = even_size(src_h / divisor)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (dst_w, dst_h))
    if not writer.isOpened():
        cap.release()
        print(f"[Skip] Cannot create output: {output_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        idx += 1
        if idx % 100 == 0:
            total = frame_count if frame_count > 0 else "?"
            print(f"  {input_path.name}: {idx}/{total}")

    writer.release()
    cap.release()
    print(f"[Done] {input_path.name}: {src_w}x{src_h} -> {dst_w}x{dst_h}, frames={idx}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Resize wound detection output videos by width/height divisor.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Folder containing videos. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        default="resized_div3_5",
        help="Subfolder name created under input_dir.",
    )
    parser.add_argument(
        "--divisor",
        type=float,
        default=SCALE_DIVISOR,
        help="Resize divisor for both width and height.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if args.divisor <= 0:
        raise ValueError("--divisor must be positive")

    output_dir = input_dir / args.output_subdir
    videos = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        print(f"No videos found in: {input_dir}")
        return

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Divisor: {args.divisor}")
    for video_path in videos:
        output_path = output_dir / f"{video_path.stem}_div{str(args.divisor).replace('.', '_')}.mp4"
        resize_video(video_path, output_path, args.divisor)


if __name__ == "__main__":
    main()
