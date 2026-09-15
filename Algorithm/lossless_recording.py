"""FFV1 recording of OpenCV's captured uint8 BGR frames, without lossy fallback."""
import math
import cv2


def open_lossless_writer(path, fps, frame_size):
    width, height = frame_size
    if (not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0 or width % 2 or height % 2):
        # OpenCV's FFmpeg writer may silently truncate an odd row/column.
        raise ValueError('無損錄影要求正的偶數寬高，避免編碼器裁掉邊緣像素')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('錄影 FPS 必須是正的有限數字')
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG,
                             cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height), True)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError('無法建立 FFV1 無損影片；請檢查 FFmpeg/FFV1 支援與儲存路徑')
    return writer
