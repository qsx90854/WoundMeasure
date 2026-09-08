import cv2
import argparse
from datetime import datetime
import os
import time

def main():
    parser = argparse.ArgumentParser(description="UVC 雙目相機預覽與存圖工具")
    parser.add_argument("--num", type=int, default=0, help="UVC 相機編號 (預設: 0)")
    args = parser.parse_args()

    # === 關鍵設定 ===
    requested_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    open_params = [
        cv2.CAP_PROP_FOURCC,
        requested_fourcc,
        cv2.CAP_PROP_FRAME_WIDTH  ,
        2560,
        cv2.CAP_PROP_FRAME_HEIGHT ,
        1024,
        cv2.CAP_PROP_FPS ,
        int(round(30)),
    ]


    # 開啟相機
    #cap = cv2.VideoCapture(args.num, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(args.num, cv2.CAP_DSHOW, open_params)
    if not cap.isOpened():
        print(f"無法開啟相機編號 {args.num}")
        return

    
    
    #cap.set(cv2.CAP_PROP_FOURCC, requested_fourcc)  # 強制 MJPEG
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝延遲
    #cap.set(cv2.CAP_PROP_FPS, 30)

    # 確認實際參數
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"相機編號: {args.num}")
    print(f"解析度  : {width} x {height}")
    print(f"FPS 設定: {fps}")
    print(f"FourCC  : {fourcc_str}")
    print("-" * 40)
    print("按 S 存圖 | 按 ESC 或 Q 離開")

    save_dir = "captured_frames"
    os.makedirs(save_dir, exist_ok=True)

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像")
            break

        # 計算實際幀率
        frame_count += 1
        if frame_count >= 30:
            now = time.time()
            actual_fps = frame_count / (now - prev_time)
            print(f"實際 FPS: {actual_fps:.1f}")
            prev_time = now
            frame_count = 0

        # 顯示（縮小方便預覽，不影響存檔解析度）
        display = cv2.resize(frame, (1280, 512))
        cv2.imshow("Stereo Camera", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(save_dir, f"stereo_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"已儲存: {filename}")

        elif key == 27 or key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程式結束")

if __name__ == "__main__":
    main()