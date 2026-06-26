import cv2
import sys
import numpy as np

# ==================== 滑鼠事件狀態 ====================
drawing = False
roi_start = None
roi_end = None
roi_rect = None  # (x, y, w, h)

def mouse_callback(event, x, y, flags, param):
    global drawing, roi_start, roi_end, roi_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        roi_end = (x, y)
        roi_rect = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)
        # 決定左上角與右下角
        x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
        x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
        # 避免單純點擊而產生過小的框
        if x2 - x1 > 10 and y2 - y1 > 10:
            roi_rect = (x1, y1, x2 - x1, y2 - y1)
        else:
            roi_rect = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右鍵清除 ROI
        roi_rect = None
        roi_start = None
        roi_end = None
# =======================================================

def main():
    # ==================== 參數設定 ====================
    CAMERA_INDEX = 1
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 30
    # ===================================================

    # 開啟相機並設定參數
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        print("❌ 無法開啟相機")
        sys.exit(1)

    # 建立 SIFT 偵測器
    # 如果覺得點太多太雜，可以調整 contrastThreshold (預設 0.04) 或 edgeThreshold (預設 10)
    sift = cv2.SIFT_create()

    # 建立視窗並綁定滑鼠事件 (使用 WINDOW_NORMAL 讓視窗可縮放，且 OpenCV 會自動處理座標映射)
    cv2.namedWindow('Live SIFT Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live SIFT Detection', 960, 540)
    cv2.setMouseCallback('Live SIFT Detection', mouse_callback)

    print("✅ 相機已開啟，開始即時 SIFT 偵測")
    print("👉 操作說明：左鍵拖曳框選 ROI，右鍵清除 ROI，按 'q' 鍵離開視窗")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取相機畫面")
            break

        # SIFT 演算法需要灰階影像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 建立 ROI 遮罩
        mask = None
        if roi_rect is not None:
            mask = np.zeros_like(gray)
            x, y, w, h = roi_rect
            mask[y:y+h, x:x+w] = 255

        # 偵測 SIFT 特徵點 (僅在 mask 範圍內偵測)
        keypoints = sift.detect(gray, mask)

        # 將特徵點畫在彩色畫面上
        output_img = cv2.drawKeypoints(
            frame, 
            keypoints, 
            None, 
            color=(0, 255, 0), 
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        # 繪製 ROI 選擇框
        if drawing and roi_start is not None and roi_end is not None:
            cv2.rectangle(output_img, roi_start, roi_end, (255, 0, 0), 2)
        elif roi_rect is not None:
            x, y, w, h = roi_rect
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(output_img, "ROI Active (Right-click to clear)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        # 在畫面上方加上當前偵測到的特徵點數量
        cv2.putText(
            output_img, 
            f"SIFT Keypoints: {len(keypoints)}", 
            (30, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 255, 255), 
            3, 
            cv2.LINE_AA
        )

        # 顯示結果 (由 WINDOW_NORMAL 自動縮放，不用手動 resize)
        cv2.imshow('Live SIFT Detection', output_img)

        # 按 'q' 鍵離開
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
