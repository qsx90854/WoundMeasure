import cv2
import numpy as np
import os
import glob
import json
import time

# ==================== 標定參數設定 ====================
# 棋盤格尺寸 (內角點數量，例如 11x8 棋盤格，內角點為 10x7)
BOARD_SIZE = (10, 16) 
# 棋盤格每個方格的真實邊長 (mm)
SQUARE_SIZE_MM = 15.0
# 存放棋盤格圖片的資料夾路徑
IMAGE_DIR = "cali_image_20260519"
# 輸出 JSON 檔名
OUTPUT_JSON = "calibration_result_c2.json"
# =====================================================

def calibrate():
    # 準備棋盤格 3D 座標
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    # 儲存偵測結果
    obj_points = []  # 3D 點
    img_points_L = [] # 左圖 2D 點
    img_points_R = [] # 右圖 2D 點
    
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"❌ 在 {IMAGE_DIR} 中找不到影像檔案。")
        return

    print(f"🔍 正在處理 {len(image_files)} 張影像...")
    
    valid_images = []
    
    for fname in image_files:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # 圖片從中間切一半：左邊是左相機，右邊是右相機
        half_w = w // 2
        img_L = img[:, :half_w]
        img_R = img[:, half_w:]
        
        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
        
        # 尋找棋盤格角點
        ret_L, corners_L = cv2.findChessboardCorners(gray_L, BOARD_SIZE, None)
        ret_R, corners_R = cv2.findChessboardCorners(gray_R, BOARD_SIZE, None)
        
        if ret_L and ret_R:
            # 亞像素精確化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_L = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
            corners_R = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            img_points_L.append(corners_L)
            img_points_R.append(corners_R)
            valid_images.append(fname)
            print(f"✅ {os.path.basename(fname)} 偵測成功")
        else:
            print(f"⚠️ {os.path.basename(fname)} 偵測失敗 (L:{ret_L}, R:{ret_R})")

    if len(img_points_L) < 5:
        print(f"❌ 有效影像不足 5 張 (目前僅 {len(img_points_L)} 張)，無法進行完整標定。")
        return

    # 1. 個別進行內部參數標定
    print("\n🎬 執行左相機內部標定...")
    ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(obj_points, img_points_L, (gray_L.shape[1], gray_L.shape[0]), None, None)
    
    print("🎬 執行右相機內部標定...")
    ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(obj_points, img_points_R, (gray_R.shape[1], gray_R.shape[0]), None, None)

    # 2. 計算每張影像的重投影誤差，找出誤差最小的 5 張
    errors = []
    for i in range(len(obj_points)):
        # 左相機誤差 (每個角點的平均歐式距離)
        imgpts2_L, _ = cv2.projectPoints(obj_points[i], rvecs_L[i], tvecs_L[i], mtx_L, dist_L)
        pts_diff_L = img_points_L[i].reshape(-1, 2) - imgpts2_L.reshape(-1, 2)
        err_L = np.mean(np.linalg.norm(pts_diff_L, axis=1))
        
        # 右相機誤差 (每個角點的平均歐式距離)
        imgpts2_R, _ = cv2.projectPoints(obj_points[i], rvecs_R[i], tvecs_R[i], mtx_R, dist_R)
        pts_diff_R = img_points_R[i].reshape(-1, 2) - imgpts2_R.reshape(-1, 2)
        err_R = np.mean(np.linalg.norm(pts_diff_R, axis=1))
        
        total_err = err_L + err_R
        errors.append((total_err, i))
    
    # 依誤差排序
    errors.sort(key=lambda x: x[0])
    best_indices = [idx for err, idx in errors[:5]]
    print(f"\n🏆 已篩選出誤差最小的 5 張影像進行外參計算：")
    for i in range(5):
        idx = best_indices[i]
        print(f"   - {os.path.basename(valid_images[idx])} (Error: {errors[i][0]:.4f})")

    # 取出 5 張精選影像的資料
    objp_best = [obj_points[i] for i in best_indices]
    imgp_L_best = [img_points_L[i] for i in best_indices]
    imgp_R_best = [img_points_R[i] for i in best_indices]

    # 3. 執行立體標定 (計算外部參數 R, T)
    print("\n🎬 執行立體標定 (Stereo Calibration)...")
    flags = cv2.CALIB_FIX_INTRINSIC # 先固定內參，只算外參
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret, mtx_L_final, dist_L_final, mtx_R_final, dist_R_final, R, T, E, F = cv2.stereoCalibrate(
        objp_best, imgp_L_best, imgp_R_best,
        mtx_L, dist_L, mtx_R, dist_R,
        (gray_L.shape[1], gray_L.shape[0]),
        criteria=criteria_stereo, flags=flags
    )

    # 4. 儲存結果為 JSON
    result = {
        "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reprojection_error": ret,
        "intrinsic_L": {
            "matrix": mtx_L_final.tolist(),
            "distortion": dist_L_final.flatten().tolist()
        },
        "intrinsic_R": {
            "matrix": mtx_R_final.tolist(),
            "distortion": dist_R_final.flatten().tolist()
        },
        "extrinsic": {
            "R": R.tolist(),
            "T": T.flatten().tolist(),
            "E": E.tolist(),
            "F": F.tolist()
        },
        "metadata": {
            "checkerboard_size": BOARD_SIZE,
            "square_size_mm": SQUARE_SIZE_MM,
            "best_5_images": [os.path.basename(valid_images[i]) for i in best_indices]
        }
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 標定成功！結果已儲存至: {OUTPUT_JSON}")
    print(f"📊 總立體重投影誤差: {ret:.4f}")

if __name__ == "__main__":
    calibrate()
