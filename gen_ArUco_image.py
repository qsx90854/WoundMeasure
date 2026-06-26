import cv2
import numpy as np

def generate_aruco_marker(dictionary_type, marker_id, marker_size, output_filename):
    """
    產生 ArUco 標籤並儲存為 PNG
    """
    # 1. 載入指定的 ArUco 字典
    # 注意：較新版本的 OpenCV (4.7+) 使用 getPredefinedDictionary
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)

    # 2. 產生標籤的像素矩陣
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)

    # 3. 增加白邊 (Quiet Zone) - 這對實體列印辨識非常重要！
    # 設定白邊寬度為標籤尺寸的 10%
    border_size = int(marker_size * 0.1) 
    
    marker_image_with_border = cv2.copyMakeBorder(
        marker_image,
        top=border_size, bottom=border_size, left=border_size, right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255] # 白色
    )

    # 4. 存檔為 PNG
    cv2.imwrite(output_filename, marker_image_with_border)
    print(f"成功產生標籤：{output_filename} (含白邊解析度：{marker_image_with_border.shape[1]}x{marker_image_with_border.shape[0]})")

if __name__ == "__main__":
    # 設定參數：使用 4x4 字典，ID 為 12，主體大小 500x500 像素
    ARUCO_DICT = cv2.aruco.DICT_4X4_100
    MARKER_ID = 2
    SIZE_PIXELS = 500
    FILENAME = f"aruco_dict4x4_id{MARKER_ID}.png"

    generate_aruco_marker(ARUCO_DICT, MARKER_ID, SIZE_PIXELS, FILENAME)