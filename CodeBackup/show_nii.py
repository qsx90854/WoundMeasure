import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def show_nii_slices(nii_path):
    print(f"正在載入: {nii_path}")
    
    # 1. 讀取 NIfTI 檔案
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    print(f"影像維度: {data.shape}")
    
    # 2. 計算三個維度 (X, Y, Z) 的正中心索引值
    # 例如維度是 (256, 256, 100)，就會抓 (128, 128, 50)
    x_mid = data.shape[0] // 2
    y_mid = data.shape[1] // 2
    z_mid = data.shape[2] // 2

    # 3. 建立畫布準備畫 3 張圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 【重要設定】：因為你的數值只有 0 到 6 (標籤)
    # 如果用預設的灰階，0~6 在 0~255 的螢幕上會是一片全黑。
    # 所以我們使用 'tab10' 這個離散色帶，讓 0~6 分別顯示成完全不同的鮮豔顏色！
    cmap_style = 'tab10' 

    # 畫第一張：Sagittal 矢狀面 (從左/右側看)
    # 固定 X 軸，顯示 Y 和 Z
    axes[0].imshow(np.rot90(data[x_mid, :, :]), cmap=cmap_style)
    axes[0].set_title(f"Sagittal Slice (X = {x_mid})")
    axes[0].axis('off') # 隱藏坐標軸讓圖更乾淨

    # 畫第二張：Coronal 冠狀面 (從正前/後方看)
    # 固定 Y 軸，顯示 X 和 Z
    axes[1].imshow(np.rot90(data[:, y_mid, :]), cmap=cmap_style)
    axes[1].set_title(f"Coronal Slice (Y = {y_mid})")
    axes[1].axis('off')

    # 畫第三張：Axial 軸狀面 (從上/下方看)
    # 固定 Z 軸，顯示 X 和 Y
    axes[2].imshow(np.rot90(data[:, :, z_mid]), cmap=cmap_style)
    axes[2].set_title(f"Axial Slice (Z = {z_mid})")
    axes[2].axis('off')

    # 顯示視窗
    plt.tight_layout()
    plt.show()

# ==========================================
# 執行範例
# ==========================================
if __name__ == "__main__":
    input_file = "nii/LDCTtest006_AirRC.nii"  # 請換成你的檔案路徑
    show_nii_slices(input_file)