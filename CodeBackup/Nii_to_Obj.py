import nibabel as nib
from skimage import measure
import numpy as np

def convert_nii_to_obj(nii_path, obj_path, threshold=100.0):
    print(f"正在載入檔案: {nii_path} ...")
    
    # 1. 讀取 NIfTI 檔案 (支援 .nii 與 .nii.gz)
    img = nib.load(nii_path)
    
    
    
    # 取得 3D 陣列資料 (Voxel 數值矩陣)
    volume_data = img.get_fdata()
    print(f"影像尺寸 (Voxel): {volume_data.shape}")
    print(np.max(volume_data))
    print(np.min(volume_data))
    # 2. 執行 Marching Cubes 演算法提取表面
    print(f"正在擷取 3D 表面 (使用的閾值 Threshold = {threshold}) ...")
    try:
        # verts: 頂點座標, faces: 組成面的頂點索引
        verts, faces, normals, values = measure.marching_cubes(volume_data, level=threshold)
    except ValueError as e:
        print(f"轉換失敗: {e}")
        print("提示：請檢查你的『閾值』是否設定在影像資料的數值範圍內。")
        return

    print(f"轉換成功！共產生 {len(verts)} 個頂點與 {len(faces)} 個面。")
    print(f"正在將模型匯出為: {obj_path} ...")

    # 3. 將資料寫入成標準的 .obj 格式檔案
    with open(obj_path, 'w') as f:
        # 寫入所有頂點 (v x y z)
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
        # 寫入所有面 (f v1 v2 v3)
        # 【重要細節】：OBJ 檔案的面索引是從 1 開始的，而 NumPy/Python 是從 0 開始，所以這裡必須 +1
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print("✅ OBJ 檔案匯出完成！你可以把它拖入 Unity 了。")

# ==========================================
# 執行範例
# ==========================================
if __name__ == "__main__":
    # 輸入你的 .nii 或 .nii.gz 檔案路徑
    input_file = "nii/LDCTtest006_Lobe.nii" 
    # 設定輸出的 OBJ 檔案名稱
    output_file = "medical_model.obj"
    
    # 【關鍵參數】：Threshold (閾值)
    # 這決定了你要提取哪種密度的組織。例如在 CT 掃描中：
    # - 提取骨頭：數值通常設在 200 到 400 之間
    # - 提取軟組織/皮膚：數值通常設在 -300 到 50 之間
    # 你可能需要根據你的 .nii 檔案實際數值多嘗試幾次
    target_threshold = 6.0 
    
    convert_nii_to_obj(input_file, output_file, threshold=target_threshold)