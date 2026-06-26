import nibabel as nib
from skimage import measure
import numpy as np

def extract_specific_label_to_obj(nii_path, obj_path, target_label):
    print(f"正在載入檔案: {nii_path} ...")
    img = nib.load(nii_path)
    volume_data = img.get_fdata()
    
    print(f"原始資料範圍: Min={np.min(volume_data)}, Max={np.max(volume_data)}")
    print(f"準備提取標籤為【{target_label}】的部位...")

    # ==========================================
    # 【關鍵修改】：過濾出你想要的特定標籤
    # 將等於 target_label 的地方設為 1，其他地方設為 0
    # ==========================================
    binary_volume = (volume_data == target_label).astype(float)
    
    # 檢查這個標籤是否存在
    if np.max(binary_volume) == 0:
        print(f"錯誤：這個檔案裡面沒有標籤 {target_label} 的資料！")
        return

    print("開始執行 Marching Cubes...")
    # 因為資料已經變成只有 0 和 1，所以閾值 (level) 固定設為 0.5 即可完美切出邊界
    verts, faces, normals, values = measure.marching_cubes(binary_volume, level=0.5)

    print(f"產生了 {len(verts)} 個頂點與 {len(faces)} 個面。")

    # 寫入 OBJ (記得 faces 要 +1)
    with open(obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"✅ 標籤 {target_label} 已成功匯出為: {obj_path}")

# ==========================================
# 執行範例
# ==========================================
if __name__ == "__main__":
    #input_file = "nii/LDCTtest006_Lobe.nii" 
    input_file = "nii/LDCTtest006_AirRC.nii" 
    # 你可以寫一個迴圈，把 1 到 6 都分別存成一個 OBJ 檔
    for i in range(1, 7):
        output_file = f"model_part_{i}.obj"
        # 提取標籤 i，並存檔
        extract_specific_label_to_obj(input_file, output_file, target_label=i)