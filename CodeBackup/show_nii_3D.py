import nibabel as nib
import pyvista as pv
import numpy as np

def show_3d_window(nii_path, target_label=1):
    print(f"正在載入檔案: {nii_path} ...")
    img = nib.load(nii_path)
    volume_data = img.get_fdata()
    
    print(f"準備提取標籤為【{target_label}】的 3D 模型...")
    # 把你要的標籤變成 1，其他的變成 0
    binary_volume = (volume_data == target_label).astype(np.uint8)
    
    if np.max(binary_volume) == 0:
        print(f"找不到標籤 {target_label} 的資料！")
        return

    # ==========================================
    # 將資料轉換為 PyVista 的 3D 網格空間
    # ==========================================
    grid = pv.ImageData()
    grid.dimensions = binary_volume.shape
    # 將 3D 陣列攤平並塞入網格中 (order="F" 是為了對齊 Fortran/MATLAB 的座標排列)
    grid.point_data["values"] = binary_volume.flatten(order="F")

    print("正在計算 3D 表面...")
    # 提取數值為 0.5 的交界面 (因為資料只有 0 和 1)
    mesh = grid.contour(isosurfaces=[0.5])

    # ==========================================
    # 開啟 3D 互動視窗
    # ==========================================
    print("開啟 3D 視窗！(你可以用滑鼠拖曳旋轉、滾輪縮放)")
    
    plotter = pv.Plotter(title="NIfTI 3D Viewer")
    
    # 將模型加入視窗。你可以自由更改 color (顏色) 和 opacity (透明度)
    plotter.add_mesh(mesh, color="lightblue", opacity=0.9, specular=0.5)
    
    # 加上一個 XYZ 座標軸提示
    plotter.add_axes()
    
    # 顯示視窗 (程式會停在這裡，直到你關閉視窗)
    plotter.show()

# ==========================================
# 執行範例
# ==========================================
if __name__ == "__main__":
    input_file = "nii/LDCTtest006_AirRC.nii" # 請換成你的檔案路徑
    
    # 假設你想看部位 1 (你可以改成 2, 3, 4...)
    show_3d_window(input_file, target_label=1)