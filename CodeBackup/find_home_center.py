import tkinter as tk
from tkinter import messagebox
import tkintermapview
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut
import numpy as np
from scipy.optimize import minimize

class MapCenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("台灣多地址中心點計算器 (地圖互動+雙模式版)")
        self.root.geometry("1000x650")
        
        self.geolocator = ArcGIS()
        self.address_markers = []
        self.center_marker = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # 左側控制面板
        control_frame = tk.Frame(self.root, width=300, padx=10, pady=10)
        control_frame.pack(side="left", fill="y")
        
        title_label = tk.Label(control_frame, text="1. 輸入地址或在地圖按【右鍵】", font=("Arial", 11, "bold"), justify="left")
        title_label.pack(anchor="w", pady=(0, 5))
        
        self.text_area = tk.Text(control_frame, height=12, width=35, font=("Arial", 10))
        self.text_area.pack(pady=5)
        self.text_area.insert("1.0", "新北市蘆洲區仁愛街108號\n新北市三重區仁安街108號\n台北市中正區重慶南路一段122號")
        
        # --- 新增：模式選擇區 ---
        self.calc_mode = tk.StringVar(value="centroid")
        mode_frame = tk.LabelFrame(control_frame, text="計算模式", font=("Arial", 10, "bold"), padx=5, pady=5)
        mode_frame.pack(fill="x", pady=10)
        
        tk.Radiobutton(mode_frame, text="重心 (整體移動距離最短)", variable=self.calc_mode, value="centroid").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="等距 (盡量公平、距離相近)", variable=self.calc_mode, value="equidistant").pack(anchor="w")
        # -------------------------
        
        calc_button = tk.Button(control_frame, text="計算中心點並更新地圖", command=self.calculate_and_draw, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
        calc_button.pack(fill="x", pady=5)
        
        clear_button = tk.Button(control_frame, text="清除所有標記與文字", command=self.clear_all, bg="#f44336", fg="white", font=("Arial", 11))
        clear_button.pack(fill="x", pady=5)
        
        self.result_label = tk.Label(control_frame, text="等待操作...", font=("Arial", 10), fg="blue", wraplength=280, justify="left")
        self.result_label.pack(pady=10, fill="x")

        # 右側地圖面板
        map_frame = tk.Frame(self.root)
        map_frame.pack(side="right", fill="both", expand=True)
        
        self.map_widget = tkintermapview.TkinterMapView(map_frame, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)
        
        self.map_widget.set_position(25.0330, 121.5654) # 預設在台北
        self.map_widget.set_zoom(11)
        self.map_widget.add_right_click_menu_command(label="在此新增地址", command=self.add_marker_from_map, pass_coords=True)

    def add_marker_from_map(self, coords):
        lat, lon = coords
        self.result_label.config(text="正在反向查詢地址...", fg="orange")
        self.root.update()
        
        try:
            location = self.geolocator.reverse((lat, lon), timeout=10)
            addr_text = location.address if location else f"{lat:.5f}, {lon:.5f}"
        except Exception:
            addr_text = f"{lat:.5f}, {lon:.5f}"
            
        current_text = self.text_area.get("1.0", tk.END).strip()
        if current_text:
            self.text_area.insert(tk.END, f"\n{addr_text}")
        else:
            self.text_area.insert(tk.END, addr_text)
            
        self.result_label.config(text=f"已新增：{addr_text}", fg="green")
        self.calculate_and_draw()

    def clear_all(self):
        self.text_area.delete("1.0", tk.END)
        self.map_widget.delete_all_marker()
        self.address_markers.clear()
        self.center_marker = None
        self.result_label.config(text="已清除所有內容。", fg="black")

    def calculate_and_draw(self):
        self.result_label.config(text="正在計算座標與更新地圖...", fg="blue")
        self.root.update()
        
        self.map_widget.delete_all_marker()
        self.address_markers.clear()
        self.center_marker = None
        
        raw_text = self.text_area.get("1.0", tk.END)
        addresses = [addr.strip() for addr in raw_text.split('\n') if addr.strip()]
        
        if not addresses:
            self.result_label.config(text="請輸入或在地圖上選擇地址。", fg="red")
            return

        lats = []
        lons = []
        failed_addresses = []
        
        for addr in addresses:
            if "," in addr and addr.replace(",","").replace(".","").replace(" ","").replace("-","").isdigit():
                try:
                    parts = addr.split(",")
                    lat, lon = float(parts[0]), float(parts[1])
                    lats.append(lat)
                    lons.append(lon)
                    self.map_widget.set_marker(lat, lon, text=addr, marker_color_circle="blue", marker_color_outside="darkblue")
                    continue
                except:
                    pass

            try:
                location = self.geolocator.geocode(addr, timeout=10)
                if location:
                    lats.append(location.latitude)
                    lons.append(location.longitude)
                    self.map_widget.set_marker(location.latitude, location.longitude, text=addr, marker_color_circle="blue", marker_color_outside="darkblue")
                else:
                    failed_addresses.append(addr)
            except Exception as e:
                failed_addresses.append(addr + f" (錯誤)")
                
        if failed_addresses:
            error_msg = "以下地址無法定位：\n" + "\n".join(failed_addresses)
            messagebox.showwarning("部分地址解析失敗", error_msg)

        if len(lats) > 0:
            # --- 核心邏輯切換 ---
            if self.calc_mode.get() == "centroid" or len(lats) < 3:
                # 模式1: 重心 (或只有1~2個點時，等距等同於重心)
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)
            else:
                # 模式2: 等距 (使用 Scipy 尋找變異數最小的點)
                # 初始猜測點設在重心
                guess = [sum(lats)/len(lats), sum(lons)/len(lons)]
                
                # 目標函數：計算候選點到所有點的距離，並回傳距離的變異數 (Variance)
                # 變異數越小，代表大家到這個點的距離越「一致」
                def distance_variance(pt):
                    dists = np.sqrt((np.array(lats) - pt[0])**2 + (np.array(lons) - pt[1])**2)
                    return np.var(dists)
                
                # 使用 Nelder-Mead 演算法尋找最佳解
                res = minimize(distance_variance, guess, method='Nelder-Mead')
                center_lat, center_lon = res.x[0], res.x[1]
            # --------------------
            
            approx_address = "未知中心點"
            try:
                center_location = self.geolocator.reverse((center_lat, center_lon), timeout=10)
                if center_location:
                    approx_address = center_location.address
            except:
                pass
            
            mode_text = "重心" if self.calc_mode.get() == "centroid" else "等距中心"
            self.center_marker = self.map_widget.set_marker(center_lat, center_lon, text=f"★ {mode_text}", marker_color_circle="red", marker_color_outside="darkred", text_color="red")
            
            self.map_widget.fit_bounding_box((min(lats), min(lons)), (max(lats), max(lons)))
            
            if len(lats) == 1 or (max(lats) - min(lats) < 0.01 and max(lons) - min(lons) < 0.01):
                 self.map_widget.set_zoom(15)
            
            self.result_label.config(text=f"計算完成 ({mode_text})！\n\n【中心座標】\n{center_lat:.5f}, {center_lon:.5f}\n\n【大約位置】\n{approx_address}", fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = MapCenterApp(root)
    root.mainloop()