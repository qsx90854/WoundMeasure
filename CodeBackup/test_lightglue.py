import time
import torch
import matplotlib.cm as cm
from lightglue import LightGlue, ALIKED, viz2d
from lightglue.utils import load_image


ex_inference_time = 0
# 1. 設定運算設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"目前使用的運算設備: {device}")

# 2. 載入模型
extractor = ALIKED(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features='aliked').eval().to(device)

# 3. 讀取影像 (請確認 img1.jpg 與 img2.jpg 存在)
image0 = load_image('glue_img1.jpg').to(device)
image1 = load_image('glue_img2.jpg').to(device)

# ================= 效能與耗時計算區塊 =================
print("開始執行 AI 特徵提取與匹配...")
start_time = time.time()  # 按下碼表開始計時

# 加入 torch.no_grad()：關閉梯度計算，省記憶體又加速！
with torch.no_grad():
    # Step A: 提取特徵
    ex_start_time = time.time()  # 按下碼表開始計時
    feats0 = extractor.extract(image0)
    ex_end_time = time.time()  # 按下碼表開始計時
    ex_inference_time = ex_end_time - ex_start_time
    print(f"⏱️ aliked AI 推論耗時: {ex_inference_time:.3f} 秒")
    feats1 = extractor.extract(image1)
    # Step B: 進行匹配
    matches01 = matcher({'image0': feats0, 'image1': feats1})

end_time = time.time()    # 按下碼表停止計時
inference_time = end_time - start_time
print(f"⏱️ lightglue！AI 推論耗時: {(inference_time-ex_inference_time):.3f} 秒")
print(f"⏱️ 運算完成！AI 推論耗時: {inference_time:.3f} 秒")
# ====================================================
# 6. 資料後處理
kpts0 = feats0['keypoints'][0].cpu()
kpts1 = feats1['keypoints'][0].cpu()
matches = matches01['matches'][0].cpu()
scores = matches01['scores'][0].cpu()

# 取得所有匹配座標
m_kpts0 = kpts0[matches[..., 0]]
m_kpts1 = kpts1[matches[..., 1]]

print(f"🎯 原始匹配了 {len(m_kpts0)} 個點對！")

# ================= 實戰升級：高分過濾器 & 顏色拉伸 =================
threshold = 0.8
valid_mask = scores > threshold

# 1. 抓出高分的座標與「分數」
best_kpts0 = m_kpts0[valid_mask]
best_kpts1 = m_kpts1[valid_mask]
best_scores = scores[valid_mask]

# 2. 顏色拉伸：把剩下的分數重新延展到 0~1，讓顏色差異最大化！
if len(best_scores) > 0:
    score_min = best_scores.min()
    score_max = best_scores.max()
    # 避免分母為 0 的保護機制
    if score_max > score_min:
        normalized_scores = (best_scores - score_min) / (score_max - score_min)
    else:
        normalized_scores = best_scores
else:
    normalized_scores = best_scores

# 3. 轉換為顏色陣列 (改用對比度更高的 turbo 色譜)
best_colors = cm.turbo(normalized_scores.detach().numpy())

print(f"🛡️ 過濾後 (分數 > {threshold})，留下 {len(best_kpts0)} 個最精準的點！")
# ========================================================

# 7. 視覺化結果並存檔
axes = viz2d.plot_images([image0.cpu(), image1.cpu()])

# 修改這裡：將線條寬度 (lw) 改成極細的 0.15
for i in range(len(best_kpts0)):
    viz2d.plot_matches(best_kpts0[i:i+1], best_kpts1[i:i+1], color=best_colors[i], lw=0.15)

viz2d.add_text(0, f'ALIKED+LightGlue\nRaw: {len(m_kpts0)} | Filtered: {len(best_kpts0)}\nTime: {inference_time:.2f}s')

viz2d.save_plot('match_result.png')
print("✅ 結果已儲存至 match_result.png，請查看極細彩色連線！")