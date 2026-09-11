請根據以下「目前已實作的 Region-SIFT」製作一張繁體中文技術流程圖，實際產生並提供可下載的 PNG 圖片，不要只輸出 Mermaid 或文字。建議使用能精確排版文字的繪圖工具；白底、清晰箭頭、適合簡報、高解析度。若一張圖太擠，主流程一張 PNG，評分細節一張 PNG。圖中文字請簡潔，但必須保留以下核心邏輯與公式。不要自行改寫算法。

標題：Region-SIFT：固定 Anchor 群組的極線匹配與三角測距
版面建議：左圖建模 → 右圖候選迴圈 → 群組評分 → 幾何輸出。左右圖用不同底色，候選迴圈用框線標示；有效性判斷用菱形。

一、輸入與前處理
輸入左右影像、左圖使用者點擊 P、相機內參、相對 R/T、參考平面、基本矩陣 F。
左右影像轉灰階並依 UI 設定做 CLAHE。下面是 Region-SIFT 主體，不是原生 SIFT 的稀疏 detectAndCompute 流程。

二、左圖：固定 28 個 anchor，建立區域描述
1. 以 P 為中心取 30×30 像素的選點區域，分成 3×3 九格，每格 10×10。
2. 每格依梯度排序選高／中／低各 1 點，共 27 點；再保留原始 P 作為第 28 點。P 屬於中央格，因此中央格 4 點，其餘各格 3 點。
3. 這些是固定位置的 dense anchors，不因低紋理而刪除，也不是重新偵測少數 SIFT 關鍵點。
4. 建立一致的 Gaussian/DoG 尺度空間。每一個 anchor 獨立比較候選尺度的局部 DoG 響應，選尺度，再在該尺度估計主方向與可靠性。
   候選 KeyPoint.size 設定為 3.2、4、5、6.4、8、10、12、16 px；對齊尺度層並套用目前名義 descriptor support 上限後，實際可用約 3.20、4.03、5.08、6.40 px。各点靠近邊界時可用尺度可能更少。不是無限制向外擴張，也不是每一尺度都算 descriptor 再比。
5. 檢查所選尺度的完整 support，包括 Gaussian、梯度、方向／尺度響應及 descriptor 所需周邊範圍；若任一左 anchor 沒有有效尺度，回報無效，不縮減點數。
6. 按各自 scale、angle、octave/layer 計算 28 個 128 維 SIFT descriptor，按固定 anchor 順序保存為 D_L，形狀 28×128。
   平坦點可產生零 descriptor 且 frame 不可靠，但仍保留該列，不另外加入「平坦度特徵」。

三、右圖：固定群組幾何，在極線帶搜尋
1. 由相對 R/T、內參與參考平面計算 H_L→R；用反向映射將原始右圖 warp 到左圖座標系，並建立 warp 有效遮罩。
2. 由 F 與預測位置建立極線搜尋帶，再轉到 warped-right 座標系。預設沿線 75 個樣本、跨線 5 個樣本，步距 1 px，最多 375 個候選；搜尋帶方向跟隨極線，並非固定水平。
3. 每個候選位移 Δ：所有 anchor 共享同一個 Δ。若左圖點為 p_i = P + o_i，右圖候選點為 q_i = Q + o_i = p_i + Δ，i=1…28。
   強調：不讓各 anchor 自由找不同位移；群組保持相對位置。左右各 anchor 的 scale/angle 則可各自不同。
4. 在這 28 個固定右圖候選位置，獨立估计各點的 scale/angle，檢查邊界及 warp 的完整 support。任何點無有效 support 就排除整個候選。
5. 用各自 frame 計算 D_R(Δ)，也是 28×128；與 D_L 按列一一對應，不做跨 anchor 的最近鄰重配對。

四、群組評分（必須畫出，目前不只是 best 21/28）
逐點距離：d_i(Δ) = || D_L[i,:] - D_R(Δ)[i,:] ||_2。
分成两個 descriptor 評分分支：
A. Trim21：將 28 個 d_i 排序，取最小的 21 個平均。
B. CellAll：每一格內全部 anchor 的 d_i 先平均，再對九格等權平均。中央格包含 P，共 4 點；其餘格各 3 點。全部 28 點均參與此分支。
GroupScore(Δ) = 0.65 × Trim21(Δ) + 0.35 × CellAll(Δ)。
因此即使低梯度點在 Trim21 中被排除，仍可透過 CellAll 約束「該相對位置應該平坦或有紋理」。

另加可靠 frame 一致性懲罰：
- 比較每個左右對應點的 log2(scale_R/scale_L) 與 wrap(angle_R-angle_L)，不是強迫 28 個點本身具有同一尺度或角度。
- 只使用左右皆可靠的點對，scale 與 angle 各需至少 3 對。
- 相對群組的穩健中心計算殘差，允許整組共同尺度比／共同旋轉；不可靠的平坦 frame 跳過此懲罰，但 descriptor 仍參與上面的距離。
- scale 容許值 0.5 log2、angle 容許值 30°，殘差平方損失截頂於 1，再按可用項平均。
- FramePenalty = 0.05 × descriptor_unit × frame_loss；目前未做 descriptor 單位正規化，descriptor_unit=512。
最終 ObjectiveScore(Δ) = GroupScore(Δ) + FramePenalty(Δ) + EpipolarPenalty(Δ)。
EpipolarPenalty 是跨線偏移的可選懲罰，目前權重為 0。
選 ObjectiveScore 最小的有效候選。不要畫成只選 GroupScore 最小，也不要誤把 28×128 畫成每列獨立選不同右圖位置。

五、品質檢查與幾何輸出
- 沒有有效候選、左圖整組 descriptor 全零、或整個有效搜尋带 GroupScore 幾乎平坦 → 回報無法可靠定位，不輸出量測點。
- max_group_score 門檻可選，目前預設未啟用；次佳分數、margin 等供 Debug 分析，不要自行加上未實作的固定接受門檻。
- 找到最佳 Q*（warped-right 座標）→ 經 H_L→R 轉回原始右圖 P' → 將 P 與 P' 交給既有三角化流程 → 3D／深度。
- 主程式若啟用多右圖候選，沿用既有融合；Wound Height 再依既有高度基準平面計算。Shared Pattern Plane 是高度讀值選項，不等於把 matcher 的 homography 參考平面自動改成 Shared。
- Debug 顯示 anchor、scale/angle、support、逐點 L2／Trim21 KEEP 排名、GroupScore／ObjectiveScore。

圖下方短註：
「30×30 是 anchor 選點區域，不是 SIFT descriptor 的完整取樣範圍。」
「點數、尺度、搜尋帶、保留比例、評分權重與容許值均可 tuning。」
「這是現行流程；角度結果快取、跨點金字塔快取等加速建議尚未實作，不要畫成既有步驟。」
請直接生成 PNG，並用一小段文字說明圖中主流程與兩個評分分支。
