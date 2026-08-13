# 白色角點中心偵測方法說明

本文件說明 `analyze_hbvcam_pattern_vs_white_pixel_gt.py` 如何在 white 影片中，找出四個白色區域的中心位置。

## 1. 方法概念

程式不是尋找「最亮的單一 pixel」，也不是直接取白色區域的矩形外框中心，而是對每個白色區域計算「去除背景後的亮度加權質心」。

這種方法可以適應下列情況：

- 近距離時白色區域較大。
- 遠距離時白色區域較小。
- 白色區域邊緣模糊或失焦。
- 白色區域形狀不完全對稱。
- 區域內像素亮度不均勻。

最後得到的中心位置是帶有小數的 subpixel 座標。

## 2. 建立四個搜尋中心

程式先處理 pattern 影片，取得每一幀 ArUco pattern 的四個 subpixel 角點。

針對左、右相機的每一個角點，分別計算跨幀座標中位數，作為 white 影片的預期搜尋中心：

\[
P_{search}=Median(P_0,P_1,\ldots,P_n)
\]

使用中位數可以降低少數角點偵測異常或抖動的影響。

## 3. 局部搜尋範圍

程式在每個預期角點位置周圍建立局部搜尋區域。

預設搜尋半徑為：

\[
r=20\text{ pixels}
\]

因此局部搜尋影像大小約為：

\[
(2r+1)\times(2r+1)=41\times41\text{ pixels}
\]

可使用以下參數調整：

```powershell
--white-search-radius-px 20
```

如果白點位置和 pattern 角點位置可能相差較多，可以適度增加搜尋半徑。但搜尋範圍太大時，也可能納入其他反光或亮點。

## 4. 背景亮度與對比

程式先將局部搜尋影像轉成灰階。

局部背景亮度使用搜尋區域灰階值的第 35 百分位估計：

\[
I_{background}=P_{35}(I)
\]

局部最高亮度為：

\[
I_{peak}=\max(I)
\]

白色區域相對於背景的對比為：

\[
Contrast=I_{peak}-I_{background}
\]

預設最低對比要求為 30 灰階值：

```powershell
--white-min-contrast-gray 30
```

如果對比低於此門檻，該角點會被判定為偵測失敗。

## 5. 動態亮度門檻

程式沒有使用固定的灰階門檻，而是依照當下局部背景與最高亮度建立動態門檻：

\[
Threshold=I_{background}+q(I_{peak}-I_{background})
\]

目前預設：

\[
q=0.5
\]

對應參數為：

```powershell
--white-threshold-ratio 0.5
```

灰階值高於或等於此門檻的像素，會被列為白色區域候選像素。

這種相對門檻可以讓近距離的大光斑與遠距離的小光斑，分別依照自己的局部亮度進行判斷。

## 6. 白色連通區域

程式對通過門檻的像素執行 connected component 分析，將相連的白色像素分成不同候選區域。

目前允許的候選區域面積為：

\[
2\le Area\le300\text{ pixels}
\]

對應參數為：

```powershell
--white-min-area-px 2
--white-max-area-px 300
```

面積小於最小值的區域通常可能是感光雜訊；面積超過最大值的區域可能是大面積反光、過度曝光或其他非目標白色區域。

## 7. 亮度加權質心

對每個通過面積限制的候選區域，先將背景亮度扣除：

\[
w_i=\max(I_i-I_{background},0)
\]

其中：

- \(I_i\) 是候選區域中第 \(i\) 個像素的灰階亮度。
- \(w_i\) 是扣除背景後的亮度權重。

白色區域中心計算如下：

\[
x_c=\frac{\sum_i x_iw_i}{\sum_iw_i}
\]

\[
y_c=\frac{\sum_i y_iw_i}{\sum_iw_i}
\]

因此輸出不是整數像素，而是浮點數的 subpixel 座標，例如：

```text
(1504.98, 402.92)
```

這個位置代表白色光斑的亮度重心，不一定等於二值化區域的幾何中心或矩形外框中心。

## 8. 多個候選區域的選擇

如果同一個搜尋範圍內出現多個通過條件的白色區域，程式會綜合考慮：

- 去除背景後的總亮度。
- 候選中心與預期 ArUco 角點搜尋中心的距離。

候選分數為：

\[
Score=\frac{TotalBrightness}{1+0.25\times Distance}
\]

總亮度越高，分數越高；距離預期位置越遠，分數越低。程式最後選擇分數最高的候選區域。

## 9. 不同拍攝距離的影響

### 近距離

近距離拍攝時，一個螢幕白色 pixel 可能因為鏡頭放大、散焦、曝光和螢幕像素結構，在相機影像中形成較大的一坨白色區域。

只要區域面積沒有超過 `--white-max-area-px`，程式會使用整個候選區域的亮度加權質心。

如果 Excel 中的 `white_blob_area_px` 經常接近或超過 300，可以提高最大面積，例如：

```powershell
--white-max-area-px 600
```

### 遠距離

遠距離拍攝時，白色區域可能縮小到少數像素。只要符合以下條件，仍可計算其中心：

- 面積不小於 `--white-min-area-px`。
- 與背景的對比不小於 `--white-min-contrast-gray`。
- 位於預期角點附近的搜尋範圍內。

如果遠距離白點經常只有一個有效像素，可考慮將最小面積改成 1，但單一像素更容易受到感光雜訊影響：

```powershell
--white-min-area-px 1
```

## 10. Excel 診斷欄位

`Corner Comparison` 分頁提供以下與白點偵測有關的欄位：

| Excel 欄位 | 意義 |
|---|---|
| `white_peak_gray` | 搜尋區域內的最高灰階亮度 |
| `white_background_gray` | 使用第 35 百分位估計的背景亮度 |
| `white_contrast_gray` | 最高亮度減去背景亮度 |
| `white_blob_area_px` | 被選中的白色連通區域面積 |
| `white_distance_from_pattern_search_center_px` | 白色質心與預期 ArUco 角點位置的距離 |
| `white_same_index_x_px` | 該幀白色質心的 x 座標 |
| `white_same_index_y_px` | 該幀白色質心的 y 座標 |
| `white_same_index_vs_gt_median_error_px` | 該幀白色質心相對於跨幀白點中位數的誤差 |

`White GT Frames` 分頁則記錄每一幀左、右相機四個白色角點的完整座標與 RT 計算結果。

## 11. 診斷影片

每組 pattern/white 影片會產生一支 960×960 的診斷 MP4：

- 上半部：pattern 影片中偵測到的 ArUco subpixel 角點。
- 下半部：white 影片中計算得到的白色區域亮度加權質心。
- 紅色十字線寬為 1 pixel。
- 偵測失敗時顯示 `MISSING`。

上下畫面使用相同的預期角點位置作為裁切中心，因此可以逐幀比較 ArUco 角點和白點 Ground Truth 的位置差異。

## 12. 建議的參數調整順序

如果不同距離的白點偵測不穩定，建議依下列順序檢查：

1. 先在 `Corner Comparison` 查看 `white_blob_area_px`。
2. 確認近距離光斑是否超過 `--white-max-area-px`。
3. 確認遠距離光斑是否低於 `--white-min-area-px`。
4. 查看 `white_contrast_gray`，確認是否經常低於最低對比。
5. 查看 `white_distance_from_pattern_search_center_px`，確認搜尋半徑是否足夠。
6. 最後再調整 `--white-threshold-ratio`，避免一開始同時改動太多條件。

建議每次只調整一個參數，並比較不同距離下的偵測成功率、白點座標標準差和 pattern 對 Ground Truth 的角點誤差。
