"""
stereo_matching.py
==================
互動式深度量測的左右圖特徵匹配演算法模組 (自主程式抽出，行為不變)。

包含：
- patch 擷取、masked 描述子與 ZNCC 評分
- 平面單應性 warp patch 評分
- epipolar band 搜尋 (含亞像素插值)
- Precise (平面 warp 模板匹配) / Improved (Harris+SIFT) / Grad-SIFT 匹配流程
- 金字塔 ECC 亞像素精修
- RGB / Opponent SIFT 描述子

主程式 import 後會把頂部可調常數注入本模組 (見 TUNABLE_CONSTANTS)。
"""
import cv2
import numpy as np

# ---- 可調常數 (預設值與主程式相同；主程式啟動時會覆寫) ----
LEFT_PATCH_SEARCH_RADIUS = 30
RIGHT_PATCH_SEARCH_RADIUS = 40
LEFT_GRADIENT_POINTS_COUNT = 150
RIGHT_GRADIENT_POINTS_COUNT = 300
LEFT_MID_GRADIENT_POINTS_COUNT = 150
RIGHT_MID_GRADIENT_POINTS_COUNT = 300
GRAD_SIFT_MAX_RT_ADJUST_PX = 40.0
GRAD_SIFT_RATIO_TEST = 0.78
GRAD_SIFT_EPIPOLAR_TOL_PX = 3.0
GRAD_SIFT_OFFSET_MEDIAN_TOL_PX = 8.0
GRAD_SIFT_RANSAC_REPROJ_PX = 2.5
GRAD_SIFT_MIN_GROUP_INLIERS = 3
GRAD_SIFT_GUIDED_RADIUS_PX = 10.0
GRAD_SIFT_GUIDED_RATIO_TEST = 0.95
EPIPOLAR_SEARCH_HALF_LEN = 55
EPIPOLAR_SEARCH_BAND_RADIUS = 2
EPIPOLAR_SEARCH_MIN_SCORE = 0.35
EPIPOLAR_SEARCH_DESC_OK = 0.30
EPIPOLAR_SEARCH_ZNCC_OK = 0.15
EPIPOLAR_SEARCH_DESC_STRONG = 0.50
EPIPOLAR_SEARCH_ZNCC_STRONG = 0.35
EPIPOLAR_SEARCH_WEAK_FLOOR = 0.05

TUNABLE_CONSTANTS = [
    'LEFT_PATCH_SEARCH_RADIUS',
    'RIGHT_PATCH_SEARCH_RADIUS',
    'LEFT_GRADIENT_POINTS_COUNT',
    'RIGHT_GRADIENT_POINTS_COUNT',
    'LEFT_MID_GRADIENT_POINTS_COUNT',
    'RIGHT_MID_GRADIENT_POINTS_COUNT',
    'GRAD_SIFT_MAX_RT_ADJUST_PX',
    'GRAD_SIFT_RATIO_TEST',
    'GRAD_SIFT_EPIPOLAR_TOL_PX',
    'GRAD_SIFT_OFFSET_MEDIAN_TOL_PX',
    'GRAD_SIFT_RANSAC_REPROJ_PX',
    'GRAD_SIFT_MIN_GROUP_INLIERS',
    'GRAD_SIFT_GUIDED_RADIUS_PX',
    'GRAD_SIFT_GUIDED_RATIO_TEST',
    'EPIPOLAR_SEARCH_HALF_LEN',
    'EPIPOLAR_SEARCH_BAND_RADIUS',
    'EPIPOLAR_SEARCH_MIN_SCORE',
    'EPIPOLAR_SEARCH_DESC_OK',
    'EPIPOLAR_SEARCH_ZNCC_OK',
    'EPIPOLAR_SEARCH_DESC_STRONG',
    'EPIPOLAR_SEARCH_ZNCC_STRONG',
    'EPIPOLAR_SEARCH_WEAK_FLOOR',
]


def select_response_keypoints(response, u0, v0, high_count, mid_count=0, exclude_mask=None):
    """
    自響應圖 (梯度強度 / cornerMinEigenVal) 選取響應最高的 high_count 點與
    排序中段的 mid_count 點。回傳 (kpts_high, kpts_mid, 有效像素數)，KeyPoint 為全圖座標。
    """
    flat = response.astype(np.float64).flatten()
    if exclude_mask is not None:
        flat[np.asarray(exclude_mask, dtype=bool).flatten()] = -np.inf
    valid = np.where(np.isfinite(flat))[0]
    if len(valid) == 0:
        return [], [], 0
    order = valid[np.argsort(flat[valid])]
    w = response.shape[1]

    def to_kpts(idxs):
        return [cv2.KeyPoint(float(u0 + px), float(v0 + py), 31.0)
                for py, px in [divmod(int(i), w) for i in idxs]]

    kpts_high = to_kpts(order[-min(len(order), high_count):])
    kpts_mid = []
    if mid_count > 0:
        mc = min(len(order), mid_count)
        ms = max(0, len(order) // 2 - mc // 2)
        kpts_mid = to_kpts(order[ms:ms + mc])
    return kpts_high, kpts_mid, len(order)


def get_patch(img, pt, size):
    h, w = img.shape; x, y = int(round(pt[0])), int(round(pt[1]))
    x0, x1 = x - size//2, x + size//2 + 1
    y0, y1 = y - size//2, y + size//2 + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h: return None
    return img[y0:y1, x0:x1]

def _masked_patch_descriptor(gray_patch):
    """Return a normalized intensity+gradient descriptor, suppressing saturated highlights."""
    if gray_patch is None or gray_patch.size == 0:
        return None
    patch = gray_patch.astype(np.float32)
    valid = patch < 245.0
    if np.count_nonzero(valid) < patch.size * 0.55:
        return None
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    patch_z = np.zeros_like(patch, dtype=np.float32)
    grad_z = np.zeros_like(grad, dtype=np.float32)
    patch_z[valid] = patch[valid] - float(patch[valid].mean())
    grad_z[valid] = grad[valid] - float(grad[valid].mean())
    vals = np.concatenate([patch_z.reshape(-1), grad_z.reshape(-1)]).astype(np.float32)
    norm = float(np.linalg.norm(vals))
    if norm < 1e-6:
        return None
    return vals / norm

def _descriptor_similarity(desc_a, desc_b):
    if desc_a is None or desc_b is None or len(desc_a) != len(desc_b):
        return -1.0
    return float(np.dot(desc_a, desc_b))

def score_patch_match(imgA_gray, imgB_gray, pt_A, pt_B, patch_size=31):
    patch_a = get_patch(imgA_gray, pt_A, patch_size)
    patch_b = get_patch(imgB_gray, pt_B, patch_size)
    desc_a = _masked_patch_descriptor(patch_a)
    desc_b = _masked_patch_descriptor(patch_b)
    return _descriptor_similarity(desc_a, desc_b)

def score_zncc_patch_match(imgA_gray, imgB_gray, pt_A, pt_B, patch_size=31):
    patch_a = get_patch(imgA_gray, pt_A, patch_size)
    patch_b = get_patch(imgB_gray, pt_B, patch_size)
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return -1.0
    score = cv2.matchTemplate(patch_b, patch_a, cv2.TM_CCOEFF_NORMED)
    return float(score[0, 0])

def _project_homography_point(H, pt):
    p = H @ np.array([float(pt[0]), float(pt[1]), 1.0], dtype=np.float64)
    if abs(float(p[2])) < 1e-9:
        return None
    return np.array([float(p[0] / p[2]), float(p[1] / p[2])], dtype=np.float64)

def get_local_homography_warped_patch(imgA_gray, pt_A, pt_B, H_AB, patch_size=31):
    """
    Warp the left patch appearance into the right-view local orientation while
    keeping the candidate center at pt_B in the original right image.
    """
    if H_AB is None:
        return None, None
    patch_b_half = patch_size // 2
    p0 = _project_homography_point(H_AB, pt_A)
    px = _project_homography_point(H_AB, (float(pt_A[0]) + 1.0, float(pt_A[1])))
    py = _project_homography_point(H_AB, (float(pt_A[0]), float(pt_A[1]) + 1.0))
    if p0 is None or px is None or py is None:
        return None, None
    J = np.column_stack((px - p0, py - p0)).astype(np.float64)
    if abs(float(np.linalg.det(J))) < 1e-6:
        return None, None
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return None, None

    coords = np.arange(-patch_b_half, patch_b_half + 1, dtype=np.float32)
    dx, dy = np.meshgrid(coords, coords)
    offsets_r = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=0).astype(np.float64)
    offsets_l = J_inv @ offsets_r
    map_x = (float(pt_A[0]) + offsets_l[0]).reshape(patch_size, patch_size).astype(np.float32)
    map_y = (float(pt_A[1]) + offsets_l[1]).reshape(patch_size, patch_size).astype(np.float32)
    hA, wA = imgA_gray.shape[:2]
    valid = (map_x >= 0) & (map_x <= wA - 1) & (map_y >= 0) & (map_y <= hA - 1)
    if np.count_nonzero(valid) < patch_size * patch_size * 0.55:
        return None, None
    warped = cv2.remap(
        imgA_gray, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )
    warped = warped.astype(np.uint8)
    warped[~valid] = 255
    return warped, valid

def score_warped_patch_match(imgA_gray, imgB_gray, pt_A, pt_B, H_AB, patch_size=31):
    patch_a, _valid = get_local_homography_warped_patch(imgA_gray, pt_A, pt_B, H_AB, patch_size)
    patch_b = get_patch(imgB_gray, pt_B, patch_size)
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return -1.0
    desc_a = _masked_patch_descriptor(patch_a)
    desc_b = _masked_patch_descriptor(patch_b)
    return _descriptor_similarity(desc_a, desc_b)

def score_warped_zncc_patch_match(imgA_gray, imgB_gray, pt_A, pt_B, H_AB, patch_size=31):
    patch_a, valid = get_local_homography_warped_patch(imgA_gray, pt_A, pt_B, H_AB, patch_size)
    patch_b = get_patch(imgB_gray, pt_B, patch_size)
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return -1.0
    valid = valid & (patch_a < 245) & (patch_b < 245)
    if np.count_nonzero(valid) < patch_size * patch_size * 0.55:
        return -1.0
    a = patch_a.astype(np.float32)[valid]
    b = patch_b.astype(np.float32)[valid]
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return -1.0
    return float(np.dot(a, b) / denom)

def project_point_to_line(pt, line):
    a, b, c = line
    denom = a * a + b * b
    if denom <= 1e-12:
        return np.array(pt, dtype=np.float32)
    x, y = float(pt[0]), float(pt[1])
    signed = (a * x + b * y + c) / denom
    return np.array([x - a * signed, y - b * signed], dtype=np.float32)

def _masked_zncc_from_patches(patch_a, valid_a, patch_b):
    """與 score_warped_zncc_patch_match 相同的 masked ZNCC，但直接使用已快取的左圖 patch。"""
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return -1.0
    valid = (patch_a < 245) & (patch_b < 245)
    if valid_a is not None:
        valid = valid_a & valid
    if np.count_nonzero(valid) < patch_a.size * 0.55:
        return -1.0
    a = patch_a.astype(np.float32)[valid]
    b = patch_b.astype(np.float32)[valid]
    a -= float(a.mean())
    b -= float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return -1.0
    return float(np.dot(a, b) / denom)

def search_match_on_epipolar_band(imgA_gray, imgB_gray, pt_A, seed_pt_B, F,
                                  patch_size=31,
                                  half_len=EPIPOLAR_SEARCH_HALF_LEN,
                                  band_radius=EPIPOLAR_SEARCH_BAND_RADIUS,
                                  min_score=EPIPOLAR_SEARCH_MIN_SCORE,
                                  search_roi=None,
                                  H_AB=None):
    """
    Use the existing candidate match only as a seed, then search along pt_A's
    epipolar band for the best combined descriptor and ZNCC patch match.
    左圖 patch/描述子只依賴 pt_A 與 H_AB，整條 band 共用一份快取；
    原始模式的 ZNCC 以單次 matchTemplate 向量化，最佳點沿極線做拋物線亞像素插值。
    """
    if seed_pt_B is None or F is None:
        return None, 0.0
    tmpl = get_patch(imgA_gray, pt_A, patch_size)
    warped_a, warped_valid = None, None
    if H_AB is not None:
        if tmpl is None:
            return None, 0.0
        warped_a, warped_valid = get_local_homography_warped_patch(imgA_gray, pt_A, seed_pt_B, H_AB, patch_size)
        if warped_a is None:
            return None, 0.0
        desc_a = _masked_patch_descriptor(warped_a)
    else:
        desc_a = _masked_patch_descriptor(tmpl)
    if desc_a is None:
        return None, 0.0

    h, w = imgB_gray.shape[:2]
    line = F @ np.array([pt_A[0], pt_A[1], 1.0], dtype=np.float64)
    a, b, c = line
    line_norm = float(np.hypot(a, b))
    if line_norm < 1e-9:
        return None, 0.0

    line = line / line_norm
    direction = np.array([-line[1], line[0]], dtype=np.float32)
    normal = np.array([line[0], line[1]], dtype=np.float32)
    center = project_point_to_line(seed_pt_B, line)

    if search_roi is not None:
        rx, ry, rw, rh = search_roi
        roi_x0, roi_y0 = float(rx), float(ry)
        roi_x1, roi_y1 = float(rx + rw), float(ry + rh)

    half = patch_size // 2
    s_vals = np.arange(-half_len, half_len + 0.5, 1.0)
    n_vals = list(range(-band_radius, band_radius + 1))

    # 先蒐集通過 ROI 與影像邊界檢查的候選像素位置
    cand_list = []  # (si, ni, x, y)
    for si, s in enumerate(s_vals):
        base = center + direction * float(s)
        for ni, n in enumerate(n_vals):
            pt = base + normal * float(n)
            x, y = int(round(pt[0])), int(round(pt[1]))
            if search_roi is not None and not (roi_x0 <= x <= roi_x1 and roi_y0 <= y <= roi_y1):
                continue
            if x < half or y < half or x >= w - half or y >= h - half:
                continue
            cand_list.append((si, ni, x, y))

    if not cand_list:
        return None, -1.0

    # 原始 (無 H_AB) 模式的 ZNCC：對候選包圍盒單次 matchTemplate，再逐點取樣
    zncc_map, map_x0, map_y0 = None, 0, 0
    if H_AB is None and tmpl is not None:
        xs = [cpt[2] for cpt in cand_list]
        ys = [cpt[3] for cpt in cand_list]
        map_x0, map_y0 = min(xs) - half, min(ys) - half
        x1_roi, y1_roi = max(xs) + half + 1, max(ys) + half + 1
        roi_img = imgB_gray[map_y0:y1_roi, map_x0:x1_roi]
        if roi_img.shape[0] >= patch_size and roi_img.shape[1] >= patch_size:
            zncc_map = cv2.matchTemplate(roi_img, tmpl, cv2.TM_CCOEFF_NORMED)

    best_pt = None
    best_score = -1.0
    best_si, best_ni = -1, -1
    score_grid = np.full((len(s_vals), len(n_vals)), -np.inf, dtype=np.float64)
    for si, ni, x, y in cand_list:
        patch_b = get_patch(imgB_gray, (x, y), patch_size)
        if patch_b is None:
            continue
        desc_b = _masked_patch_descriptor(patch_b)
        desc_score = _descriptor_similarity(desc_a, desc_b)
        if H_AB is not None:
            zncc_score = _masked_zncc_from_patches(warped_a, warped_valid, patch_b)
        elif zncc_map is not None:
            zncc_score = float(zncc_map[y - half - map_y0, x - half - map_x0])
        else:
            zncc_score = -1.0

        desc_pos = max(0.0, float(desc_score))
        zncc_pos = max(0.0, float(zncc_score))
        both_ok = desc_score >= EPIPOLAR_SEARCH_DESC_OK and zncc_score >= EPIPOLAR_SEARCH_ZNCC_OK
        desc_rescue = desc_score >= EPIPOLAR_SEARCH_DESC_STRONG and zncc_score >= EPIPOLAR_SEARCH_WEAK_FLOOR
        zncc_rescue = zncc_score >= EPIPOLAR_SEARCH_ZNCC_STRONG and desc_score >= EPIPOLAR_SEARCH_WEAK_FLOOR
        if not (both_ok or desc_rescue or zncc_rescue):
            continue

        # Reward agreement, but still allow a strong score to rescue a slightly weak one.
        score = desc_pos + zncc_pos + 0.25 * min(desc_pos, zncc_pos)
        score_grid[si, ni] = score
        if score > best_score:
            best_score = score
            best_pt = np.array([float(x), float(y)], dtype=np.float32)
            best_si, best_ni = si, ni

    if best_pt is None or best_score < min_score * 0.6:
        return None, best_score

    # 沿極線方向做拋物線亞像素插值 (ECC 關閉時也能得到亞像素結果)
    if 0 < best_si < len(s_vals) - 1:
        s_m = score_grid[best_si - 1, best_ni]
        s_p = score_grid[best_si + 1, best_ni]
        if np.isfinite(s_m) and np.isfinite(s_p):
            denom_p = s_m - 2.0 * best_score + s_p
            if abs(denom_p) > 1e-9:
                delta = 0.5 * (s_m - s_p) / denom_p
                if abs(delta) <= 0.5:
                    best_pt = best_pt + direction * float(delta)

    best_pt = project_point_to_line(best_pt, line)
    return best_pt, best_score

def point_in_roi(pt, roi):
    if pt is None or roi is None:
        return True
    x, y = float(pt[0]), float(pt[1])
    rx, ry, rw, rh = roi
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def plane_homography_from_cand(cand, K_L):
    if cand.get('plane_n') is None or cand.get('plane_c') is None:
        return None
    d_plane = float(np.dot(cand['plane_n'], cand['plane_c']))
    if abs(d_plane) <= 1e-6:
        return None
    return cand['K_R'] @ (
        cand['R_rel'] + (cand['t_rel'] @ cand['plane_n'].reshape(1, 3)) / d_plane
    ) @ np.linalg.inv(K_L)

def predict_right_seed_from_geometry(pt_A, cand, K_L, return_debug=False):
    """Predict a right-image seed from RT and the marker/global plane when available."""
    u, v = float(pt_A[0]), float(pt_A[1])
    debug = {'source': 'RawSeed', 'd_plane': None, 'hom_w': None}
    if cand.get('plane_n') is not None and cand.get('plane_c') is not None:
        d_plane = float(np.dot(cand['plane_n'], cand['plane_c']))
        debug['d_plane'] = d_plane
        if abs(d_plane) > 1e-6:
            H_AB = plane_homography_from_cand(cand, K_L)
            if H_AB is None:
                seed = np.array([u, v], dtype=np.float32)
                return (seed, "RawSeed", debug) if return_debug else (seed, "RawSeed")
            pt_h = H_AB @ np.array([u, v, 1.0], dtype=np.float64)
            debug['hom_w'] = float(pt_h[2])
            if abs(pt_h[2]) > 1e-9:
                debug['source'] = 'PlaneSeed'
                seed = np.array([pt_h[0] / pt_h[2], pt_h[1] / pt_h[2]], dtype=np.float32)
                return (seed, "PlaneSeed", debug) if return_debug else (seed, "PlaneSeed")
    seed = np.array([u, v], dtype=np.float32)
    return (seed, "RawSeed", debug) if return_debug else (seed, "RawSeed")

def enforce_point_on_epipolar(pt_A, pt_B, F):
    if pt_B is None or F is None:
        return pt_B
    line = F @ np.array([pt_A[0], pt_A[1], 1.0], dtype=np.float64)
    denom = float(line[0] * line[0] + line[1] * line[1])
    if denom < 1e-9:
        return pt_B
    dist = (line[0] * pt_B[0] + line[1] * pt_B[1] + line[2]) / np.sqrt(denom)
    return np.array([
        pt_B[0] - line[0] / np.sqrt(denom) * dist,
        pt_B[1] - line[1] / np.sqrt(denom) * dist
    ], dtype=np.float32)

def compute_rgb_sift_descriptors(img_bgr, kpts, sift_detector):
    """
    計算指定 KeyPoints 在 BGR 影像的 R, G, B 通道上的 SIFT 描述子，並予以串接。
    回傳的描述子維度為 N x 384 (128 * 3)。
    """
    if not kpts or img_bgr is None:
        return None, None
        
    # 分離 B, G, R 通道
    b, g, r = cv2.split(img_bgr)
    
    # 分別對三個通道計算 SIFT 描述子
    _, des_r = sift_detector.compute(r, kpts)
    _, des_g = sift_detector.compute(g, kpts)
    _, des_b = sift_detector.compute(b, kpts)
    
    # 邊界狀況檢查
    if des_r is None or des_g is None or des_b is None:
        return None, None
    if len(des_r) != len(kpts) or len(des_g) != len(kpts) or len(des_b) != len(kpts):
        return None, None
        
    # 串接描述子 (維度: N x 384)
    des_rgb = np.hstack([des_r, des_g, des_b])
    return kpts, des_rgb

def compute_opponent_sift_descriptors(img_bgr, kpts, sift_detector):
    """
    計算指定 KeyPoints 在 BGR 影像的 Opponent 色彩空間 (O1, O2, O3) 通道上的 SIFT 描述子並串接。
    回傳的描述子維度為 N x 384。
    """
    if not kpts or img_bgr is None:
        return None, None
        
    # 分離 B, G, R 通道
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    
    # O1 = (R - G) / sqrt(2)
    o1 = (r - g) / np.sqrt(2.0)
    o1 = ((o1 + 255.0 / np.sqrt(2.0)) / (510.0 / np.sqrt(2.0)) * 255.0).astype(np.uint8)
    
    # O2 = (R + G - 2*B) / sqrt(6)
    o2 = (r + g - 2.0 * b) / np.sqrt(6.0)
    o2 = ((o2 + 510.0 / np.sqrt(6.0)) / (1020.0 / np.sqrt(6.0)) * 255.0).astype(np.uint8)
    
    # O3 = (R + G + B) / sqrt(3)
    o3 = (r + g + b) / np.sqrt(3.0)
    o3 = (o3 / (765.0 / np.sqrt(3.0)) * 255.0).astype(np.uint8)
    
    # 分別對三個通道計算 SIFT 描述子
    _, des_o1 = sift_detector.compute(o1, kpts)
    _, des_o2 = sift_detector.compute(o2, kpts)
    _, des_o3 = sift_detector.compute(o3, kpts)
    
    if des_o1 is None or des_o2 is None or des_o3 is None:
        return None, None
    if len(des_o1) != len(kpts) or len(des_o2) != len(kpts) or len(des_o3) != len(kpts):
        return None, None
        
    des_opponent = np.hstack([des_o1, des_o2, des_o3])
    return kpts, des_opponent

def check_color_histogram_similarity(left_bgr, pt_A, pt_B, cand, patch_size=16, threshold=0.45):
    """
    計算左圖 pt_A (自 left_bgr 取得，BGR格式)
    與右圖 pt_B (從 cand['rgb'] 中取得，RGB格式) 的色彩直方圖巴氏距離。
    """
    if left_bgr is None or 'rgb' not in cand:
        return True
        
    h_A, w_A = left_bgr.shape[:2]
    h_B, w_B = cand['rgb'].shape[:2]
    
    xA, yA = int(round(pt_A[0])), int(round(pt_A[1]))
    xB, yB = int(round(pt_B[0])), int(round(pt_B[1]))
    
    r = patch_size // 2
    
    if xA - r < 0 or xA + r >= w_A or yA - r < 0 or yA + r >= h_A:
        return False
    if xB - r < 0 or xB + r >= w_B or yB - r < 0 or yB + r >= h_B:
        return False
        
    patch_A = left_bgr[yA-r:yA+r, xA-r:xA+r]
    patch_B = cand['rgb'][yB-r:yB+r, xB-r:xB+r]
    
    hsv_A = cv2.cvtColor(patch_A, cv2.COLOR_BGR2HSV)
    hsv_B = cv2.cvtColor(patch_B, cv2.COLOR_RGB2HSV)
    
    hist_A = cv2.calcHist([hsv_A], [0, 1], None, [18, 16], [0, 180, 0, 256])
    hist_B = cv2.calcHist([hsv_B], [0, 1], None, [18, 16], [0, 180, 0, 256])
    
    cv2.normalize(hist_A, hist_A, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_B, hist_B, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    dist = cv2.compareHist(hist_A, hist_B, cv2.HISTCMP_BHATTACHARYYA)
    return dist <= threshold

def find_precise_match(imgA_gray, imgB_gray, pt_A, F, K_L, K_R, R_rel, t_rel, plane_normal, plane_center):
    if plane_normal is None or plane_center is None: return None
    h, w = imgA_gray.shape
    d = np.dot(plane_normal, plane_center)
    H_3D = R_rel + (t_rel @ plane_normal.reshape(1, 3)) / d
    H_AB = K_R @ H_3D @ np.linalg.inv(K_L.astype(np.float64))
    H_BA = np.linalg.inv(H_AB)
    imgB_warped = cv2.warpPerspective(imgB_gray, H_BA, (w, h), flags=cv2.INTER_LINEAR)
    patch_size = 20
    u, v = int(round(pt_A[0])), int(round(pt_A[1]))
    u0, u1 = max(0, u-patch_size), min(w, u+patch_size+1)
    v0, v1 = max(0, v-patch_size), min(h, v+patch_size+1)
    patch_l = imgA_gray[v0:v1, u0:u1]
    if patch_l.size == 0: return None
    search = 45
    x0, y0 = max(0, u-search-patch_size), max(0, v-search-patch_size)
    x1, y1 = min(w, u+search+patch_size+1), min(h, v+search+patch_size+1)
    roi_r = imgB_warped[y0:y1, x0:x1]
    if roi_r.shape[0] < patch_l.shape[0] or roi_r.shape[1] < patch_l.shape[1]: return None
    res = cv2.matchTemplate(roi_r, patch_l, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.4: return None
    pt_w_h = np.array([x0 + max_loc[0] + patch_size, y0 + max_loc[1] + patch_size, 1.0])
    pt_B_h = H_AB @ pt_w_h
    return (float(pt_B_h[0]/pt_B_h[2]), float(pt_B_h[1]/pt_B_h[2]))

def pyramid_ecc_refinement(imgA_gray, imgB_gray, pt_A, pt_B, patch_size_tmpl=45, patch_size_roi=91):
    """
    使用二層 Gaussian 金字塔 (Coarse-to-Fine) 的 ECC 亞像素級精修
    """
    tmpl = get_patch(imgA_gray, pt_A, patch_size_tmpl)
    roi = get_patch(imgB_gray, pt_B, patch_size_roi)
    
    if tmpl is None or roi is None:
        return pt_B, "+ECC失敗(Patch無效)"
        
    # 第一層 (降採樣至 1/2)
    tmpl_down = cv2.pyrDown(tmpl)
    roi_down = cv2.pyrDown(roi)
    
    # 初始平移矩陣，預估中心對齊的偏置量
    init_offset_x = (patch_size_roi - patch_size_tmpl) / 2.0
    init_offset_y = (patch_size_roi - patch_size_tmpl) / 2.0
    
    warp_down = np.eye(2, 3, dtype=np.float32)
    warp_down[0, 2] = init_offset_x / 2.0
    warp_down[1, 2] = init_offset_y / 2.0
    
    # 低解析度下進行粗對齊 (限制較少迭代次數)
    criteria_down = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-3)
    try:
        _, warp_down = cv2.findTransformECC(tmpl_down, roi_down, warp_down, cv2.MOTION_TRANSLATION, criteria_down)
        # 將低解析度估計的位移縮放回原圖尺寸
        warp_up = np.eye(2, 3, dtype=np.float32)
        warp_up[0, 2] = warp_down[0, 2] * 2.0
        warp_up[1, 2] = warp_down[1, 2] * 2.0
    except:
        # 降採樣失敗則退回直接用原圖預估位移初始化
        warp_up = np.eye(2, 3, dtype=np.float32)
        warp_up[0, 2] = init_offset_x
        warp_up[1, 2] = init_offset_y
        
    # 第二層 (原圖解析度精修)
    criteria_up = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp_final = cv2.findTransformECC(tmpl, roi, warp_up, cv2.MOTION_TRANSLATION, criteria_up)
        offset_shift_x = warp_final[0, 2] - init_offset_x
        offset_shift_y = warp_final[1, 2] - init_offset_y
        refined_pt_B = np.array([pt_B[0] + offset_shift_x, pt_B[1] + offset_shift_y])
        return refined_pt_B, "+ECC(Improved)"
    except:
        return pt_B, "+ECC失敗"

def run_improved_matching_flow(imgA_gray, imgB_gray, u, v, cand, K_L, use_hamming, orb, sift,
                               use_color_hist=False, use_rgb_sift=False, use_opponent_sift=False,
                               left_spec_mask=None, right_spec_mask=None, reject_specular_candidates=False,
                               left_bgr=None):
    """
    改良版局部特徵匹配流程 (高光遮罩 + Harris Corner 響應 + 收緊幾何門檻)
    """
    ui, vi = int(round(u)), int(round(v))
    sobel_range = LEFT_PATCH_SEARCH_RADIUS
    
    # 1. 提取左圖 Patch 並進行高光遮罩與 Harris Corner 篩選
    v0_A = max(0, vi - sobel_range)
    v1_A = min(imgA_gray.shape[0], vi + sobel_range + 1)
    u0_A = max(0, ui - sobel_range)
    u1_A = min(imgA_gray.shape[1], ui + sobel_range + 1)
    patch_g = imgA_gray[v0_A:v1_A, u0_A:u1_A]
    
    if patch_g.size == 0:
        return None, "Improved-Failed", None, None, None
        
    # 計算高光遮罩 (灰度值 >= 220)
    specular_mask = (patch_g >= 220)
    if reject_specular_candidates and left_spec_mask is not None:
        specular_mask = specular_mask | (left_spec_mask[v0_A:v1_A, u0_A:u1_A] > 0)
    
    # 使用 cornerMinEigenVal (Harris 響應的基礎) 提取特徵顯著度，排除高光遮罩
    corner_resp = cv2.cornerMinEigenVal(patch_g, blockSize=3, ksize=3)
    kpts_inj, _, _ = select_response_keypoints(corner_resp, u0_A, v0_A,
                                               LEFT_GRADIENT_POINTS_COUNT, 0, specular_mask)

    if not kpts_inj:
        return None, "Improved-NoFeatures", None, None, None
                
    if use_hamming:
        _, des_inj = orb.compute(imgA_gray, kpts_inj)
    else:
        if use_rgb_sift:
            _, des_inj = compute_rgb_sift_descriptors(left_bgr, kpts_inj, sift)
        elif use_opponent_sift:
            _, des_inj = compute_opponent_sift_descriptors(left_bgr, kpts_inj, sift)
        else:
            _, des_inj = sift.compute(imgA_gray, kpts_inj)
        
    if des_inj is None or len(des_inj) == 0:
        return None, "Improved-NoDescriptor", None, None, None

    # 2. 預估右圖投影位置
    u_exp, v_exp = u, v
    if cand['plane_n'] is not None and cand['plane_c'] is not None:
        d_plane = np.dot(cand['plane_n'], cand['plane_c'])
        if abs(d_plane) > 1e-6:
            H_AB = cand['K_R'] @ (cand['R_rel'] + (cand['t_rel'] @ cand['plane_n'].reshape(1, 3)) / d_plane) @ np.linalg.inv(K_L)
            pt_exp = H_AB @ np.array([u, v, 1.0])
            if abs(pt_exp[2]) > 1e-6:
                u_exp, v_exp = pt_exp[0]/pt_exp[2], pt_exp[1]/pt_exp[2]

    # 3. 提取右圖 Patch 並進行高光遮罩與 Harris Corner 篩選
    rad = RIGHT_PATCH_SEARCH_RADIUS
    uei, vei = int(round(u_exp)), int(round(v_exp))
    u0_B = max(0, uei - rad)
    u1_B = min(imgB_gray.shape[1], uei + rad)
    v0_B = max(0, vei - rad)
    v1_B = min(imgB_gray.shape[0], vei + rad)
    patch_r = imgB_gray[v0_B:v1_B, u0_B:u1_B]
    
    if patch_r.size == 0:
        return None, "Improved-Failed", None, None, None
        
    specular_mask_r = (patch_r >= 220)
    if reject_specular_candidates and right_spec_mask is not None:
        specular_mask_r = specular_mask_r | (right_spec_mask[v0_B:v1_B, u0_B:u1_B] > 0)
    corner_resp_r = cv2.cornerMinEigenVal(patch_r, blockSize=3, ksize=3)
    kpts_r, _, _ = select_response_keypoints(corner_resp_r, u0_B, v0_B,
                                             RIGHT_GRADIENT_POINTS_COUNT, 0, specular_mask_r)

    if not kpts_r:
        return None, "Improved-NoFeaturesB", None, None, None
              
    g_kptsB = [kp.pt for kp in kpts_r]
    g_rect = (u0_B, v0_B, u1_B - u0_B, v1_B - v0_B)
    
    if use_hamming:
        _, des_r = orb.compute(imgB_gray, kpts_r)
        if des_r is None or len(des_r) == 0:
            return None, "Improved-NoDescriptorB", None, None, g_rect
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des_inj, des_r)
        good = [m for m in matches if m.distance < 100]
    else:
        if use_rgb_sift:
            imgB_bgr = cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR)
            _, des_r = compute_rgb_sift_descriptors(imgB_bgr, kpts_r, sift)
        elif use_opponent_sift:
            imgB_bgr = cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR)
            _, des_r = compute_opponent_sift_descriptors(imgB_bgr, kpts_r, sift)
        else:
            _, des_r = sift.compute(imgB_gray, kpts_r)
            
        if des_r is None or len(des_r) == 0:
            return None, "Improved-NoDescriptorB", None, None, g_rect
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.match(des_inj, des_r)
        sift_thres = 780 if (use_rgb_sift or use_opponent_sift) else 450
        good = [m for m in matches if m.distance < sift_thres]
        
    # 4. 幾何過濾與 RANSAC 估計 (收緊門檻)
    pts_info = []
    for m in good:
        pL, pR = np.array(kpts_inj[m.queryIdx].pt), np.array(kpts_r[m.trainIdx].pt)
        if np.linalg.norm(pL - np.array([u, v])) < 50:
            if use_color_hist:
                if not check_color_histogram_similarity(left_bgr, pL, pR, cand, patch_size=16, threshold=0.45):
                    continue
            pts_info.append({'pL': pL, 'pR': pR, 'off': pR - pL})
            
    # 收緊的視差過濾閾值 (2.5 像素)
    if len(pts_info) >= 3:
        offs = np.array([x['off'] for x in pts_info])
        med_off = np.median(offs, axis=0)
        pts_info = [x for x in pts_info if np.linalg.norm(x['off'] - med_off) < 2.5]
        
    if not pts_info:
        return None, "Improved-NoInliers", None, None, g_rect
        
    ptsA_m = np.array([x['pL'] for x in pts_info], dtype=np.float32)
    ptsB_m = np.array([x['pR'] for x in pts_info], dtype=np.float32)
    mapped = None
    
    # 降低 RANSAC 重投影誤差閾值至 2.0 像素
    if len(ptsA_m) >= 6:
        H_local, _ = cv2.findHomography(ptsA_m, ptsB_m, cv2.RANSAC, 2.0)
        if H_local is not None:
            pt_h = H_local @ np.array([u, v, 1.0])
            if abs(pt_h[2]) > 1e-6:
                mapped = np.array([pt_h[0]/pt_h[2], pt_h[1]/pt_h[2]])
    if mapped is None and len(ptsA_m) >= 3:
        M_local, _ = cv2.estimateAffinePartial2D(ptsA_m, ptsB_m, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M_local is not None:
            pt_a = M_local @ np.array([u, v, 1.0])
            mapped = pt_a[:2]
    if mapped is None:
        wts = 1.0 / (np.sum((ptsA_m - np.array([u, v]))**2, axis=1) + 1e-5)
        mapped = np.array([u, v]) + np.sum((ptsB_m - ptsA_m) * wts[:, np.newaxis], axis=0) / np.sum(wts)
        
    if use_rgb_sift:
        method_name = "RGB-SIFT (Improved)"
    elif use_opponent_sift:
        method_name = "Opponent-SIFT (Improved)"
    else:
        method_name = "Grad-SIFT (Improved)"
    return mapped, method_name, ptsA_m, ptsB_m, g_rect

def run_grad_sift_matching_flow(imgA_gray, imgB_gray, u, v, cand, K_L, view_state, orb, sift,
                                left_bgr, right_bgr, left_spec_mask=None, right_spec_mask=None,
                                is_best_cand=True, left_cache=None):
    """
    Grad-SIFT 匹配流程 (自 compute_measure 內嵌區塊抽出，行為不變)：
    取點選點周圍梯度最高/中段的像素為特徵點，計算 SIFT/ORB 描述子做高、中梯度雙群匹配，
    經極線、RT seed、視差中位數與 RANSAC 過濾後加權融合出右圖對應點。
    注意：filter_specular* 開啟時會就地塗黑 left_bgr/right_bgr 的高光區 (供 UI 顯示)。
    回傳 dict: m_pt/method/g_ptsA/g_ptsB/g_groups/g_refA/g_refB/g_refA_groups/
    g_refB_groups/g_kptsB/g_rect/reject_reason
    """
    m_pt, method = None, ""
    g_ptsA = g_ptsB = g_groups = None
    g_refA = g_refB = g_refA_groups = g_refB_groups = None
    g_kptsB = None
    g_rect = None
    rt_bound_reject_reason = None
    ui, vi = int(round(u)), int(round(v))
    sobel_range = LEFT_PATCH_SEARCH_RADIUS
    v0_A = max(0, vi - sobel_range)
    v1_A = min(imgA_gray.shape[0], vi + sobel_range + 1)
    u0_A = max(0, ui - sobel_range)
    u1_A = min(imgA_gray.shape[1], ui + sobel_range + 1)
    patch_g = imgA_gray[v0_A:v1_A, u0_A:u1_A]
    # 左側特徵在同一次點擊的多候選影格間完全相同，透過 left_cache 重用。
    # filter_specular* 會就地塗黑 left_bgr (跨候選順序相依的既有行為)，該情況停用快取以維持原結果。
    cache_ok = (left_cache is not None
                and not view_state.get('filter_specular', False)
                and not view_state.get('filter_specular_hsv_mser', False))
    lc = left_cache.get('grad_left') if cache_ok else None
    if patch_g.size > 0:
        if lc is not None:
            kpts_inj_high = lc['kpts_high']
            kpts_inj_mid = lc['kpts_mid']
            sorted_left_idx = lc['sorted_left_idx']
        else:
            gx, gy = cv2.Sobel(patch_g, cv2.CV_32F, 1, 0), cv2.Sobel(patch_g, cv2.CV_32F, 0, 1)
            mag = cv2.sqrt(gx**2 + gy**2)
            if view_state.get('filter_specular_hsv_mser', False):
                # 1. 結合 HSV 進行高光判斷
                patch_L_color = left_bgr[v0_A:v1_A, u0_A:u1_A]
                patch_L_hsv = cv2.cvtColor(patch_L_color, cv2.COLOR_BGR2HSV)
                H_L, S_L, V_L = cv2.split(patch_L_hsv)

                # 2. 自適應雙閾值
                mean_v = np.mean(V_L)
                std_v = np.std(V_L)
                high_light_mask = (V_L > 190) & (S_L < 60) & (V_L > (mean_v + 1.0 * std_v))

                # 3. 連通域斑點過濾與動態形狀膨脹
                excluded_mask = np.zeros_like(high_light_mask, dtype=np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_light_mask.astype(np.uint8))
                for label_idx in range(1, num_labels):
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    if area >= 2: # 濾除微小噪聲
                        k_size = 3 if area < 10 else (5 if area < 50 else 7)
                        comp_mask = (labels == label_idx).astype(np.uint8)
                        kernel = np.ones((k_size, k_size), dtype=np.uint8)
                        dilated_comp = cv2.dilate(comp_mask, kernel)
                        excluded_mask = cv2.bitwise_or(excluded_mask, dilated_comp)

                left_bgr[v0_A:v1_A, u0_A:u1_A][excluded_mask == 1] = [0, 0, 0]
                mag[excluded_mask == 1] = -99999.0
            elif view_state.get('filter_specular', False):
                thresh = np.percentile(patch_g, 80)
                high_light_mask = (patch_g >= thresh)
                kernel = np.ones((3, 3), dtype=np.uint8)
                excluded_mask = cv2.dilate(high_light_mask.astype(np.uint8), kernel)
                left_bgr[v0_A:v1_A, u0_A:u1_A][excluded_mask == 1] = [0, 0, 0]
                mag[excluded_mask == 1] = -99999.0
            if view_state.get('reject_specular_candidates', False) and left_spec_mask is not None:
                spec_patch = left_spec_mask[v0_A:v1_A, u0_A:u1_A] > 0
                mag[spec_patch] = -99999.0
            flat = mag.flatten()
            valid_left_idx = np.where(flat > -99990.0)[0]
            sorted_left_idx = valid_left_idx[np.argsort(flat[valid_left_idx])] if len(valid_left_idx) > 0 else np.array([], dtype=np.int64)
            idx_g_high = sorted_left_idx[-min(len(sorted_left_idx), LEFT_GRADIENT_POINTS_COUNT):]
            mid_count_l = min(len(sorted_left_idx), LEFT_MID_GRADIENT_POINTS_COUNT)
            mid_start_l = max(0, len(sorted_left_idx) // 2 - mid_count_l // 2)
            idx_g_mid = sorted_left_idx[mid_start_l:mid_start_l + mid_count_l]
            kpts_inj_high = [cv2.KeyPoint(float(u0_A + px), float(v0_A + py), 31.0)
                             for py, px in [divmod(int(idx), patch_g.shape[1]) for idx in idx_g_high]]
            kpts_inj_mid = [cv2.KeyPoint(float(u0_A + px), float(v0_A + py), 31.0)
                            for py, px in [divmod(int(idx), patch_g.shape[1]) for idx in idx_g_mid]]
            if cache_ok:
                lc = {'kpts_high': kpts_inj_high, 'kpts_mid': kpts_inj_mid,
                      'sorted_left_idx': sorted_left_idx, 'des': {}}
                left_cache['grad_left'] = lc
        u_exp, v_exp = u, v
        if cand['plane_n'] is not None:
            H_AB = cand['K_R'] @ (cand['R_rel'] + (cand['t_rel'] @ cand['plane_n'].reshape(1,3))/np.dot(cand['plane_n'], cand['plane_c'])) @ np.linalg.inv(K_L)
            pt_exp = H_AB @ np.array([u, v, 1.0]); u_exp, v_exp = pt_exp[0]/pt_exp[2], pt_exp[1]/pt_exp[2]
        rt_seed_pt = np.array([float(u_exp), float(v_exp)], dtype=np.float32)
        rad = RIGHT_PATCH_SEARCH_RADIUS; uei, vei = int(round(u_exp)), int(round(v_exp))
        v0_B = max(0, vei - rad)
        v1_B = min(imgB_gray.shape[0], vei + rad)
        u0_B = max(0, uei - rad)
        u1_B = min(imgB_gray.shape[1], uei + rad)
        patch_r = imgB_gray[v0_B:v1_B, u0_B:u1_B]
        if patch_r.size > 0:
            gxr, gyr = cv2.Sobel(patch_r, cv2.CV_32F, 1, 0), cv2.Sobel(patch_r, cv2.CV_32F, 0, 1)
            magr = cv2.sqrt(gxr**2 + gyr**2)
            if view_state.get('filter_specular_hsv_mser', False):
                # 1. 結合 HSV 進行高光判斷
                patch_R_color = right_bgr[v0_B:v1_B, u0_B:u1_B]
                patch_R_hsv = cv2.cvtColor(patch_R_color, cv2.COLOR_BGR2HSV)
                H_R, S_R, V_R = cv2.split(patch_R_hsv)

                # 2. 自適應雙閾值
                mean_v_r = np.mean(V_R)
                std_v_r = np.std(V_R)
                high_light_mask_r = (V_R > 190) & (S_R < 60) & (V_R > (mean_v_r + 1.0 * std_v_r))

                # 3. 連通域斑點過濾與動態形狀膨脹
                excluded_mask_r = np.zeros_like(high_light_mask_r, dtype=np.uint8)
                num_labels_r, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(high_light_mask_r.astype(np.uint8))
                for label_idx in range(1, num_labels_r):
                    area = stats_r[label_idx, cv2.CC_STAT_AREA]
                    if area >= 2:
                        k_size = 3 if area < 10 else (5 if area < 50 else 7)
                        comp_mask = (labels_r == label_idx).astype(np.uint8)
                        kernel = np.ones((k_size, k_size), dtype=np.uint8)
                        dilated_comp = cv2.dilate(comp_mask, kernel)
                        excluded_mask_r = cv2.bitwise_or(excluded_mask_r, dilated_comp)

                if is_best_cand:
                    right_bgr[v0_B:v1_B, u0_B:u1_B][excluded_mask_r == 1] = [0, 0, 0]
                magr[excluded_mask_r == 1] = -99999.0
            elif view_state.get('filter_specular', False):
                thresh_r = np.percentile(patch_r, 80)
                high_light_mask_r = (patch_r >= thresh_r)
                kernel_r = np.ones((3, 3), dtype=np.uint8)
                excluded_mask_r = cv2.dilate(high_light_mask_r.astype(np.uint8), kernel_r)
                if is_best_cand:
                    right_bgr[v0_B:v1_B, u0_B:u1_B][excluded_mask_r == 1] = [0, 0, 0]
                magr[excluded_mask_r == 1] = -99999.0
            if view_state.get('reject_specular_candidates', False) and right_spec_mask is not None:
                spec_patch_r = right_spec_mask[v0_B:v1_B, u0_B:u1_B] > 0
                magr[spec_patch_r] = -99999.0
            flatr = magr.flatten()
            valid_right_idx = np.where(flatr > -99990.0)[0]
            sorted_right_idx = valid_right_idx[np.argsort(flatr[valid_right_idx])] if len(valid_right_idx) > 0 else np.array([], dtype=np.int64)
            idx_gr_high = sorted_right_idx[-min(len(sorted_right_idx), RIGHT_GRADIENT_POINTS_COUNT):]
            mid_count_r = min(len(sorted_right_idx), RIGHT_MID_GRADIENT_POINTS_COUNT)
            mid_start_r = max(0, len(sorted_right_idx) // 2 - mid_count_r // 2)
            idx_gr_mid = sorted_right_idx[mid_start_r:mid_start_r + mid_count_r]
            kpts_r_high = [cv2.KeyPoint(float(u0_B + px), float(v0_B + py), 31.0)
                           for py, px in [divmod(int(idx), patch_r.shape[1]) for idx in idx_gr_high]]
            kpts_r_mid = [cv2.KeyPoint(float(u0_B + px), float(v0_B + py), 31.0)
                          for py, px in [divmod(int(idx), patch_r.shape[1]) for idx in idx_gr_mid]]
            g_refA = np.array([kp.pt for kp in (kpts_inj_high + kpts_inj_mid)], dtype=np.float32)
            g_refB = np.array([kp.pt for kp in (kpts_r_high + kpts_r_mid)], dtype=np.float32)
            g_refA_groups = np.array(
                (["high"] * len(kpts_inj_high)) + (["mid"] * len(kpts_inj_mid)),
                dtype=object
            )
            g_refB_groups = np.array(
                (["high"] * len(kpts_r_high)) + (["mid"] * len(kpts_r_mid)),
                dtype=object
            )
            g_kptsB = [kp.pt for kp in (kpts_r_high + kpts_r_mid)]
            g_rect = (u0_B, v0_B, u1_B-u0_B, v1_B-v0_B)
            print(
                f"   [Grad-SIFT refs] "
                f"L_high={len(kpts_inj_high)}/{LEFT_GRADIENT_POINTS_COUNT}, "
                f"L_mid={len(kpts_inj_mid)}/{LEFT_MID_GRADIENT_POINTS_COUNT}, "
                f"R_high={len(kpts_r_high)}/{RIGHT_GRADIENT_POINTS_COUNT}, "
                f"R_mid={len(kpts_r_mid)}/{RIGHT_MID_GRADIENT_POINTS_COUNT}, "
                f"valid_L={len(sorted_left_idx)}, valid_R={len(sorted_right_idx)}"
            )

            def compute_left_descriptors(kpts_left):
                if view_state.get('use_hamming', False):
                    return orb.compute(imgA_gray, kpts_left)
                if view_state.get('use_rgb_sift', False):
                    return compute_rgb_sift_descriptors(left_bgr, kpts_left, sift)
                if view_state.get('use_opponent_sift', False):
                    return compute_opponent_sift_descriptors(left_bgr, kpts_left, sift)
                return sift.compute(imgA_gray, kpts_left)

            def compute_group_descriptors(kpts_left, kpts_right, group_name):
                if not kpts_left or not kpts_right:
                    return None, None, None, None
                # 左側描述子快取：同一次點擊各候選的左圖/特徵點/描述子模式皆相同
                desc_key = (group_name,
                            view_state.get('use_hamming', False),
                            view_state.get('use_rgb_sift', False),
                            view_state.get('use_opponent_sift', False))
                if lc is not None and desc_key in lc['des']:
                    kpts_left_desc, des_left = lc['des'][desc_key]
                else:
                    kpts_left_desc, des_left = compute_left_descriptors(kpts_left)
                    if lc is not None:
                        lc['des'][desc_key] = (kpts_left_desc, des_left)
                if view_state.get('use_hamming', False):
                    kpts_right_desc, des_right = orb.compute(imgB_gray, kpts_right)
                elif view_state.get('use_rgb_sift', False):
                    imgB_bgr = cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR)
                    kpts_right_desc, des_right = compute_rgb_sift_descriptors(imgB_bgr, kpts_right, sift)
                elif view_state.get('use_opponent_sift', False):
                    imgB_bgr = cv2.cvtColor(cand['rgb'], cv2.COLOR_RGB2BGR)
                    kpts_right_desc, des_right = compute_opponent_sift_descriptors(imgB_bgr, kpts_right, sift)
                else:
                    kpts_right_desc, des_right = sift.compute(imgB_gray, kpts_right)
                return kpts_left_desc, des_left, kpts_right_desc, des_right

            def map_from_gradient_group(kpts_left, kpts_right, group_name):
                kpts_left_desc, des_left, kpts_right_desc, des_right = compute_group_descriptors(kpts_left, kpts_right, group_name)
                if des_left is None or des_right is None or len(des_left) == 0 or len(des_right) == 0:
                    print(f"   [Grad-SIFT {group_name}] no descriptors: left_kpts={len(kpts_left)}, right_kpts={len(kpts_right)}")
                    return None, None, None, 0.0
                kpts_left = kpts_left_desc
                kpts_right = kpts_right_desc
                if len(kpts_left) != len(des_left):
                    n_left_desc = min(len(kpts_left), len(des_left))
                    kpts_left = kpts_left[:n_left_desc]
                    des_left = des_left[:n_left_desc]
                if len(kpts_right) != len(des_right):
                    n_right_desc = min(len(kpts_right), len(des_right))
                    kpts_right = kpts_right[:n_right_desc]
                    des_right = des_right[:n_right_desc]
                if view_state.get('use_hamming', False):
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                    match_thres = 100
                else:
                    bf = cv2.BFMatcher(cv2.NORM_L2)
                    match_thres = 780 if (view_state.get('use_rgb_sift', False) or view_state.get('use_opponent_sift', False)) else 450

                knn_lr = bf.knnMatch(des_left, des_right, k=2)
                knn_rl = bf.knnMatch(des_right, des_left, k=1)
                reverse_best = {
                    m.queryIdx: m.trainIdx
                    for pair in knn_rl for m in pair[:1]
                }
                good = []
                reject_dist = 0
                reject_ratio = 0
                reject_mutual = 0
                for pair in knn_lr:
                    if not pair:
                        continue
                    m = pair[0]
                    if m.distance >= match_thres:
                        reject_dist += 1
                        continue
                    if len(pair) > 1 and m.distance >= GRAD_SIFT_RATIO_TEST * pair[1].distance:
                        reject_ratio += 1
                        continue
                    if reverse_best.get(m.trainIdx) != m.queryIdx:
                        reject_mutual += 1
                        continue
                    good.append(m)

                guided_added = 0
                if len(good) < GRAD_SIFT_MIN_GROUP_INLIERS:
                    used_query = {m.queryIdx for m in good}
                    best_by_train = {m.trainIdx: m for m in good}
                    pts_right_arr = np.array([kp.pt for kp in kpts_right], dtype=np.float32)
                    for qi, kp_left in enumerate(kpts_left):
                        if qi in used_query:
                            continue
                        pL = np.array(kp_left.pt, dtype=np.float32)
                        if np.linalg.norm(pL - np.array([u, v], dtype=np.float32)) >= 50:
                            continue
                        local_seed = rt_seed_pt + (pL - np.array([u, v], dtype=np.float32))
                        spatial_d = np.linalg.norm(pts_right_arr - local_seed, axis=1)
                        cand_idx = np.where(spatial_d <= GRAD_SIFT_GUIDED_RADIUS_PX)[0]
                        if cand.get('F') is not None and len(cand_idx) > 0:
                            epi_line = cand['F'] @ np.array([pL[0], pL[1], 1.0], dtype=np.float64)
                            denom = float(np.hypot(epi_line[0], epi_line[1]))
                            if denom > 1e-8:
                                epi_d = np.abs(
                                    epi_line[0] * pts_right_arr[cand_idx, 0]
                                    + epi_line[1] * pts_right_arr[cand_idx, 1]
                                    + epi_line[2]
                                ) / denom
                                cand_idx = cand_idx[epi_d <= GRAD_SIFT_EPIPOLAR_TOL_PX]
                        if len(cand_idx) == 0:
                            continue
                        if view_state.get('use_hamming', False):
                            dists = np.array([
                                cv2.norm(des_left[qi], des_right[ri], cv2.NORM_HAMMING)
                                for ri in cand_idx
                            ], dtype=np.float32)
                        else:
                            diff = des_right[cand_idx].astype(np.float32) - des_left[qi].astype(np.float32)
                            dists = np.linalg.norm(diff, axis=1)
                        order = np.argsort(dists)
                        best_pos = int(order[0])
                        best_ri = int(cand_idx[best_pos])
                        best_dist = float(dists[best_pos])
                        second_dist = float(dists[int(order[1])]) if len(order) > 1 else float("inf")
                        if best_dist >= match_thres:
                            continue
                        if np.isfinite(second_dist) and best_dist >= GRAD_SIFT_GUIDED_RATIO_TEST * second_dist:
                            continue
                        new_match = cv2.DMatch(_queryIdx=int(qi), _trainIdx=best_ri, _imgIdx=0, _distance=best_dist)
                        prev = best_by_train.get(best_ri)
                        if prev is None or new_match.distance < prev.distance:
                            best_by_train[best_ri] = new_match
                    guided_good = list(best_by_train.values())
                    guided_added = max(0, len(guided_good) - len(good))
                    good = guided_good

                pts_info = []
                reject_epi = 0
                reject_seed = 0
                reject_color = 0
                for m in good:
                    pL, pR = np.array(kpts_left[m.queryIdx].pt), np.array(kpts_right[m.trainIdx].pt)
                    if np.linalg.norm(pL - np.array([u, v])) < 50:
                        if cand.get('F') is not None:
                            epi_line = cand['F'] @ np.array([pL[0], pL[1], 1.0], dtype=np.float64)
                            denom = float(np.hypot(epi_line[0], epi_line[1]))
                            if denom > 1e-8:
                                epi_dist = abs(float(epi_line[0] * pR[0] + epi_line[1] * pR[1] + epi_line[2])) / denom
                                if epi_dist > GRAD_SIFT_EPIPOLAR_TOL_PX:
                                    reject_epi += 1
                                    continue
                        local_seed = rt_seed_pt + (pL - np.array([u, v], dtype=np.float32))
                        if float(np.linalg.norm(pR - local_seed)) > GRAD_SIFT_MAX_RT_ADJUST_PX:
                            reject_seed += 1
                            continue
                        if view_state.get('use_color_hist', False):
                            if not check_color_histogram_similarity(left_bgr, pL, pR, cand, patch_size=16, threshold=0.45):
                                reject_color += 1
                                continue
                        pts_info.append({'pL': pL, 'pR': pR, 'off': pR - pL, 'dist': float(m.distance)})
                if len(pts_info) >= 3:
                    offs = np.array([x['off'] for x in pts_info])
                    med_off = np.median(offs, axis=0)
                    pts_info = [x for x in pts_info if np.linalg.norm(x['off'] - med_off) < GRAD_SIFT_OFFSET_MEDIAN_TOL_PX]
                if len(pts_info) < GRAD_SIFT_MIN_GROUP_INLIERS:
                    print(
                        f"   [Grad-SIFT {group_name}] rejected: "
                        f"left_kpts={len(kpts_left)}, right_kpts={len(kpts_right)}, "
                        f"knn={len(knn_lr)}, good={len(good)}, inliers={len(pts_info)}, "
                        f"guided_add={guided_added}, "
                        f"dist={reject_dist}, ratio={reject_ratio}, mutual={reject_mutual}, "
                        f"epi={reject_epi}, seed={reject_seed}, color={reject_color}"
                    )
                    return None, None, None, 0.0

                ptsA_m = np.array([x['pL'] for x in pts_info], dtype=np.float32)
                ptsB_m = np.array([x['pR'] for x in pts_info], dtype=np.float32)
                wts = 1.0 / (np.sum((ptsA_m - np.array([u, v]))**2, axis=1) + 1e-5)
                weighted_group = np.array([u, v]) + np.sum((ptsB_m - ptsA_m) * wts[:, np.newaxis], axis=0) / np.sum(wts)
                mapped_group = None
                inlier_mask = np.ones(len(ptsA_m), dtype=bool)
                if len(ptsA_m) >= 3:
                    M_local, inliers = cv2.estimateAffinePartial2D(
                        ptsA_m, ptsB_m, method=cv2.RANSAC,
                        ransacReprojThreshold=GRAD_SIFT_RANSAC_REPROJ_PX
                    )
                    if M_local is not None:
                        if inliers is not None:
                            inlier_mask = inliers.ravel().astype(bool)
                            if np.count_nonzero(inlier_mask) >= GRAD_SIFT_MIN_GROUP_INLIERS:
                                ptsA_m = ptsA_m[inlier_mask]
                                ptsB_m = ptsB_m[inlier_mask]
                                pts_info = [x for x, keep in zip(pts_info, inlier_mask) if keep]
                            else:
                                print(f"   [Grad-SIFT {group_name}] rejected by RANSAC: inliers={np.count_nonzero(inlier_mask)}")
                                return None, None, None, 0.0
                        pt_a = M_local @ np.array([u, v, 1.0])
                        mapped_group = pt_a[:2]
                if mapped_group is not None:
                    center_b = np.mean(ptsB_m, axis=0)
                    radius_b = max(float(np.percentile(np.linalg.norm(ptsB_m - center_b, axis=1), 90)), 3.0)
                    if float(np.linalg.norm(mapped_group - center_b)) > radius_b * 1.25:
                        mapped_group = weighted_group
                if mapped_group is None:
                    mapped_group = weighted_group
                if float(np.linalg.norm(mapped_group - rt_seed_pt)) > GRAD_SIFT_MAX_RT_ADJUST_PX:
                    print(f"   [Grad-SIFT {group_name}] rejected by final RT bound")
                    return None, None, None, 0.0
                offset_spread = float(np.median(np.linalg.norm((ptsB_m - ptsA_m) - np.median(ptsB_m - ptsA_m, axis=0), axis=1)))
                mean_dist = float(np.mean([x['dist'] for x in pts_info]))
                score = len(pts_info) / (1.0 + offset_spread + mean_dist / max(match_thres, 1.0))
                print(
                    f"   [Grad-SIFT {group_name}] accepted: "
                    f"inliers={len(pts_info)}, guided_add={guided_added}, spread={offset_spread:.2f}, score={score:.2f}"
                )
                return mapped_group, ptsA_m, ptsB_m, score

            mapped_high, ptsA_high, ptsB_high, score_high = map_from_gradient_group(kpts_inj_high, kpts_r_high, "HIGH")
            mapped_mid, ptsA_mid, ptsB_mid, score_mid = map_from_gradient_group(kpts_inj_mid, kpts_r_mid, "MID")
            mapped_candidates = [
                (mapped_high, score_high),
                (mapped_mid, score_mid),
            ]
            mapped_candidates = [(p, s) for p, s in mapped_candidates if p is not None and s > 0]
            if mapped_candidates:
                total_score = sum(s for _, s in mapped_candidates)
                mapped = sum(np.array(p, dtype=np.float32) * (s / total_score) for p, s in mapped_candidates)
                grad_dev = float(np.linalg.norm(mapped - rt_seed_pt))
                if grad_dev > GRAD_SIFT_MAX_RT_ADJUST_PX:
                    print(f"   [Grad-SIFT] 融合映射點偏離 RT/平面預測 {grad_dev:.1f}px，放棄此匹配結果。")
                    rt_bound_reject_reason = f"匹配點偏離RT/平面預測 {grad_dev:.0f}px"
                    mapped_candidates = []
            if mapped_candidates:
                if view_state.get('use_rgb_sift', False):
                    method_name = "RGB-SIFT"
                elif view_state.get('use_opponent_sift', False):
                    method_name = "Opponent-SIFT"
                else:
                    method_name = "Grad-SIFT"
                if mapped_high is not None and mapped_mid is not None:
                    method_name += "+MidGradInterp"
                elif mapped_mid is not None:
                    method_name += "+MidGrad"
                else:
                    method_name += "+HighGrad"
                m_pt, method = mapped, method_name
                ptsA_groups = [pts for pts in (ptsA_high, ptsA_mid) if pts is not None]
                ptsB_groups = [pts for pts in (ptsB_high, ptsB_mid) if pts is not None]
                if ptsA_groups and ptsB_groups:
                    g_ptsA = np.vstack(ptsA_groups)
                    g_ptsB = np.vstack(ptsB_groups)
                    group_labels = []
                    if ptsA_high is not None:
                        group_labels.extend(["high"] * len(ptsA_high))
                    if ptsA_mid is not None:
                        group_labels.extend(["mid"] * len(ptsA_mid))
                    g_groups = np.array(group_labels, dtype=object)
    return {'m_pt': m_pt, 'method': method, 'g_ptsA': g_ptsA, 'g_ptsB': g_ptsB, 'g_groups': g_groups,
            'g_refA': g_refA, 'g_refB': g_refB, 'g_refA_groups': g_refA_groups,
            'g_refB_groups': g_refB_groups, 'g_kptsB': g_kptsB, 'g_rect': g_rect,
            'reject_reason': rt_bound_reject_reason}
