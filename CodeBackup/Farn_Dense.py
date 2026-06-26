def find_correspondence(img_left, img_right, pt_left, 
                         P_rt, K, R_rel, t_rel, radius=60):
    
    # Step 1: 右圖做旋轉補償，消除視角差異
    H = K @ R_rel @ np.linalg.inv(K)
    h, w = img_right.shape[:2]
    img_right_comp = cv.warpPerspective(img_right, H, (w, h))
    
    # Step 2: 補償後重新計算Prt位置
    P_rt_comp = cv.perspectiveTransform(
        np.array([[P_rt]], dtype=np.float32), H)[0][0]
    
    # Step 3: 裁ROI
    cx_l, cy_l = int(pt_left[0]), int(pt_left[1])
    cx_r, cy_r = int(P_rt_comp[0]), int(P_rt_comp[1])
    
    roi_l = cv.cvtColor(
        img_left[cy_l-radius:cy_l+radius, cx_l-radius:cx_l+radius],
        cv.COLOR_BGR2GRAY).astype(np.float32)
    roi_r = cv.cvtColor(
        img_right_comp[cy_r-radius:cy_r+radius, cx_r-radius:cx_r+radius],
        cv.COLOR_BGR2GRAY).astype(np.float32)
    
    # Step 4: Dense flow
    flow = cv.calcOpticalFlowFarneback(
        roi_l, roi_r, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    
    # Step 5: 查詢中心點的位移
    dx, dy = flow[radius, radius]
    
    P_prime = np.array([cx_r + dx, cy_r + dy])
    
    return P_prime