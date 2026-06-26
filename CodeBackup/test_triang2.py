import numpy as np
import cv2

# Approximate KL
KL = np.array([[523., 0., 512.], [0., 523., 288.], [0., 0., 1.]], dtype=np.float64)
KR = KL.copy()

u_m, v_m = 541.7, 438.8
m_pt_m = np.array([462.5, 446.7])

u_a, v_a = 542.0, 439.0
m_pt_a = np.array([462.8, 447.0])

R_rel = np.array([[ 0.9991, -0.0182,  0.0374],
                  [ 0.0181,  0.9998,  0.0031],
                  [-0.0375, -0.0024,  0.9993]])

t_rel = np.array([[-55.32], [3.19], [-8.8]])

def triangulate_point_3d(pt_A, pt_B, K_L, K_R, R_rel, t_rel):
    P0 = (K_L.astype(np.float64) @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
    P1 = (K_R.astype(np.float64) @ np.hstack([R_rel, t_rel])).astype(np.float32)
    pts4d = cv2.triangulatePoints(P0, P1, np.array([[pt_A[0]], [pt_A[1]]]), np.array([[pt_B[0]], [pt_B[1]]]))
    pt3d = (pts4d[:3] / pts4d[3]).flatten()
    return pt3d

p3d_m = triangulate_point_3d((u_m, v_m), m_pt_m, KL, KR, R_rel, t_rel)
p3d_a = triangulate_point_3d((u_a, v_a), m_pt_a, KL, KR, R_rel, t_rel)

print(f"Manual depth: {p3d_m[2]:.2f}")
print(f"Auto depth: {p3d_a[2]:.2f}")
