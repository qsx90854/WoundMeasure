import cv2
import numpy as np
import sys

K = np.array([[523, 0, 512], [0, 523, 512], [0, 0, 1]], dtype=np.float32)
canon = np.array([[-16.5, 16.5, 0], [16.5, 16.5, 0], [16.5, -16.5, 0], [-16.5, -16.5, 0]], dtype=np.float32)

rvecA = np.zeros(3, dtype=np.float32)
tvecA = np.array([0, 0, 150], dtype=np.float32)
imgA, _ = cv2.projectPoints(canon, rvecA, tvecA, K, np.zeros(5))
imgA = imgA.reshape(4, 2)

rvecB = np.zeros(3, dtype=np.float32)
tvecB = np.array([-30, 0, 150], dtype=np.float32)
imgB, _ = cv2.projectPoints(canon, rvecB, tvecB, K, np.zeros(5))
imgB = imgB.reshape(4, 2)

ok, rvA, tvA = cv2.solvePnP(canon, imgA, K, np.zeros(5))
RA, _ = cv2.Rodrigues(rvA)
objA = (RA @ canon.T).T + tvA.T
objA = objA.astype(np.float32)

ok, rv_rel, tv_rel = cv2.solvePnP(objA, imgB, K, np.zeros(5))
R_rel, _ = cv2.Rodrigues(rv_rel)

P0 = (K @ np.hstack([np.eye(3), np.zeros((3, 1))])).astype(np.float32)
P1 = (K @ np.hstack([R_rel, tv_rel])).astype(np.float32)

pt_A = imgA[0]
pt_B = imgB[0]

pts4d = cv2.triangulatePoints(P0, P1, pt_A.reshape(2, 1), pt_B.reshape(2, 1))
pt3d = (pts4d[:3] / pts4d[3]).flatten()
print("Triangulated Z:", pt3d[2])
