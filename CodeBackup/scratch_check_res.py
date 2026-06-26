import json, cv2
with open("calibration_result_c2.json") as f:
    d = json.load(f)
if "intrinsic_L" in d:
    K = d['intrinsic_L']['matrix']
else:
    K = d['K']
print("K cx:", K[0][2], "cy:", K[1][2])
cap = cv2.VideoCapture("test_video/1.mp4")
ret, frame = cap.read()
if ret:
    print("Video shape:", frame.shape)
else:
    print("Video not found")
