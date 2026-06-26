import cv2
import numpy as np
import onnxruntime as ort

def load_image(img_path, input_size=256):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # MiDaS Small 的正規化參數（ImageNet mean/std）
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    img_norm = (img_resized / 255.0 - mean) / std
    
    # HWC → CHW → NCHW
    img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    
    return img, img_tensor

def run_midas(session, img_tensor, orig_size):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_tensor})[0]  # shape: (1, H, W)
    
    depth = output[0]  # (H, W)
    
    # Resize 回原圖尺寸
    depth_resized = cv2.resize(depth, (orig_size[1], orig_size[0]))
    
    return depth_resized

def visualize_depth(depth, colormap=cv2.COLORMAP_INFERNO):
    # 正規化到 0~255
    d_min, d_max = depth.min(), depth.max()
    depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    
    # 注意：MiDaS 輸出是 inverse depth（值大 = 近，值小 = 遠）
    depth_colored = cv2.applyColorMap(depth_norm, colormap)
    return depth_colored

def main():
    model_path = "midas_v21_small_256.onnx"
    img_path   = "captured_images/image_5.png"   # ← 換成你的圖片
    
    # 載入模型
    print("載入模型...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"輸入 shape: {session.get_inputs()[0].shape}")
    
    # 載入圖片
    img_orig, img_tensor = load_image(img_path, input_size=256)
    orig_h, orig_w = img_orig.shape[:2]
    
    # 推論
    import time
    t0 = time.time()
    depth = run_midas(session, img_tensor, orig_size=(orig_h, orig_w))
    print(f"推論時間：{time.time() - t0:.3f} 秒")
    
    # 視覺化
    depth_vis = visualize_depth(depth)
    
    # 並排顯示原圖 + 深度圖
    combined = np.hstack([img_orig, depth_vis])
    
    cv2.imwrite("midas_result.jpg", combined)
    print("結果已儲存為 midas_result.jpg")
    
    # 顯示（等比例縮放視窗）
    display_w = 1280
    display_h = int(orig_h * display_w / (orig_w * 2))
    cv2.imshow("MiDaS Depth", cv2.resize(combined, (display_w, display_h)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()