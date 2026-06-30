import cv2
import numpy as np
import random
from typing import List, Optional

class MaskDrawer:
    def __call__(self, 
                 img: np.ndarray,
                 predicts: List[np.ndarray],
                 idx2label: Optional[list] = None) -> np.ndarray:
        
        img = img.copy()
        h, w = img.shape[:2]
        
        font_scale = max(0.5, h / 1000.0)
        thickness = max(1, int(h / 300))
        
        if isinstance(predicts, list) and len(predicts) > 0 and isinstance(predicts[0], list):
            predicts = predicts[0]
        if not len(predicts):
            return img

        def get_color(cls_id):
            random.seed(int(cls_id))
            return (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

        classes, bboxes, scores, masks = predicts

        for class_id, mask, conf in zip(classes, masks, scores):
            conf_val = float(conf[0] if isinstance(conf, np.ndarray) else conf)
            
            if conf_val > 0.01:
                cls_id = int(class_id[0] if isinstance(class_id, np.ndarray) else class_id)
                color = np.array(get_color(cls_id), dtype=np.float32)

                if mask.shape != (h, w):
                    mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    mask_resized = mask.astype(np.float32)
                
                bool_mask = mask_resized > 0.5

                img[bool_mask] = img[bool_mask] * 0.5 + color * 0.5
        
        for class_id, bbox, conf in zip(classes, bboxes, scores):
            conf_val = float(conf[0] if isinstance(conf, np.ndarray) else conf)
            
            if conf_val > 0.01:
                cls_id = int(class_id[0] if isinstance(class_id, np.ndarray) else class_id)
                color = get_color(cls_id)
                
                x_min, y_min, x_max, y_max = map(int, bbox)
                x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                y_min, y_max = min(y_min, y_max), max(y_min, y_max)
                
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
                
                class_text = str(idx2label[cls_id] if idx2label else cls_id)
                label_text = f"{class_text} {conf_val:.0%}"
                
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_y_min = max(y_min - text_h - baseline - 5, 0)
                
                cv2.rectangle(img, (x_min, text_y_min), (x_min + text_w, text_y_min + text_h + baseline + 5), color, -1)
                cv2.putText(img, label_text, (x_min, text_y_min + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
        return img.astype(np.uint8)