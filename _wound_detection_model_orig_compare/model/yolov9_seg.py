import cv2
import numpy as np
import onnxruntime as ort

from typing import List, Optional, Tuple

from utils.bounding_box_utils import Vec2Mask, bbox_mc_nms, sigmoid

class NMSConfig:
    min_confidence: float = 0.5
    min_iou: float = 0.5
    max_bbox: int = 10

class PostProcess:
    """
    TODO: function document
    scale back the prediction and do nms for pred_bbox
    """

    def __init__(self, converter: Vec2Mask, nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg
    
    def __call__(
        self, predict, rev_tensor: Optional[List] = None, image_size: Optional[List[int]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict)
        pred_class, _, pred_bbox, pred_mc, protos = prediction
        pred_conf = prediction[5] if len(prediction) == 6 else None
        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        pred_bbox_mc = bbox_mc_nms(pred_class, pred_bbox, pred_mc, self.nms, pred_conf)
        final_results = []
        for (classes, bboxes, scores, mcs), proto in zip(pred_bbox_mc,protos):
            _, c, h, w = protos.shape
            # convert to mask
            if mcs.shape[0] == 0:
                continue
            masks = sigmoid(mcs @ proto.reshape(c, -1)).reshape(-1, h, w)
            # scale to input size
            masks_hwc = masks.transpose(1, 2, 0)
            masks_hwc = cv2.resize(masks_hwc, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_LINEAR)
            if masks_hwc.ndim == 2:
                masks_hwc = masks_hwc[:, :, np.newaxis]
            masks = masks_hwc.transpose(2, 0, 1)
        
            # scale to origin size
            if rev_tensor is not None:
                pad_x1 = max(0,int(rev_tensor[0, 1]))
                pad_y1 = max(0,int(rev_tensor[0, 2]))
                pad_x2 = int(masks.shape[2] - rev_tensor[0, 3])
                pad_y2 = int(masks.shape[1] - rev_tensor[0, 4])
                masks = masks[:, pad_y1:pad_y2, pad_x1:pad_x2]
                masks_hwc = masks.transpose(1, 2, 0)
                masks_hwc = cv2.resize(masks_hwc, None, fx=1/rev_tensor[0, 0], fy=1/rev_tensor[0, 0], interpolation=cv2.INTER_LINEAR)
                if masks_hwc.ndim == 2:
                    masks_hwc = masks_hwc[:, :, np.newaxis]
                masks = masks_hwc.transpose(2, 0, 1)
            _, h, w = masks.shape
            x = np.arange(w).reshape(1, 1, w)
            y = np.arange(h).reshape(1, h, 1)
            inside_mask = (x >= bboxes[:, 0:1, None]) & (x < bboxes[:, 2:3, None]) & \
                (y >= bboxes[:, 1:2, None]) & (y < bboxes[:, 3:4, None])
            final_results.append((classes, bboxes, scores, (masks * inside_mask) > 0.5))
        return final_results


class YOLOV9Seg:
    def __init__(self, model_path='model/assets/v9-c-seg.onnx', providers=['CPUExecutionProvider']):
        """
        初始化 ONNX Runtime Session
        :param model_path: ONNX 模型檔案路徑 (例如: 'best.onnx')
        """
        # 1. 建立推論 Session
        self.session = ort.InferenceSession(model_path)
        
        # 2. 獲取模型的輸入與輸出資訊
        self.inputs_info = self.session.get_inputs()
        
        # 3. 紀錄輸入/輸出名稱，這在執行 session.run() 時必須用到
        self.input_name = self.inputs_info[0].name
        self.input_shape = self.inputs_info[0].shape[2:]
        
        # 印出模型資訊，方便 Debug 與確認 Shape
        # print(f"✅ 模型 {model_path} 載入成功！")
        # print(f"👉 輸入節點: {[(inp.name, inp.shape, inp.type) for inp in self.inputs_info]}")
        # print(f"👉 輸出節點: {[(out.name, out.shape, out.type) for out in self.outputs_info]}")

        self.vec2box = Vec2Mask(self.input_shape, [8, 16, 32])
        self.post_process = PostProcess(self.vec2box, NMSConfig)

    def letterbox(self, input_image, background_color = (114, 114, 114)):
        img_height, img_width, _ = input_image.shape
        scale = min(self.input_shape[0] / img_height, self.input_shape[1] / img_width)
        new_height, new_width = int(img_height * scale), int(img_width * scale)
        
        input_image = cv2.resize(input_image, (new_width, new_height), cv2.INTER_AREA)
        pad_left = (self.input_shape[0] - new_width) // 2
        pad_top = (self.input_shape[1] - new_height) // 2
        padded_image = cv2.copyMakeBorder(
            input_image, 
            pad_top, pad_top, pad_left, pad_left, 
            cv2.BORDER_CONSTANT, 
            value=background_color
        )

        # return resized image and transform info
        return padded_image, np.asarray([[scale, pad_left, pad_top, pad_left, pad_top]])

    def preprocess(self, input_image):
        image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
        processed_image, rev_tensor = self.letterbox(image)
        processed_image = processed_image.astype(np.float32) / 255

        return np.expand_dims(processed_image.transpose(2,0,1),axis=0), rev_tensor

    def predict(self, input_image):
        input_data, rev_tensor = self.preprocess(input_image)
        outputs = self.session.run(None, {self.input_name: input_data})
        
        det_output = [[outputs[0],outputs[1],outputs[2]],
                      [outputs[3],outputs[4],outputs[5]],
                      [outputs[6],outputs[7],outputs[8]]]
        seg_output = outputs[9:]
        predicts = self.post_process((det_output,seg_output), rev_tensor=rev_tensor)
        return predicts