import numpy as np

from typing import List, Optional
from einops import rearrange


def generate_anchors(image_size: List[int], strides: List[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(np.full((anchor_num,), stride))
        shift = stride // 2
        h = np.arange(0, H, stride) + shift
        w = np.arange(0, W, stride) + shift
        anchor_h, anchor_w = np.meshgrid(h, w, indexing="ij")
        anchor = np.stack([anchor_w.flatten(), anchor_h.flatten()], axis=-1)
        anchors.append(anchor)
    all_anchors = np.concatenate(anchors, axis=0)
    all_scalers = np.concatenate(scaler, axis=0)
    return all_anchors, all_scalers

def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)

    return np.where(x >= 0, 
                    1.0 / (1.0 + np.exp(-np.where(x >= 0, x, 0))), 
                    np.exp(np.where(x < 0, x, 0)) / (1.0 + np.exp(np.where(x < 0, x, 0))))

def nms_numpy(boxes, scores, iou_threshold):

    if boxes.size == 0:
        return np.empty((0,), dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)

def batched_nms(boxes, scores, idxs, iou_threshold):

    if boxes.size == 0:
        return np.empty((0,), dtype=int)
    
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    return nms_numpy(boxes_for_nms, scores, iou_threshold)

def bbox_mc_nms(cls_dist: np.ndarray, bbox: np.ndarray, mc: np.ndarray, nms_cfg, confidence: Optional[np.ndarray] = None):
    cls_dist = sigmoid(cls_dist) * (1 if confidence is None else confidence)
    batch_idx, valid_grid, valid_cls = np.where(cls_dist > nms_cfg.min_confidence)
    valid_con = cls_dist[batch_idx, valid_grid, valid_cls]
    valid_box = bbox[batch_idx, valid_grid]
    valid_mc = mc[batch_idx, valid_grid]
    nms_idx = batched_nms(valid_box, valid_con, batch_idx + valid_cls * bbox.shape[0], nms_cfg.min_iou)
    predicts_nms = []
    for idx in range(cls_dist.shape[0]):
        instance_idx = nms_idx[idx == batch_idx[nms_idx]][: nms_cfg.max_bbox]
        predicts_nms.append((
            valid_cls[instance_idx][:, None],
            valid_box[instance_idx],
            valid_con[instance_idx][:, None],
            valid_mc[instance_idx]
        ))
    return predicts_nms

class Vec2Mask:
    def __init__(self, image_size, strides = None):

        if strides:
            print(f":japanese_not_free_of_charge_button: Found stride of model {strides}")
            self.strides = strides
        else:
            print(":teddy_bear: Found no stride of model, performed a dummy test for auto-anchor size")
            # self.strides = self.create_auto_anchor(model, image_size)

        self.anchor_grid, self.scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size

    # TODO
    # def create_auto_anchor(self, model: YOLO, image_size):
        # W, H = image_size
        # # TODO: need accelerate dummy test
        # dummy_input = torch.zeros(1, 3, H, W)
        # dummy_output = model(dummy_input)
        # strides = []
        # for predict_head in dummy_output["Main"]:
        #     _, _, *anchor_num = predict_head[2].shape
        #     strides.append(W // anchor_num[1])
        # return strides

    def update(self, image_size):
        """
        image_size: W, H
        """
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(self.device), scaler.to(self.device)

    def __call__(self, predicts):
        pred, seg  = predicts
        preds_cls, preds_anc, preds_box, preds_mc = [], [], [], []
        protos = seg[-1]
        for pred_output, seg_mc in zip(pred, seg[:-1]):
            pred_cls, pred_anc, pred_box = pred_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
            preds_mc.append(rearrange(seg_mc, "B A h w -> B (h w) A"))
        preds_cls = np.concatenate(preds_cls, axis=1)
        preds_anc = np.concatenate(preds_anc, axis=1)
        preds_box = np.concatenate(preds_box, axis=1)
        preds_mc = np.concatenate(preds_mc, axis=1)
        # xy2h to x1y1x2y2
        pred_LTRB = preds_box * self.scaler.reshape(1, -1, 1)
        lt, rb = pred_LTRB[..., :2], pred_LTRB[..., 2:]
        preds_box = np.concatenate([self.anchor_grid - lt, self.anchor_grid + rb], axis=-1)
        return preds_cls, preds_anc, preds_box, preds_mc, protos
