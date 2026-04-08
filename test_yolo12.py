# ...existing code...
"""
Reworked YOLO12 ONNX inference (Ultralytics-inspired, more robust handling)
- Handles common ONNX outputs: (1, N, 5+nc), (N, 5+nc), (M,6) (model-NMS)
- Detects normalized coords and maps between padded/original image correctly
- Uses pure-numpy NMS by default, optional torchvision nms if available
Dependencies: onnxruntime, numpy, opencv-python (torch & torchvision optional)
"""
import onnxruntime as ort
import numpy as np
import cv2
import sys

# -----------------------
# CONFIG
# -----------------------
ONNX_MODEL = "yolo12n.onnx"
IMAGE_PATH = "/home/faith/yolov5/data/images/bus.jpg"
OUT_PATH = "result.jpg"
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
CLASS_NAMES = None  # e.g. ["person","car",...]

USE_TORCH_NMS = False
try:
    import torch
    from torchvision.ops import nms as torch_nms
    USE_TORCH_NMS = True
except Exception:
    USE_TORCH_NMS = False

# -----------------------
# UTIL FUNCTIONS
# -----------------------
def letterbox(img, new_size=IMG_SIZE, color=(114,114,114)):
    h0, w0 = img.shape[:2]
    r = min(new_size / w0, new_size / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = new_size - new_unpad[0]
    dh = new_size - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (left, top)

def xywh2xyxy(x):
    y = x.copy()
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y

def nms_numpy(boxes, scores, iou_threshold=IOU_THRESH):
    if boxes.shape[0] == 0:
        return np.array([], dtype=int)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)

def non_max_suppression(prediction, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH):
    if prediction is None or prediction.shape[0] == 0:
        return np.zeros((0,6))
    # prediction: (N, 5+nc) => [x,y,w,h,obj_conf, cls_scores...]
    nc = prediction.shape[1] - 5
    # objectness filter
    mask = prediction[:,4] > conf_thres
    pred = prediction[mask]
    if pred.shape[0] == 0:
        return np.zeros((0,6))
    boxes = pred[:, :4].copy()
    # class scores * objectness
    scores_all = pred[:, 5:] * pred[:, 4:5]
    class_ids = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(scores_all.shape[0]), class_ids]
    keep_mask = scores > conf_thres
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]
    if boxes.shape[0] == 0:
        return np.zeros((0,6))
    # convert xywh->xyxy
    xyxy = np.stack([xywh2xyxy(b) for b in boxes], axis=0)
    output = []
    for c in np.unique(class_ids):
        idxs = np.where(class_ids == c)[0]
        boxes_c = xyxy[idxs]
        scores_c = scores[idxs]
        if boxes_c.shape[0] == 0:
            continue
        if USE_TORCH_NMS:
            bt = torch.tensor(boxes_c, dtype=torch.float32)
            st = torch.tensor(scores_c, dtype=torch.float32)
            keep = torch_nms(bt, st, iou_thres).cpu().numpy()
        else:
            keep = nms_numpy(boxes_c, scores_c, iou_thres)
        for i in keep:
            j = idxs[i]
            x1,y1,x2,y2 = xyxy[j]
            output.append([x1,y1,x2,y2, float(scores[j]), int(c)])
    if len(output) == 0:
        return np.zeros((0,6))
    return np.array(output)

def scale_coords_from_padded(coords, pad, ratio, orig_shape):
    pad_x, pad_y = pad
    orig_h, orig_w = orig_shape
    coords[:, [0,2]] = (coords[:, [0,2]] - pad_x) / ratio
    coords[:, [1,3]] = (coords[:, [1,3]] - pad_y) / ratio
    coords[:, 0] = np.clip(coords[:, 0], 0, orig_w - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, orig_h - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, orig_w - 1)
    coords[:, 3] = np.clip(coords[:, 3], 0, orig_h - 1)
    return coords

# -----------------------
# LOAD ONNX
# -----------------------
session = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

# -----------------------
# READ IMAGE & PREPROCESS
# -----------------------
img0 = cv2.imread(IMAGE_PATH)
if img0 is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
orig_h, orig_w = img0.shape[:2]

img, ratio, pad = letterbox(img0, IMG_SIZE)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_input = img_rgb.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2,0,1))[None]  # 1x3xHxW
img_input = np.ascontiguousarray(img_input)

# ...existing code...
# -----------------------
# INFERENCE
# -----------------------
outputs = session.run(output_names, {input_name: img_input})

# Helper to draw & save detections (xyxy,img-scaled or normalized)
def _draw_and_save(dets_array):
    if dets_array.shape[0] == 0:
        print("No detections")
        cv2.imwrite(OUT_PATH, img0)
        sys.exit(0)
    dets_local = dets_array.astype(float).copy()
    # if coords normalized (<=1) -> scale to IMG_SIZE
    if dets_local[:, :4].max() <= 1.01:
        dets_local[:, :4] *= IMG_SIZE
    dets_local = scale_coords_from_padded(dets_local, pad, ratio, (orig_h, orig_w))
    out_img_local = img0.copy()
    for x1,y1,x2,y2,score,cls in dets_local:
        x1i,y1i,x2i,y2i = map(int, (x1,y1,x2,y2))
        cls_i = int(cls)
        label = f"{cls_i} {score:.2f}" if CLASS_NAMES is None else f"{CLASS_NAMES[cls_i]} {score:.2f}"
        color = (0,255,0)
        cv2.rectangle(out_img_local, (x1i, y1i), (x2i, y2i), color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(out_img_local, (x1i, y1i - t_size[1] - 6), (x1i + t_size[0] + 6, y1i), color, -1)
        cv2.putText(out_img_local, label, (x1i + 3, y1i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.imwrite(OUT_PATH, out_img_local)
    print(f"Inference done. Results saved to {OUT_PATH}")
    sys.exit(0)

# Case A: ONNX exported with model-NMS often returns multiple outputs (boxes, scores, classes)
if len(outputs) >= 3:
    boxes = np.asarray(outputs[0])
    scores = np.asarray(outputs[1])
    classes = np.asarray(outputs[2])

    # flatten batch dim if present
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if classes.ndim == 2 and classes.shape[0] == 1:
        classes = classes[0]

    if boxes.shape[0] == 0:
        _draw_and_save(np.zeros((0,6)))

    # boxes may be xyxy or xywh depending on export - try to detect
    if boxes.shape[1] == 4:
        # assume boxes are xyxy (model-NMS) -> build dets [x1,y1,x2,y2,score,class]
        dets = np.concatenate([boxes, scores.reshape(-1,1), classes.reshape(-1,1)], axis=1)
        _draw_and_save(dets)

# Otherwise take first output as either model-NMS or raw predictions
pred = np.asarray(outputs[0])

# handle (1,N,C) or (B,N,C)
if pred.ndim == 3 and pred.shape[0] == 1:
    pred = pred[0]
elif pred.ndim == 3:
    pred = pred.reshape(-1, pred.shape[-1])

# If model already did NMS and outputs (M,6): [x1,y1,x2,y2,score,class]
if pred.ndim == 2 and pred.shape[1] == 6:
    dets = pred.copy()
    _draw_and_save(dets)

# Otherwise expect raw predictions (N,5+nc) in xywh center format
if pred.ndim != 2 or pred.shape[1] < 6:
    raise RuntimeError(f"Unexpected model output shape: {pred.shape}")

# If coords are normalized (<=1), scale them to IMG_SIZE
coords_max = pred[:,:4].max()
if coords_max <= 1.01:
    pred[:,:4] *= IMG_SIZE

# Run NMS (only if model didn't include it)
detections = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)  # (K,6) [x1,y1,x2,y2,score,class]
if detections.shape[0] == 0:
    _draw_and_save(np.zeros((0,6)))

# Map back to original image coordinates and save
_draw_and_save(detections)
# ...existing code...