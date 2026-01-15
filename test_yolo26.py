from ultralytics import YOLO
from time import time


import onnxruntime as ort
import numpy as np
import cv2

# 1. 加载 ONNX 模型
# session = ort.InferenceSession("yolo26n.onnx", providers=["CPUExecutionProvider"])

# # 2. 读取并预处理图片
# img = cv2.imread("/home/faith/yolov5/data/images/bus.jpg")
# img_resized = cv2.resize(img, (640, 640))   # YOLO26 默认输入 640x640
# img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# img_input = img_rgb.astype(np.float32) / 255.0
# img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
# img_input = np.expand_dims(img_input, axis=0)   # 增加 batch 维度

# # 3. 获取输入输出名称
# input_name = session.get_inputs()[0].name
# output_names = [o.name for o in session.get_outputs()]
# start = time()
# # 4. 推理
# outputs = session.run(output_names, {input_name: img_input})
# end = time()
# inference_time = end - start
# print(f"Inference Time: {inference_time * 1000:.2f} ms")
# # 5. 解析结果（YOLO26 输出格式为 [batch, num_boxes, 85]）
# # 其中 85 = 4 (bbox) + 1 (objectness) + 80 (类别数)
# pred = outputs[0][0]

# # 6. 简单后处理：筛选置信度 > 0.5 的目标
# conf_threshold = 0.5
# for det in pred:
#     conf = det[4] * np.max(det[5:])  # objectness * class score
#     if conf > conf_threshold:
#         x, y, w, h = det[:4]
#         cls_id = np.argmax(det[5:])
#         print(f"检测到类别 {cls_id}, 置信度 {conf:.2f}, 坐标: {x:.0f},{y:.0f},{w:.0f},{h:.0f}")



# model = YOLO("yolo26n.pt")
model = YOLO("yolo26s.pt")

# model.export(format="onnx", opset=17, dynamic=False)

# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

start = time()
results = model("/home/faith/yolov5/data/images/bus.jpg")
end = time()
print(f"Inference time: {end - start} seconds")
# print(results)
# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    
results[0].save(filename="bus_detected.jpg")  # 保存带框的图片
