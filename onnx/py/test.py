import os
import cv2
import logging
import argparse
import numpy as np
from pathlib import Path

from typing import List, Tuple

from models import YOLOv5
from utils.general import check_img_size, scale_boxes, draw_detections, colors, increment_path, LoadMedia

weights = "/home/faith/yolov5/yolov5n6-6.2.onnx"
# weights = '/home/faith/yolov5/exp4/weights/best.onnx'
# weights = '/home/faith/yolov5/exp3/weights/best.onnx'
source = "/home/faith/fux.png"
source = '/home/faith/yolov5/data/images/zidane.jpg'
source = '/home/faith/yolov5/data/images/bus.jpg'
project = "test"
conf_thres = 0.15
iou_thres = 0.45
max_det = 1000
img_size = [640, 640]
save_dir = increment_path(Path(project))
save_dir.mkdir(parents=True, exist_ok=True)


model = YOLOv5(weights, conf_thres, iou_thres, max_det)
img_size = check_img_size(img_size, s=model.stride)
dataset = LoadMedia(source, img_size=img_size)



for resized_image, original_image, status in dataset:
    # Model Inference
    boxes, scores, class_ids = model(resized_image)

    # Scale bounding boxes to original image size
    boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

    # Draw bunding boxes
    for box, score, class_id in zip(boxes, scores, class_ids):
        draw_detections(original_image, box, score, model.names[int(class_id)], colors(int(class_id)))

    # Print results
    for c in np.unique(class_ids):
        n = (class_ids == c).sum()  # detections per class
        status += f"{n} {model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

    
    # if view_img:
    #     # Display the image with detections
    #     cv2.imshow('Webcam Inference', original_image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
    #         break

    print(status)

    if dataset.type == "image":
        save_path = str(save_dir / f"frame_{dataset.frame:04d}.jpg")
        print(save_path)
        cv2.imwrite(save_path, original_image)