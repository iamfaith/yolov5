"""
Author: Yakhyokhuja Valikhujaev
Date: 2024-08-07
Description: YOLOv5 ONNX inference
Copyright (c) 2024 Yakhyokhuja Valikhujaev. All rights reserved.
"""

import cv2
import onnxruntime
import numpy as np
from typing import Tuple, List
from time import time

class YOLOv5:

    def warmup(self, imgsz=(3, 640, 640)):
        # Warmup model by running inference once
        # im = np.zeros(imgsz)  # input
        # im = im.astype(np.float32) / 255.0  # Normalize the input
        # for _ in range(1):  #
        #     self(im)  # warmup
        pass
    
    
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300, nms_mode: str = 'dnn', class_id = None, verbose=False) -> None:
        """YOLOv5 class initialization

        Args:
            model_path (str): Path to .onnx file
            conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): IOU threshold. Defaults to 0.45.
            max_det (int, optional): Maximum number of detections. Defaults to 300.
            nms_mode (str, optional): NMS calculation method. Defaults to `torchvision`
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.max_det = max_det
        self.nms_mode = nms_mode
        self.class_id = class_id
        self.verbose = verbose

        # YOLOv5 default anchors and strides
        

        self.anchors = [
            np.array([[10., 13.], [16., 30.], [33., 23.]], dtype=np.float32),
            np.array([[30., 61.], [62., 45.], [59., 119.]], dtype=np.float32),
            np.array([[116., 90.], [156., 198.], [373., 326.]], dtype=np.float32)
        ]
        self.na = 3 #self.anchors[0].shape[0]
        self.grid = [None] * self.na
        self.anchor_grid = [None] * self.na
        

        # Initialize model
        self._initialize_model(model_path=model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the model on the given image and return predictions.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple: boxes, confidence scores, class indexes
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input image must be a numpy array with 3 dimensions (H, W, C).")

        outputs = self.inference(image)
        predictions = self.postprocess(outputs)
        return predictions

    def inference(self, image: np.ndarray) -> List[np.ndarray]:
        """Run inference on the given image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            List[np.ndarray]: Model outputs.
        """
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def _initialize_model(self, model_path: str) -> None:
        """Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get model metadata
            metadata = self.session.get_modelmeta().custom_metadata_map
            if len(self.output_names) == 1:
                self.stride = int(metadata.get("stride", 32))  # Default stride value
            else:
                self.stride = [8, 16, 32]
                # 归一化 anchors
                self.anchors = [a / s for a, s in zip(self.anchors, self.stride)]
            
            self.names = eval(metadata.get("names", "{}"))  # Default to empty dict
            self.nc = len(self.names)
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: HWC -> CHW, BGR to RGB, Normalize and Add batch dimension.
        """
        # image = image.transpose(2, 0, 1)  # Convert from HWC -> CHW
        # image = image[::-1]  # Convert BGR to RGB
        # image = np.ascontiguousarray(image)
        # image = image.astype(np.float32) / 255.0  # Normalize the input
        # image_tensor = image[np.newaxis, ...]  # Add batch dimension
        
        if len(image.shape) == 3:
            image = image[np.newaxis, ...]  # Add batch dimension

        return image

    def postprocess(self, prediction: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post processing

        Args:
            prediction (List[np.ndarray]): Model raw outputs

        Returns:
            Tuple: boxes, confidence scores, class indexes
        """
        if len(prediction) == 1:
            # Single output: assume concatenated (1, 25200, 85)
            outputs = np.squeeze(prediction[0])
        else:
            # Sort prediction by size descending (largest first, corresponding to smallest stride)
            prediction = sorted(prediction, key=lambda p: p.shape[2] * p.shape[3], reverse=True)
            # Multiple outputs: process each layer
            outputs = []
            for i, pred in enumerate(prediction):
                y = pred[0]  # (na, ny, nx, no)
                na, ny, nx, no = y.shape


                # Make grid
                # xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
                # grid = np.stack((xv, yv), 2).astype(np.float32) - 0.5  # (ny, nx, 2)
                # grid = np.expand_dims(grid, 0)  # (1, ny, nx, 2)

                # Anchor grid
                # anchors = np.array(self.anchors[i]).reshape(na, 2)  # (na, 2)
                # anchor_grid = np.zeros((na, ny, nx, 2))  # (na, ny, nx, 2)
                # for a in range(na):
                    # anchor_grid[a, :, :, :] = anchors[a] * self.stride[i]

                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                # Split y into xy, wh, conf
                xy = y[..., :2]
                wh = y[..., 2:4]
                conf = y[..., 4:]

                # Compute coordinates
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]


                # multiply batch size
                # conf = conf[None, ...]

                # single batch
                xy = np.squeeze(xy, axis=0)
                wh = np.squeeze(wh, axis=0)
                # Concatenate back
                y_processed = np.concatenate((xy, wh, conf), axis=-1)
                
                # multiply batch size                
                # bs = 1
                # y_processed = y_processed.reshape(bs, -1, no)

                # single batch  
                y_processed = y_processed.reshape(-1, no)

                outputs.append(y_processed)

            # single batch
            outputs = np.concatenate(outputs, axis=0)

            # multiply batch size
            # Concatenate all layers
            # outputs = np.concatenate(outputs, axis=1)
            # outputs = np.squeeze(outputs[0])

        # Extract boxes, scores, and classes
        boxes = outputs[:, :4]  # xywh
        scores = outputs[:, 4]  # confidence scores
        classes = outputs[:, 5:]  # class probabilities

        boxes = self.xywh2xyxy(boxes)

        # Apply confidence threshold
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        # Get class with highest probability for each detection
        class_ids = np.argmax(classes, axis=1)[:self.max_det]

        # Apply NMS
        if self.nms_mode == "torchvision":
            import torch
            import torchvision
            # better performance
            indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), self.iou_threshold).numpy()
        else:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

        return boxes, scores, class_ids

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """xywh -> xyxy

        Args:
            x (np.ndarray): [x, y, w, h]

        Returns:
            np.ndarray: [x1, y1, x2, y2]
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def _make_grid(self, nx=20, ny=20, i=0):
        shape = (1, self.na, ny, nx, 2)
        y = np.arange(ny, dtype=np.float32)
        x = np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x, indexing='ij')
        grid = np.stack((xv, yv), axis=2)
        grid = np.broadcast_to(grid, shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.broadcast_to(anchor_grid, shape)
        return grid, anchor_grid