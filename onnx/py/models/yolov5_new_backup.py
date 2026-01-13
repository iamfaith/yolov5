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


class YOLOv5:
    def __init__(self, model_path: str, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300, nms_mode: str = 'dnn', class_id = None) -> None:
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

        # YOLOv5 default anchors and strides
        

        self.anchors = [
            np.array([[10., 13.], [16., 30.], [33., 23.]], dtype=np.float32),
            np.array([[30., 61.], [62., 45.], [59., 119.]], dtype=np.float32),
            np.array([[116., 90.], [156., 198.], [373., 326.]], dtype=np.float32)
        ]
        
        self.grid = []
        self.anchor_grid = []
        

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

            # Set na based on number of outputs
            self.na = len(self.output_names)
            
            
            # Get model metadata
            metadata = self.session.get_modelmeta().custom_metadata_map
            if len(self.output_names) == 1:
                self.stride = int(metadata.get("stride", 32))  # Default stride value
            else:
                self.stride = [8, 16, 32]
                # 归一化 anchors
                self.anchors = [a / s for a, s in zip(self.anchors, self.stride)]
                
                # Build grids based on output shapes
                for i, output in enumerate(self.session.get_outputs()):
                    shape = output.shape  # e.g., (1, 3, ny, nx, no)
                    ny, nx = shape[2], shape[3]
                    grid, anchor_grid = self._make_grid(nx, ny, i)
                    self.grid.append(grid)
                    self.anchor_grid.append(anchor_grid)
            
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
        image = image.transpose(2, 0, 1)  # Convert from HWC -> CHW
        image = image[::-1]  # Convert BGR to RGB
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0  # Normalize the input
        image_tensor = image[np.newaxis, ...]  # Add batch dimension

        return image_tensor

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
        elif len(prediction) == 3:
            # Sort prediction by size descending (largest first, corresponding to smallest stride)
            prediction = sorted(prediction, key=lambda p: p.shape[2] * p.shape[3], reverse=True)
            # Multiple outputs: process each layer
            outputs = []
            for i, pred in enumerate(prediction):
                y = pred[0]  # (na, ny, nx, no)
                
                if len(prediction[0].shape) == 4:

                    _, ny, nx = y.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                    y = y.reshape(self.na, -1, ny, nx)
                    _, no, _, _ = y.shape
                    
                    scores = y[:, 4, ...] # confidence scores
                    mask = scores > self.conf_threshold
                    if not mask.any():
                        continue

                    # 原始做法，tranpose再过滤，现在改为直接过滤
                    # y = np.transpose(y, (0, 2, 3, 1))
                    # y = np.ascontiguousarray(y)
                    
                    mask_flat = mask.ravel() 
                    # 24, 6 可能变成两维
                    y = np.transpose(y, (0, 2, 3, 1))[mask]

                    y = np.ascontiguousarray(y)
                    xy_filtered = y[..., :2]
                    wh_filtered = y[..., 2:4]
                    conf_filtered = y[..., 4:]
                    
                elif len(prediction[0].shape) == 5: # with transpose
                    na, ny, nx, no = y.shape

                    # self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                    # Split y into xy, wh, conf
                    xy = y[..., :2]
                    wh = y[..., 2:4]
                    conf = y[..., 4:]

                    # Apply confidence threshold mask
                    scores = conf[..., 0]  # confidence scores
                    mask = scores > self.conf_threshold
                    if not mask.any():
                        continue


                    # Flatten mask and arrays for filtering
                    mask_flat = mask.ravel()  # (na*ny*nx,)
                    xy_flat = xy.reshape(-1, 2)  # (na*ny*nx, 2)
                    wh_flat = wh.reshape(-1, 2)  # (na*ny*nx, 2)
                    conf_flat = conf.reshape(-1, conf.shape[-1])  # (na*ny*nx, nc+1)

                    # Filter using flat mask
                    xy_filtered = xy_flat[mask_flat]  # (num_filtered, 2)
                    wh_filtered = wh_flat[mask_flat]  # (num_filtered, 2)
                    conf_filtered = conf_flat[mask_flat]  # (num_filtered, nc+1)
                    
                    
                    

                # Filter grid and anchor_grid accordingly (flatten them too)
                grid_flat = self.grid[i].reshape(-1, 2)  # (na*ny*nx, 2)
                anchor_grid_flat = self.anchor_grid[i].reshape(-1, 2)  # (na*ny*nx, 2)
                grid_filtered = grid_flat[mask_flat]  # (num_filtered, 2)
                anchor_grid_filtered = anchor_grid_flat[mask_flat]  # (num_filtered, 2)
     
                # Compute coordinates
                xy_filtered = (xy_filtered * 2 + grid_filtered) * self.stride[i]
                wh_filtered = (wh_filtered * 2) ** 2 * anchor_grid_filtered

                # Concatenate back
                y_processed = np.concatenate((xy_filtered, wh_filtered, conf_filtered), axis=-1)
                y_processed = y_processed.reshape(-1, no)
                

                outputs.append(y_processed)

            # single batch
            if not outputs:
                outputs = np.empty((0, no))
            else:
                outputs = np.concatenate(outputs, axis=0)

            # Extract boxes, scores, and classes
            boxes = outputs[:, :4]  # xywh
            scores = outputs[:, 4]  # confidence scores
            classes = outputs[:, 5:]  # class probabilities
            boxes = self.xywh2xyxy(boxes)


        # Get class with highest probability for each detection
        class_ids = np.argmax(classes, axis=1)[:self.max_det]

        # Filter by specific class_id if provided
        if self.class_id is not None:
            if isinstance(self.class_id, list):
                mask = np.isin(class_ids, self.class_id)  # 支持多个类别
            else:
                mask = class_ids == self.class_id  # 单个类别
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]
            class_ids = class_ids[mask]


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