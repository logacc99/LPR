import numpy as np
import torch
import cv2
import sys
import sys
sys.path.insert(0, "yolov5")

from    utils.augmentations import letterbox
from    models.common       import DetectMultiBackend
from    utils.general       import (LOGGER, non_max_suppression, scale_coords)
from    utils.plots         import Annotator, colors
from    utils.torch_utils   import time_sync


class Vehicle_Detector():
    def __init__(self, weights = 'yolov5s.pt', threshold = 0.25):
        # """ Model """
        self.conf_thres = threshold
        self.weights = weights
        self.max_distance = 30
        self.obj_cnt = 0
        self.car_count = 0
        self.truck_count = 0
        self.bike_count = 0
        self.device = torch.device('cpu')
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False)
        self.stride, self.names, self.pt, self.jit, self.onnx = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx

    @torch.no_grad()
    def detect(self,
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        image=False  # input image
        ):
        # Tracker
        # tracker = cv2.TrackerCSRT_create()
        
        # Dataloader
        dt, seen = [0.0, 0.0, 0.0], 0
        label_vehicle = []

        # Padded resize
        im0s = image
        im = letterbox(im0s, auto=self.pt and not self.jit)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        # Run detected
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if False else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, iou_thres, None, False, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # boxes detected
            filter_boxes = []
            boxes_d = []
            seen += 1
            im0 = im0s.copy()

            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    # Filter by distance
                    box = annotator.box_info(xyxy)
                    boxes_d.append(box)
                    if (((box[1] + box[3]) // 2) > im0.shape[0] * 0.05) and (self.names[c] in ['car', 'motorcycle', 'bus', 'truck']):
                        filter_boxes.append(box)
                        label_vehicle.append(self.names[c])
        return filter_boxes, label_vehicle


def demo_image(path):
    detector = Vehicle_Detector()
    image = cv2.imread(path)
    # TEST STEP 1
    vehicle_boxes_detected, label_vehicle = detector.vehicle_detected_model(image=image)
    print("Vehicle detected = ", vehicle_boxes_detected)
    print("Label detected = ", label_vehicle)


if __name__ == '__main__':
    demo_image('data/images/test.png')