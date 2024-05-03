from ultralytics import YOLO
from ultralytics import YOLOWorld

import cv2
# Load a model
model = YOLO('yolov8n-seg.pt') 
model_world= YOLOWorld('yolov8s-world.pt')
classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
model_world.set
cap=cv2.VideoCapture(0)
while cap.isOpened():
    success,frame=cap.read()
    if success:
        results=model(frame)
        anotate=results[0].plot()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1)==ord('q'):
            break