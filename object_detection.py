from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
# print(cap.isOpened()) # False
# print(cap.read())
# ret, img = cap.read()
# cv2.imshow('img', img)
model = YOLO("yolo_weights/yolov8n.pt")
results = model("me.jpg", show=True)
print(type(results[0]))
# cv2.resizeWindow("me.jpg", 200, 200)
cv2.waitKey(0)