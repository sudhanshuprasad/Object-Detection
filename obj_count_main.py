import cv2
import torch
from ultralytics import YOLO
from tracker import *
import numpy as np
import pandas
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = YOLO("yolo_weights/yolov8l.pt")


cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()

detection_area= [(0,0),(0,720),(1280,0),(1280,720)]
detected_ppl=set()

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1280,720))

    results=model(frame)
    # frame=np.squeeze(results.render())
    dataframe=results.pandas().xyxy[0]
    # print(dataframe)

    list=[]
    for index,row in dataframe.iterrows():
        # print(row)
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['xmax'])
        obj_name=str(row['name'])
        # cv2.rectangle(frame, (x1,y1),(x2,y2), (255,0,0), 2)
        # cv2.putText(frame, obj_name, (x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),2)
        if 'person' in obj_name:
            list.append([x1,y1,x2,y2])
        
    boxes_id=tracker.update(list)
    for box_id in boxes_id:
        x,y,w,h,id=box_id
        cv2.rectangle(frame, (x,y),(w,h), (255,0,0), 2)
        cv2.putText(frame, str(id), (x1,y1), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255),2)
        detected_ppl.add(id)

        # print(detected_ppl)
    print(len(boxes_id))

    cv2.imshow('FRAME',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    
