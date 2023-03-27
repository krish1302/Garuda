# Bottom to Top (Vertical)
# Line Crossing (if someone crosses the line,detect the person and change the color from Green to red)

import cv2, time
import datetime
import imutils
import numpy as np
import pandas as pd
from centroidtracker import CentroidTracker
import os
from statistics import mean

current_time = str(datetime.datetime.now().time())
current_hrs = int(current_time[:2])

# loading predefined model from MobilNetSSD
protopath = r"D:\Kushagramati\Garuda-Angular-NodeJS-MySQL\usecase\MobileNetSSD_deploy.prototxt"
modelpath = r"D:\Kushagramati\Garuda-Angular-NodeJS-MySQL\usecase\MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# # Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# cap = cv2.VideoCapture(r'D:\work\garuda\DYNDNS\DYN_Line11.mp4')
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
# Start(x,y) and end(x,y) coordinates


# start_point = (280, 200)
# end_point = (250, 300)

# start_point = (360, 120)
# end_point = (350, 148)

#########################################

# fps_start_time = datetime.datetime.now()
# fps = 0
# total_frames = 0
# person_center=(0,0)
# line_center=(0,0)

# alarm_trigger = False

# loading class that can be detect by our model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=25, maxDistance=70)


# 
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


#cap = cv2.VideoCapture(r'D:\Kushagramati\New folder\Garuda_vivek_final\Garuda_vivek_final\DYN_Line.mp4')
cap = cv2.VideoCapture(0)


motion_list = [None, None]
time = []
df = pd.DataFrame(columns=["Start", "End"])
total_frames = 0
person_center = (0, 0)
line_center = (0, 0)

start_point = (360, 120)
end_point = (350, 148)


mask_start_point = (50, 150)


mask_end_point = (550, 300)


color = (0, 0, 0)

line_thickness = -1

while True:

    ret, frame1 = cap.read()
    # 
    crossing_detection = 0

    frame1 = imutils.resize(frame1, width=600)
    frame = frame1.copy()
    #         frame = cv2.rectangle(frame, start_point, end_point, color, line_thickness)
    # 
    frame = cv2.rectangle(frame, mask_start_point, mask_end_point, color, line_thickness)

    #     start_point = (360, 120)
    #     end_point = (350, 148)
    cv2.line(img=frame, pt1=(start_point), pt2=(end_point), color=(0, 0, 255), thickness=2, lineType=8, shift=0)
    ### 
    cv2.line(img=frame1, pt1=(start_point), pt2=(end_point), color=(0, 0, 255), thickness=2, lineType=8, shift=0)

    total_frames = total_frames + 1
    (H, W) = frame.shape[:2]
    # print(f"Height of the frame : {H} and width : {W}") # 337X600
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

    detector.setInput(blob)
    # detection result from our file
    person_detections = detector.forward()
    rects = []
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            #             alarm_trigger = True

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            #             (startX, startY, endX, endY) = person_box.astype("int")
            rects.append(person_box)

    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)
    
    objects = tracker.update(rects)

    # 
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox

        startX = int(x1)
        startY = int(y1)
        endX = int(x2)
        endY = int(y2)

        person_center = ((startX + endX) / 2, (startY + endY) / 2)

        ## line_center = ((2 * x + w) / 2, (2 * y + h) / 2)
        line_center = ((start_point[0] + start_point[1]) / 2, (end_point[0] + end_point[1]) / 2)
        #         print(f'line center x : {line_center[0]}, y : {line_center[1]}')
        #         print("xs,ys,xe,ye : ",startX,startY,endX,endY)

        ### Bbox in cal. frame(i.e frame)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text = "Alert !! Line Crossed!"

        cv2.rectangle(frame1, (startX, startY), (endX, endY), (0, 255, 0), 2)
        text = "Alert !! Line Crossed!"

        #             if(person_center[0] < (start_point[0]+end_point[0])/2):
        #                 cv2.putText(frame, text, (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 65), 1)
        #                 cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        if (person_center[0] < (start_point[0] + end_point[0]) / 2):
            cv2.putText(frame1, text, (180, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 65), 1)
            cv2.rectangle(frame1, (startX, startY), (endX, endY), (0, 0, 255), 2)
            crossing_detection = 1
    #                 current_time1 = str(datetime.datetime.now().time())
    # #                 current_hrs = int(current_time[:2])
    #                 print("detected at : ",current_time1[:8])

    motion_list.append(crossing_detection)
    motion_list = motion_list[-2:]

    if motion_list[-1] == 1 and motion_list[-2] == 0:
        time.append(datetime.datetime.now())

    if motion_list[-1] == 0 and motion_list[-2] == 1:
        time.append(datetime.datetime.now())

    # 
    cv2.imshow("Line_Cross_Test", frame1)

    ## ESC to exit
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        if crossing_detection == 1:
            time.append(datetime.datetime.now())
        break

#         key = cv2.waitKey(10)
#         if key == 27:
#             break


#     saving_video.release()
cap.release()
cv2.destroyAllWindows()

