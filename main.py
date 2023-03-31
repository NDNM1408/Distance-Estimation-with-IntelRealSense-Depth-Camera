import cv2
import pyrealsense2
from ultralytics import YOLO
import cvzone

from realsense_camera import *
from detector import *

rsc = RealsenseCamera()
dt = Detector()
while True:
    ret, bgr_frame, depth_frames = rsc.get_frame_stream()
    res = dt.detect_object(bgr_frame)
    bgr_frame = dt.draw_object_info(bgr_frame, depth_frames)

    cv2.imshow("BGR Frame", bgr_frame)
    # cv2.imshow("Depth frame", depth_frames)
    print(depth_frames)
    # cv2.imshow("Depth Frame", depth_frames)
    key = cv2.waitKey(1)
    if key == 27:
        break
