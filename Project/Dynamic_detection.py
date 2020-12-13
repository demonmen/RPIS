import os
import cv2
from base_camera import BaseCamera
import datetime
import numpy as np


class Camera(BaseCamera):
    video_source = 0
    
    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        sdThresh = 4 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cap = cv2.VideoCapture(Camera.video_source)
        if not cap.isOpened():
            raise RuntimeError('Could not start camera.')
            
        def distMap(frame1, frame2):
            frame1_32 = np.float32(frame1)
            frame2_32 = np.float32(frame2)
            diff32 = frame1_32 - frame2_32
            norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + \
                     diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
            dist = np.uint8(norm32*255)
            return dist
        while True:
            _, frame1 = cap.read()
            _, frame2 = cap.read()
            while True:
                _, frame3 = cap.read()
                rows, cols, _ = np.shape(frame3)
                dist = distMap(frame1, frame3)
    
                frame1 = frame2
                frame2 = frame3
    
                mod = cv2.GaussianBlur(dist, (9,9), 0)
    
                _, thresh = cv2.threshold(mod, 100, 255, 0)
                _, stDev = cv2.meanStdDev(mod)

                if stDev > sdThresh:
                    now = str(datetime.datetime.now())[:19].replace(':', ':')
                    cv2.putText(frame2, now, (5, 20), font, 0.5, (255, 255, 255), 1)
                yield cv2.imencode('.jpg', cv2.resize(frame2, (int(frame2.shape[1]/3),int(frame2.shape[0]/3))))[1].tobytes()