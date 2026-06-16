import cv2
import numpy as np

def extract_frames(video_path):

    frames = []

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224,224))

        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames)