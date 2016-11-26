import cv2


class CameraInputGenerator:
    def __init__(self, index = 0):
        self.cap = cv2.VideoCapture(index)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        return frame

    def __del__(self):
        self.cap.release()