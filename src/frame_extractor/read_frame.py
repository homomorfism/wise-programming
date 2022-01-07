import time

import cv2
import matplotlib.pyplot as plt


def read_frame(video_capture):
    flag, frame = video_capture.read()

    if not flag:
        raise ValueError("Can not read from video camera")

    frame = frame[:, :, ::-1]  # Converting BGR to RGB

    return frame


def repeat_read_frame():
    capture = cv2.VideoCapture(0)
    try:
        while True:
            time.sleep(0.5)
            image = read_frame(capture)
            plt.imshow(image)
            plt.show()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Attempting graceful shutdown...")


if __name__ == '__main__':
    repeat_read_frame()
