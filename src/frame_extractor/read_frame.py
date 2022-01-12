import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from skimage import io


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


def make_screenshot(saving_path: Path):
    capture = cv2.VideoCapture(0)
    image = read_frame(capture)
    io.imsave(str(saving_path), image)


if __name__ == '__main__':
    saving_path = Path("/home/shamil/PycharmProjects/wise-programming/demo_images/emotion_examples/shamil-sad.png")
    make_screenshot(saving_path)
