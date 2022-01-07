from unittest import TestCase

import cv2
import matplotlib.pyplot as plt

from src.frame_extractor.read_frame import read_frame


class ReadFrameTest(TestCase):
    def setUp(self) -> None:
        self.video_capture = cv2.VideoCapture(0)

    def test_read_frame(self):
        image = read_frame(self.video_capture)
        plt.imshow(image)
        plt.show()

    def tearDown(self):
        self.video_capture.release()
