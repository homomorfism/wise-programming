from pathlib import Path
from unittest import TestCase

import numpy as np
from skimage import io

from detect_face import load_model
from src.emotion_predictor.face_detection.detect import extract_box


class TestDetection(TestCase):

    def setUp(self) -> None:
        image_path = Path("demo_images/shamil-smile.jpg")
        weights_path = Path("weights/yolov5n-face.pt")
        reference_crop_path = Path("demo_images/shamil-cropped.png")

        self.image = io.imread(str(image_path))
        self.model = load_model(weights=str(weights_path), device="cpu")
        self.reference_crop = io.imread(str(reference_crop_path))

    def test_face_detection(self):
        coords = extract_box(self.image, self.model)

        x1, y1, x2, y2 = coords[0]
        cropped_face = self.image[y1:y2, x1:x2]

        self.assertTrue(np.all(self.reference_crop == cropped_face), "Face detection boxes mismatch.")
