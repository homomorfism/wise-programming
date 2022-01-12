from pathlib import Path
from unittest import TestCase

from skimage import io

from src.emotion_predictor.emotion_analysis.classifier import preprocess, get_onnx_session, predict_emotion


class TestClassifier(TestCase):
    def setUp(self) -> None:
        model_path = Path("weights/emotion-ferplus-8.onnx")
        image_path = Path("demo_images/shamil-cropped.png")

        image = io.imread(str(image_path))
        self.image = preprocess(image)

        self.session = get_onnx_session(model_path)

    def test_classifier(self):
        prediction = predict_emotion(self.image, self.session)

        self.assertIsInstance(prediction, dict)
        self.assertTrue(len(prediction) == 8)
