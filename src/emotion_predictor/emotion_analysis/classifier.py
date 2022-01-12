from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from scipy.special import softmax
from skimage import color
from skimage import io
from skimage.transform import resize

emotions_order = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']


def predict_emotion(image: np.array, session: onnxruntime.InferenceSession):
    ort_inputs = {session.get_inputs()[0].name: image}
    scores = session.run(None, ort_inputs)[0][0]
    scores = softmax(scores, axis=0)

    emotions_predictions = {emotion: score for emotion, score in zip(emotions_order, scores)}

    return emotions_predictions


def preprocess(image: np.array):
    image = color.rgb2gray(image)
    image = resize(image, (64, 64), anti_aliasing=True)
    image = (image - 0.485) / 0.229

    image = image.reshape(1, 1, 64, 64)
    image = image.astype(np.float32)

    return image


def get_onnx_session(model_path: Path):
    assert model_path.is_file(), f"Weights file not found in {model_path.absolute()}!"

    model = onnx.load(str(model_path))
    session = onnxruntime.InferenceSession(model.SerializeToString())
    return session


def make_classification(model_path: Path, image_path: Path):
    session = get_onnx_session(model_path)
    image = io.imread(str(image_path))
    image = preprocess(image)
    predictions = predict_emotion(image, session)
    print(predictions)


if __name__ == '__main__':
    make_classification(
        model_path=Path('/home/shamil/PycharmProjects/wise-programming/weights/emotion-ferplus-8.onnx'),
        image_path=Path(
            "/home/shamil/PycharmProjects/wise-programming/demo_images/emotion_examples/shamil-sad-crop.png")
    )
