from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage import io

from detect_face import load_model
from yolov5_face.utils.datasets import letterbox
from yolov5_face.utils.general import non_max_suppression_face, check_img_size, \
    scale_coords, xyxy2xywh


def rescale_coordinates(img, xywh):
    h, w, c = img.shape

    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

    return np.asarray([x1, y1, x2, y2])


def extract_box(image: np.array, model: nn.Module, device="cpu") -> np.array:
    """Extracts numpy array - box with face on image"""

    # Load model
    img_size = 800
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = image
    img0 = image.copy()

    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    coords_predicted = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                coords_pred = rescale_coordinates(orgimg, xywh).astype(int)

                coords_predicted.append(coords_pred)

    return np.asarray(coords_predicted)


def make_crop(src_image: Path, weights_path: Path, dst_image: Path):
    image = io.imread(str(src_image))
    model = load_model(str(weights_path), device="cpu")

    x1, y1, x2, y2 = extract_box(image, model)[0]
    crop = image[y1:y2, x1:x2]

    io.imsave(str(dst_image), crop)


if __name__ == '__main__':
    make_crop(
        src_image=Path('/home/shamil/PycharmProjects/wise-programming/demo_images/emotion_examples/shamil-sad.png'),
        weights_path=Path("/home/shamil/PycharmProjects/wise-programming/weights/yolov5n-face.pt"),
        dst_image=Path("/home/shamil/PycharmProjects/wise-programming/demo_images/emotion_examples/shamil-sad-crop.png")
    )
