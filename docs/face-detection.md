How to run face detection algorithm?

1. Install packages from current `requirements.txt` file
2. Install yolo5-face by command: `git clone https://github.com/deepcam-cn/yolov5-face.git`
3. Load pretrained weights
   from [drive.google.com](https://drive.google.com/file/d/18oenL6tjFkdR1f5IgpYeQfDFqU4w3jEr/view)
4. Run command

```bash 
python3.9 detect_face.py \
    --weights /home/shamil/PycharmProjects/wise-programming/weights/yolov5n-face.pt \
    --image /home/shamil/PycharmProjects/wise-programming/demo_images/shamil-smile.jpg
```

4. Now `result.jpg` contains the predictions