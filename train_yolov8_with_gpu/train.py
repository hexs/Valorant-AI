import time
from datetime import datetime
from ultralytics import YOLO


def main():
    # model = YOLO(r'D:\Python_Projects\valo-ai\train_yolov8_with_gpu\runs\detect\train4\weights\last.pt')
    model = YOLO('yolov8m.pt')
    results = model.train(
        data='data.yaml',
        epochs=600,
        device=0,  # for gpu
        # device='cpu'  # for cpu
    )
    success = model.export(format='onnx')

    print(results)
    print(success)


if __name__ == '__main__':
    main()
