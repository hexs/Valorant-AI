import cv2
import numpy as np
import multiprocessing
import serial


def control(data):
    '''
        if (x == 'w') Mouse.move(0, -10);
        if (x == 'a') Mouse.move(-10, 0);
        if (x == 's') Mouse.move(0, 10);
        if (x == 'd') Mouse.move(10, 0);

        if (x == 'W') Mouse.move(0, -100);
        if (x == 'A') Mouse.move(-100, 0);
        if (x == 'S') Mouse.move(0, 100);
        if (x == 'D') Mouse.move(100, 0);


        if (x == 'c') Mouse.click();
        if (x == 'p') Mouse.press();
        if (x == 'r') Mouse.release();
    '''
    port = 'COM3'
    baud_rate = 9600
    ser = serial.Serial(port, baud_rate)
    while data['play']:
        s = data['control']
        data['control'] = []
        if s:
            print(s, '--------------------------------------------------------------')
            ser.write((''.join(s)).encode())

    ser.close()


def predict(data):
    from ultralytics import YOLO
    import mss

    model = YOLO(r'train_yolov8_with_gpu/runs/detect/train2/weights/best.pt')
    # cap = cv2.VideoCapture(r'D:\Python_Projects\valo-ai\videos\VALORANT   2024-07-04 05-27-48.mp4')
    while data['play']:
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[0])
        image = np.array(screenshot)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # image = cap.read()[1]
        results = model(image)

        boxes = results[0].boxes
        xywh = boxes.xywh
        xyxy = boxes.xyxy
        conf = boxes.conf
        if conf.tolist():
            xyxy = xyxy[0].tolist()
            xywh = xywh[0].tolist()
            print(f'xywh = {xywh}')
            print(xyxy)
            x1, y1, x2, y2 = [int(x) for x in xyxy]
            annotated_frame = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            center = np.array([1920, 1080]) / 2
            s = []
            if center[0] < xywh[0] - 10:
                s.append('ddd')
            if center[0] > xywh[0] + 10:
                s.append('aaa')
            if center[1] < xywh[1] - 10:
                s.append('sss')
            if center[1] > xywh[1] + 10:
                s.append('www')

            if xywh[0] - 10 < center[0] < xywh[0] + 10 and xywh[1] - 10 < center[1] < xywh[1] + 10:
                s.append('c')

            data['control'] = s


        else:
            annotated_frame = image
        cv2.imshow('img', cv2.resize(annotated_frame, (0, 0), fx=.25, fy=.25))
        # cv2.imshow('img', cv2.resize(annotated_frame, (0, 0), fx=.75, fy=.75))
        key = cv2.waitKey(1)
        if key == ord('q'):
            data['play'] = False


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    data = manager.dict()
    data['play'] = True
    data['control'] = []

    predict_process = multiprocessing.Process(target=predict, args=(data,))
    control_process = multiprocessing.Process(target=control, args=(data,))

    predict_process.start()
    control_process.start()

    predict_process.join()
