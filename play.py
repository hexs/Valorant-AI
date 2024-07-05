import time
from datetime import datetime

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
            ser.write(s.encode())

    ser.close()


def predict(data):
    from ultralytics import YOLO
    import mss

    model = YOLO(r'train_yolov8_with_gpu/runs/detect/train2/weights/best.pt')
    # cap = cv2.VideoCapture(r'D:\Python_Projects\valo-ai\videos\VALORANT   2024-07-04 05-27-48.mp4')
    center = np.array([1920, 1080]) / 2
    t1 = datetime.now()
    t2 = datetime.now()
    time = 0.05
    while data['play']:
        t2 = datetime.now()
        delta_t = (t2 - t1).total_seconds()
        if delta_t > time:
            t1 = t2
            print()
            print()
            print(delta_t)
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[0])
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            # image = cap.read()[1]
            results = model(image)

            boxes = results[0].boxes
            print(boxes.xywh)  # tensor([[1225.1987,  542.2114,   28.5000,   23.3500]], device='cuda:0')
            boxes_xywh = boxes.xywh.cpu().numpy()
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf
            if conf.tolist():
                print(boxes_xywh)  # [[     1225.2      542.21        28.5       23.35]]
                distances = np.sqrt(((boxes_xywh[:, :2] - center) ** 2).sum(axis=1))
                print('distances', distances)  # list of distance
                nearest_index = np.argmin(distances)
                nearest_box_xywh = boxes_xywh[nearest_index]
                nearest_box_xyxy = boxes_xyxy[nearest_index]
                xywh = nearest_box_xywh
                xyxy = nearest_box_xyxy
                print(f'xywh = {xywh}')  # xywh = [     1229.9      549.18      64.294      56.323]
                x, y, w, h = xywh
                x1, y1, x2, y2 = [int(x) for x in xyxy]

                annotated_frame = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)

                distance = (abs(x - center[0]), abs(y - center[1]))
                ###########################################################
                # y = mx
                # m = 8 / 20
                # a = 0.001
                # vx = int(m * distance[0] - a * distance[0] ** 2)
                # vy = int(m * distance[1] - a * distance[0] ** 2)

                ##########################################################
                # y = ax^p + bx
                a = 1.11  # 0-2
                p = 0.68  # 0-2
                b = 0  # 0-1
                vx = int((a * distance[0]) ** p + b * distance[0])
                vy = int((a * distance[1]) ** p + b * distance[1])

                ###################################################
                print('v', vx, vy)

                w_h = w * 0.4 / 2
                h_h = h / 2
                s = ''
                if center[0] < x - w_h:
                    s += 'd' * vx
                if center[0] > x + w_h:
                    s += 'a' * vx
                if center[1] < y - h_h:
                    s += 's' * vy
                if center[1] > y + h_h:
                    s += 'w' * vy
                if x - w_h <= center[0] <= x + w_h and y - h_h <= center[1] <= y + h_h:
                    s += 'c'
                print('s', s)
                s = s.replace('w' * 10, 'W').replace('a' * 10, 'A').replace('s' * 10, 'S').replace('d' * 10, 'D')
                data['control'] = s
                cv2.putText(annotated_frame, f'{s}', (100, 100), 1, 4, (255, 0, 0), 4)

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
