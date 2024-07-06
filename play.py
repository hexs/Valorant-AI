import os
import time
from datetime import datetime

import cv2
import numpy as np
import multiprocessing
import serial


def control(data):
    '''
        <+xxx-yyy>
        Mouse.move(x, y);

        if <c>
        Mouse.click();
    '''
    port = 'COM3'
    baud_rate = 9600
    ser = serial.Serial(port, baud_rate)
    while data['play']:
        s = data['control']
        if s:
            print(s, '--------------------------------------------------------------')
            ser.write(s.encode())
        data['control'] = ''

    ser.close()


def predict(data):
    from ultralytics import YOLO
    import mss
    from math import atan

    model = YOLO(r'train_yolov8_with_gpu/runs/detect/train3/weights/best.pt')
    # cap = cv2.VideoCapture(r'D:\Python_Projects\valo-ai\videos\VALORANT   2024-07-04 05-27-48.mp4')
    center = np.array([0.5, 0.5])
    t1 = datetime.now()
    t2 = datetime.now()
    time = 0.05
    x, y, w, h = 0.5, 0.5, 0.7, 0.6
    x1_ = int((x - w / 2) * 1920)
    y1_ = int((y - h / 2) * 1080)
    x2_ = int((x + w / 2) * 1920)
    y2_ = int((y + h / 2) * 1080)

    while data['play']:
        # t2 = datetime.now()
        # delta_t = (t2 - t1).total_seconds()
        if True:
            # t1 = t2
            print()
            print()
            # print(delta_t)
            with mss.mss() as sct:
                screenshot = sct.grab((x1_, y1_, x2_, y2_), )
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            # image = cap.read()[1]
            results = model(image)
            cv2.rectangle(image, (0, 0), (600, 100), (255, 255, 255), -1)

            boxes = results[0].boxes
            print(boxes.xywhn)  # tensor([[0.3384, 0.5006, 0.0160, 0.0301]], device='cuda:0')
            boxes_xywh = boxes.xywhn.cpu().numpy()
            boxes_xyxy_px = boxes.xyxy.cpu().numpy()
            conf = boxes.conf
            if conf.tolist():
                print(boxes_xywh)  # [[    0.33843     0.50057    0.015984     0.03014]]
                distances = np.sqrt(((boxes_xywh[:, :2] - center) ** 2).sum(axis=1))
                print('distances', distances)  # distances [    0.16157]
                nearest_index = np.argmin(distances)
                nearest_box_xywh = boxes_xywh[nearest_index]
                nearest_box_xyxy_px = boxes_xyxy_px[nearest_index]
                xywh = nearest_box_xywh
                xyxy_px = nearest_box_xyxy_px
                print(f'xywh = {xywh}')  # xywh = [    0.33843     0.50057    0.015984     0.03014]
                x, y, w, h = xywh
                x1_px, y1_px, x2_px, y2_px = [int(x) for x in xyxy_px]

                distance = (x - center[0], y - center[1])
                print('distance', distance)  # distance (-0.16157126426696777, 0.0005723237991333008)
                distance = distance * np.array((x2_ - x1_, y2_ - y1_), )
                print('distance', distance)  # distance [    -310.22     0.61798]

                # px
                ###########################################################
                ### y = b arctan(ax)
                a = 0.001306
                b = 455
                vx_px = int(3 * b * atan(a * distance[0]))
                vy_px = int(3 * b * atan(a * distance[1]))
                #########################################################

                print('v', vx_px, vy_px)

                w_h = w * 0.4 / 2
                h_h = h / 2.8
                # h_h = (h - h * 0.1) / 2
                y -= h / 4

                s = ''
                if x - w_h <= center[0] <= x + w_h and y - h_h <= center[1] <= y + h_h:
                    s += '<c>'
                else:
                    s += f'<{"+" if vx_px >= 0 else "-"}{abs(vx_px):03}{"+" if vy_px >= 0 else "-"}{abs(vy_px):03}>'

                data['control'] = s

                cv2.rectangle(image, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 255), 1)
                cv2.rectangle(image, (int((x - w_h / 2) * (x2_ - x1_)), int((y - h_h / 2) * (y2_ - y1_))),
                              (int((x + w_h / 2) * (x2_ - x1_)), int((y + h_h / 2) * (y2_ - y1_))), (0, 255, 0), 1)

                cv2.putText(image, f'{s}', (10, 60), 1, 4, (255, 0, 0), 4)
                os.makedirs('img_out', exist_ok=True)
                cv2.imwrite(datetime.now().strftime('img_out/%H%M%S.png'), image)

            cv2.imshow('img', cv2.resize(image, (0, 0), fx=.25, fy=.25))
            # cv2.imshow('img', image)
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
