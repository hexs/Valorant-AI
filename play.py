import os
import time
from datetime import datetime, timedelta
import keyboard
import cv2
import numpy as np
import multiprocessing
import serial


def setup():
    '''
        <+xxx-yyy>
        Mouse.move(x, y);

        if <c>
        Mouse.click();
    '''
    port = 'COM3'
    baud_rate = 9600
    ser = serial.Serial(port, baud_rate)
    return ser


def loop(ser, s):
    if s:
        print(f'--- {s} ---')
        ser.write(s.encode())


def predict(data):
    from ultralytics import YOLO
    import mss
    ser = setup()
    last_ser_time = datetime.now()

    model = YOLO(r'train_yolov8_with_gpu/runs/detect/train4/weights/best.pt')
    center = np.array([0.5, 0.5])
    focus_xywh = np.array([0.5, 0.5, 0.7, 0.6])
    WH_ = np.array([1920, 1080])

    focus_xy = np.array(focus_xywh[:2])
    focus_wh = np.array(focus_xywh[2:])

    xy1_ = (focus_xy - focus_wh / 2) * WH_
    xy2_ = (focus_xy + focus_wh / 2) * WH_
    xyxy_ = np.concatenate((xy1_, xy2_))
    xyxy_int_tuple = tuple(map(int, xyxy_))

    with mss.mss() as sct:
        screenshot = sct.grab(xyxy_int_tuple)
        WH_ = np.array(screenshot.size)
    print(WH_)  # [1344  648]
    center_ = center * WH_

    enable = False
    while data['play']:
        if enable:
            print()
            print()
            with mss.mss() as sct:
                screenshot = sct.grab(xyxy_int_tuple)
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            results = model(image)
            cv2.rectangle(image, (0, 0), (600, 100), (255, 255, 255), -1)

            boxes = results[0].boxes
            print(boxes.xywhn)  # tensor([[0.0263, 0.2256, 0.0524, 0.1227]], device='cuda:0')
            boxes_xywh = boxes.xywhn.cpu().numpy()

            conf = boxes.conf
            if conf.tolist():
                print(boxes_xywh)  # [[   0.026281      0.2256    0.052399     0.12273]]
                distances = np.sqrt(((boxes_xywh[:, :2] - center) ** 2).sum(axis=1))
                print('distances', distances)  # distances [    0.54745]
                nearest_index = np.argmin(distances)
                nearest_box_xywh = boxes_xywh[nearest_index]

                xy, wh = np.split(nearest_box_xywh, 2)
                print(f'xy | wh = {xy} | {wh}')  # xy | wh = [   0.026281      0.2256] | [   0.052399     0.12273]
                distance = xy - center
                print('distance', distance)  # distance (-0.47371928952634335, -0.2744021415710449)
                distance_ = distance * WH_
                print('distance', distance_)  # distance [    -636.68     -177.81]

                ###########################################################
                ### y = b arctan(ax)
                a = 0.001306
                b = 455
                v_ = 3 * b * np.arctan(a * distance_)
                v_ = v_.astype(int)
                #########################################################

                xy_head = xy - [0, wh[1] / 4]
                wh_head = wh * [0.2, 0.35]
                print(xy_head, wh_head)
                s = ''
                if np.all(xy_head - wh_head <= center) and np.all(center <= xy_head + wh_head):
                    s += '<c>'
                else:
                    if last_ser_time+timedelta(milliseconds=60) <datetime.now():
                        last_ser_time = datetime.now()
                        s += f'<{"+" if v_[0] >= 0 else "-"}{abs(v_[0]):03}{"+" if v_[1] >= 0 else "-"}{abs(v_[1]):03}>'
                print('s', s)
                loop(ser, s)
                xy_ = xy * WH_
                xy1_ = (xy - wh / 2) * WH_
                xy2_ = (xy + wh / 2) * WH_
                xy1_head_ = (xy_head - wh_head / 2) * WH_
                xy2_head_ = (xy_head + wh_head / 2) * WH_
                cv2.rectangle(image, xy1_.astype(int), xy2_.astype(int), (0, 0, 255), 1)
                cv2.rectangle(image, xy1_head_.astype(int), xy2_head_.astype(int), (0, 255, 0), 1)
                cv2.line(image, xy_.astype(int), center_.astype(int), (200, 200, 0), 1)
                cv2.putText(image, f'{s}', (10, 60), 1, 3, (255, 0, 0), 3)
                os.makedirs('img_out', exist_ok=True)
                cv2.imwrite(datetime.now().strftime('img_out/%y%m%d %H%M%S %f.png'), image)

            # cv2.imshow('img', cv2.resize(image, (0, 0), fx=.5, fy=.5))
            # cv2.imshow('img', image)
            # cv2.waitKey(1)

        if data['last_key'] == 'f4':
            data['play'] = False
        if data['last_key'] == 'ctrl':
            enable = True
        if data['last_key'] == 'alt':
            enable = False
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    ser.close()


def on_key_press(event, data):
    data['last_key'] = event.name


def kb_listener(data):
    keyboard.hook(lambda event: on_key_press(event, data))
    keyboard.wait('f4')


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    data = manager.dict()
    data['play'] = True
    data['last_key'] = ''

    predict_process = multiprocessing.Process(target=predict, args=(data,))
    keyboard_process = multiprocessing.Process(target=kb_listener, args=(data,))

    predict_process.start()
    keyboard_process.start()

    predict_process.join()
