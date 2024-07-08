import os
import time
from datetime import datetime, timedelta
import keyboard
import cv2
import mouse
import numpy as np
import multiprocessing
import serial


def setup():
    '''
        <+xxx-yyy>
        Mouse.move(x, y);

        <c>
        Mouse.click();
    '''
    port = 'COM3'
    baud_rate = 921600
    ser = serial.Serial(port, baud_rate)
    return ser


def loop(ser, s):
    if s:
        print(f'--- {s} ---')
        ser.write(s.encode())


def predict(play, m_right, k_ctrl):
    from ultralytics import YOLO
    import mss

    debug = {
        'imwrite': True,
        'imshow': False,
        'send_data': True
    }
    # debug = {
    #     'imwrite': False,
    #     'imshow': True,
    #     'send_data': False
    # }

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
    center_ = (center * WH_).astype(int)

    while play.value:
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
            conf = float(boxes.conf[nearest_index])
            print('conf', conf)

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
            min_wh_head = np.array([6, 8]) / WH_  # min_wh_head 6 8 px
            max_wh_enable_auto = [33, 40] / WH_  # max_wh  enable 30 40 px
            enable_auto = all(wh > max_wh_enable_auto)
            enable = (m_right.value, k_ctrl.value, enable_auto)

            xy_head = xy - [0, wh[1] / 4]
            wh_head = wh * [0.2, 0.35]
            color_head = (0, 255, 255) if all(wh_head < min_wh_head) else (0, 255, 0)
            wh_head = np.maximum(wh_head, min_wh_head)

            print(xy_head, wh_head)
            send_data = ''
            if np.all(xy_head - wh_head <= center) and np.all(center <= xy_head + wh_head):
                last_ser_time = datetime.now()
                send_data += '<c>'
            else:
                if last_ser_time + timedelta(milliseconds=60) < datetime.now() and any(enable):
                    last_ser_time = datetime.now()
                    send_data += f'<{"+" if v_[0] >= 0 else "-"}{abs(v_[0]):03}{"+" if v_[1] >= 0 else "-"}{abs(v_[1]):03}>'
            if debug['send_data']:
                loop(ser, send_data)

            xy_ = (xy * WH_).astype(int)
            xy1_ = ((xy - wh / 2) * WH_).astype(int)
            xy2_ = ((xy + wh / 2) * WH_).astype(int)
            xy1_head_ = ((xy_head - wh_head / 2) * WH_).astype(int)
            xy2_head_ = ((xy_head + wh_head / 2) * WH_).astype(int)
            cv2.rectangle(image, xy1_, xy2_, (0, 0, 255), 1)
            cv2.rectangle(image, xy1_head_, xy2_head_, color_head, 1)
            cv2.line(image, xy_, center_, (200, 200, 0), 1)
            cv2.putText(image, f'{send_data}', (10, 60), 1, 3, (255, 0, 0), 3)
            cv2.putText(image, f'{conf:.1f}', xy_, 1, 1, color_head, 1)
            cv2.putText(image, f'{enable}', (100, 100), 1, 2, color_head, 2)

            os.makedirs('img_out', exist_ok=True)
            if debug['imwrite']:
                cv2.imwrite(datetime.now().strftime('img_out/%y%m%d %H%M%S %f.png'), image)

        if debug['imshow']:
            cv2.imshow('img', cv2.resize(image, (0, 0), fx=.5, fy=.5))
            # cv2.imshow('img', image)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    ser.close()


def input_listener(play, m_right, k_ctrl):
    def on_mouse_click(event):
        if isinstance(event, mouse.ButtonEvent):
            if event.button == 'right':
                m_right.value = True if event.event_type == 'down' else False

    def on_key_press(event):
        if event.name == 'ctrl':
            k_ctrl.value = True if event.event_type == 'down' else False

    keyboard.hook(on_key_press)
    mouse.hook(on_mouse_click)

    keyboard.wait('f4')
    play.value = False


if __name__ == '__main__':
    manager = multiprocessing.Manager()

    queue = multiprocessing.Queue()
    m_right = multiprocessing.Value('b', False)
    k_ctrl = multiprocessing.Value('b', False)
    play = multiprocessing.Value('b', True)

    predict_process = multiprocessing.Process(target=predict, args=(play, m_right, k_ctrl))
    keyboard_process = multiprocessing.Process(target=input_listener, args=(play, m_right, k_ctrl))

    predict_process.start()
    keyboard_process.start()

    predict_process.join()
    keyboard_process.join()
