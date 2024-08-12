import multiprocessing
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
import serial
from train_keras.train_classification import classify_predict

BLACK = '\033[90m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PINK = '\033[95m'
CYAN = '\033[96m'
ENDC = '\033[0m'
BOLD = '\033[1m'
ITALICIZED = '\033[3m'
UNDERLINE = '\033[4m'
YOLO_model_path = r'train_yolov8_with_gpu/runs/detect/train7/weights/best.pt'
keras_model_path = r'train_keras/model.h5'


def setup():
    '''
        <+xxx-yyy>
        Mouse.move(x, y);

        <c>
        Mouse.click();
    '''
    port = 'COM3'
    baud_rate = 921600
    while True:
        try:
            ser = serial.Serial(port, baud_rate)
            return ser
        except:
            print(f'ERROR ser = serial.Serial({port}, {baud_rate})')
            time.sleep(1)


def loop(ser, s):
    if s:
        print(f'--- {s} ---')
        ser.write(s.encode())


def check_time(data, val):
    print(PINK, 'check_time', val, ENDC)
    setSW = data['setSW']
    profile = setSW['profile']  # 1,2
    return setSW[val][0] + setSW[val][1][profile] < datetime.now()


def check_time_reset(data, *args):
    setSW = data['setSW']
    print(CYAN, 'check_time_reset', args, ENDC)
    for val in args:
        setSW[val][0] = datetime.now()
    data['setSW'] = setSW


def predict(data, play):
    from ultralytics import YOLO
    from keras import models
    import mss
    import math
    import os

    debug = {
        'imwrite': True,
        'imshow': False,
        'send_data': True
    }

    ser = setup()

    model = YOLO(YOLO_model_path)
    classify_model = models.load_model(keras_model_path)
    center = np.array([0.5, 0.5])
    focus_xywh = np.array([0.5, 0.5, 0.6, 0.6])
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

    a = data['setSW']['a']
    b = data['setSW']['b']
    distance_to_shooting = data['setSW']['distance_to_shooting']

    t2 = datetime.now()
    while play.value:
        t1 = t2
        t2 = datetime.now()
        print()
        print()
        print((t2 - t1).total_seconds())
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
            v_ = 3 * b * np.arctan(a * distance_)
            v_ = v_.astype(int)
            sv_ = math.sqrt(v_[0] ** 2 + v_[1] ** 2)
            #########################################################
            min_wh_head = np.array([6, 8]) / WH_  # min_wh_head 6 8 px

            xy_head = xy - [0, wh[1] / 4]
            wh_head = wh * [0.25, 0.35]
            color_head = (0, 255, 255) if all(wh_head < min_wh_head) else (0, 255, 0)
            wh_head = np.maximum(wh_head, min_wh_head)

            xy_ = (xy * WH_).astype(int)
            xy1_ = ((xy - wh / 2) * WH_).astype(int)
            xy2_ = ((xy + wh / 2) * WH_).astype(int)
            xy1_head_ = ((xy_head - wh_head / 2) * WH_).astype(int)
            xy2_head_ = ((xy_head + wh_head / 2) * WH_).astype(int)

            crop_image = image[xy1_[1]: xy2_[1], xy1_[0]: xy2_[0]]
            index, percent = classify_predict(classify_model, crop_image)
            print(f'classify_predict {index}, {percent}')

            enable = data['auto'] == 'on' or (
                    data['auto'] == 'auto' and any((data['m_right'], data['k_ctrl']))) and sv_ < distance_to_shooting
            send_data = ''

            if np.all(xy_head - wh_head <= center) and np.all(center <= xy_head + wh_head):
                if enable and index == 1:
                    if data['auto'] != 'off':
                        if check_time(data, 'shooting_last_time') \
                                and check_time(data, 'right_click_before_shooting_time'):
                            check_time_reset(data, 'shooting_last_time', 'move_last_time')
                            send_data += '<c>'
            else:
                if enable and index == 1:
                    if check_time(data, 'move_last_time'):
                        print('++')
                        check_time_reset(data, 'move_last_time')
                        send_data += f'<{"+" if v_[0] >= 0 else "-"}{abs(v_[0]):03}{"+" if v_[1] >= 0 else "-"}{abs(v_[1]):03}>'

            if debug['send_data']:
                loop(ser, send_data)

            if data['setHW']:
                commands = data['setHW'].copy()
                data['setHW'] = []
                for setHW in commands:
                    loop(ser, f"{setHW}")

            # save image fro debug image classify
            os.makedirs(f'train_keras/{index}', exist_ok=True)
            cv2.imwrite(f'train_keras/{index}/{datetime.now().strftime("%y%m%d %H%M%S %f.png")}', crop_image)

            cv2.rectangle(image, xy1_, xy2_, (0, 0, 255), 1)
            cv2.rectangle(image, xy1_head_, xy2_head_, color_head, 1)
            cv2.line(image, xy_, center_, (200, 200, 0), 1)
            cv2.putText(image, f'{conf:.1f}', xy_, 1, 1, color_head, 1)
            enable_ = f"{data['auto']}, m{data['m_right']}, k{data['k_ctrl']}"
            cv2.putText(image, f"{data['setSW']['profile']} {enable_} -> {enable}",
                        (5, 30), 1, 2, (255, 0, 0), 2)
            cv2.putText(image, f'{send_data}', (5, 60), 1, 2, (255, 0, 0), 2)

            if debug['imwrite']:
                os.makedirs('img_output_for_monitor/img_output', exist_ok=True)
                cv2.imwrite(datetime.now().strftime('img_output_for_monitor/img_output/%y%m%d %H%M%S %f.png'), image)

        if debug['imshow']:
            cv2.imshow('img', cv2.resize(image, (0, 0), fx=.5, fy=.5))
            # cv2.imshow('img', image)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    ser.close()


def input_listener(data, play):
    from pynput import mouse
    import keyboard

    def on_mouse_click(x, y, button, pressed):
        if button == mouse.Button.right:
            if pressed:
                data['m_right'] = 1
                check_time_reset(data, 'right_click_before_shooting_time')
            else:
                data['m_right'] = 0

        if button == mouse.Button.middle:
            if pressed:
                setSW = data['setSW']
                setSW['profile'] = '2'
                data['setSW'] = setSW

    def on_key_press(event):
        # print(event)
        if event.name == 'ctrl':
            data['k_ctrl'] = 1 if event.event_type == 'down' else 0

        if event.name == 'alt' and event.event_type == 'down':
            data['auto'] = 'off'
            data['setSW']['distance_to_shooting'] = 250
        if event.name == 'right alt' and event.event_type == 'down':
            data['auto'] = 'on'
            data['setSW']['distance_to_shooting'] = 600
        if event.name in ('w', 'a', 's', 'd') and event.event_type == 'down':
            if data['auto'] == 'off':
                data['auto'] = 'auto'
        if event.name == '1' and event.event_type == 'down':
            setSW = data['setSW']
            setSW['profile'] = '1'
            data['setSW'] = setSW
        if event.name in ('2', 'x') and event.event_type == 'down':
            setSW = data['setSW']
            setSW['profile'] = '2'
            data['setSW'] = setSW

    keyboard.hook(on_key_press)
    listener = mouse.Listener(on_click=on_mouse_click)
    listener.start()

    keyboard.wait('f4')
    play.value = False


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    data = manager.dict()
    data['m_right'] = 0
    data['k_ctrl'] = 0
    data['auto'] = 'auto'  # on<right alt> ,off<alt> ,if off auto<w a s d>,
    data['setHW'] = ['<vel120>', '<delay000>']
    data['setSW'] = {
        'a': np.array([0.0030, 0.001306]),  # 0.001306
        'b': np.array([250, 455]),  # 455
        'profile': '1',
        'move_last_time': [datetime.now(), {
            '1': timedelta(milliseconds=30),
            '2': timedelta(milliseconds=50)
        }],
        'shooting_last_time': [datetime.now(), {
            '1': timedelta(milliseconds=150),
            '2': timedelta(milliseconds=300)
        }],
        'right_click_before_shooting_time': [datetime.now(), {
            '1': timedelta(milliseconds=250),
            '2': timedelta(milliseconds=300)
        }],
        'distance_to_shooting': 250,
    }

    queue = multiprocessing.Queue()
    play = multiprocessing.Value('b', True)

    predict_process = multiprocessing.Process(target=predict, args=(data, play))
    keyboard_process = multiprocessing.Process(target=input_listener, args=(data, play))

    predict_process.start()
    keyboard_process.start()

    predict_process.join()
    keyboard_process.join()
