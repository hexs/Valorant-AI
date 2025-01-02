import time
from datetime import datetime, timedelta
import cv2
import numpy as np
import serial
from hexss.constants.cml import *
from flask import Flask, render_template, Response
import mss

YOLO_MODEL_PATH = r'train_yolov8_with_gpu/runs/detect/train7/weights/best.pt'
KERAS_MODEL_PATH = r'train_keras/model.h5'


def get_img(xyxy_int_tuple):
    with mss.mss() as sct:
        screenshot = sct.grab(xyxy_int_tuple)
    image = np.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def check_time(data, *args):
    # print(PINK, 'check_time', args, ENDC)
    setup = data['setup']
    mode = data['mode']  # 1,2,3
    return all(setup[val][0] + timedelta(milliseconds=setup[val][1][mode]) < datetime.now() for val in args)


def check_time_reset(data, *args):
    # print(CYAN, 'check_time_reset', args, ENDC)
    for val in args:
        data['setup'][val][0] = datetime.now()


def load_models():
    from ultralytics import YOLO
    from keras import models
    yolo_model = YOLO(YOLO_MODEL_PATH)
    keras_model = models.load_model(KERAS_MODEL_PATH)
    return yolo_model, keras_model


class Arduino:
    def __init__(self):
        while True:
            try:
                self.ser = serial.Serial('COM3', 921600)
                break
            except serial.SerialException as e:
                print(f"Error: {e}")
                print("Retrying in 1 second...")
                time.sleep(1)

    def send(self, s):
        print(f'send_to_arduino: {s}')
        self.ser.write(s.encode())
        return s


def p(show=False):
    t = f"Mode={data['mode']}  R={data['m_right']}"
    if show:
        print(t)
    return t


app = Flask(__name__)


@app.route('/')
def index():
    data = app.config['data']
    return render_template('index.html')


@app.route('/video')
def get_video():
    def generate():
        while True:
            frame = app.config['data']['img']
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_server(data):
    app.config['data'] = data
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def predict(data, arduino):
    from train_keras.train_classification import classify_predict
    import math
    import os

    yolo_model, classify_model = load_models()

    center = np.array([0.5, 0.5])
    focus_xywh = np.array([0.5, 0.5, 0.6, 0.6])
    WH_ = np.array([1920, 1080])

    focus_xy = np.array(focus_xywh[:2])
    focus_wh = np.array(focus_xywh[2:])

    xy1_ = (focus_xy - focus_wh / 2) * WH_
    xy2_ = (focus_xy + focus_wh / 2) * WH_
    xyxy_ = np.concatenate((xy1_, xy2_))
    xyxy_int_tuple = tuple(map(int, xyxy_))

    WH_ = get_img(xyxy_int_tuple).shape[1::-1]
    center_ = (center * WH_).astype(int)
    sendlog = ''

    while data['play']:
        image = get_img(xyxy_int_tuple)
        image_show = image.copy()
        results = yolo_model(image, verbose=False)

        boxes = results[0].boxes
        boxes_xywh = boxes.xywhn.cpu().numpy()
        index, percent = '-', '0.00%'

        conf = boxes.conf
        if conf.tolist():
            # print(boxes_xywh)  # [[   0.026281      0.2256    0.052399     0.12273]]
            distances = np.sqrt(((boxes_xywh[:, :2] - center) ** 2).sum(axis=1))
            # print('distances', distances)  # distances [    0.54745]
            nearest_index = np.argmin(distances)
            nearest_box_xywh = boxes_xywh[nearest_index]
            conff = float(boxes.conf[nearest_index])
            # print('conf', conff)

            xy, wh = np.split(nearest_box_xywh, 2)
            # print(f'xy | wh = {xy} | {wh}')  # xy | wh = [   0.026281      0.2256] | [   0.052399     0.12273]
            distance = xy - center
            # print('distance', distance)  # distance (-0.47371928952634335, -0.2744021415710449)
            distance_ = distance * WH_
            # print('distance', distance_)  # distance [    -636.68     -177.81]
            sv_ = math.sqrt(distance_[0] ** 2 + distance_[1] ** 2)
            min_wh_head = np.array([6, 8]) / WH_  # min_wh_head 6 8 px

            xy_head = xy - [0, wh[1] / 4]
            wh_head = wh * [0.25, 0.35]
            color_head = (0, 255, 255) if all(wh_head < min_wh_head) else (0, 255, 0)
            # wh_head = np.maximum(wh_head, min_wh_head)

            xy_ = (xy * WH_).astype(int)
            xy1_ = ((xy - wh / 2) * WH_).astype(int)
            xy2_ = ((xy + wh / 2) * WH_).astype(int)
            xy1_head_ = ((xy_head - wh_head / 2) * WH_).astype(int)
            xy2_head_ = ((xy_head + wh_head / 2) * WH_).astype(int)

            crop_image = image[xy1_[1]: xy2_[1], xy1_[0]: xy2_[0]]
            index, percent = classify_predict(classify_model, crop_image)
            # print(f'classify_predict {index}, {percent}')
            sendlog = ''
            if data['mode'] != 0 and index == 1 and sv_ < data['distance_to_shooting']:
                if data['mode'] == 3 and data['m_right'] == False:
                    ...
                    sendlog = "mode == 3 and m_right == False"
                elif data['mode'] == 3 and any(data['move'].values()):
                    ...
                    sendlog = "mode == 3 and any(data['move'].values())"
                elif np.all(xy_head - wh_head <= center) and np.all(center <= xy_head + wh_head):
                    if check_time(data, 'shooting_to_shooting', 'right_click_to_shooting'):
                        check_time_reset(data, 'shooting_to_shooting', 'shooting_to_move')
                        if data['mode'] == 1:
                            sendlog = arduino.send('<click,200>')
                        else:
                            sendlog = arduino.send('<click,10>')
                else:
                    if check_time(data, 'move_to_move', 'shooting_to_move'):
                        check_time_reset(data, 'move_to_move')
                        if datetime.now() - data['last_move_datetime'] < timedelta(milliseconds=200):  # fast move
                            if data['move_before_shooting_n'] >= 0 and distance[0] > 0:
                                data['move_before_shooting_n'] += 1
                            elif data['move_before_shooting_n'] <= 0 and distance[0] < 0:
                                data['move_before_shooting_n'] -= 1
                            else:
                                data['move_before_shooting_n'] = 0
                        else:
                            data['move_before_shooting_n'] = 0
                        data['last_move_datetime'] = datetime.now()

                        if abs(data['move_before_shooting_n']) == 0:
                            data['mul'] = np.array([1.10, 1])
                        elif abs(data['move_before_shooting_n']) == 1:
                            data['mul'] = np.array([1.20, 1])
                        elif abs(data['move_before_shooting_n']) == 2:
                            data['mul'] = np.array([1.30, 1])
                        elif abs(data['move_before_shooting_n']) == 3:
                            data['mul'] = np.array([1.40, 1])
                        else:
                            data['mul'] = np.array([1.50, 1])

                        ###########################################################
                        a = np.array([0.00133, 0.00133]) * data['mul']
                        b = np.array([1620, 1620]) * data['mul']

                        ### y = b arctan(ax)
                        v_ = b * np.arctan(a * distance_)
                        v_ = v_.astype(int)
                        #########################################################
                        print(f"mul = {data['mul']} {data['move_before_shooting_n']}")
                        sendlog = arduino.send(f'<move,{v_[0]},{v_[1]},10,50>')

        cv2.rectangle(image_show, (0, 0), (600, 100), (255, 255, 255), -1)
        cv2.putText(image_show, f"{p()} move{any(data['move'].values())}", (5, 30),
                    0, 1, (255, 0, 0), 1)
        cv2.putText(image_show, f"mul {data['mul']} {data['move_before_shooting_n']}", (5, 60),
                    0, 1, (255, 0, 0), 1)
        cv2.putText(image_show, f"{sendlog}{index, percent}", (5, 90),
                    0, 1, (255, 0, 0), 1)

        if conf.tolist():
            cv2.rectangle(image_show, xy1_, xy2_, (0, 0, 255), 1)
            cv2.rectangle(image_show, xy1_head_, xy2_head_, (0, 0, 255), 1)
            cv2.line(image_show, xy_, center_, (200, 200, 0), 1)

            cv2.putText(image_show, f'{conff:.1f}', xy_, 1, 1, color_head, 1)
            os.makedirs(f'train_keras/{index}', exist_ok=True)
            cv2.imwrite(f'train_keras/{index}/{datetime.now().strftime("%y%m%d %H%M%S %f.png")}', crop_image)

            os.makedirs('img_output_for_monitor/img_output', exist_ok=True)
            cv2.imwrite(datetime.now().strftime('img_output_for_monitor/img_output/%y%m%d %H%M%S %f.png'), image_show)

        if data['move']['up']:
            arduino.send(f'<move,0,-50,5,50>')
        if data['move']['down']:
            arduino.send(f'<move,0,50,5,50>')
        if data['move']['left']:
            arduino.send(f'<move,-100,0,5,50>')
        if data['move']['right']:
            arduino.send(f'<move,100,0,5,50>')

        data['img'] = image_show.copy()


def input_listener(data, ser):
    from pynput import mouse
    import keyboard

    def on_scroll(x, y, dx, dy):
        if dx == -1:
            data['mode'] = 1
            # data['distance_to_shooting'] = 400
        if dx == 1:
            data['mode'] = 2
            # data['distance_to_shooting'] = 150
        if dy == 1:
            data['mode'] = 3
            # data['distance_to_shooting'] = 150
        if dy == -1:
            data['mode'] = 0

    def on_click(x, y, button, pressed):
        if button == mouse.Button.right:
            if pressed:
                data['m_right'] = True
                check_time_reset(data, 'right_click_to_shooting')
            else:
                data['m_right'] = False

    def on_key_press(event):
        for k in ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right']:
            if event.name == k:
                if event.event_type == 'down':
                    data['move'][k] = True
                else:
                    data['move'][k] = False

        if event.event_type == 'down':
            if event.name == 'right ctrl':
                data['mode'] = 0
                arduino.send('<click,10>')
            if event.name == 'right shift':
                data['mode'] = 1
            if event.name == 'alt':
                data['mode'] = 0
        else:
            if event.name == "l":
                data['n'] = 0
            if event.name == "j":
                arduino.send(f'<move,-1,0,5,50>')
                data['n'] -= 1
                print('j', data['n'])
            if event.name == "k":
                arduino.send(f'<move,+1,0,5,50>')
                data['n'] += 1
                print('k', data['n'])
            if event.name == "u":
                arduino.send(f'<move,-10,0,5,50>')
                data['n'] -= 10
                print('j', data['n'])
            if event.name == "i":
                arduino.send(f'<move,10,0,5,50>')
                data['n'] += 10
                print('k', data['n'])
            if event.name == "7":
                arduino.send(f'<move,-100,0,5,50>')
                data['n'] -= 100
                print('j', data['n'])
            if event.name == "8":
                arduino.send(f'<move,100,0,5,50>')
                data['n'] += 100
                print('k', data['n'])

    keyboard.hook(on_key_press)
    listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    listener.start()

    keyboard.wait('f4')
    data['play'] = False


if __name__ == '__main__':
    from hexss.threading import Multithread

    now = datetime.now()
    m = Multithread()
    data = {
        'n': 0,
        'play': True,
        'img': np.zeros((720, 1280, 3), np.uint8),

        'mode': 0,  # 0, 1AR, 2C, 3s
        'm_right': False,

        'move': {
            'w': False, 'a': False, 's': False, 'd': False,
            'up': False, 'down': False, 'left': False, 'right': False
        },

        'distance_to_shooting': 200,
        'a': np.array([0.00133, 0.00133]),  # 0.00133
        'b': np.array([1620, 1620]),  # 1620
        'mul': np.array([1, 1]),

        'move_before_shooting_n': 0,
        'last_move_datetime': now,

        'setup': {
            'move_to_move': [now, {
                1: 100,
                2: 100,
                3: 100
            }],
            'shooting_to_move': [now, {
                1: 150,
                2: 500,
                3: 700
            }],
            'shooting_to_shooting': [now, {
                1: 300,
                2: 500,
                3: 900
            }],
            'right_click_to_shooting': [now, {
                1: 20,
                2: 20,
                3: 250,
            }]
        }
    }
    arduino = Arduino()

    m.add_func(predict, (data, arduino))
    m.add_func(input_listener, (data, arduino))
    m.add_func(run_server, (data,), join=False)

    m.start()
    m.join()
