import time
from datetime import datetime, timedelta
import cv2
import numpy as np
import multiprocessing
import serial


def classify_predict(model, img_bgr):
    img_bgr = cv2.resize(img_bgr, (40, 40))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    images = np.expand_dims(img_rgb, axis=0)
    predictions = model.predict_on_batch(images)
    exp_x = [2.7 ** x for x in predictions[0]]
    percent_score_list = [round(x * 100 / sum(exp_x)) for x in exp_x]
    highest_score_index = np.argmax(predictions[0])  # 3
    highest_score_percent = percent_score_list[highest_score_index]
    return highest_score_index, highest_score_percent


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


def predict(data, play, m_right, k_ctrl):
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
    move_last_datetime = datetime.now()
    shoot_last_datetime = datetime.now()

    model = YOLO(r'train_yolov8_with_gpu/runs/detect/train4/weights/best.pt')
    # model = YOLO(r'train_yolov8_with_gpu/runs/detect/train4/weights/best.onnx')
    classify_model = models.load_model(r'train_keras/model.h5')
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

    a = data['setSW']['a']
    b = data['setSW']['b']
    move_last_timedelta = data['setSW']['move_last_timedelta']
    shooting_last_timedelta = data['setSW']['shooting_last_timedelta']
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
            wh_head = wh * [0.2, 0.35]
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
                    data['auto'] == 'auto' and any((m_right.value, k_ctrl.value))) and sv_ < distance_to_shooting
            send_data = ''

            if np.all(xy_head - wh_head <= center) and np.all(center <= xy_head + wh_head):
                if enable and index == 1:
                    if data['auto'] != 'off':
                        if shoot_last_datetime + timedelta(milliseconds=shooting_last_timedelta) < datetime.now():
                            shoot_last_datetime = datetime.now()
                            move_last_datetime = datetime.now()
                            send_data += '<c>'
            else:
                if enable and index == 1:
                    if move_last_datetime + timedelta(milliseconds=move_last_timedelta) < datetime.now():
                        move_last_datetime = datetime.now()
                        send_data += f'<{"+" if v_[0] >= 0 else "-"}{abs(v_[0]):03}{"+" if v_[1] >= 0 else "-"}{abs(v_[1]):03}>'

            if debug['send_data']:
                loop(ser, send_data)

            if data['setHW']:
                commands = data['setHW'].copy()
                data['setHW'] = []
                for setHW in commands:
                    loop(ser, f"{setHW}")

            if data['setSW']:
                commands = data['setSW'].copy()
                data['setSW'] = {}
                for k, v in commands.items():
                    if k == 'a':
                        a = v
                    if k == 'b':
                        b = v
                    if k == 'move_last_timedelta':
                        move_last_timedelta = v
                    if k == 'shooting_last_timedelta':
                        shooting_last_timedelta = v
                    if k == 'distance_to_shooting':
                        distance_to_shooting = v

            # save image fro debug image classify
            os.makedirs(f'train_keras/{index}', exist_ok=True)
            cv2.imwrite(f'train_keras/{index}/{datetime.now().strftime("%y%m%d %H%M%S %f.png")}', crop_image)

            cv2.rectangle(image, xy1_, xy2_, (0, 0, 255), 1)
            cv2.rectangle(image, xy1_head_, xy2_head_, color_head, 1)
            cv2.line(image, xy_, center_, (200, 200, 0), 1)
            cv2.putText(image, f'{send_data}', (10, 60), 1, 3, (255, 0, 0), 3)
            cv2.putText(image, f'{conf:.1f}', xy_, 1, 1, color_head, 1)
            enable_ = data['auto'], m_right.value, k_ctrl.value
            cv2.putText(image, f'{enable_, sv_ < 50, enable}', (10, 50), 1, 2, (255, 0, 0), 2)

            if debug['imwrite']:
                os.makedirs('img_output_for_monitor/img_output', exist_ok=True)
                cv2.imwrite(datetime.now().strftime('img_output_for_monitor/img_output/%y%m%d %H%M%S %f.png'), image)

        if debug['imshow']:
            cv2.imshow('img', cv2.resize(image, (0, 0), fx=.5, fy=.5))
            # cv2.imshow('img', image)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    ser.close()


def input_listener(data, play, m_right, k_ctrl):
    from pynput import mouse
    import keyboard

    def on_mouse_click(x, y, button, pressed):
        if button == mouse.Button.right:
            m_right.value = True if pressed else False

    def on_key_press(event):
        # print(event)
        if event.name == 'ctrl':
            k_ctrl.value = True if event.event_type == 'down' else False

        if event.name == 'alt' and event.event_type == 'down':
            data['auto'] = 'off'
            data['setSW'] = {'distance_to_shooting': 150}
        if event.name == 'right alt' and event.event_type == 'down':
            data['auto'] = 'on'
            data['setSW'] = data['setSW'].update({'distance_to_shooting': 600})
        if event.name in 'wasd' and event.event_type == 'down':
            if data['auto'] == 'off':
                data['auto'] = 'auto'

    keyboard.hook(on_key_press)
    listener = mouse.Listener(on_click=on_mouse_click)
    listener.start()

    keyboard.wait('f4')
    play.value = False


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    data = manager.dict()
    data['m_right'] = False
    data['k_ctrl'] = False
    data['auto'] = 'auto'  # on<right alt> ,off<alt> ,if off auto<w a s d>,
    data['setHW'] = ['<vel120>', '<delay000>']
    data['setSW'] = {
        'a': 0.001306,
        'b': 550,  # 455
        'move_last_timedelta': 30,
        'shooting_last_timedelta': 300,
        'distance_to_shooting': 600,
    }

    queue = multiprocessing.Queue()
    m_right = multiprocessing.Value('b', False)
    k_ctrl = multiprocessing.Value('b', False)
    play = multiprocessing.Value('b', True)

    predict_process = multiprocessing.Process(target=predict, args=(data, play, m_right, k_ctrl))
    keyboard_process = multiprocessing.Process(target=input_listener, args=(data, play, m_right, k_ctrl))

    predict_process.start()
    keyboard_process.start()

    predict_process.join()
    keyboard_process.join()
