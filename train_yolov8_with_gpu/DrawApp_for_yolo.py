import json
import logging
import re
import random
import os
import cv2
import numpy as np
from pygame import Rect
from pygame_gui import UI_BUTTON_PRESSED, UI_FILE_DIALOG_PATH_PICKED
from pygame_gui.elements import UIButton
import DrawApp
from DrawApp import random_text
import pygame as pg
import shutil
from PIL import ImageEnhance, Image
from ultralytics import YOLO
from play import YOLO_model_path


# remove File extension
def remove_extension(file_name, new_file_extension=''):
    name_without_extension = re.sub(r'\.[^.]+$', new_file_extension, file_name)
    return name_without_extension


def get_text_label(frames, focus_xywh):
    XY_crop = np.array(focus_xywh[:2])
    WH_crop = np.array(focus_xywh[2:])
    XY_ori = np.array([0.5, 0.5])
    WH_ori = np.array([1, 1])
    XY_ori_ = np.array([1920 / 2, 1080 / 2])
    WH_ori_ = np.array([1920, 1080])
    WH_crop_ = WH_crop * WH_ori_

    # กรอบเหลือง
    XY1_crop = (XY_crop - WH_crop / 2)
    # XY1_crop_ = (XY1_crop * WH_ori_)
    # XY2_crop = (XY_crop + WH_crop / 2)
    # XY2_crop_ = (XY2_crop * WH_ori_)
    # cv2.rectangle(img, (XY1_crop_).astype(int), (XY2_crop_).astype(int), (0, 255, 255), 2)
    # cv2.circle(img, XY1_crop_.astype(int), 6, (0, 100, 255), -1)
    # imgcop = img[XY1_crop_.astype(int)[1]:XY2_crop_.astype(int)[1],
    #          XY1_crop_.astype(int)[0]:XY2_crop_.astype(int)[0]]

    txt = ''
    txt_crop = ''
    for name, v in frames.items():
        xywh = v['xywh']
        xy_ori = np.array(xywh[:2])
        wh_ori = np.array(xywh[2:])
        txt += f'{0} {xy_ori[0]} {xy_ori[1]} {wh_ori[0]} {wh_ori[1]}\n'

        # กรอบแดง ori
        xy1_ori = xy_ori - wh_ori / 2
        xy1_ori_ = xy1_ori * WH_ori_
        xy2_ori = xy_ori + wh_ori / 2
        xy2_ori_ = xy2_ori * WH_ori_
        # cv2.rectangle(img, xy1_ori_.astype(int), xy2_ori_.astype(int), (0, 0, 255), 2)

        # กรอบน้ำเงิน crop
        xy1_crop = (xy1_ori - XY1_crop) * WH_ori_ / WH_crop_
        xy1_crop_ = (xy1_crop * WH_crop_)
        xy2_crop = (xy2_ori - XY1_crop) * WH_ori_ / WH_crop_
        xy2_crop_ = xy2_crop * WH_crop_

        # cv2.rectangle(imgcop, xy1_crop_.astype(int), xy2_crop_.astype(int), (255, 0, 0), 2)

        xy_crop = (xy2_crop + xy1_crop) / 2
        wh_crop = xy2_crop - xy1_crop
        txt_crop += f'{0} {xy_crop[0]} {xy_crop[1]} {wh_crop[0]} {wh_crop[1]}\n'

    # while True:
    #     cv2.imshow('ori', cv2.resize(img, (0, 0), fx=0.6, fy=0.6))
    #     cv2.imshow('crop', cv2.resize(imgcop, (0, 0), fx=0.6, fy=0.6))
    #     cv2.waitKey(1)

    return txt, txt_crop


def write_data_YOLO(self):
    base_path = 'output_for_YOLO'
    base_crop_path = 'output_for_YOLO_crop'

    # delete old file
    if base_path in os.listdir():
        shutil.rmtree(base_path)
    if base_crop_path in os.listdir():
        shutil.rmtree(base_crop_path)

    # write new data
    paths = {
        'train': {'images': os.path.join(base_path, 'train', 'images'),
                  'labels': os.path.join(base_path, 'train', 'labels')},
        'valid': {'images': os.path.join(base_path, 'valid', 'images'),
                  'labels': os.path.join(base_path, 'valid', 'labels')},
        'train_crop': {'images': os.path.join(base_crop_path, 'train', 'images'),
                       'labels': os.path.join(base_crop_path, 'train', 'labels')},
        'valid_crop': {'images': os.path.join(base_crop_path, 'valid', 'images'),
                       'labels': os.path.join(base_crop_path, 'valid', 'labels')}
    }

    for path in paths.values():
        for subpath in path.values():
            os.makedirs(subpath, exist_ok=True)

    for file_name in [remove_extension(file) for file in os.listdir('videos_data') if '.json' in file]:
        print()
        print(f'--- {file_name} ---')
        cap = cv2.VideoCapture(os.path.join('videos_data', file_name + '.mp4'))
        with open(os.path.join('videos_data', file_name + '.json')) as f:
            frame_dict_time = json.load(f)
        for frame_n, frames in frame_dict_time.items():
            frame_n = int(frame_n)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
            img = cap.read()[1]

            txt, txt_crop = get_text_label(frames, self.focus_xywh)

            for i in range(5):
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                set_type = 'train' if random.randint(0, 5) else 'valid'
                if i:  # if i == 0 save original image
                    enhancers = [
                        ImageEnhance.Brightness(pil_img),
                        ImageEnhance.Contrast(pil_img),
                        ImageEnhance.Sharpness(pil_img),
                        ImageEnhance.Color(pil_img)
                    ]

                    for enhancer in enhancers:
                        factor = random.uniform(0.5, 1.5)
                        pil_img = enhancer.enhance(factor)

                # save data
                base_name = f'{file_name} {frame_n} {i}'
                pil_img.save(os.path.join(paths[set_type]['images'], f'{base_name}.png'))
                with open(os.path.join(paths[set_type]['labels'], f'{base_name}.txt'), 'w') as f:
                    f.write(txt)

                # save data crop
                W, H = pil_img.size
                x_center, y_center, width, height = self.focus_xywh
                left = int((x_center - width / 2) * W)
                upper = int((y_center - height / 2) * H)
                right = int((x_center + width / 2) * W)
                lower = int((y_center + height / 2) * H)
                cropped_img = pil_img.crop((left, upper, right, lower), )
                cropped_img.save(os.path.join(paths[set_type + '_crop']['images'], f'{base_name}.png'))

                with open(os.path.join(paths[set_type + '_crop']['labels'], f'{base_name}.txt'), 'w') as f:
                    f.write(txt_crop)

            print(frame_n, '\n'.join(f"{0} {' '.join(map(str, v['xywh']))}" for name, v in frames.items()), '',
                  sep='\n')
        print(f'--- END ---')


class Manage(DrawApp.DrawApp):
    def __init__(self):
        super().__init__()
        self.video_file_path = None
        # video_file_name = '240626-141535.avi'
        # json_file_name = remove_extension(video_file_name, '.json')

    def show_predict_rects_to_surface(self):
        # yellow frame to surface
        x, y, w, h = self.focus_xywh
        W, H = self.scaled_img_surface.get_size()
        x1, y1 = x - w / 2, y - h / 2
        x1y1wh_ = x1 * W, y1 * H, w * W, h * H
        pg.draw.rect(self.scaled_img_surface, (255, 255, 0), Rect(x1y1wh_), 3)

        # show predict rects to surface
        if self.predict_YOLO_button.text == 'start predict YOLO':
            return False, None
        results = model(self.img_np)

        boxes = results[0].boxes
        conf = boxes.conf
        if conf.tolist():
            print(boxes.xywhn)  # tensor([[0.0263, 0.2256, 0.0524, 0.1227]], device='cuda:0')
        boxes_xywh = boxes.xywhn.cpu().numpy()

        for xywh in boxes_xywh:
            x, y, w, h = xywh
            x1y1wh = xywh - [w / 2, h / 2, 0, 0]
            x1y1wh_ = x1y1wh * np.tile(self.img_size_vector, 2)
            pg.draw.rect(self.scaled_img_surface, (200, 255, 0), Rect(x1y1wh_.tolist()), 1)
        return True if conf.tolist() else False, boxes_xywh

    def get_frame_from_frame_dict_time(self):
        frames = self.frame_dict_time.get(f'{self.current_frame_n}')
        if frames:
            self.frame_dict = frames
        else:
            self.frame_dict = {}
        self.set_item_list()

    def setup_ui(self):
        super().setup_ui()
        self.save_data_for_YOLO_button = UIButton(relative_rect=Rect(10, 5, 150, 25),
                                                  text='Save data for YOLO',
                                                  container=self.panel3,
                                                  anchors={'left_target': self.fast_forward_button})

        self.predict_YOLO_button = UIButton(relative_rect=Rect(0, 5, 150, 25),
                                            text='start predict YOLO',
                                            container=self.panel3,
                                            anchors={'left_target': self.save_data_for_YOLO_button})
        self.auto_add_frame_button = UIButton(relative_rect=Rect(0, 5, 150, 25),
                                              text='auto add frame',
                                              container=self.panel3,
                                              anchors={'left_target': self.predict_YOLO_button})
        self.auto_add_frame_button.disable()

        if model == None:
            self.predict_YOLO_button.disable()

    def run(self):
        have_boxes = False
        boxes_xywh = []
        old_current_frame_n = 0
        while self.is_running:
            self.time_delta = self.clock.tick(60) / 1000.0
            self.dp.fill((180, 180, 180))
            self.mouse_pos = np.array(pg.mouse.get_pos())
            self.can_wheel = self.get_can_wheel()
            events = pg.event.get()
            self.right_click.events(events, self.can_wheel)

            if self.video_file_path:
                if old_current_frame_n != self.current_frame_n:
                    old_current_frame_n = self.current_frame_n
                    self.get_frame_from_frame_dict_time()
                self.get_surface_form_video_file()
            else:
                self.get_surface_form_np()

            for event in events:
                self.manager.process_events(event)
                self.handle_window_resize(event)
                self.wheel_drawing_moving(event)

            self.update_panels(events)
            for event in events:
                if event.type == pg.QUIT:
                    self.is_running = False

                if event.type == UI_FILE_DIALOG_PATH_PICKED:
                    self.video_file_path = event.text
                    self.setup_video_file(self.video_file_path)
                    self.json_file_path = remove_extension(self.video_file_path, '.json')
                    self.frame_dict_time = self.load_frame_json(self.json_file_path)

                if event.type == UI_BUTTON_PRESSED:
                    if event.ui_element in [self.add_button, self.delete_button, self.auto_add_frame_button] \
                            or event.type == 32867 and '#rect_list.panel_window.selection_list.#delete_button.' in event.ui_object_id:
                        # update json_file_name
                        self.frame_dict_time[f'{self.current_frame_n}'] = self.frame_dict
                        self.write_frame_json(self.frame_dict_time, self.json_file_path)

                    if event.ui_element == self.save_data_for_YOLO_button:
                        write_data_YOLO(self)

                    if event.ui_element == self.predict_YOLO_button:
                        self.predict_YOLO_button.set_text(
                            'stop predict YOLO' if self.predict_YOLO_button.text == 'start predict YOLO' else 'start predict YOLO')

                    if event.ui_element == self.auto_add_frame_button:
                        if have_boxes:
                            for xywh in boxes_xywh:
                                name = self.name_entry.get_text() or random_text(skip_list=list(self.frame_dict.keys()))
                                self.name_entry.set_text('')
                                if not self.frame_dict.get(name):
                                    self.frame_dict[name] = {}
                                self.frame_dict[name]['xywh'] = xywh.tolist()
                                self.set_item_list()

                    if event.ui_element == self.fast_forward_button:
                        self.status_button.set_text('||>')
                        self.current_frame_n += 1
                    if event.ui_element == self.rewind_button:
                        self.status_button.set_text('<||')
                        self.current_frame_n -= 1
                    if event.ui_element == self.status_button:
                        self.status_button.set_text('||')

                if event.type == pg.KEYDOWN:
                    selection_name = self.rect_list.get_single_selection()
                    if event.key in [pg.K_RIGHT, pg.K_LEFT, pg.K_DOWN, pg.K_UP] and selection_name:
                        w_, h_ = 1920, 1080
                        index, value = {
                            pg.K_RIGHT: (0, 1 / w_),
                            pg.K_LEFT: (0, -1 / w_),
                            pg.K_DOWN: (1, 1 / h_),
                            pg.K_UP: (1, -1 / h_)
                        }[event.key]
                        if event.mod & pg.KMOD_SHIFT:
                            value *= 10
                        if event.mod & pg.KMOD_CTRL:
                            index += 2
                        self.frame_dict_time[f'{self.current_frame_n}'][selection_name]['xywh'][index] += value

                if event.type == pg.TEXTINPUT:
                    if event.text in 'wasd':
                        if event.text == 'a':
                            self.current_frame_n -= 1
                        if event.text == 'd':
                            self.current_frame_n += 1
                        if event.text == 's':
                            self.current_frame_n -= 10
                        if event.text == 'w':
                            self.current_frame_n += 10

                        if self.current_frame_n < 0:
                            self.current_frame_n = 0
                        if self.current_frame_n > self.max_frame_n:
                            self.current_frame_n = self.max_frame_n
                        self.panel3_slider.set_current_value(self.current_frame_n)

                # if event.type != 1024:
                #     print(event)
            if model:
                have_boxes, boxes_xywh = self.show_predict_rects_to_surface()
            if have_boxes:
                self.auto_add_frame_button.enable()

                self.status_button.set_text('||')
                # print(boxes_xywh)
                for xywh in boxes_xywh:
                    ...

            else:
                self.auto_add_frame_button.disable()
                if self.status_button.text == '||>':
                    self.current_frame_n += 1
                    self.panel3_slider.set_current_value(self.current_frame_n)
                if self.status_button.text == '<||':
                    self.current_frame_n -= 1
                    self.panel3_slider.set_current_value(self.current_frame_n)

            self.show_rects_to_surface(self.frame_dict)
            self.blit_to_display()
            self.manager.update(self.time_delta)
            self.manager.draw_ui(self.dp)

            pg.display.update()


if __name__ == '__main__':
    try:
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        model_path = os.path.relpath(YOLO_model_path, 'train_yolov8_with_gpu')
        model = YOLO(model_path)
    except:
        model = YOLO('yolov8m.pt')

    app = Manage()
    app.run()
