#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import cv2
import math

import config
from model import NvidiaNet
from utils import load_checkpoint, single_prediction


# Global Variables
SOURCE_FOLDER = '../dataset/images/03_16_2020_0/output_0000'
OUTPUT = 'output/output_0000.mp4'
SHAPE = (360, 640) # height, width
QUIT_KEY = ord('q')
MAX_ANGLE = 90
NEW_DATA = False


# Functions
def draw_line(image, value, origin, angle, line_length, color, thiccness):
    cv2.line(image,
             (origin[0],
              origin[1]),
             (origin[0] - int(math.sin(angle*(5-value)/10)*line_length),
              origin[1] - int(math.cos(angle*(5-value)/10)*line_length)),
             color, thiccness)


def put_text(image, origin, target_color, predict_color, servo, servo_pred):
    
    image = cv2.putText(image,
                        '{0:02d}'.format(servo),
                        (origin[0] - 60, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, target_color, 3, cv2.LINE_AA)    
    image = cv2.putText(image,
                        "(Target)",
                        (origin[0] - 200, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, target_color, 3, cv2.LINE_AA)
    image = cv2.putText(image,
                        '{0:02d}'.format(servo_pred),
                        (origin[0] + 15, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, predict_color, 3, cv2.LINE_AA)
    image = cv2.putText(image,
                        "(Predict)",
                        (origin[0] + 70, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, predict_color, 3, cv2.LINE_AA)
    return image


# Main Method
def main():
    origin = [SHAPE[1]//2, int(0.80*SHAPE[0])] # x,y
    line_length = SHAPE[0]//2
    thiccness = 10
    angle = math.pi*MAX_ANGLE/90
    target_color = (0,255,127)
    predict_color = (0,127,255)
    out = cv2.VideoWriter(OUTPUT, -1, 13, (SHAPE[1], SHAPE[0]))
    
    data = os.listdir(SOURCE_FOLDER)
    model = NvidiaNet(in_channels=3).to(config.DEVICE)
    load_checkpoint(config.CHECKPOINT, model)
    model.eval()
    
    for i in range(len(data)):
        image = cv2.imread(os.path.join(SOURCE_FOLDER, data[i]))
        if NEW_DATA:
            servo = int(data[i].split('_')[-3])
        else:
            servo = int(data[i].split('_')[-2][1:])

        servo_pred = single_prediction(model, image)
        
        image = cv2.resize(image, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_CUBIC)
    
        # line and text display
        draw_line(image, servo, origin, angle, line_length, target_color, thiccness)
        draw_line(image, servo_pred, origin, angle, line_length, predict_color, thiccness)
        image = put_text(image, origin, target_color, predict_color, servo, servo_pred)
        
        # write image in video
        out.write(image)
    
    out.release()


if __name__ == "__main__":
    main()