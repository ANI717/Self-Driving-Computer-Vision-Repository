#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to Create Video Clip from Prediction.

Contains a pipeline to create a video clip to visualize prediction.

Revision History:
        2021-11-20 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 prediction_visualization.py

"""


#___Import Modules:
import os
import cv2
import math
import torch
from pathlib import Path

import config
from model import NvidiaNet


#___Global Variables
SOURCE_FOLDER = '../dataset/images/03_16_2020_0/output_0000'
OUTPUT = 'output/output_0000.mp4'
SHAPE = (360, 640) # height, width
QUIT_KEY = ord('q')
MAX_ANGLE = 90
NEW_DATA = False


#___Functions
def draw_line(image, value, origin, angle, line_length, color, thiccness):
    cv2.line(image,
             (origin[0],
              origin[1]),
             (origin[0] - int(math.sin(angle*(5-value)/10)*line_length),
              origin[1] - int(math.cos(angle*(5-value)/10)*line_length)),
             color, thiccness)


def put_text(image, origin, target_color, predict_color, targets, predictions):
    
    image = cv2.putText(image,
                        '{0:02d}'.format(targets),
                        (origin[0] - 60, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, target_color, 1, cv2.LINE_AA)    
    image = cv2.putText(image,
                        "(Target)",
                        (origin[0] - 200, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, target_color, 1, cv2.LINE_AA)
    image = cv2.putText(image,
                        '{0:02d}'.format(predictions),
                        (origin[0] + 15, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, predict_color, 1, cv2.LINE_AA)
    image = cv2.putText(image,
                        "(Predict)",
                        (origin[0] + 70, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, predict_color, 1, cv2.LINE_AA)
    return image


def single_prediction(model, image):
    # Prediction on Single Image
    return round(10*model(config.TEST_TRANSFORMS(image=image)["image"].unsqueeze(0).to(config.DEVICE)).item())


#___Main Method:
def main():
    
    # Initialize variables for drawing
    origin = [SHAPE[1]//2, int(0.80*SHAPE[0])] # x,y
    line_length = SHAPE[0]//2
    thiccness = 10
    angle = math.pi*MAX_ANGLE/90
    target_color = (0,255,127)
    predict_color = (0,127,255)
    
    # Create video writer object
    Path(OUTPUT.split('/')[0]).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(OUTPUT, -1, 13, (SHAPE[1], SHAPE[0]))
    
    # Set source folder for visualization
    data = os.listdir(SOURCE_FOLDER)
    
    # Load Model
    model = NvidiaNet(in_channels=3).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE)["state_dict"])
    model.eval()
    
    # loop through dataset
    for index in range(len(data)):
        
        # Load image
        image = cv2.imread(os.path.join(SOURCE_FOLDER, data[index]))
        
        # Load targets
        if NEW_DATA:
            targets = int(data[index].split('_')[-3])
        else:
            targets = int(data[index].split('_')[-2][1:])
        
        # Make prediction
        predictions = single_prediction(model, image)
        
        # Reshape image for visualization
        image = cv2.resize(image, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_CUBIC)
    
        # Draw line and text for display
        draw_line(image, targets, origin, angle, line_length, target_color, thiccness)
        draw_line(image, predictions, origin, angle, line_length, predict_color, thiccness)
        image = put_text(image, origin, target_color, predict_color, targets, predictions)
        
        # write image in video
        out.write(image)
    
    # Release video writer object
    out.release()


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""