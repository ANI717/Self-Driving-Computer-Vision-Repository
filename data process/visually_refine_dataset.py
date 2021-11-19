#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visually Dataset Refining Tool.

Drops unwanted data by visual inspection.

Revision History:
        2021-11-18 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 visually_refine_dataset.py

"""


#___Import Modules:
import os
import cv2
import math
import json
import pandas as pd
from pathlib import Path


#___Global Variables:
START_INDEX = 0
INDEX_JSON = 'settings/index.json'
USE_INDEX_JSON = True

CSV_FILE = 'output/refined_dataset.csv'
REFINED_CSV_FILE = 'output/visually_refined_dataset.csv'

IMAGE_DIR = '../dataset/images'
SHAPE = (480, 720) # height, width
POSITION = (500, 500)

DROP_KEY = ord('d')
SAVE_KEY = ord('s')
QUIT_KEY = ord('q')

MAX_ANGLE = 90


#___Functions
def draw_line(image, value, origin, angle, line_length, color, thiccness):
    cv2.line(image,
             (origin[0],
              origin[1]),
             (origin[0] - int(math.sin(angle*(5-value)/10)*line_length),
              origin[1] - int(math.cos(angle*(5-value)/10)*line_length)),
             color, thiccness)


def put_text(image, z, x, origin, color):
    # puts angular z value as text
    image = cv2.putText(image,
                        'Angular Z: {0:02d}'.format(z),
                        (origin[0] - 250, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
    
    # puts linear x value as text
    image = cv2.putText(image,
                        'Linear X: {0:02d}'.format(x),
                        (origin[0] + 30, origin[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
    
    return image


#___Main Method:
def main():
    
    # initialize drwaing parameters
    origin = [SHAPE[1]//2, int(0.80*SHAPE[0])] # x,y
    line_length = SHAPE[0]//2
    thiccness = 10
    angle = math.pi*MAX_ANGLE/90
    color = (0, 127, 255)
    
    # load dataset
    df = pd.read_csv(CSV_FILE)
    
    # set starting point in the dataset
    start_index = START_INDEX
    if USE_INDEX_JSON:
        with open(INDEX_JSON, 'r') as fp:
            start_index = json.load(fp)["start_index"]
    
    # initialize a list to contain indexes to drop
    # loop through dataset
    drop_index = []
    for index in range(start_index, len(df)):
        
        # read and process image
        image = cv2.imread(os.path.join(IMAGE_DIR, df['images'][index]))
        image = cv2.resize(image, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        draw_line(image, df['z'][index], origin, angle, line_length, color, thiccness)
        image = put_text(image, df['z'][index], df['x'][index], origin, color)
        
        # show image and set waitkey
        cv2.imshow('picture {}'.format(index), image)
        cv2.moveWindow('picture {}'.format(index), *POSITION)
        key = cv2.waitKey(0) & 0xFF
        
        if (key == QUIT_KEY):
            # break if QUIT_KEY is pressed
            break
        
        if (key == DROP_KEY):
            # drop row with current index if DROP_KEY is pressed
            drop_index.append(index)
            cv2.destroyWindow('picture {}'.format(index))
        
        elif (key == SAVE_KEY):
            # save dataframe if SAVE_KEY is pressed
            df_dropped = df.drop(index=drop_index)
            df_dropped.to_csv(REFINED_CSV_FILE, index=False)
            cv2.destroyWindow('picture {}'.format(index))
        
        else:
            # go to next frame
            cv2.destroyWindow('picture {}'.format(index))
        
    
    # destroy all windows and drop rows
    cv2.destroyAllWindows()
    Path(REFINED_CSV_FILE.split('/')[0]).mkdir(parents=True, exist_ok=True)
    df = df.drop(index=drop_index)
    
    # save modified dataframe and ending index
    df.to_csv(REFINED_CSV_FILE, index=False)
    with open(INDEX_JSON, 'w') as f:
        json.dump({'start_index': index}, f, ensure_ascii=False)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""