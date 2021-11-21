#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config File.

Contains Hyperparameters and Transformations.

Revision History:
        2021-11-20 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ import config

"""


#___Import Modules:
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


#___Hyperparameters:
TRAIN_TYPE = 'z'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 1
PIN_MEMORY = True

IMG_SHAPE = (3, 75, 75) # channels, rows, columns

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 150

IMG_SOURCE = '../../dataset/images'
TRAIN_CSV = '../../dataset/lists/random/train.csv'
TEST_CSV = '../../dataset/lists/random/test.csv'
VAL_CSV = '../../dataset/lists/random/val.csv'

LOAD_MODEL = True
SAVE_MODEL = True
WRITE_LOG = True
SAVE_ONNX = True

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = 'output'
MODEL_FILE = "checkpoints/epoch_36.pth.tar"
ONNX_MODEL_FILE = "checkpoints/epoch_36.onnx"

ERROR_TOLERENCE = 0.1


#___Transformations
TRAIN_TRANSFORMS = A.Compose([
    A.Resize(height=IMG_SHAPE[1], width=IMG_SHAPE[2]),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),])

TEST_TRANSFORMS = A.Compose([
    A.Resize(height=IMG_SHAPE[1], width=IMG_SHAPE[2]),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),])


#                                                                              
# end of file
"""ANI717"""