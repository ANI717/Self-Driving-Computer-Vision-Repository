#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config for Deep Learning.

Configuration file for Deep Lerning Procedure.

Revision History:
        2021-11-20 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ from dataset import ANI717Dataset

"""


#___Import Modules:
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


#___Classes
class ANI717Dataset(Dataset):
    def __init__(self, csv_file, img_source_dir, transforms=None):
        self.img_source_dir = img_source_dir
        self.annotations = pd.read_csv(csv_file)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.img_source_dir,
                                        self.annotations.iloc[index, 0]))
        z = self.annotations.iloc[index, 1]/10
        x = self.annotations.iloc[index, 2]/10
        
        if self.transforms:
            return self.transforms(image=image)["image"], z, x

        return image, z, x





#___Test Block
# import config
# from torch.utils.data import DataLoader
# dataset = ANI717Dataset(config.VAL_CSV, config.IMG_SOURCE,
#                         transforms=config.train_transforms)
# loader = DataLoader(dataset, batch_size=100, shuffle=False)
# images, z, x = next(iter(loader))
# print(images.shape)
# print(z.shape)
# print(x.shape)


#                                                                              
# end of file
"""ANI717"""