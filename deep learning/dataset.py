#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


# Custom Dataset Class
class ANI717Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root_dir, self.annotations.iloc[index, 0]))
        servo = self.annotations.iloc[index, 1]
        motor = self.annotations.iloc[index, 2]
        
        if self.transforms:
            return self.transforms(image=image)["image"], servo, motor

        return image, servo, motor





# # Test Block
# import config
# from torch.utils.data import DataLoader
# dataset = ANI717Dataset('../dataset/val.csv', '../dataset/images', transforms=config.train_transforms)
# loader = DataLoader(dataset, batch_size=100, shuffle=False)
# images, servo, motor = next(iter(loader))