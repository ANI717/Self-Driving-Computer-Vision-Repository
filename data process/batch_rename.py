#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Naming Format Change Tool.

Changes file-name of all images in a directory tree.

Revision History:
        2021-11-18 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 create_dataset.py

"""


#___Import Modules:
import os


#___Global Variables:
SOURCE = '../dataset/images'


#___Functions
def batch_rename_level1(source='.'):
    # rename images in a directory

    for file in os.listdir(source):
        if file.split('.')[-1] == 'jpg':
            
            # index value, z value and x value
            num = int(file.split('_')[0][1:])
            z = int(file.split('_')[1][1:])
            x = int(file.split('_')[2][1:3])
            
            # rename with proper format
            old_name = os.path.join(source, file)
            new_name = os.path.join(source, '{0:07d}_z{1:02d}_x{2:02d}.jpg'.format(num, z, x))
            os.rename(old_name, new_name)


def batch_rename_level2(source='.'):
    # rename images in directories in a directory
    
    for dirs in os.listdir(source):
        if os.path.isdir(os.path.join(source, dirs)):     
            batch_rename_level1(os.path.join(source, dirs))


def batch_rename_level3(source='.'):
    # rename images in directory depth is 3
    
    for dirs in os.listdir(source):
        if os.path.isdir(os.path.join(source, dirs)):     
            batch_rename_level2(os.path.join(source, dirs))


#___Main Method:
def main():
    batch_rename_level3(SOURCE)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""