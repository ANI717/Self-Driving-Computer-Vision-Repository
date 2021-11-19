#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Creation Tool.

Creates dataset from directory tree.

Revision History:
        2021-11-18 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 create_dataset.py

"""


#___Import Modules:
import os
import numpy as np
import pandas as pd
from pathlib import Path


#___Global Variables:
SOURCE = '../dataset/images'
OUTPUT = 'output/dataset.csv'


#___Functions
def create_list_level1(source='.'):
    # create dataset of images in a directory
    
    flist = []
    zlist = []
    xlist = []
    for file in os.listdir(source):
        if file.split('.')[-1] == 'jpg':
            
            # parse file-name, z value and x value
            # append then in propper aray
            flist.append(os.path.join(source, file))
            zlist.append(int(file.split('_')[1][1:]))
            xlist.append(int(file.split('_')[2][1:3]))
            
    return flist, zlist, xlist


def create_list_level2(source='.'):
    # create dataset of images in directories in a directory
    
    flist = []
    zlist = []
    xlist = []
    for dirs in os.listdir(source):
        if os.path.isdir(os.path.join(source, dirs)):
            flist_sub, zlist_sub, xlist_sub = create_list_level1(os.path.join(source, dirs))
            flist.extend(flist_sub)
            zlist.extend(zlist_sub)
            xlist.extend(xlist_sub)
    
    return flist, zlist, xlist


def create_list_level3(source='.'):
    # create dataset of images in directory depth is 3
    
    flist = []
    zlist = []
    xlist = []
    for dirs in os.listdir(source):
        if os.path.isdir(os.path.join(source, dirs)):
            flist_sub, zlist_sub, xlist_sub = create_list_level2(os.path.join(source, dirs))
            flist.extend(flist_sub)
            zlist.extend(zlist_sub)
            xlist.extend(xlist_sub)
    
    return flist, zlist, xlist


#___Main Method:
def main():
    
    # create list of images in directory with proper file-name format
    flist, zlist, xlist = create_list_level3(SOURCE)
    for index, file in enumerate(flist):        
        flist[index] = ('\\').join(file.split('/')[-1].split('\\')[1:])
    
    # reshape in proper format
    content = np.array([flist, zlist, xlist]).T
    
    # save dataset
    Path(OUTPUT.split('/')[0]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(content, columns =['images', 'z', 'x']).to_csv(OUTPUT, index=False)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""