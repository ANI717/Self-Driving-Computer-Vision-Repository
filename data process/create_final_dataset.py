#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train-Test-Val Dataset Creation Tool.

Creates final dataset from directory tree.

Revision History:
        2021-11-19 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 create_final_dataset.py

"""


#___Import Modules:
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle


#___Global Variables:
SOURCE_CSV = 'output/visually_refined_dataset.csv'
ODIR = 'output'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
TEST_CSV = 'test.csv'

SEED = 717
TRAIN_TEST_RATIO = [0.8, 0.2]
TRAIN_VAL_RATIO = [0.8, 0.2]


#___Functions
def devide_dataset(df):
    df_train_val = df[:int(len(df)*TRAIN_TEST_RATIO[0])]
    df_test = df[int(len(df)*TRAIN_TEST_RATIO[0])+1:]
    df_train = df_train_val[:int(len(df_train_val)*TRAIN_VAL_RATIO[0])]
    df_val = df_train_val[int(len(df_train_val)*TRAIN_VAL_RATIO[0])+1:]
    return df_train, df_val, df_test


def randomly_shuffled_dataset(df):
    # shuffle dataset
    np.random.seed(SEED)
    df = shuffle(df)
    
    # devide dataset
    df_train, df_val, df_test = devide_dataset(df)
    
    # write in csv files
    Path(os.path.join(ODIR, 'random')).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(os.path.join(ODIR, 'random', TRAIN_CSV), index=False)
    df_val.to_csv(os.path.join(ODIR, 'random', VAL_CSV), index=False)
    df_test.to_csv(os.path.join(ODIR, 'random', TEST_CSV), index=False)


#___Main Method:
def main():
    
    # load dataset
    df = pd.read_csv(SOURCE_CSV)
    randomly_shuffled_dataset(df)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""