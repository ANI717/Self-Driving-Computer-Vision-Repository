#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Refining Tool.

Drops unwanted data for different conditions.

Revision History:
        2021-11-18 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 refine_dataset.py

"""


#___Import Modules:
import pandas as pd
from pathlib import Path


#___Global Variables:
CSV_FILE = 'output/dataset.csv'
REFINED_CSV_FILE = 'output/refined_dataset.csv'

DROP_ZERO_THRESHOLD = 5
DROP_NEGATIVE_THRESHOLD = 6


#___Functions
def drop_equal(value):
    # drops if value = threshold
    return (value == DROP_ZERO_THRESHOLD)

def drop_less(value):
    # drops if value < threshold
    return (value < DROP_NEGATIVE_THRESHOLD)


def drop_function(x):
    # drops when linear x < threshol (when robot car doesn't move forward)
    return drop_less(int(x))


#___Main Method:
def main():
    
    # load dataset
    df = pd.read_csv(CSV_FILE)
    
    # initialize a list to contain indexes to drop
    # loop through dataset
    drop_index = []
    for index in range(len(df)):
        if drop_function(df['x'][index]):
            drop_index.append(index)
    
     # drop rows and save modified dataframe
    df = df.drop(index=drop_index)
    Path(REFINED_CSV_FILE.split('/')[0]).mkdir(parents=True, exist_ok=True)
    df.to_csv(REFINED_CSV_FILE, index=False)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""