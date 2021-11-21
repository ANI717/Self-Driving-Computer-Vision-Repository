#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to Test ONNX Model.

Contains a pipeline to test a onnx model.

Revision History:
        2021-11-20 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 test_onnx.py

"""


#___Import Modules:
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ANI717Dataset


#___Main Method:
def main():
    
    # Load Data
    dataset = ANI717Dataset(config.TEST_CSV, config.IMG_SOURCE, transforms=config.TEST_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize ONNX session
    ort_session = ort.InferenceSession(config.ONNX_MODEL_FILE)
    
    # Initialize total correct number and counter
    num_correct = 0.0
    count = 0
    
    # Loop through dataset
    loop = tqdm(loader, position=0, leave=True)
    for batch_idx, (inputs, z, x) in enumerate(loop):
        inputs = inputs.numpy()
        if config.TRAIN_TYPE == 'z':
            targets = z.numpy()[0]
        else:
            targets = x.numpy()[0]
        
        # Calculate prediction
        predictions = ort_session.run(None, {"input.1": inputs.astype(np.float32)},)[0][0][0]
        
        # Update total correct number and counter
        num_correct += abs(round(targets/config.ERROR_TOLERENCE) - round(predictions/config.ERROR_TOLERENCE)) <= 1
        count += 1
        
        # Calculate accuracy
        loop.set_postfix(accuracy=100*num_correct/count)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""