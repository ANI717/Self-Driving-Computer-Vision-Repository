#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility File.

Contains Utility Functions.

Revision History:
        2021-11-20 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ import utils

"""


#___Import Modules:
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import config


#___Functions
def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    # Loads Checkpoint to PyTorch Model
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Load optimizer
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def save_checkpoint(model, optimizer, epoch):
    # Saves Serialized Model as PyTorch Checkpoint and ONNX Model
    
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save as pth.tar format
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'epoch_{epoch+1}.pth.tar'))
    
    # Save as ONNX format
    if config.SAVE_ONNX:
        torch.onnx.export(model.to('cpu'),
                          torch.randn(1, *config.IMG_SHAPE), 
                          os.path.join(config.CHECKPOINT_DIR, f'epoch_{epoch+1}.onnx'), 
                          verbose=False)
        model.to(config.DEVICE)


def calculate_accuracy(loader, model):
    # Calculates Accuracy
    
    # Initialize total correct number and counter
    num_correct = 0.0
    count = 0
    
    # Loop through dataset
    with torch.no_grad():
        
        loop = tqdm(loader, position=0, leave=True)
        for batch_idx, (inputs, z, x) in enumerate(loop):
            
            # Enable GPU support is available
            inputs = inputs.to(config.DEVICE)
            if config.TRAIN_TYPE == 'z':
                targets = z.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            else:
                targets = x.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            
            # Calculate prediction
            predictions = model(inputs)
            
            # Update total correct number and counter
            num_correct += sum(abs(torch.round(targets/config.ERROR_TOLERENCE) - torch.round(predictions/config.ERROR_TOLERENCE)) <= 1).item()
            count += predictions.shape[0]
            
            loop.set_postfix(accuracy=100*num_correct/count)
    
    # Return accuracy
    return 100*num_correct/count


def plot_result(accuracy, loss, epoch):
    # Plots Accuracy and Loss fror each Epoch
    
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Plot accuracy
    plt.figure()
    plt.plot(range(1, epoch+2), accuracy, linewidth = 4)
    plt.title("Training (" + config.TRAIN_TYPE + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.savefig(os.path.join(config.OUTPUT_DIR, config.TRAIN_TYPE + " accuracy curve.png"))
    plt.close()
    
    # Plot loss
    plt.figure()
    plt.plot(range(1, epoch+2), loss, linewidth = 4)
    plt.title("Training (" + config.TRAIN_TYPE + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Loss")
    plt.savefig(os.path.join(config.OUTPUT_DIR, config.TRAIN_TYPE + " loss curve.png"))
    plt.close()


def write_log(epoch, loss, accuracy, best_epoch):
    # Writea Log File
    
    best_epoch.append(np.argmax(accuracy) + 1)
    data = np.array([range(1, epoch+2), loss, accuracy, best_epoch]).T
    pd.DataFrame(data, columns=["epoch", 'loss', 'accuracy', 'best_epoch']).to_csv(os.path.join(config.OUTPUT_DIR, 'log.csv'), index=False) 
    return best_epoch


#                                                                              
# end of file
"""ANI717"""