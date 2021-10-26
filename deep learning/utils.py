#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import config


# Load Checkpoint
def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


# Save Checkpoint
def save_checkpoint(model, optimizer, epoch):
    Path(config.CHECKPOINT).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(config.CHECKPOINT, f'epoch_{epoch+1}.pth.tar'))


# Calculate Accuracy
def calculate_accuracy(loader, model):
    num_correct = 0.0
    count = 0
    
    with torch.no_grad():
        loop = tqdm(loader, position=0, leave=True)
        for batch_idx, (inputs, servo, motor) in enumerate(loop):
            inputs = inputs.to(config.DEVICE)
            if config.TRAIN_TYPE == 'servo':
                targets = servo.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            else:
                targets = motor.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            
            predictions = model(inputs)
            
            num_correct += sum(abs(torch.round(targets) - torch.round(predictions)) <= config.TOLERENCE).item()
            count += predictions.shape[0]
            
            loop.set_postfix(accuracy=100*num_correct/count)
        
    return 100*num_correct/count


# Plot Accuracy
def plot_accuracy(accuracy, loss, epoch):
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    plt.plot(range(1, epoch+2), accuracy, linewidth = 4)
    plt.title("Training (" + config.TRAIN_TYPE + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.savefig(os.path.join(config.OUTPUT, config.TRAIN_TYPE + " accuracy curve.png"))
    plt.close()
    
    plt.figure()
    plt.plot(range(1, epoch+2), loss, linewidth = 4)
    plt.title("Training (" + config.TRAIN_TYPE + ")")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Loss")
    plt.savefig(os.path.join(config.OUTPUT, config.TRAIN_TYPE + " loss curve.png"))
    plt.close()


# Write Log
def write_log(epoch, loss, accuracy, best_epoch):
    best_epoch.append(np.argmax(accuracy) + 1)
    data = np.array([range(1, epoch+2), loss, accuracy, best_epoch]).T
    pd.DataFrame(data, columns=["epoch", 'loss', 'accuracy', 'best_epoch']).to_csv(os.path.join(config.OUTPUT, 'log.csv'), index=False) 
    return best_epoch