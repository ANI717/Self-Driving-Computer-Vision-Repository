#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ANIMESH BALA ANI"""

# Import Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import NvidiaNet
from dataset import ANI717Dataset
from utils import save_checkpoint, load_checkpoint
from utils import calculate_accuracy, plot_accuracy, write_log


# Seed for Reproducibility
torch.manual_seed(0)


# Main Method
def main():
    
    # Load Data
    train_dataset = ANI717Dataset(config.TRAIN_CSV, config.IMG_SOURCE, transforms=config.train_transforms)
    val_dataset = ANI717Dataset(config.VAL_CSV, config.IMG_SOURCE, transforms=config.test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)#, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)#, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    
    # Initialize Model, Optimizer and Loss
    model = NvidiaNet(in_channels=3).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # Load Checkpoint
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, optimizer, config.LEARNING_RATE)
    
    # # Test Block
    # images, servo, motor = next(iter(train_loader))
    # print(images.shape, servo.unsqueeze(1).shape, motor.unsqueeze(1).shape)
    # print(images.dtype, servo.dtype, motor.dtype)
    # print(model(images.to(config.DEVICE)).dtype)
    # print(model(images.to(config.DEVICE)).shape)
    # print(sum(abs(servo.unsqueeze(1) - motor.unsqueeze(1)) <= 0.1).item())
    # import sys
    # sys.exit()
    
    accuracy_holder = []
    loss_holder = []
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, position=0, leave=True)
        for batch_idx, (inputs, servo, motor) in enumerate(loop):
            inputs = inputs.to(config.DEVICE)
            if config.TRAIN_TYPE == 'servo':
                targets = servo.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            else:
                targets = motor.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            
            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                loss = criterion(predictions, targets)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update tqdm Loop
            loop.set_postfix(loss=loss.item(), epoch=epoch+1)
        
        # Plot Loss and Accuracy
        loss_holder.append(loss.item())
        accuracy_holder.append(calculate_accuracy(val_loader, model))
        plot_accuracy(accuracy_holder, loss_holder, epoch)
        
        # Write Log
        if config.WRITE_LOG:
            write_log(epoch, accuracy_holder, loss_holder)
        
        # Save Model
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, epoch)


if __name__ == "__main__":
    main()