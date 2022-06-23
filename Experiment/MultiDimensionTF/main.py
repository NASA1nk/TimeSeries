import os
import sys
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from get_data import get_data, get_batch
from models import TransformerModel
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)
np.random.seed(0)


def train(train_data, batch_size, input_window):
    model.train()
    total_loss = 0.
    avg_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size, input_window, feature)
        optimizer.zero_grad()
        output = model(data)        
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
            avg_loss += len(data[0]) * loss.item()
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
            avg_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            cost_time = time.time() - start_time
            print('-' * 75)
            print('| epoch {:2d} | {:5d} / {:} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} |'.format(
                  epoch, batch, len(train_data) // batch_size, scheduler.get_last_lr()[0], cost_time / log_interval * 1000, cur_loss))
            total_loss = 0
            start_time = time.time()
    if calculate_loss_over_all_values:
        avg_loss = avg_loss / len(train_data)
    else:
        avg_loss = avg_loss / batch
    writer.add_scalar('train_loss', avg_loss, epoch) 


def evaluate(eval_model, val_data, batch_size, input_window, output_window):
    eval_model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, batch_size):
            data, targets = get_batch(val_data, i, batch_size, input_window, feature)
            output = eval_model(data) 
            print(output[-output_window:].size(),targets[-output_window:].size())
            if calculate_loss_over_all_values:
                loss = criterion(output, targets).cpu().item()
                total_loss += len(data[0]) * loss
            else:
                loss = criterion(output[-output_window:], targets[-output_window:]).cpu().item()          
                total_loss += len(data[0]) * loss
    if calculate_loss_over_all_values:
        avg_loss = total_loss / len(val_data)
    else:
        avg_loss = total_loss / i        
    writer.add_scalar('eval_loss', avg_loss, epoch)
    return avg_loss


def plot(eval_model, val_data, batch_size, scaler, input_window, output_window):
    eval_model.eval() 
    total_loss = 0.
    predict = torch.Tensor(0)    
    ground_truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(val_data) - 1):
            data, target = get_batchget_batch(val_data, i, 1, input_window)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)            
            output = eval_model(data)    
            if calculate_loss_over_all_values:                                
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            predict = torch.cat((predict, output[-output_window,:].squeeze(1).cpu()), 0)
            ground_truth = torch.cat((ground_truth, target[-output_window,:].squeeze(1).cpu()), 0)
    predict = scaler.inverse_transform(predict)
    ground_truth = scaler.inverse_transform(ground_truth)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.plot(ground_truth, c='blue', label='ground_truth')
    ax.plot(predict, c='red', label='predict')
    ax.plot(predict-ground_truth, color="green", label="diff")
    ax.legend() 
    plt.savefig(f'./img/Epoch_{epoch}.png')
    return total_loss / i


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_window = 50
    output_window = 1
    batch_size = 32
    data_path = './Experiment/data/MultiTSData.xlsx'
    train_data, val_data, test_data, scaler = get_data(data_path, input_window, output_window)
    train_data, val_data = train_data.to(device), val_data.to(device)
    feature = 10
    layers = 1
    name = f'{input_window}_{output_window}_{feature}_{layers}_{batch_size}'
    model = TransformerModel(feature_size=feature, num_layers=layers).to(device)
    criterion = nn.MSELoss()
    lr = 0.005
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    writer = SummaryWriter(comment=f'{name}', flush_secs=10)
    best_val_loss = float("inf")
    epochs = 100
    best_model = None
    calculate_loss_over_all_values = False
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, batch_size, input_window)
        if(epoch % 10 == 0):
            val_loss = plot(model, val_data, batch_size, scaler, input_window, output_window)
        else:
            val_loss = evaluate(model, val_data, batch_size, input_window, output_window)
            
        print('-' * 75)
        print('|   End of epoch {:2d}   |   avg_loss {:5.5f}   |   time: {:5.2f}s   |'.format(epoch,
                                                                                              loss,
                                                                                              (time.time() - epoch_start_time)))
        if val_loss < best_val_loss:
           best_val_loss = val_loss
           best_model = model
        scheduler.step() 

    torch.save(best_model.state_dict(), f'./best_model/{name}.pth')
    print(f'total time: {time.time()-start_time},  best loss: {best_val_loss}')