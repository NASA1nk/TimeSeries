import os
import sys
import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
import torch
import torch.nn as nn
from get_data import get_data, get_batch
from models import TransformerModel


torch.manual_seed(0)
np.random.seed(0)

# The flag decides if the loss will be calculted over all 
# or just the predicted values.
calculate_loss_over_all_values = False


def train(train_data):
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)        

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

 
def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data) 
            print(output[-output_window:].size(),targets[-output_window:].size())
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)

def plot(eval_model, data_source,epoch,scaler):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)            
            # look like the model returns static values for the output window
            output = eval_model(data)    
            if calculate_loss_over_all_values:                                
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()
            
            test_result = torch.cat((test_result, output[-1,:].squeeze(1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1,:].squeeze(1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy()
    len(test_result)
    
    test_result_=scaler.inverse_transform(test_result[:700])
    truth_=scaler.inverse_transform(truth)
    print(test_result.shape,truth.shape)
    for m in range(9):
        test_result = test_result_[:,m]
        truth = truth_[:,m]
        fig = pyplot.figure(1, figsize=(20, 5))
        fig.patch.set_facecolor('xkcd:white')
        pyplot.plot([k + 510                 for k in range(190)],test_result[510:],color="red")
        pyplot.title('Prediction uncertainty')
        pyplot.plot(truth[:700],color="black")
        pyplot.legend(["prediction", "true"], loc="upper left")
        ymin, ymax = pyplot.ylim()
        pyplot.vlines(510, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        pyplot.ylim(ymin, ymax)
        pyplot.xlabel("Periods")
        pyplot.ylabel("Y")
        pyplot.show()
        pyplot.close()
    return total_loss / i


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    input_window = 100
    output_window = 5
    batch_size = 32
    data_path = './Experiment/data/MultiTSData.xlsx'
    train_data, val_data, test_data, scaler = get_data(data_path, input_window, output_window)
    train_data, val_data = train_data.to(device), val_data.to(device)
    feature = 512
    layers = 1
    model = TransformerModel(feature_size=feature, num_layers=layers).to(device)
    criterion = nn.MSELoss()
    lr = 0.005 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    best_val_loss = float("inf")
    epochs = 20
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data)
        if(epoch % 10 == 0):
            val_loss = plot(model, val_data, epoch, scaler)
        else:
            val_loss = evaluate(model, val_data)
            
        print('-' * 85)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f}'.format(epoch, time.time()-epoch_start_time, val_loss))
        print('-' * 85)
        if val_loss < best_val_loss:
           best_val_loss = val_loss
           best_model = model
        scheduler.step() 
    torch.save(best_model.state_dict(), f'./best_model/{input_window}_{output_window}_{feature}_{layers}_{batch_size}.pth')
    print(f'total time: {time.time()-start_time},  best loss: {best_val_loss}')