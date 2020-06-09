import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from dataloader import *




###############
# train iteration
def train(train_set, batch_size, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    l = len(train_set)
    correct = 0
    # for X, y_true in train_loader:
    while len(train_set)>0:
        batch = get_batch(train_set,batch_size)
        normalize_batch(batch)
        X, y_true = convert_batch_to_tensors(batch)
        # print(type(np.array(y_true)[0][0]))
        
        optimizer.zero_grad()


        X = X.to(device)
        y_true = y_true.to(device)
    
        
        # Forward pass
        y_hat,_ = model(X) 
        # print(y_hat.size(),y_true.size())
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item()
        for i in range(len(y_hat)):
            y_hat_rounded = round(y_hat[i].cpu().detach().numpy()[0])
            if y_hat_rounded == y_true[i]:
                correct += 1

        

        
        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / l
    epoch_acc = correct / l
    return model,optimizer, epoch_loss, epoch_acc


# validate 
def validate(test_set, batch_size, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    l = len(test_set)
    correct = 0
    while len(test_set)>0:
        batch = get_batch(test_set,batch_size)
        normalize_batch(batch)
        X, y_true = convert_batch_to_tensors(batch)    
        # print(type(y_true))

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat,_= model(X) 
        # print(type(y_hat))
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item()

        for i in range(len(y_hat)):
            y_hat_rounded = round(y_hat[i].cpu().detach().numpy()[0])
            if y_hat_rounded == y_true[i]:
                correct += 1

    epoch_loss = running_loss / l
    epoch_acc = correct / l
        
    return model, epoch_loss, epoch_acc



def training_loop(model, criterion, batch_size, optimizer, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # load dataset
        train_set, test_set = load_data()
        # training
        model, optimizer, train_loss, train_acc = train(train_set, batch_size, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(test_set, batch_size, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_set, test_set = load_data()

            train_acc = get_accuracy(model, train_set,batch_size, device=device)
            valid_acc = get_accuracy(model, test_set,batch_size, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, train_losses, valid_losses



############
# helper functions
def get_accuracy(model, dataset,batch_size, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        while len(dataset)>0:
            batch = get_batch(dataset,batch_size)
            normalize_batch(batch)
            X, y_true = convert_batch_to_tensors(batch)    
            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
    