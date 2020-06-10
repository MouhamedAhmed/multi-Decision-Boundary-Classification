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
def train(train_set, batch_size, model, cross_entropy_loss_criterion, optimizer,contrastive_ratio,margin, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    cross_entropy_loss = 0
    contrastive_l = 0

    l = len(train_set)
    correct = 0
    # for X, y_true in train_loader:
    while len(train_set)>0:
        # get batch 1
        batch1 = get_batch(train_set,batch_size)
        normalize_batch(batch1)
        X1, y_true1 = convert_batch_to_tensors(batch1)

        # get batch 2
        # batch2 = get_batch(train_set,batch_size)
       
        
        optimizer.zero_grad()


        X1 = X1.to(device)
        y_true1 = y_true1.to(device)
        
        
        # Forward pass
        y_hat1,logits = model(X1) 
        # print(y_hat.size(),y_true.size())

        loss1 = cross_entropy_loss_criterion(y_hat1, y_true1) 
        loss2 = contrastive_loss(logits,y_true1,device,margin)
        cross_entropy_loss += loss1.item()
        contrastive_l += loss2.item()

        # batch 2
        # if len(batch2) != 0:
        #     normalize_batch(batch2)
        #     X2, y_true2 = convert_batch_to_tensors(batch2)
        #     X2 = X2.to(device)
        #     y_true2 = y_true2.to(device) 
        #     y_hat2,_ = model(X2) 
        #     loss2 = cross_entropy_loss_criterion(y_hat2, y_true2)
        #     cross_entropy_loss += loss2.item()
        #     loss = loss1 + loss2
        # else:
        loss = (contrastive_ratio * loss2) + ((1-contrastive_ratio) * loss1)
        # print(batch1)



        # for i in range(len(y_hat)):
        #     y_hat_rounded = round(y_hat[i].cpu().detach().numpy()[0])
        #     if y_hat_rounded == y_true[i]:
        #         correct += 1

        

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = ((contrastive_ratio * contrastive_l) + ((1-contrastive_ratio) * cross_entropy_loss)) / l
    epoch_acc = correct / l
    return model,optimizer, epoch_loss, epoch_acc


# validate 
def validate(test_set, batch_size, model, cross_entropy_loss_criterion,contrastive_ratio,margin, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    cross_entropy_loss = 0
    contrastive_l = 0
    
    l = len(test_set)
    correct = 0
    while len(test_set)>0:
        # get batch 1
        batch1 = get_batch(test_set,batch_size)
        normalize_batch(batch1)
        X1, y_true1 = convert_batch_to_tensors(batch1)

        # get batch 2
        # batch2 = get_batch(test_set,batch_size)
        

        X1 = X1.to(device)
        y_true1 = y_true1.to(device)

        # Forward pass
        y_hat1,logits = model(X1)

        # loss
        loss1 = cross_entropy_loss_criterion(y_hat1, y_true1) 
        loss2 = contrastive_loss(logits,y_true1,device,margin)
        cross_entropy_loss += loss1.item()
        contrastive_l += loss2.item()
        # batch 2
        # if len(batch2) != 0:
        #     normalize_batch(batch2)
        #     X2, y_true2 = convert_batch_to_tensors(batch2)
        #     X2 = X2.to(device)
        #     y_true2 = y_true2.to(device) 
        #     y_hat2,_ = model(X2) 
        #     loss2 = cross_entropy_loss_criterion(y_hat2, y_true2)
        #     cross_entropy_loss += loss2.item()



        # for i in range(len(y_hat)):
        #     y_hat_rounded = round(y_hat[i].cpu().detach().numpy()[0])
        #     if y_hat_rounded == y_true[i]:
        #         correct += 1

    epoch_loss = ((contrastive_ratio * contrastive_l) + ((1-contrastive_ratio) * cross_entropy_loss)) / l
    epoch_acc = correct / l
        
    return model, epoch_loss, epoch_acc



def training_loop(model, cross_entropy_loss_criterion, batch_size, optimizer, epochs,contrastive_ratio,margin, device, print_every=1):
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
        model, optimizer, train_loss, train_acc = train(train_set, batch_size, model, cross_entropy_loss_criterion, optimizer,contrastive_ratio,margin, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(test_set, batch_size, model, cross_entropy_loss_criterion,contrastive_ratio,margin, device)
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
# contrastive loss
def contrastive_loss(y_hatt, y_truee,device, margin = 0):
    y_true = y_truee.cpu().detach().numpy()
    y_hat = y_hatt.cpu().detach().numpy()
    indices_1 = random.sample(range(0, len(y_true)), len(y_true)//2)
    indices = list(range(0, len(y_true)))
    indices_2 = list(set(indices) - set(indices_1)) 
    y_hat1 = [y_hat[i] for i in indices_1]
    y_hat2 = [y_hat[i] for i in indices_2]
    y_true1 = [y_true[i] for i in indices_1]
    y_true2 = [y_true[i] for i in indices_2]
    y = [int(i == j) for i, j in zip(y_true1, y_true2)]
    distance = list(np.abs(np.array(y_hat1)-np.array(y_hat2)))
    d1 = np.array(distance)**2
    d1 = np.sum(d1,axis = 1)
    # print(d1.shape)
    margins = np.ones(d1.shape) * margin
    z = np.zeros(d1.shape)
    # print(margins.shape)
    # print(np.array(distance).shape)
    d2 = margins - d1
    d2 = np.maximum(d2,z)
    # print(d2)
    # d2 = d2 ** 2
    # d1 = d1 ** 2
    loss = 0
    for i in range(len(y)):
        if y[i] == 0:
            loss += d2[i]
        else:
            loss += d1[i]
    loss = torch.Tensor([loss])
    loss = loss.to(device)
    return loss







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
    