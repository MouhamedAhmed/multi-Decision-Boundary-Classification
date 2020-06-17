import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from mnist_dataloader import *
import json


###############
# train iteration
def train(train_set, batch_size, model, cross_entropy_loss_criterion,contrastive_loss_criterion, optimizer,contrastive_ratio,lossLayer,margin, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    cross_entropy_loss_epoch = 0
    contrastive_loss_epoch = 0

    l = len(train_set)/batch_size
    correct = 0
    i = 0

    contrastive_losses = []
    cross_losses = []

    while len(train_set)>0:
         # get batch
        batch = get_batch(train_set,batch_size)
        if len(batch) == 1:
            print("ffff")
        normalize_batch(batch)
        X1, y_true1, X2, y_true2, y, X1_ord, X2_ord, y_ord = convert_batch_to_tensors(batch)
        X1 = X1.to(device)
        y_true1 = y_true1.to(device)
        X2 = X2.to(device)
        y_true2 = y_true2.to(device)
        y = y.float()
        y = y.to(device)
        

        optimizer.zero_grad()


        # Forward pass
        logits1,y_hat1 = model(X1)
        logits2,y_hat2 = model(X2)
        
        if X1_ord.numpy().shape[0] != 0 and X2_ord.numpy().shape[0] != 0 and y_ord.numpy().shape[0]:
            X1_ord = X1_ord.to(device)
            X2_ord = X2_ord.to(device)
            y_ord = y_ord.float()
            y_ord = y_ord.to(device)
            logits1_ord,_ = model(X1_ord)
            logits2_ord,_ = model(X2_ord)
            contrastive_loss_unord = contrastive_loss_criterion(logits1,logits2,y)
            contrastive_loss_ord = contrastive_loss_criterion(logits1_ord,logits2_ord,y_ord)
            contrastive_loss = (contrastive_loss_unord + contrastive_loss_ord)/2
            contrastive_loss_epoch += (contrastive_loss_unord.item() + contrastive_loss_ord.item())/2
        else:
            contrastive_loss = 0

        # loss
        cross_entropy_loss1 = cross_entropy_loss_criterion(y_hat1, y_true1) 
        cross_entropy_loss2 = cross_entropy_loss_criterion(y_hat2, y_true2) 

        
        cross_entropy_loss = cross_entropy_loss1+ cross_entropy_loss2

        cross_entropy_loss_epoch += cross_entropy_loss.item()
        
        contrastive_losses.append(contrastive_loss)
        cross_losses.append(cross_entropy_loss)

        contrastive_loss.to(device)
        cross_entropy_loss.to(device)
        loss = lossLayer(contrastive_loss, cross_entropy_loss, contrastive_ratio)
        # loss = (contrastive_ratio * contrastive_loss) + ((1-contrastive_ratio) * cross_entropy_loss)
        i += 1        

        # Backward pass
        loss.backward()
        optimizer.step()

        with open('cross_losses.json', 'a+') as outfile:
            json.dump(cross_entropy_loss, outfile)
        with open('contrastive_losses.json', 'a+') as outfile:
            json.dump(contrastive_loss, outfile)
        
    epoch_loss = ((contrastive_ratio * contrastive_loss_epoch) + ((1-contrastive_ratio) * cross_entropy_loss_epoch)) / l
    epoch_acc = correct / l


    
    
    return model,optimizer, epoch_loss, epoch_acc


# validate 
def validate(test_set, batch_size, model, cross_entropy_loss_criterion,contrastive_loss_criterion,contrastive_ratio,lossLayer,margin, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    cross_entropy_loss_epoch = 0
    contrastive_loss_epoch = 0
    
    l = len(test_set)/batch_size
    correct = 0
    while len(test_set)>0:
        # get batch
        batch = get_batch(test_set,batch_size)
        normalize_batch(batch)
        X1, y_true1, X2, y_true2, y, X1_ord, X2_ord, y_ord = convert_batch_to_tensors(batch)

        X1 = X1.to(device)
        y_true1 = y_true1.to(device)
        X2 = X2.to(device)
        y_true2 = y_true2.to(device)
        y = y.float()
        y = y.to(device)
        

        # Forward pass
        logits1,y_hat1 = model(X1)
        logits2,y_hat2 = model(X2)
        
        
        if X1_ord.numpy().shape[0] != 0 and X2_ord.numpy().shape[0] != 0 and y_ord.numpy().shape[0]:
            X1_ord = X1_ord.to(device)
            X2_ord = X2_ord.to(device)
            y_ord = y_ord.to(device)
            y_ord = y_ord.float()
            logits1_ord,_ = model(X1_ord)
            logits2_ord,_ = model(X2_ord)
            contrastive_loss_unord = contrastive_loss_criterion(logits1,logits2,y)
            contrastive_loss_ord = contrastive_loss_criterion(logits1_ord,logits2_ord,y_ord)
            contrastive_loss = (contrastive_loss_unord + contrastive_loss_ord)/2
            contrastive_loss_epoch += (contrastive_loss_unord.item() + contrastive_loss_ord.item())/2
        else:
            contrastive_loss = 0
            
         
        # loss
        cross_entropy_loss1 = cross_entropy_loss_criterion(y_hat1, y_true1) 
        cross_entropy_loss2 = cross_entropy_loss_criterion(y_hat2, y_true2) 

        
        cross_entropy_loss = cross_entropy_loss1 + cross_entropy_loss2

        cross_entropy_loss_epoch += cross_entropy_loss.item()

       
    epoch_loss = ((contrastive_ratio * contrastive_loss_epoch) + ((1-contrastive_ratio) * cross_entropy_loss_epoch)) / l
    epoch_acc = correct / l
        
    return model, epoch_loss, epoch_acc



def training_loop(model, cross_entropy_loss_criterion,contrastive_loss_criterion,lossLayer, batch_size, optimizer, epochs,contrastive_ratio,margin, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    
    train_set_start, test_set_start,_ = load_data()

    # Train model
    for epoch in range(0, epochs):

        # load dataset
        train_set = copy.deepcopy(train_set_start)
        test_set = copy.deepcopy(test_set_start)

        # training
        model, optimizer, train_loss, train_acc = train(train_set, batch_size, model, cross_entropy_loss_criterion,contrastive_loss_criterion, optimizer,contrastive_ratio,lossLayer,margin, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(test_set, batch_size, model, cross_entropy_loss_criterion,contrastive_loss_criterion,contrastive_ratio,lossLayer,margin, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            # train_set, test_set = load_data()
            train_set = copy.deepcopy(train_set_start)
            test_set = copy.deepcopy(test_set_start)
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
def contrastive_loss_prep(y_hatt, y_truee,device, margin = 0):
    y_true = y_truee.cpu().detach().numpy()
    y_hat = y_hatt.cpu().detach().numpy()
    indices_1 = random.sample(range(0, len(y_true)), len(y_true)//2)
    indices = list(range(0, len(y_true)))
    indices_2 = list(set(indices) - set(indices_1)) 
    y_hat1 = [y_hat[i] for i in indices_1]
    y_hat2 = [y_hat[i] for i in indices_2]
    y_true1 = [y_true[i] for i in indices_1]
    y_true2 = [y_true[i] for i in indices_2]
    y = [int(i != j) for i, j in zip(y_true1, y_true2)]
    y_hat1 = torch.Tensor(y_hat1)
    y_hat2 = torch.Tensor(y_hat2)
    y = torch.Tensor(y)
    return y_hat1,y_hat2,y
    # distance = list(np.abs(np.array(y_hat1)-np.array(y_hat2)))
    # d1 = np.array(distance)**2
    # d1 = np.sum(d1,axis = 1)
    # # print(d1.shape)
    # margins = np.ones(d1.shape) * margin
    # z = np.zeros(d1.shape)
    # # print(margins.shape)
    # # print(np.array(distance).shape)
    # d2 = margins - d1
    # d2 = np.maximum(d2,z)
    # # print(d2)
    # # d2 = d2 ** 2
    # # d1 = d1 ** 2
    # loss = 0
    # for i in range(len(y)):
    #     if y[i] == 0:
    #         loss += d2[i]
    #     else:
    #         loss += d1[i]
    # loss = torch.Tensor([loss])
    # loss = loss.to(device)
    # return loss







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
            # get batch
            batch = get_batch(dataset,batch_size)
            normalize_batch(batch)
            X1, y_true1, X2, y_true2, y,_,_,_ = convert_batch_to_tensors(batch)

            X1 = X1.to(device)
            y_true1 = y_true1.to(device)
            X2 = X2.to(device)
            y_true2 = y_true2.to(device)
            y = y.to(device)


            # Forward pass
            y_hat1,y_prob1 = model(X1)
            y_hat2,y_prob2 = model(X2)
            _, predicted_labels1 = torch.max(y_prob1, 1)
            _, predicted_labels2 = torch.max(y_prob2, 1)


            n += y_true1.size(0) + y_true2.size(0)
            correct_pred += (predicted_labels1 == y_true1).sum() + (predicted_labels2 == y_true2).sum()

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
    