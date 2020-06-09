import os
import numpy as np
import random
import copy
from PIL import Image

import torch
import matplotlib.pyplot as plt



def load_data():
    datapath = "mnist_png-master/mnist_png"
    # get training data
    training_path = datapath + "/training"
    training_labels = os.listdir(training_path)
    training_classes_paths = []
    for i in training_labels:
        training_classes_paths.append(training_path + "/" + i)
    training_images = []
    for i in range(len(training_labels)):
        label = int(training_labels[i])
        label_dataset_paths = os.listdir(training_classes_paths[i])
        for p in label_dataset_paths:
            d = {
                "path": training_classes_paths[i] + '/' + p,
                "label": label
            }
            training_images.append(d)
    
    # get testing data
    testing_path = datapath + "/testing"
    testing_labels = os.listdir(testing_path)
    testing_classes_paths = []
    for i in testing_labels:
        testing_classes_paths.append(testing_path + "/" + i)
    testing_images = []
    for i in range(len(testing_labels)):
        label = int(testing_labels[i])
        label_dataset_paths = os.listdir(testing_classes_paths[i])
        for p in label_dataset_paths:
            d = {
                "path": testing_classes_paths[i] + '/' + p,
                "label": label
            }
            testing_images.append(d)

    return training_images,testing_images

################

def get_batch (dataset,batch_size):
    '''
    dataset: list of lists each containig data of a label
    '''

    batch_size = min(batch_size,len(dataset))
    
    # get random indices = batch_size at max
    indices = random.sample(range(0, len(dataset)), batch_size)

    # get paths
    batch = []
    indices.sort(reverse=True) 
    for i in range (len(indices)):
        c = copy.deepcopy(dataset[indices[i]])
        path = c["path"]
        # load the image
        image = Image.open(path)
        #resize
        image = image.resize((32,32))
        # convert image to numpy array
        image = np.asarray(image)
        d = {
            "path": c["path"],
            "image": image,
            "label": c["label"]
        }
        batch.append(d)
        del dataset[indices[i]]
        
   
    return batch

##################
def normalize_batch(batch):
    for i in batch:
        i["image"] =  i["image"]/255

##################
def convert_batch_to_tensors(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for i in batch:
        image = i["image"]
        label = i["label"]

        images.append(image)
        labels.append(label)

    images = np.asarray(images)
    labels = np.asarray(labels)
    images = np.expand_dims(images, axis=1)
    # labels = np.reshape(labels,(labels.shape[0],1))
    images = torch.Tensor(images)
    # print(labels)
    labels = torch.LongTensor(labels)
    # print(labels)

    return images,labels
    

