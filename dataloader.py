import os
import numpy as np
import random
import copy
from PIL import Image

import torch
import matplotlib.pyplot as plt



def load_data():
    training_class1_path = "mnist_png-master/mnist_png/training/8"
    training_class2_path = "mnist_png-master/mnist_png/training/6"
    testing_class1_path = "mnist_png-master/mnist_png/testing/8"
    testing_class2_path = "mnist_png-master/mnist_png/testing/6"
    train_images_8_names = os.listdir(training_class1_path)
    train_images_6_names = os.listdir(training_class2_path)
    test_images_8_names = os.listdir(testing_class1_path)
    test_images_6_names = os.listdir(testing_class2_path)

    # get paths
    training_images_8 = []
    for i in range (len(train_images_8_names)):
        path = training_class1_path + '/' + train_images_8_names[i]
        label = 1
        d = {
            "path": path,
            "label": label
        }
        training_images_8.append(d)

    training_images_6 = []
    for i in range (len(train_images_6_names)):
        path = training_class2_path + '/' + train_images_6_names[i]
        label = 0
        d = {
            "path": path,
            "label": label
        }
        training_images_6.append(d)


    testing_images_8 = []
    for i in range (len(test_images_8_names)):
        path = testing_class1_path + '/' + test_images_8_names[i]
        label = 1
        d = {
            "path": path,
            "label": label
        }
        testing_images_8.append(d)


    testing_images_6 = []
    for i in range (len(test_images_6_names)):
        path = testing_class2_path + '/' + test_images_6_names[i]
        label = 0
        d = {
            "path": path,
            "label": label
        }
        testing_images_6.append(d)
    
    training_images_8 = np.array(training_images_8)
    training_images_6= np.array(training_images_6)
    train_images = np.append(training_images_8,training_images_6)
    train_images = list(train_images)

    testing_images_8 = np.array(testing_images_8)
    testing_images_6= np.array(testing_images_6)
    test_images = np.append(testing_images_8,testing_images_6)
    test_images = list(test_images)

    # train_images = [training_images_8,training_images_6]
    # test_images = [testing_images_8,testing_images_6]
    return train_images,test_images

################

def get_batch (dataset,batch_size):
    '''
    dataset: list of lists each containig data of a label
    '''
    # batch_0_size = random.randint(batch_size//2 - batch_size//6, batch_size//2 + batch_size//6)
    # batch_1_size = batch_size - batch_0_size

    batch_size = min(batch_size,len(dataset))
    
    # get random indices = batch_size at max
    indices = random.sample(range(0, len(dataset)), batch_size)
    # indices_1 = random.sample(range(0, len(dataset[1])), min(len(dataset[1]),batch_1_size))
    # print(indices)
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
        
    # for i in (indices_1):
    #     c = copy.deepcopy(dataset[1][i])
    #     path = c["path"]
    #     # load the image
    #     image = Image.open(path)
    #     #resize
    #     image = image.resize((32,32))
    #     # convert image to numpy array
    #     image = np.asarray(image)
    #     d = {
    #         "path": c["path"],
    #         "image": image,
    #         "label": c["label"]
    #     }
    #     batch.append(d)    
    
    # delete the batch from dataset
    # dataset = [i for j, i in enumerate(dataset) if j not in indices]
    # dataset = d0
    
    # d1 = [i for j, i in enumerate(dataset[1]) if j not in indices_1]
    # dataset[1] = d1
    
    
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
    labels = np.reshape(labels,(labels.shape[0],1))
    images = torch.Tensor(images)
    labels = torch.Tensor(labels)
    return images,labels
    

