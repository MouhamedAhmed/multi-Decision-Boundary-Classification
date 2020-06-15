import os
import numpy as np
import random
import copy
from PIL import Image

import torch
import matplotlib.pyplot as plt
import random


def load_data():
    training = []
    testing = []
    validation = []
    datapath = "CUB_200_2011/CUB_200_2011/images"
    labels_str = os.listdir(datapath)
    labels = np.arange(len(labels_str))
    for i in range(len(labels_str)):
        label = labels[i]
        label_path = datapath + "/" + labels_str[i]
        label_images = os.listdir(label_path)
        label_images_paths = [label_path + "/" + j for j in label_images]
        label_images_paths_train = label_images_paths[0:int(0.8*len(label_images_paths))]
        label_images_paths_valid = label_images_paths[int(0.8*len(label_images_paths)):int(0.9*len(label_images_paths))]
        label_images_paths_test = label_images_paths[int(0.9*len(label_images_paths)):len(label_images_paths)]
        for x in label_images_paths_train:
            d = {
                    "path": x,
                    "label": label
                }
            training.append(d)
        for x in label_images_paths_valid:
            d = {
                    "path": x,
                    "label": label
                }
            testing.append(d)
        for x in label_images_paths_test:
            d = {
                    "path": x,
                    "label": label
                }
            validation.append(d)
    return training,validation,testing

################

def get_batch (dataset,batch_size):
    '''
    dataset: list of lists each containig data of a label
    '''
    if len(dataset) == 0:
        return []
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
        if len(list(np.array(image).shape)) != 3:
#             print(path)
#             print(np.array(image).shape)
            image = image.convert('RGB')
#             print(np.array(image).shape)
        #resize
        image = image.resize((32,32))
        # convert image to numpy array
        # if np.array(image).shape != (32,32,3):
        #     image = image.convert('RGB')

        image = np.asarray(image)
        
        d = {
            "path": c["path"],
            "image": image,
            "label": c["label"]
        }
        batch.append(d)
        del dataset[indices[i]]

    random.shuffle(batch)
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
        if np.array(image).shape != (32,32,3) :
            print(np.array(image).shape)

    images = np.asarray(images)
    labels = np.asarray(labels)
    # images = np.expand_dims(images, axis=1)
    images = np.moveaxis(images, -1, 1)
    # labels = np.reshape(labels,(labels.shape[0],1))
    # print(labels)
    # print(images.shape)
    indices_1 = random.sample(range(0, images.shape[0]), images.shape[0]//2)
    indices = list(range(0, images.shape[0]))
    indices_2 = list(set(indices) - set(indices_1)) 
    images1 = [images[i] for i in indices_1]
    images2 = [images[i] for i in indices_2]
    labels1 = [labels[i] for i in indices_1]
    labels2 = [labels[i] for i in indices_2]
    
    images1 = images1[0:min(len(images1),len(images2))]
    images2 = images2[0:min(len(images1),len(images2))]
    labels1 = labels1[0:min(len(labels1),len(labels2))]
    labels2 = labels2[0:min(len(labels1),len(labels2))]

    y = [int(i != j) for i, j in zip(labels1, labels2)]

    
    
    
    images1 = torch.Tensor(images1)
    images2 = torch.Tensor(images2)
    labels1 = torch.LongTensor(labels1)
    labels2 = torch.LongTensor(labels2)
    y = torch.LongTensor(y)


    images = list(images)
    labels = list(labels)

    images_ordered1 = []
    images_ordered2 = []
    labels_ordered1 = []
    labels_ordered2 = []
    labels_set = set(labels)
    for x in labels_set:
        indices_of_label = [i for i in range(len(labels)) if labels[i] == x]
        imgs1 = [images[i] for i in indices_of_label[0:len(indices_of_label)//2]]
        imgs2 = [images[i] for i in indices_of_label[len(indices_of_label)//2:len(indices_of_label)]]
        # print(np.asarray(imgs1).shape)
        # imgs1 = np.asarray(imgs1)
        # imgs2 = np.asarray(imgs2)
        for i in imgs1:
            images_ordered1.append(i)
        for i in imgs2:
            images_ordered2.append(i)
        # print("hhhh",np.array(images_ordered1).shape)

        lbls1 = [labels[i] for i in indices_of_label[0:len(indices_of_label)//2]]
        lbls2 = [labels[i] for i in indices_of_label[len(indices_of_label)//2:len(indices_of_label)]]
        # lbls1 = np.array(lbls1)
        # lbls2 = np.array(lbls2)
        for i in lbls1:
            labels_ordered1.append(i)
        for i in lbls2:
            labels_ordered2.append(i)
        
        labels_ordered1 = labels_ordered1[0:min(len(labels_ordered1),len(labels_ordered2))]
        labels_ordered2 = labels_ordered2[0:min(len(labels_ordered1),len(labels_ordered2))]
        images_ordered1 = images_ordered1[0:min(len(images_ordered1),len(images_ordered2))]
        images_ordered2 = images_ordered2[0:min(len(images_ordered1),len(images_ordered2))]
    # print(labels_ordered1)
    # print(labels_ordered2)
    

    y_ordered = [int(i != j) for i, j in zip(labels_ordered1, labels_ordered2)]

    images_ordered1 = torch.Tensor(images_ordered1)
    images_ordered2 = torch.Tensor(images_ordered2)
    y_ordered = torch.LongTensor(y_ordered)

    # print(images_ordered1.size())
    # print(images1.size())

    return images1,labels1,images2,labels2,y,images_ordered1,images_ordered2,y_ordered
    

