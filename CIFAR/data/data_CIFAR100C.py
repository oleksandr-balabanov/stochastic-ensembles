"""
	LOAD THE CIFARC DATASET
"""
import os
import torch
import torch.utils.data as data_utils
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import numpy as np


# download cifar100C
def get_cifar100C(data_dir, corruption_file_name, alpha_level):
    
    """

    LOAD CIFAR10C

    Input: data_dir, corruption_file_name, alpha_level
    Output: test_dataset 

    """

    # CIFAR C DATA
    test_data = np.load(os.path.join(data_dir, 'CIFAR-100-C', corruption_file_name))[alpha_level*10000:(alpha_level+1)*10000]
    test_labels = np.load(os.path.join(data_dir, 'CIFAR-100-C',  "labels.npy"))

    test_data = torch.from_numpy(test_data).reshape(-1, 3, 32, 32)
    test_labels = torch.from_numpy(test_labels).long()

    return [test_data, test_labels]


# create test CIFAR100 dataset    
def create_test_dataset_CIFARC(test_data, transform):

    """

    TEST CIFARC

    Input: test_data, transform
    Output: test_dataset

    """ 

    # test set
    test_images = test_data[0].type(torch.float)
    test_labels = test_data[1]

    # match the conventional ordering of the elements (the same as in the original CIFAR100)
    test_images_numpy = test_images.numpy().reshape(-1, 32,32,3).transpose(0, 3, 1, 2)
    test_images = torch.from_numpy(test_images_numpy)

    # datasets
    test_dataset = preprocessed_dataset(test_images, test_labels, transform)

    return test_dataset

# preprocess the images 
class preprocessed_dataset(Dataset):

    def __init__(self, x, y, transform):
        self.data = x.type(torch.float) / 255
        self.target = y

        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index, :, :, :]
        y = self.target[index]

        # preprocess the data
        x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# unpickle file
def unpickle(file):
    file_pickle = open (file, "rb")
    return pickle.load(file_pickle, encoding='latin1')



