"""
	LOAD THE TEST DATASETS (CIFAR10, CIFAR100, SVHN)
"""

import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import numpy as np


# create test CIFAR dataset
def create_test_dataset_CIFAR(test_data, transform):

    """

    TEST CIFAR

    Input: test_data, transform
    Output: test_dataset

    """

    # test set
    test_images = test_data[0].type(torch.float)
    test_labels = test_data[1]

    # datasets
    test_dataset = preprocessed_dataset(test_images, test_labels, transform)

    return test_dataset


# create SVHN test dataset
def create_test_dataset_SVHN(data_dir, transform):

    """

    TEST SVHN

    Input: torchvision data_dir, transform
    Output: test_dataset

    """

    transform_SVHN = transforms.Compose([transforms.ToTensor(), transform])
    test_dataset = torchvision.datasets.SVHN(
        root=data_dir, split="test", download=False, transform=transform_SVHN
    )

    return test_dataset


# download cifar10
def get_cifar10(data_dir):

    """

    LOAD CIFAR10

    Input: data_dir
    Output: train_data, test_data

    """

    # CIFAR TRAIN DATA
    train_data = np.array([])
    train_labels = np.array([])
    for i in range(5):
        data_batch = unpickle(
            os.path.join(data_dir, "CIFAR/cifar-10-batches-py/data_batch_" + str(i + 1))
        )

        train_data = np.append(train_data, data_batch["data"])
        train_labels = np.append(train_labels, data_batch["labels"])

    train_data = torch.Tensor(train_data.reshape(-1, 3, 32, 32))
    train_labels = torch.Tensor(train_labels).long()

    # CIFAR TEST DATA
    test_batch = unpickle(os.path.join(data_dir, "CIFAR/cifar-10-batches-py/test_batch"))

    test_data = torch.Tensor(test_batch["data"].reshape(-1, 3, 32, 32))
    test_labels = torch.Tensor(test_batch["labels"]).long()

    return [train_data, train_labels], [test_data, test_labels]


# download cifar100
def get_cifar100(data_dir):

    """

    LOAD CIFAR100

    Input: data_dir
    Output: train_valid_data, test_dataset

    """

    # CIFAR TRAIN DATA
    train_batch = unpickle(os.path.join(data_dir, "CIFAR/cifar-100-python/train"))

    train_data = torch.Tensor(train_batch["data"].reshape(-1, 3, 32, 32))
    train_labels = torch.Tensor(train_batch["fine_labels"]).long()

    # CIFAR TEST DATA
    test_batch = unpickle(os.path.join(data_dir, "CIFAR/cifar-100-python/test"))

    test_data = torch.Tensor(test_batch["data"].reshape(-1, 3, 32, 32))
    test_labels = torch.Tensor(test_batch["fine_labels"]).long()

    return [train_data, train_labels], [test_data, test_labels]


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
    file_pickle = open(file, "rb")
    return pickle.load(file_pickle, encoding="latin1")
