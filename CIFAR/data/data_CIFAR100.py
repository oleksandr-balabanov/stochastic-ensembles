"""
	LOAD AND PREPROCESS THE CIFAR100 DATASET
"""
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import numpy as np

# unpickle file
def unpickle(file):
    file_pickle = open(file, "rb")
    return pickle.load(file_pickle, encoding="latin1")


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


# create torch datasets
def create_datasets(
    train_data,
    test_data,
    num_train,
    shuffle=True,
    do_augmentation_train=False,
    do_augmentation_test=False,
):

    """

    PREPROCESS CIFAR100

    Input: train_data, test_data, num_train, shuffle = True, do_augmentation_train = False, do_augmentation_test = False
    Output: train_dataset, valid_dataset, test_dataset

    """

    # PREPROCESS
    # shuffle
    if shuffle:
        perm = torch.randperm(train_data[0].shape[0])
        shuffled_images = train_data[0][perm, :, :, :]
        shuffled_labels = train_data[1][perm]
    else:
        shuffled_images = train_data[0]
        shuffled_labels = train_data[1]

    # training set
    train_images = shuffled_images.type(torch.float)[:num_train]
    train_labels = shuffled_labels[:num_train]

    # validation set
    valid_images = shuffled_images.type(torch.float)[num_train:]
    valid_labels = shuffled_labels[num_train:]

    # test set
    test_images = test_data[0].type(torch.float)
    test_labels = test_data[1]

    # datasets
    train_dataset = preprocessed_CIFAR100_dataset(
        train_images, train_labels, test=False, do_augmentation=do_augmentation_train
    )
    valid_dataset = preprocessed_CIFAR100_dataset(
        valid_images, valid_labels, test=False, do_augmentation=do_augmentation_train
    )
    test_dataset = preprocessed_CIFAR100_dataset(
        test_images, test_labels, test=True, do_augmentation=do_augmentation_test
    )

    return train_dataset, valid_dataset, test_dataset


# train transform
def train_transform(do_augmentation):
    if do_augmentation == True:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762])]
        )


# test transform
def test_transform(do_augmentation):
    if do_augmentation == True:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762]),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762])]
        )


# preprocess the images
class preprocessed_CIFAR100_dataset(Dataset):
    def __init__(self, x, y, test=False, do_augmentation=False):
        self.data = x.type(torch.float) / 255
        self.target = y

        if test == False:
            self.transform = train_transform(do_augmentation)
        else:
            self.transform = test_transform(do_augmentation)

    def __getitem__(self, index):
        x = self.data[index, :, :, :]
        y = self.target[index]

        # preprocess the data
        x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
