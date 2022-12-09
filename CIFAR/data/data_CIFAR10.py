"""
	LOAD AND PREPROCESS THE CIFAR10 DATASET
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

    PREPROCESS CIFAR10

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
    train_dataset = preprocessed_CIFAR10_dataset(
        train_images, train_labels, test=False, do_augmentation=do_augmentation_train
    )
    valid_dataset = preprocessed_CIFAR10_dataset(
        valid_images, valid_labels, test=False, do_augmentation=do_augmentation_train
    )
    test_dataset = preprocessed_CIFAR10_dataset(
        test_images, test_labels, test=True, do_augmentation=do_augmentation_test
    )

    return train_dataset, valid_dataset, test_dataset


# train transform (augmented)
def train_transform(do_augmentation):

    if do_augmentation == True:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                    std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                    std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628],
                )
            ]
        )


# test transform
def test_transform(do_augmentation):
    if do_augmentation == True:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                    std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
                    std=[0.24703225141799082, 0.24348516474564, 0.26158783926049628],
                )
            ]
        )


# preprocess the images
class preprocessed_CIFAR10_dataset(Dataset):
    def __init__(self, x, y, test=False, do_augmentation=False):
        self.data = x.type(torch.float) / 255
        self.target = y

        if test == False:
            self.transform = train_transform(do_augmentation)
        else:
            self.transform = test_transform(do_augmentation)

    # mean
    def mean(self, data):

        return torch.mean(self.transform(data), (0, 2, 3))

    # std
    def std(self, data):

        return torch.std(self.transform(data), (0, 2, 3))

    def __getitem__(self, index):
        x = self.data[index, :, :, :]
        y = self.target[index]

        # preprocess the data
        x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
