"""

	CREATE THE DATASET

"""

import os

# torch
import torch

# other
import numpy as np
import pickle

# modules
from run_create_data import create_data, get_train_dataset_name, get_eval_dataset_name
from run_create_data import get_train_dataset_basename, get_eval_dataset_basename


# create the dataset
class ToyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        num_data_train=2000,
        num_data_eval=2000,
        mode="train",
        scaling_factor=None,
    ):

        self.num_data_train = num_data_train
        self.num_data_eval = num_data_eval
        self.scaling_factor = scaling_factor
        self.mode = mode

        # list with data
        self.examples = []

        x_train_name, y_train_name = get_train_dataset_name(num_data_train)

        if scaling_factor != None:
            x_eval_name, y_eval_name = get_eval_dataset_name(num_data_eval, scaling_factor)
        else:
            # create dummy eval names = train names
            x_eval_name, y_eval_name = get_train_dataset_name(num_data_train)

        # get the data
        print("Looking for data at ", data_dir)

        x_train = torch.load(os.path.join(data_dir, x_train_name + ".pt"))
        y_train = torch.load(os.path.join(data_dir, y_train_name + ".pt"))
        x_eval = torch.load(os.path.join(data_dir, x_eval_name + ".pt"))
        y_eval = torch.load(os.path.join(data_dir, y_eval_name + ".pt"))

        print("The data is found at", data_dir)

        if mode == "train":

            # the ratio between 0 and 1 labels
            x_train_false = x_train[y_train == 0]
            x_train_true = x_train[y_train == 1]
            print("num_false: %d" % x_train_false.shape[0])
            print("num_true: %d" % x_train_true.shape[0])

            # create a list of data
            for i in range(x_train.shape[0]):
                example = {}
                example["x"] = x_train[i]
                example["y"] = y_train[i]
                self.examples.append(example)

        self.num_examples = len(self.examples)

        if mode == "eval":
            # create a list of data
            for i in range(x_eval.shape[0]):
                example = {}
                example["x"] = x_eval[i]
                example["y"] = y_eval[i]
                self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (index, x, y)

    def __len__(self):
        return self.num_examples

    def get_identifier(self):
        return dict(
            train=get_train_dataset_basename(self.num_data_train),
            eval=get_eval_dataset_basename(self.num_data_eval, self.scaling_factor),
        )[self.mode]
