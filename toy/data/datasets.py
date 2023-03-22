"""

	CREATE THE DATASET

"""

# torch
import torch

# create the dataset
class ToyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x,
        y,
    ):
        # create a list of data
        self.examples=[]
        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i].float()
            example["y"] = y[i].long()
            self.examples.append(example)

        self.num_examples = len(self.examples)
        

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (index, x, y)

    def __len__(self):
        return self.num_examples
