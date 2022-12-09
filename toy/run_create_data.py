"""

	CREATE AND SAVE THE DATA

"""

import argparse
import torch
import pickle
import os
import pathlib


def get_train_dataset_basename(num_data_train):
    return f"train_n{num_data_train}"


def get_eval_dataset_basename(num_data_eval, scaling_factor):
    return f"eval_n{num_data_eval}_sf_{scaling_factor}"


def get_train_dataset_name(num_data_train):
    basename = get_train_dataset_basename(num_data_train)
    return f"x_{basename}", f"y_{basename}"


def get_eval_dataset_name(num_data_eval, scaling_factor):
    basename = get_eval_dataset_basename(num_data_eval, scaling_factor)
    return f"x_{basename}", f"y_{basename}"


def create_train_data(data_dir, num_data_train=2000):

    # data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # uniform from  x1 = [0, 3], x2 = [-3, 3]
    x_1 = 3 * torch.rand(size=(num_data_train,), dtype=torch.float32)
    x_2 = 6 * torch.rand(size=(num_data_train,), dtype=torch.float32) - 3

    # train data
    x_train = torch.zeros((num_data_train, 2))
    x_train[:, 0] = x_1
    x_train[:, 1] = x_2

    y_train = []
    for i in range(num_data_train):
        x_value = x_train[i]

        if torch.linalg.norm(x_value) < 2.4:
            y_value = 0
        else:
            y_value = 1
        y_train.append(y_value)

    y_train = torch.tensor(y_train)

    x_name, y_name = get_train_dataset_name(num_data_train)
    print(f"Saving {os.path.join(data_dir, x_name)} and {y_name}")
    torch.save(x_train, os.path.join(data_dir, x_name) + ".pt")
    torch.save(y_train, os.path.join(data_dir, y_name) + ".pt")

    return x_train, y_train


def create_eval_data(data_dir, num_data_eval=2000, scaling_factor=6):

    # data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # loop while we won't produce enough datapoints (num_data_eval)
    x_1 = None
    x_2 = None
    while True:

        # uniform from  x1 = [-3, 3], x2 = [-3, 3]
        x_1_all = 6 * torch.rand(size=(num_data_eval,), dtype=torch.float32) - 3
        x_2_all = 6 * torch.rand(size=(num_data_eval,), dtype=torch.float32) - 3

        # scale the evaluation data domain
        if scaling_factor < 1.0:
            # scaling_factor = 0 meaning x1 = [0, 3], x2 = [-3, 3]
            x_1 = (x_1_all + 3.0) / 2.0
            x_2 = x_2_all
            break
        else:
            # scaling_factor = 1 meaning x1 = [-3, 0], x2 = [-3, 3]
            if scaling_factor < 2.0:
                x_1 = -(x_1_all + 3.0) / 2.0
                x_2 = x_2_all
                break
            else:
                # other cases x1 = [-3*scaling_factor, 3*scaling_factor],
                # x2 = [-3*scaling_factor, 3*scaling_factor]
                x_1_all *= scaling_factor
                x_2_all *= scaling_factor

        # remove points overlaping with x1 = [-3*(scaling_factor-1), 3*(scaling_factor-1)]
        # x2 = [-3*(scaling_factor-1), 3*(scaling_factor-1)]
        mask_x1 = torch.lt(torch.abs(x_1_all), 3 * (scaling_factor - 1))
        mask_x2 = torch.lt(torch.abs(x_2_all), 3 * (scaling_factor - 1))
        data_mask = torch.logical_not(torch.logical_and(mask_x1, mask_x2))

        # add obtained data to the dataset
        if x_1 != None:
            x_1 = torch.cat((x_1, x_1_all[data_mask]), 0)
            x_2 = torch.cat((x_2, x_2_all[data_mask]), 0)
        else:
            x_1 = x_1_all[data_mask]
            x_2 = x_2_all[data_mask]

        # break the loop
        if x_1.shape[0] > num_data_eval:
            x_1 = x_1[:num_data_eval]
            x_2 = x_2[:num_data_eval]
            break

    # eval data
    x_eval = torch.zeros((num_data_eval, 2))
    x_eval[:, 0] = x_1
    x_eval[:, 1] = x_2

    y_eval = []
    for i in range(num_data_eval):
        x_value = x_eval[i]

        if torch.linalg.norm(x_value) < 2.4:
            y_value = 0
        else:
            y_value = 1
        y_eval.append(y_value)

    y_eval = torch.tensor(y_eval)

    x_name, y_name = get_eval_dataset_name(num_data_eval, scaling_factor)
    print(f"Saving {os.path.join(data_dir, x_name)} and {y_name}")
    torch.save(x_eval, os.path.join(data_dir, x_name + ".pt"))
    torch.save(y_eval, os.path.join(data_dir, y_name + ".pt"))

    return x_eval, y_eval


def create_data(data_dir, num_data_train, num_data_eval, scaling_factor):
    x_train, y_train = create_train_data(data_dir, num_data_train)
    x_eval, y_eval = create_eval_data(data_dir, num_data_eval, scaling_factor)
    return x_eval, y_eval, x_train, y_train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_data_train", type=int, default=2000, help="Number of train data points"
    )
    parser.add_argument(
        "--num_data_eval", type=int, default=2000, help="Number of eval data points"
    )
    parser.add_argument(
        "--scaling_factors",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        help="List of scaling factors, e.g. --scaling factors 1 2 3 6 12",
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="The output folder")
    parser.add_argument("--data_folder", type=str, default="datasets/", help="The data folder")
    args = parser.parse_args()
    data_dir = str(pathlib.Path(os.path.join(args.output_dir, args.data_folder)).absolute())
    scaling_factors = [float(f) for f in args.scaling_factors]

    print(f"Creating training data at {data_dir}")
    create_train_data(data_dir=data_dir, num_data_train=args.num_data_train)

    print(f"Creating evaluation data at {data_dir}")
    for scaling_factor in scaling_factors:
        create_eval_data(
            data_dir=data_dir, num_data_eval=args.num_data_eval, scaling_factor=scaling_factor
        )
