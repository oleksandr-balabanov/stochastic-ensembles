"""
	TRAIN OR EVAL A STOCHASITC ENSEMBLE OF NETWORKS
"""

# torch
import torch

# wandb
import wandb

# args and sys
import os
import argparse
import pathlib
import sys

# import modules
import eval.eval_ens_CIFAR as eval_ens_CIFAR

# str to bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Parsing and configuration
def parse_args():
    desc = (
        "Eval an ensamble of stochastic neural networks for outputing "
        "accuracy, loss, calibration, OOD, entropy, mutual information, distribution shift"
    )
    parser = argparse.ArgumentParser(description=desc)

    # input
    parser.add_argument(
        "--num_nets", type=int, default=50, help="The number of nets in the ensemble"
    )

    # reload softmax probabilities?
    parser.add_argument(
        "--compute_save_softmax_probs",
        type=str2bool,
        default=True,
        help="Compute and save the softmax probabilities?",
    )

    # stochastic method
    parser.add_argument(
        "--method",
        choices=["regular", "dropout", "np_dropout", "dropconnect"],
        default="regular",
        help="stochastic method",
    )

    # CIFAR10 or CIFAR100?
    parser.add_argument(
        "--cifar_mode",
        choices=["CIFAR10", "CIFAR100"],
        default="CIFAR10",
        help="CIFAR10 or CIFAR100?",
    )

    # augmentation?
    parser.add_argument(
        "--do_augmentation_train", type=str2bool, default=False, help="do augmentation train?"
    )
    parser.add_argument(
        "--do_augmentation_test", type=str2bool, default=False, help="do augmentation test?"
    )

    # stochastic parameters
    parser.add_argument(
        "--drop_rate_conv", type=float, default=0, help="Dropout rate for conv layers"
    )
    parser.add_argument(
        "--drop_rate_linear", type=float, default=0, help="Dropout rate for the output linear layer"
    )

    # prior lambda = var^(-1)
    parser.add_argument("--lambda_prior", type=float, default=5, help="The lambda_prior value")

    # evaluation
    parser.add_argument("--test_batch", type=int, default=200, help="The test batch size")
    parser.add_argument("--eval_batch", type=int, default=200, help="The eval batch size")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle the data")
    parser.add_argument(
        "--num_net_passes", type=int, default=1, help="The number of eval inferences per one net"
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda?")

    # save / load folders
    parser.add_argument(
        "--load_folder", type=str, default="models_best", help="load models from ..."
    )
    parser.add_argument("--save_folder", type=str, default="models_best", help="save res to ...")
    parser.add_argument("--data_dir_CIFAR", type=str, default="./datasets/CIFAR")
    parser.add_argument("--data_dir_CIFARC", type=str, default="./datasets/CIFAR_C")
    parser.add_argument("--data_dir_SVHN", type=str, default="./datasets/SVHN")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


# EVAL METHOD
def eval():
    """
    EVAL A STOCHASITC ENSEMBLE OF RESNET-20-FRN NETWORKS ON CIFAR10 or CIFAR100
    """

    # random seed
    seed_index = torch.randint(1000, (1,))
    torch.manual_seed(seed_index)
    print("SEED: ", seed_index)

    # parse the arguments
    args = parse_args()
    args.seed = seed_index.item()

    # eval and save
    eval_ens_CIFAR.eval_ens_CIFAR(args)


# RUN
if __name__ == "__main__":
    eval()
