"""
	EVAL AN ENSEMBLE OF NETWORKS (TOY MODEL)
"""

# TORCH
import torch

# other
import sys
import os
import argparse
import pathlib

# import modules
import HMC.eval as eval_HMC
import stochastic_methods.eval as eval_all
import plot.plot_uncertainty_metrics as plot_uncertainty_metrics


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
    desc = "Evaluate an ensemble of networks on a toy dataset."
    parser = argparse.ArgumentParser(description=desc)

    # mode
    parser.add_argument(
        "--method",
        choices=["HMC", "regular", "dropout", "np_dropout", "dropconnect"],
        default="regular",
    )

    # data
    parser.add_argument(
        "--num_data_train", type=int, default=2000, help="Number of train data points"
    )
    parser.add_argument(
        "--num_data_eval", type=int, default=2000, help="Number of eval data points"
    )
    parser.add_argument(
        "--eval_scaling_factors",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        help="Eval scaling factors",
    )

    # model
    parser.add_argument("--input_dim", type=int, default=2, help="Input data")
    parser.add_argument("--num_classes", type=int, default=2, help="Output classes")
    parser.add_argument(
        "--hidden_dim", type=int, default=10, help="Number of hidden neurons (per layer)"
    )
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help='["cuda", "cpu"]')
    parser.add_argument("--drop_rate", type=float, default=0, help="Dropout rate")

    # evaluation
    parser.add_argument(
        "--num_samples_pass",
        type=int,
        default=1,
        help="Number of stochastic forward passes per net",
    )
    parser.add_argument(
        "--num_samples_ens", type=int, default=1024, help="Number of nets in the ensemble"
    )
    parser.add_argument("--num_samples_HMC", type=int, default=1024, help="Number of HMC samples")
    parser.add_argument("--eval_batch", type=int, default=256, help="Eval batch size")

    # reload softmax probabilities?
    parser.add_argument(
        "--compute_save_softmax_probs",
        type=str2bool,
        default=True,
        help="Compute and save the softmax probs?",
    )

    # plot
    parser.add_argument(
        "--plot_scaling_factors",
        nargs="+",
        default=None,
        help="Plot scaling factors (for the error to HMC plot)",
    )
    parser.add_argument(
        "--plot_scaling_factor",
        type=int,
        default=None,
        help="Plot scaling factor (for the raw entropy and mutual info plots)",
    )
    parser.add_argument(
        "--eval_folder_HMC",
        type=str,
        default=None,
        help="Folder for loading HMC eval results when plotting the error to HMC plot",
    )

    # general folders
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prefix", type=str, default="toy")

    # data folder
    parser.add_argument("--data_folder", type=str, default="datasets/", help="The data folder")

    # train folder
    parser.add_argument(
        "--train_folder", type=str, default=None, help="Folder for saving trained networks"
    )

    # eval folder
    parser.add_argument(
        "--eval_folder", type=str, default=None, help="Folder for saving eval results"
    )

    args = parser.parse_args()

    if args.train_folder is None:
        args.train_folder = f"train_toy_{args.method}"

    if args.eval_folder is None:
        args.eval_folder = f"eval_{args.method}"

    # for plotting only
    if args.eval_folder_HMC is None:
        args.eval_folder_HMC = f"eval_HMC"

    return args


# MAIN METHOD
def main():
    """
    MAIN PROCEDURE
    """

    # Parse the arguments
    args = parse_args()

    # random seed
    seed_index = torch.randint(1000, (1,)).item()
    args.seed = seed_index
    print("SEED: ", args.seed)

    # create the output folder
    model_folder = args.output_dir
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    ############################################# RUN #############################################

    # softmax probs
    if args.compute_save_softmax_probs:
        if args.method == "HMC":
            eval_HMC.save_all_softmax_probs_HMC(args)
        else:
            eval_all.save_all_softmax_probs(args)

    # creates plots
    if args.plot_scaling_factors != None:
        plot_uncertainty_metrics.compute_and_save_abs_mean_error_to_HMC(
            args, args.plot_scaling_factors
        )

    if args.plot_scaling_factor != None:
        plot_uncertainty_metrics.compute_and_plot_entropy_mutual_info(
            args, args.plot_scaling_factor
        )


# RUN
if __name__ == "__main__":
    main()
