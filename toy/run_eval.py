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
import stochastic_methods.eval as eval
#import plot.plot_uncertainty_metrics as plot_uncertainty_metrics


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
        choices=["HMC", "regular", "dropout", "np_dropout", "dropconnect", "multiswa", "multiswag"],
        default="regular",
    )

    # data
    parser.add_argument(
        "--case", type=int, default=1, help="The data case number"
    )

    parser.add_argument(
        "--size_grid", type=int, default=100, help="Number of eval grid points per side"
    )

    parser.add_argument(
        "--domain", choices=["in", "out"], default="in", help="In-domain [-1, 1]**2 or out-of-domain [-10, 10]**2 test data"
    )


    # model
    parser.add_argument("--input_dim", type=int, default=2, help="Input data")
    parser.add_argument("--num_classes", type=int, default=2, help="Output classes")
    parser.add_argument("--hidden_dim", type=int, default=10, help="Number of hidden neurons (per layer)")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help='["cuda", "cpu"]')
    parser.add_argument("--drop_rate", type=float, default=0, help="Dropout rate")

    # ens evaluation
    parser.add_argument("--num_samples_pass",type=int,default=1,help="Number of stochastic forward passes per net")
    parser.add_argument("--num_samples_ens", type=int, default=1024, help="Number of nets in the ensemble")
    parser.add_argument("--eval_batch", type=int, default=256, help="Eval batch size")

    # HMC 
    parser.add_argument("--number_samples", type=int, default=2000, help="Number of HMC samples")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--number_chains", type=int, default=4, help="Parallel HMC chains")
    parser.add_argument("--total_num_samples_HMC", type=int, default=8000, help="Total number of HMC samples, including all chains")

    # multiswag
    parser.add_argument("--num_swa_models",type=int,default=2000,help="Number of models used for calculating Stochastic Weight Average")
    parser.add_argument("--num_swag_samples", type=int, default=20, help="Number of samples created per swag model")

    # reload softmax probabilities?
    parser.add_argument(
        "--compute_save_softmax_probs",
        type=str2bool,
        default=True,
        help="Compute and save the softmax probs?",
    )

    # general folders
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prefix", type=str, default="toy")

    # data folder
    parser.add_argument("--data_folder", type=str, default="datasets/", help="The data folder")

    args = parser.parse_args()

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
            eval_HMC.save_softmax_probs_HMC(args)
        else:
            eval.save_softmax_probs(args)

# RUN
if __name__ == "__main__":
    main()
