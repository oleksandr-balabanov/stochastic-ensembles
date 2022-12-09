"""
	TRAIN AN ENSEMBLE OF NETWORKS (TOY MODEL)
"""

# TORCH
import torch

# other
import sys
import os
import argparse
import wandb
import pathlib

# import modules
import HMC.train as train_HMC
import stochastic_methods.train as train_all


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
    desc = "Train an ensemble of networks on a toy dataset."
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

    # model
    parser.add_argument("--input_dim", type=int, default=2, help="Input data")
    parser.add_argument("--num_classes", type=int, default=2, help="Output classes")
    parser.add_argument(
        "--hidden_dim", type=int, default=10, help="Number of hidden neurons (per layer)"
    )
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help='["cpu", "cuda"]')

    # HMC
    parser.add_argument("--number_samples", type=int, default=2000, help="Number of HMC samples")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--number_chains", type=int, default=1, help="Parallel HMC chains")

    # ens
    parser.add_argument("--num_nets", type=int, default=1024, help="Number of nets in the ensemble")

    # training
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--train_batch", type=int, default=100, help="Training batch size")
    parser.add_argument("--total_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle the data")
    parser.add_argument("--drop_rate", type=float, default=0, help="Dropout rate")
    parser.add_argument("--lambda_prior", type=float, default=1.0, help="lambda prior value")

    # general folders
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prefix", type=str, default="toy")

    # data folder
    parser.add_argument("--data_folder", type=str, default="datasets/", help="Data folder")

    # train folder
    parser.add_argument(
        "--train_folder", type=str, default=None, help="Folder for saving trained networks"
    )

    # debug mode?
    parser.add_argument("--fast", action="store_true", help="Minimal run to test code")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable WandB logging")

    args = parser.parse_args()

    if args.train_folder is None:
        args.train_folder = f"train_toy_{args.method}"

    if args.fast:
        args.num_data_train = 10
        args.num_data_eval = 10
        args.eval_scaling_factors = ["1", "2"]

        args.number_samples = 10
        args.warmup_steps = 1
        args.number_chains = 1

        args.num_nets = 2

        args.train_batch = 5
        args.total_epochs = 2

        args.num_samples_pass = 1
        args.num_samples_ens = 2
        args.num_samples_HMC = 2

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

    # wandb outputs
    group_name = "toy-" + wandb.util.generate_id()
    wandb_config_to_report = dict(
        absolute_output=str(pathlib.Path(args.output_dir).absolute()), **args.__dict__
    )
    wandb_config = dict(
        group=group_name,
        config=wandb_config_to_report,
        dir=os.getenv("WANDB_OUTPUT", "./"),
        mode="disabled" if not args.enable_wandb else "online",
    )
    wandb_run = wandb.init(**wandb_config)
    wandb.log(dict(command=" ".join(sys.argv)))
    wandb.run.log_code(".", name=wandb.run.name)

    ############################################# RUN #############################################

    # HMC
    if args.method == "HMC":
        train_HMC.train_HMC(args)
    # other methods
    else:
        train_all.train_ens(args, wandb_config)


# RUN
if __name__ == "__main__":
    main()
