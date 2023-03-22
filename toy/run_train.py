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
import stochastic_methods.train as train


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
        choices=["HMC", "regular", "dropout", "np_dropout", "dropconnect", "multiswa", "multiswag"],
        default="HMC",
    )
    
    # data
    parser.add_argument(
        "--case", type=int, default=1, help="The data case number"
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
    parser.add_argument("--number_chains", type=int, default=4, help="Parallel HMC chains")
    parser.add_argument("--step_size", type=float, default=1, help="Step size of NUTS")
    parser.add_argument("--target_accept_prob", type=float, default=0.8, help="Target acceptance probability of step size adaptation scheme")
    parser.add_argument("--adapt_step_size", type=str2bool, default=True, help="A flag to decide if we want to adapt step_size during warm-up phase using Dual Averaging scheme")

    # ens
    parser.add_argument("--num_nets", type=int, default=1024, help="Number of nets in the ensemble")

    # training
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--train_batch", type=int, default=100, help="Training batch size")
    parser.add_argument("--total_epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle the data")
    parser.add_argument("--drop_rate", type=float, default=0, help="Dropout rate")
    parser.add_argument("--lambda_prior", type=float, default=1.0, help="lambda prior value")

    # multiswa and multiswag
    parser.add_argument("--swa_swag_lr1", type=float, default=0.001, help="Top learning rate for swa and swag")
    parser.add_argument("--swa_swag_lr2", type=float, default=0.001, help="Bottom learning rate for swa and swag")
    parser.add_argument("--swa_swag_cycle_epochs", type=int, default=1, help="Number of epochs per swa(g) learning cycle")
    parser.add_argument("--swa_swag_total_epochs", type=int, default=2000, help="Total number of epochs for swa(g) protocol")

    # general folders
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prefix", type=str, default="toy")

    # data folder
    parser.add_argument("--data_folder", type=str, default="./train_datasets", help="Data folder")

    # debug mode?
    parser.add_argument("--fast", action="store_true", help="Minimal run to test code")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable WandB logging")

    args = parser.parse_args()

    if args.fast:
        args.num_data_train = 10
        args.num_data_eval = 10
        args.eval_scaling_factors = ["1", "2"]

        args.number_samples = 10
        args.warmup_steps = 1
        args.number_chains = 4

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
    
    # multiswag
    elif args.method == "multiswa" or args.method == "multiswag":
        train.train_multiswag(args, wandb_config)

    # ens    
    else:
        train.train_ens(args, wandb_config)


# RUN
if __name__ == "__main__":
    main()
