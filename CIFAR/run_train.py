"""
	TRAIN A STOCHASITC ENSEMBLE OF RESNET-20-FRN NETWORKS ON CIFAR10 or CIFAR100
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
import train.train_CIFAR as train_CIFAR

# str to bool type
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_wandb_credentials():
    """Set WandB API credentials for the uncertainty project login
    Typically this would be done by every user using `wandb login` from
    the command line. Since this is a private repo we can add the credentials here
    directly instead.
    """
    os.environ["WANDB_API_KEY"] = "a654547139d950a9595d41177611b7b57f28d289"

# Parsing and configuration
def parse_args():
    desc = "Train an ensamble of stochastic neural networks on the CIFAR dataset"
    parser = argparse.ArgumentParser(description=desc)

    # input
    parser.add_argument('--num_nets', type=int, default=50, help='The number of nets in the ensemble')

    # stochastic method
    parser.add_argument(
        '--method',
        choices=["regular", "dropout", "np_dropout", "dropconnect", "multiswa", "multiswag"], 
        default="regular", 
        help='stochastic method'
    )

    # CIFAR10 or CIFAR100?
    parser.add_argument('--cifar_mode', 
                            choices=["CIFAR10", "CIFAR100"], default="CIFAR10", help="CIFAR10 or CIFAR100?")

    # augmentation?
    parser.add_argument('--do_augmentation_train', type=str2bool, default=False, help='do augmentation train?')
    parser.add_argument('--do_augmentation_test', type=str2bool, default=False, help='do augmentation test?')

    # training
    parser.add_argument('--lr', type=float, default=0.1, help='The learning rate')
    parser.add_argument('--train_batch', type=int, default=64, help='The training batch size')
    parser.add_argument('--test_batch', type=int, default=256, help='The training batch size')
    parser.add_argument('--total_epochs', type=int, default=300, help='Total number of epochs')
    parser.add_argument('--num_train', type=int, default=40960, help='Number of images used for the training')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Shuffle the data')
    parser.add_argument('--device', type=str, default="cuda", help='cpu or cuda?')

    # stochastic parameters
    parser.add_argument('--drop_rate_conv', type=float, default=0, help='Dropout rate for conv layers')
    parser.add_argument('--drop_rate_linear', type=float, default=0, help='Dropout rate for the output linear layer')

    # prior lambda = var^(-1)
    parser.add_argument('--lambda_prior', type=float, default=5, help='The lambda_prior value')


    # multiswa and multiswag
    parser.add_argument("--swa_swag_lr1", type=float, default=0.03, help="Top learning rate for swa and swag")
    parser.add_argument("--swa_swag_lr2", type=float, default=0.03, help="Bottom learning rate for swa and swag")
    parser.add_argument("--swa_swag_cycle_epochs", type=int, default=1, help="Number of epochs per swa(g) learning cycle")
    parser.add_argument("--swa_swag_total_epochs", type=int, default=300, help="Total number of epochs for swa(g) protocol")


    # save to the file
    parser.add_argument('--train_folder', type=str, default="models_best")
    parser.add_argument('--data_dir_CIFAR', type=str, default="/cephyr/NOBACKUP/Datasets", help='folder with CIFAR datasets ...')
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--debug', action="store_true")

    # fast
    parser.add_argument('--fast', type=str2bool, default=False, help='Fast?')
    parser.add_argument('--enable_wandb', type=str2bool, default=False, help='wandb_config?') 

    return parser.parse_args()


# TRAIN METHOD
def train():
    """
        TRAIN A STOCHASITC ENSEMBLE OF RESNET-20-FRN NETWORKS ON CIFAR10 or CIFAR100
    """

    # random seed
    seed_index = torch.randint(1000, (1,))
    torch.manual_seed(seed_index)
    print('SEED: ', seed_index)

    # parse the arguments
    args = parse_args()
    args.seed = seed_index.item()

    if args.fast:
        args.num_train = 256
        args.total_epochs = 10
        args.enable_wandb = False

    # wandb
    set_wandb_credentials()
    group_name = f"{args.method}-" + wandb.util.generate_id()
    wandb_config_to_report = dict(
        absolute_output=str(pathlib.Path(args.output_dir).absolute()),
        **args.__dict__
    )
    wandb_config = dict(
        group=group_name,
        project="CIFAR",
        entity="uncertainty",
        config=wandb_config_to_report,
        dir=os.getenv("WANDB_OUTPUT", "./"),
        mode="disabled" if not args.enable_wandb else "online"
    )
    wandb_run = wandb.init(**wandb_config)
    wandb.log(dict(command=" ".join(sys.argv)))
    wandb.run.log_code(".", name=wandb.run.name)

    # train and save
    # multiswag
    if args.method == "multiswag" or args.method == "multiswa":
        train_CIFAR.train_multiswag_CIFAR(args, wandb_config)
    # ens    
    else:
        train_CIFAR.train_ens_CIFAR(args, wandb_config)
    
# RUN
if __name__ == '__main__':
    train()