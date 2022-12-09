"""
	TRAIN AN ENSEMBLE
"""

# torch
import torch
import torch.nn.functional as F
import os
import json
import pathlib

# import modules
from data.datasets import ToyDataset
import stochastic_methods.model as model

# save output
import wandb


# train torch ens
def train_ens(args, wandb_config):

    # torch random seed
    torch.manual_seed(args.seed)

    # create the datasets
    data_dir = os.path.join(args.output_dir, args.data_folder)
    train_dataset = ToyDataset(
        data_dir, num_data_train=args.num_data_train, num_data_eval=args.num_data_eval, mode="train"
    )

    # loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True
    )

    # device
    device = args.device

    # create the output folder
    model_folder = os.path.join(args.output_dir, args.train_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    args_file = pathlib.Path(os.path.join(model_folder, "args.json"))
    with open(args_file, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    for model_id in range(args.num_nets):
        wandb_run = wandb.init(**wandb_config, reinit=True)

        net = model.train_create_net(args)
        net.to(device)
        net.train()

        # training
        learning_rate = args.lr
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        for iepoch in range(args.total_epochs):

            # simple learning rate scheduler
            if iepoch == round(args.total_epochs / 2):
                learning_rate *= 0.1
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

            if iepoch == round(3 * args.total_epochs / 4):
                learning_rate *= 0.1
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

            # train the net
            train_loss = 0
            train_loss_class = 0
            train_loss_prior = 0
            for (indices, x, y) in train_loader:

                x = x.to(device)
                y = y.to(device)

                outputs = net(x)
                loss_class = F.nll_loss(F.log_softmax(outputs, dim=1), y)

                loss_prior = 0.0
                for name, layer in net.named_modules():
                    if isinstance(layer, torch.nn.Linear):

                        # L2 factor from the variational inference analysis
                        l2_dr = l2_drop_rate(args, name)

                        # weight
                        a = 0.5 * l2_dr / len(train_dataset)
                        loss_prior += a * torch.sum(torch.pow(layer.weight, 2))

                        # bias
                        b = 0.5 * l2_dr / len(train_dataset)
                        loss_prior += b * torch.sum(torch.pow(layer.bias, 2))

                loss = loss_class + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_loss_class += loss_class.item()
                train_loss_prior += loss_prior.item()

            wandb.log(
                dict(
                    loss_prior=train_loss_prior,
                    loss_class=loss_class,
                    loss=train_loss,
                    learning_rate=learning_rate,
                    epoch=iepoch,
                )
            )

        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"model_id_{model_id}.pt")).absolute()
        )
        print(f"Saving model at {model_path}")
        torch.save(net.state_dict(), model_path)
        print("Net: ", model_id, " Loss: ", train_loss)
        wandb_run.finish()


def l2_drop_rate(args, layer_name):

    if args.method == "regular":
        return 1

    if args.method == "dropout":
        if layer_name != "output":
            return 1 - args.drop_rate
        else:
            return 1

    if args.method == "np_dropout":
        if layer_name != "output":
            return 0.5
        else:
            return 1

    if args.method == "dropconnect":
        if layer_name != "output":
            return 1 - args.drop_rate
        else:
            return 1
