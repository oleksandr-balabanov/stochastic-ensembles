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
from data.data import load_train_data
from data.datasets import ToyDataset
import stochastic_methods.model as model

# save output
import wandb
import copy


# train torch ens
def train_ens(args, wandb_config):

    # torch random seed
    torch.manual_seed(args.seed)


    # create the dataset
    x_train, y_train = load_train_data(args)
    train_dataset = ToyDataset(x_train, y_train)

    # loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True
    )

    # device
    device = args.device

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
            train_loss, train_loss_class, train_loss_prior = train_one_epoch(net, train_dataset, train_loader, optimizer, args)

            wandb.log(
                dict(
                    loss_prior=train_loss_prior,
                    loss_class=train_loss_class,
                    loss=train_loss,
                    learning_rate=learning_rate,
                    epoch=iepoch,
                )
            )

        # save model
        model_folder = create_folder(args, args.method)
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"model_id_{model_id}.pt")).absolute()
        )
        print(f"Saving model at {model_path}")
        torch.save(net.state_dict(), model_path)

        print("Net: ", model_id, " Loss: ", train_loss)
        wandb_run.finish()


# train multiswag
def train_multiswag(args, wandb_config):

    # load models
    nets = load_nets(args):

    # train multiswag
    for net in nets:
        
        # init params
        num_swa_models = 1
        sum_state_dict = copy.deepcopy(net.state_dict())
        sum_square_state_dict = copy.deepcopy(net.state_dict())
        layer_names = list(sum_state_dict.keys())
        for layer_name in layer_names:
            sum_square_state_dict[layer_name] = torch.square(sum_state_dict[layer_name])

        # train swag
        for iepoch in range(args.swa_swag_total_epochs):

            # swa-cycle learning rate scheduler
            t = ( iepoch % args.swa_swag_cycle_epochs)/args.swa_swag_cycle_epochs
            learning_rate = (1-t)*args.swa_swag_lr1 + t*args.swa_swag_lr2
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

            # train
            train_loss, train_loss_class, train_loss_prior = train_one_epoch(net, train_dataset, train_loader, optimizer, args)
            wandb.log(
                dict(
                    loss_prior=train_loss_prior,
                    loss_class=train_loss_class,
                    loss=train_loss,
                    learning_rate=learning_rate,
                    epoch=iepoch,
                )
            )

            # update swa weights
            if t == 0.0:
                update_swa_params(net, sum_state_dict,  sum_square_state_dict)
                num_swa_models += 1

        # save swa model
        args.num_swa_models = num_swa_models
        model_folder = create_folder(args, "multiswa")
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"model_id_{model_id}.pt")).absolute()
        )
        print(f"Saving swa model at {model_path}")
        torch.save(get_mean_model(sum_state_dict, num_swa_models), model_path)

        # save swag model
        model_folder = create_folder(args, "multiswag")
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"sum_sq_model_{model_id}.pt")).absolute()
        )
        torch.save(sum_square_state_dict, model_path)

        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"sum_model_{model_id}.pt")).absolute()
        )
        torch.save(sum_state_dict, model_path)

        print("Net: ", model_id, " Loss: ", train_loss)
        wandb_run.finish()

# load nets
def load_nets(args):

    nets = []
    model_folder = create_folder(args, "regular")
    for model_id in range(args.num_samples_ens):

        net = model.train_create_net(args)
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"model_id_{model_id}.pt")).absolute()
        )
        net.load_state_dict(torch.load(model_path))
        net.to(args.device)
        net.train()
        nets.append(net)

    return nets


def create_folder(args, method):

    # create the output folder
    model_folder = os.path.join(args.output_dir, f"{args.case}", method)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    args_file = pathlib.Path(os.path.join(model_folder, "args.json"))
    with open(args_file, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    return model_folder

def update_swa_params(net, sum_state_dict, sum_square_state_dict):

    with torch.no_grad():
        layer_names = list(sum_state_dict.keys())
        net_state_dict = net.state_dict()
        for layer_name in layer_names:
            # add
            sum_state_dict[layer_name] += net_state_dict[layer_name]
            sum_square_state_dict[layer_name] += torch.square(net_state_dict[layer_name])

def get_mean_model(sum_state_dict, num_swa_models):

    with torch.no_grad():
        res = copy.deepcopy(sum_state_dict)
        layer_names = list(sum_state_dict.keys())
        for layer_name in layer_names:
            res[layer_name] = sum_state_dict[layer_name]/num_swa_models

    return res      



def train_one_epoch(net, train_dataset, train_loader, optimizer, args):

    # train the net
    train_loss = 0
    train_loss_class = 0
    train_loss_prior = 0
    device = args.device
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

        return train_loss, train_loss_class, train_loss_prior

def l2_drop_rate(args, layer_name):

    if (args.method == "regular" or args.method == "multiswa" or args.method == "multiswag"):
        return 1

    if args.method == "dropout":
        if layer_name != "output" and layer_name != "input":
            return 1 - args.drop_rate
        else:
            return 1

    if args.method == "np_dropout":
        return 0.5

    if args.method == "dropconnect":
        if layer_name != "output" and layer_name != "input":
            return 1 - args.drop_rate
        else:
            return 1

