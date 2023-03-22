"""
	EVAL THE MODEL (TORCH)
"""

# torch
import torch
import torch.nn.functional as F

# modules
from data.datasets import ToyDataset
import stochastic_methods.model as model

# other
import os
import argparse
import json
import pathlib
import copy

# run over one test dataset and save softmax outputs
def save_softmax_probs(args):

    # in-domain [-1, 1]**2 or out-of-domain [-10, 10]**2
    if args.domain == "in":
        sf = 1
    else:
        sf = 10

    one_side = sf * torch.arange(-1, 1, 2/args.size_grid)
    random_x = torch.cartesian_prod(one_side, one_side)
    random_y = torch.zeros(args.size_grid**2).long()

    # dataset
    eval_dataset = ToyDataset(random_x, random_y)

    # loaders
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch, shuffle=False
    )
    
    # device
    device = args.device


    with torch.no_grad():

        nets = load_nets(args)
        softmax_probs = torch.zeros(len(nets) * args.num_samples_pass, len(eval_dataset), args.num_classes).to(device)
        data = torch.zeros(len(eval_dataset), 2).to(device)
        for (indices, x, y) in eval_loader:

            # batch
            x, y = x.to(device), y.to(device)

            # outputs
            softmax_probs_batch = torch.zeros(len(nets) * args.num_samples_pass, len(indices), args.num_classes).to(device)
            for model_id in range(args.num_samples_ens):

                # do multiple stochastic forward passes and avarage
                for i_pass in range(args.num_samples_pass):
                    logits = nets[model_id](x)
                    softmax_probs_batch[i_pass + model_id * args.num_samples_pass, :, :] = F.softmax(logits, dim=1)

            softmax_probs[:, indices, :] = softmax_probs_batch
            data[indices, :] = x

    # save softmax_probs
    model_folder = create_folder(args, "eval_" + args.method)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    softmax_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"softmax_probs_{args.method}_domain_{args.domain}_case_{args.case}.pt")
        ).absolute()
    )
    print(f"Saving softmax at {softmax_path}")
    torch.save(softmax_probs, softmax_path)

    # save data
    data_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"data_{args.method}_domain_{args.domain}_case_{args.case}.pt")
        ).absolute()
    )
    torch.save(data, data_path)


# load nets
def load_nets(args):

    if args.method != "multiswag":
        nets = []
        model_folder = create_folder(args, args.method)
        for model_id in range(args.num_samples_ens):

            net = model.train_create_net(args)
            model_path = os.path.join(model_folder, f"model_id_{model_id}.pt")
            net.load_state_dict(torch.load(model_path))
            net.to(args.device)
            net.train()
            nets.append(net)

        return nets
    else:
        return load_multiswag_nets(args)



# load nets
def load_multiswag_nets(args):

    nets = []
    model_folder = create_folder(args, args.method)
    for model_id in range(args.num_samples_ens):

        model_path = os.path.join(model_folder, f"sum_model_{model_id}.pt")
        sum_state_dict = torch.load(model_path)

        model_path = os.path.join(model_folder, f"sum_sq_model_{model_id}.pt")
        sum_square_state_dict = torch.load(model_path)

        for i_swag in range(args.num_swag_samples):
            
            # swag sample
            swag_sample = sample_swag_model(sum_state_dict, sum_square_state_dict, args.num_swa_models)

            net = model.train_create_net(args)
            net.load_state_dict(swag_sample)
            net.to(args.device)
            net.train()
            nets.append(net)

    return nets


def sample_swag_model(sum_state_dict, sum_square_state_dict, num_swa_models, var_clamp=1e-16):

    with torch.no_grad():
        res = copy.deepcopy(sum_state_dict)
        layer_names = list(sum_state_dict.keys())
        for layer_name in layer_names:

            var = sum_square_state_dict[layer_name]/num_swa_models - (sum_state_dict[layer_name]/num_swa_models)**2
            var = torch.clamp(var, var_clamp)
            res[layer_name] = torch.normal(
                sum_state_dict[layer_name]/num_swa_models, # mean
                torch.sqrt(var), # std
            )

    return res


def create_folder(args, method):

    # create the output folder
    model_folder = os.path.join(args.output_dir, f"{args.case}", method)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    args_file = pathlib.Path(os.path.join(model_folder, "args.json"))
    with open(args_file, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    return model_folder
