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

# load nets
def load_nets(args):

    model_folder = os.path.join(args.output_dir, args.train_folder)
    nets = []
    for model_id in range(args.num_samples_ens):

        net = model.train_create_net(args)
        net.to(args.device)
        net.train()

        model_path = os.path.join(model_folder, f"model_id_{model_id}.pt")
        net.load_state_dict(torch.load(model_path))
        nets.append(net)

    return nets


# run over the entire dataset and return softmax outputs
def get_softmax_probs(args, nets, eval_dataset):

    # loaders
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch, shuffle=False
    )

    # device
    device = args.device

    # loop over the evaluation dataset
    softmax_probs = torch.zeros(
        len(nets) * args.num_samples_pass, len(eval_dataset), args.num_classes
    ).to(device)
    data = torch.zeros(len(eval_dataset), 2).to(device)
    with torch.no_grad():
        for (indices, x, y) in eval_loader:

            # batch
            x, y = x.to(device), y.to(device)

            # outputs
            softmax_probs_batch = torch.zeros(
                len(nets) * args.num_samples_pass, len(indices), args.num_classes
            ).to(device)
            for model_id in range(args.num_samples_ens):

                # do multiple stochastic forward passes and avarage
                for i_pass in range(args.num_samples_pass):
                    logits = nets[model_id](x)
                    softmax_probs_batch[
                        i_pass + model_id * args.num_samples_pass, :, :
                    ] = F.softmax(logits, dim=1)

            softmax_probs[:, indices, :] = softmax_probs_batch
            data[indices, :] = x

    return data, softmax_probs


# loop over the test datasets labeled by the scaling_factor variable and save softmax outputs
def save_all_softmax_probs(args):
    scaling_factors = [float(f) for f in args.eval_scaling_factors]
    eval_folder = os.path.join(args.output_dir, args.eval_folder)
    os.makedirs(eval_folder, exist_ok=True)
    eval_folder_arg = os.path.join(eval_folder, "args.json")
    open(eval_folder_arg, "w").write(json.dumps(args.__dict__, indent=2))
    for scaling_factor in scaling_factors:
        args_with_sf = dict(scaling_factor=scaling_factor, **args.__dict__)
        namespace_with_sf = argparse.Namespace(**args_with_sf)
        print(f"Eval torch {args.method} with scaling factor {scaling_factor}...")
        save_softmax_probs(namespace_with_sf)


# save_softmax_probs
def save_softmax_probs(args):

    # load the datasets
    data_dir = os.path.join(args.output_dir, args.data_folder)
    eval_dataset = ToyDataset(
        data_dir,
        num_data_train=args.num_data_train,
        num_data_eval=args.num_data_eval,
        mode="eval",
        scaling_factor=args.scaling_factor,
    )

    # load nets
    nets = load_nets(args)

    # load softmax outputs
    data, softmax_probs = get_softmax_probs(args, nets, eval_dataset)

    # save softmax_probs
    model_folder = os.path.join(args.output_dir, args.eval_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    softmax_path = str(
        pathlib.Path(
            os.path.join(
                model_folder,
                f"softmax_probs_{eval_dataset.get_identifier()}_num_pass_{args.num_samples_pass}.pt",
            )
        ).absolute()
    )
    print(f"Saving softmax at {softmax_path}")
    torch.save(softmax_probs, softmax_path)

    # save data
    torch.save(data, os.path.join(model_folder, f"data_{eval_dataset.get_identifier()}.pt"))

    with open(os.path.join(model_folder, f"args_{eval_dataset.get_identifier()}.json"), "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))
