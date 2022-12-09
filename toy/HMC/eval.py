"""
	EVAL THE MODEL (PYRO)
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

# loop over the test datasets labeled by the scaling_factor variable and save softmax outputs
def save_all_softmax_probs_HMC(args):
    scaling_factors = [float(f) for f in args.eval_scaling_factors]
    eval_folder = os.path.join(args.output_dir, args.eval_folder)
    os.makedirs(eval_folder, exist_ok=True)
    eval_folder_arg = os.path.join(eval_folder, "args.json")
    open(eval_folder_arg, "w").write(json.dumps(args.__dict__, indent=2))
    for scaling_factor in scaling_factors:
        args_with_sf = dict(scaling_factor=scaling_factor, **args.__dict__)
        namespace_with_sf = argparse.Namespace(**args_with_sf)
        print(f"Eval HMC with scaling factor {scaling_factor}...")
        save_softmax_probs_HMC(namespace_with_sf)


# run over one test dataset and save softmax outputs
def save_softmax_probs_HMC(args):

    # create the datasets
    data_dir = os.path.join(args.output_dir, args.data_folder)
    eval_dataset = ToyDataset(
        data_dir,
        num_data_train=args.num_data_train,
        num_data_eval=args.num_data_eval,
        mode="eval",
        scaling_factor=args.scaling_factor,
    )

    # loaders
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch, shuffle=False
    )

    # read HMC data
    model_folder = os.path.join(args.output_dir, args.train_folder)
    model_file_name = os.path.join(model_folder, f"model_HMC.json")
    with open(model_file_name, "r") as model_file:
        samples = json.load(model_file)

    # device
    device = args.device

    # torch network
    net = model.ToyNet(
        num_hidden_layers=args.num_hidden_layers,
        hidden_dim=args.hidden_dim,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
    )
    net.to(device)

    for name, dict_ in samples.items():
        print("the name of the dictionary is ", name)

    # loop over the HMC realizations
    softmax_probs = torch.zeros(args.num_samples_HMC, len(eval_dataset), args.num_classes).to(
        device
    )
    data = torch.zeros(len(eval_dataset), 2).to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for inet in range(args.num_samples_HMC):

            # load layer weight and bias
            for name, layer in net.named_modules():
                if isinstance(layer, torch.nn.Linear):
                    weight = torch.FloatTensor(samples[f"{name}.weight"][inet])
                    bias = torch.FloatTensor(samples[f"{name}.bias"][inet])

                    layer.weight = torch.nn.Parameter(weight.to(device))
                    layer.bias = torch.nn.Parameter(bias.to(device))

            # loop over the batches
            for (indices, x, y) in eval_loader:

                # data
                batch = x.shape[0]
                x, y = x.to(device), y.to(device)

                # output
                logits = net(x)
                softmax_probs_batch = F.softmax(logits, dim=1)

                # add to the main tensor
                softmax_probs[inet, indices, :] = softmax_probs_batch
                data[indices, :] = x

    # save softmax_probs
    model_folder = os.path.join(args.output_dir, args.eval_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    softmax_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"softmax_probs_HMC_{eval_dataset.get_identifier()}.pt")
        ).absolute()
    )
    print(f"Saving softmax at {softmax_path}")
    torch.save(softmax_probs, softmax_path)

    # save data
    torch.save(data, os.path.join(model_folder, f"data_{eval_dataset.get_identifier()}.pt"))

    with open(os.path.join(model_folder, f"args_{eval_dataset.get_identifier()}.json"), "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))
