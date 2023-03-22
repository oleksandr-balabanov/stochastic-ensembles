"""

	EVAL THE HMC MODEL

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


# run over one test dataset and save softmax outputs
def save_softmax_probs_HMC(args):


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
    
    # read HMC data
    model_folder = os.path.join(args.output_dir, f"{args.case}", "HMC")
    model_file_name = str(pathlib.Path(os.path.join(model_folder, f"model_HMC_chains_{args.number_chains}_s_{args.number_samples}_w_{args.warmup_steps}_case_{args.case}.json")).absolute())
    
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
    softmax_probs = torch.zeros(args.total_num_samples_HMC, len(eval_dataset), args.num_classes).to(device)
    data = torch.zeros(len(eval_dataset), 2).to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for inet in range(args.total_num_samples_HMC):

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
    model_folder = os.path.join(args.output_dir, f"{args.case}", "eval_HMC")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    softmax_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"softmax_probs_HMC_domain_{args.domain}_case_{args.case}.pt")
        ).absolute()
    )
    print(f"Saving softmax at {softmax_path}")
    torch.save(softmax_probs, softmax_path)

    # save data
    data_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"data_domain_{args.domain}_case_{args.case}.pt")
        ).absolute()
    )
    torch.save(data, data_path)

    # save args
    args_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"args_domain_{args.domain}_case_{args.case}.json")
        ).absolute()
    )

    with open(args_path, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))
