"""s
	TRAIN MODULE (PYRO)
"""

# torch
import torch
import os
import json
import pathlib

# pyro
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import pyro.poutine as poutine

# import modules
from data.datasets import ToyDataset
import HMC.model as model


# train pyro model with HMC
def train_HMC(args):
    """
    TRAIN PROCEDURE (HMC)
    """

    # pyro
    pyro.enable_validation(True)
    pyro.clear_param_store()

    # pyro random seed
    pyro.set_rng_seed(args.seed)

    # create the datasets
    data_dir = os.path.join(args.output_dir, args.data_folder)
    train_dataset = ToyDataset(
        data_dir, num_data_train=args.num_data_train, num_data_eval=args.num_data_eval, mode="train"
    )

    # loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))

    # device
    device = args.device

    # create the output folder
    model_folder = os.path.join(args.output_dir, args.train_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    args_file = pathlib.Path(os.path.join(model_folder, "args.json"))
    with open(args_file, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    pyro_model = model.ToyNetPyroPosterior(
        num_hidden_layers=args.num_hidden_layers,
        hidden_dim=args.hidden_dim,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        device=args.device,
    )

    # posterior parameters and samples
    print("--------------TRUE POSTERIOR-------------------------")
    (indices, x, y) = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    with poutine.trace() as tr:
        pyro_model(x)
    for site in tr.trace.nodes.values():
        print(site["type"], site["name"], site["value"].shape)

    # NUTS(HMC)
    nuts_kernel = NUTS(
        pyro_model,
        jit_compile=False,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=args.number_samples,
        warmup_steps=args.warmup_steps,
        num_chains=args.number_chains,
        mp_context="spawn",
    )
    mcmc.run(x, y)

    # get samples
    samples = mcmc.get_samples()

    # save to file
    serialisable = {}
    for k, v in samples.items():
        print(v.shape)
        serialisable[k] = v.tolist()

    model_file_name = str(pathlib.Path(os.path.join(model_folder, f"model_HMC.json")).absolute())
    print(f"Saving HMC models at {model_file_name}")
    with open(model_file_name, "w") as model_file:
        json.dump(serialisable, model_file)
