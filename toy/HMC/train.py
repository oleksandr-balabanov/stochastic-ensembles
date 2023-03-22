"""

	TRAIN HMC (NUTS)

"""

# torch
import torch
import os
import json
import pathlib
import dill

# pyro
import pyro
from pyro.infer.mcmc import MCMC, NUTS
import pyro.poutine as poutine

# import modules
import HMC.model as model
from data.data import load_train_data
from data.datasets import ToyDataset


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
    x_train, y_train = load_train_data(args)

    # device
    device = args.device

    # create the output folder
    model_folder = os.path.join(args.output_dir, f"{args.case}", "HMC")
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
    x, y = x_train.float().to(device), y_train.long().to(device)
    with poutine.trace() as tr:
        pyro_model(x)
    for site in tr.trace.nodes.values():
        print(site["type"], site["name"], site["value"].shape)

    # NUTS(HMC)
    nuts_kernel = NUTS(
        pyro_model,
        step_size=args.step_size,
        target_accept_prob = args.target_accept_prob,
        adapt_step_size = args.adapt_step_size,
        jit_compile=False,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_samples=args.number_samples,
        warmup_steps=args.warmup_steps,
        num_chains=args.number_chains,
        mp_context="spawn",
        disable_progbar=True,
    )
    mcmc.run(x, y)

    # get samples
    samples = mcmc.get_samples()

    # save to file
    serialisable = {}
    for k, v in samples.items():
        print(v.shape)
        serialisable[k] = v.tolist()

    model_file_name = str(pathlib.Path(os.path.join(model_folder, f"model_HMC_chains_{args.number_chains}_s_{args.number_samples}_w_{args.warmup_steps}_case_{args.case}.json")).absolute())
    print(f"Saving HMC models at {model_file_name}")
    with open(model_file_name, "w") as model_file:
        json.dump(serialisable, model_file)


    # convergence statistics 
    diag = mcmc.diagnostics()
    model_file_name = str(pathlib.Path(os.path.join(model_folder, f"diagnostics_HMC_chains_{args.number_chains}_s_{args.number_samples}_w_{args.warmup_steps}_case_{args.case}.dill")).absolute())
    with open(model_file_name, 'wb') as f:
	    dill.dump(diag, f)


    # save args
    args_path = str(
        pathlib.Path(
            os.path.join(model_folder, f"args_domain_{args.domain}_case_{args.case}.json")
        ).absolute()
    )

    with open(args_path, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))
    
    print("Convergence statistics: ")
    print(diag)

    
