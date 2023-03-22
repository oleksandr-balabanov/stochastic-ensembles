"""
        EVALUATE THE MODEL      
"""

import os
import copy
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
import pathlib
import json

import data.create_datasets as create_datasets
import data.data_OOD as data_OOD
import data.data_CIFAR10 as data_CIFAR10
import data.data_CIFAR10C as data_CIFAR10C
import data.data_CIFAR100 as data_CIFAR100
import data.data_CIFAR100C as data_CIFAR100C
import models.create_model as create_model
import utilities.serialize as serialize
from utilities.folders import get_model_folder, get_eval_folder


def save_softmax_probs_CIFAR(args):

    print("CIFAR mode: ", args.cifar_mode)
    print("stochastic method: ", args.method)

    # load nets
    nets = load_nets(args)

    # compute and save the softmax probs for CIFAR and SVHN
    serialize.save(ens_softmax_probs(nets, args), get_eval_folder(args) / f"softmax_probs_targets_nn{args.num_nets:02d}.dill")

    # compute and save the softmax probs for the corrupted CIFAR
    serialize.save(ens_softmax_probs_CIFARC(nets, args), get_eval_folder(args) / f"softmax_probs_targets_nn{args.num_nets:02d}_{args.cifar_mode}C.dill")

    with open(get_eval_folder(args) / f"args_nn{args.num_nets:02d}.json", "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    
def load_nets(args):
    
    # get the model folder
    model_folder = get_model_folder(args)

    # load
    nets=[]
    for inet in range(args.num_nets):
        net = create_model.create_resnet20(args)
        net.load_state_dict(torch.load(model_folder / f'model_{inet}.pt'))
        net.train()
        nets.append(net)

    return nets



# output softmax probs associated with CIFAR10, CIFAR100, SVHN sets
def ens_softmax_probs(nets, args):
    
    """

        EVALUATE THE PERFORMANCE: get softmax_probs obtained over the entire datasets. 
        Input: nets, args
        Output: {
            "CIFAR10":softmax_probs_targets_CIFAR10,
            "CIFAR100":softmax_probs_targets_CIFAR100,
            "SVHN":softmax_probs_targets_SVHN,
        }

    """

    # datasets
    if args.cifar_mode == "CIFAR10":
        num_classes = 10
        transform = data_CIFAR10.test_transform(args.do_augmentation_test)
    else:
        num_classes = 100
        transform = data_CIFAR100.test_transform(args.do_augmentation_test)
    
    # SVHN torchvision and CIFAR data folders
    data_dir_SVHN = args.data_dir_SVHN
    data_dir_CIFAR = args.data_dir_CIFAR

    # datasets
    test_dataset_SVHN = data_OOD.create_test_dataset_SVHN(data_dir_SVHN, transform)

    train_data_CIFAR10, test_data_CIFAR10 = data_OOD.get_cifar10(data_dir_CIFAR)
    test_dataset_CIFAR10 = data_OOD.create_test_dataset_CIFAR(test_data_CIFAR10, transform)

    train_data_CIFAR100, test_data_CIFAR100 = data_OOD.get_cifar100(data_dir_CIFAR)
    test_dataset_CIFAR100 = data_OOD.create_test_dataset_CIFAR(test_data_CIFAR100, transform)

    # Loaders
    testLoaderCIFAR10 = torch.utils.data.DataLoader(test_dataset_CIFAR10, batch_size=args.eval_batch, shuffle=False)
    testLoaderCIFAR100 = torch.utils.data.DataLoader(test_dataset_CIFAR100, batch_size=args.eval_batch, shuffle=False)
    testLoaderSVHN = torch.utils.data.DataLoader(test_dataset_SVHN, batch_size=args.eval_batch, shuffle=False)

    # CIFAR 10
    softmax_probs_CIFAR10, targets_CIFAR10 = get_softmax_probs(nets, args, testLoaderCIFAR10, num_classes)
    softmax_probs_CIFAR10 =  torch.mean(softmax_probs_CIFAR10, 0)
    softmax_probs_targets_CIFAR10 = {
        "softmax_probs":softmax_probs_CIFAR10, 
        "targets":targets_CIFAR10,
    }

    # CIFAR 100
    softmax_probs_CIFAR100, targets_CIFAR100 = get_softmax_probs(nets, args, testLoaderCIFAR100, num_classes)
    softmax_probs_CIFAR100 =  torch.mean(softmax_probs_CIFAR100, 0)
    softmax_probs_targets_CIFAR100 = {
        "softmax_probs":softmax_probs_CIFAR100, 
        "targets":targets_CIFAR100,
    }

    # SVHN
    softmax_probs_SVHN, targets_SVHN = get_softmax_probs(nets, args, testLoaderSVHN, num_classes)
    softmax_probs_SVHN =  torch.mean(softmax_probs_SVHN, 0)
    softmax_probs_targets_SVHN = {
        "softmax_probs":softmax_probs_SVHN, 
        "targets":targets_SVHN,
    }

    return {
        "CIFAR10":softmax_probs_targets_CIFAR10,
        "CIFAR100":softmax_probs_targets_CIFAR100,
        "SVHN":softmax_probs_targets_SVHN,
    }



# mean softmaxprobs evaluated on the corrupted CIFAR dataset
def ens_softmax_probs_CIFARC(nets, args):
    
    """

        EVALUATE THE PERFORMANCE: CORRUPTED CIFAR10 or CIFAR100

        Input: nets, args
        Output: 
        {
            "softmax_probs_targets_CIFARC":softmax_probs_targets_CIFARC,
        }

    """

    # selected corruptions
    CORRUPTION_FILES = [
            'fog.npy',
            'zoom_blur.npy',
            'speckle_noise.npy',
            'glass_blur.npy',
            'spatter.npy',
            'shot_noise.npy',
            'defocus_blur.npy',
            'elastic_transform.npy',
            'gaussian_blur.npy',
            'frost.npy',
            'saturate.npy',
            'brightness.npy',
            'gaussian_noise.npy',
            'contrast.npy',
            'impulse_noise.npy',
            'pixelate.npy'
    ]

    # device
    device = args.device

    # CIFARC data folder
    data_dir_CIFARC = args.data_dir_CIFARC

    result_accuracy = torch.zeros(len(CORRUPTION_FILES), 5)
    result_loss = torch.zeros(len(CORRUPTION_FILES), 5)
    softmax_probs_targets_CIFARC = {}
    for file_count, corruption_file in enumerate(CORRUPTION_FILES):
        softmax_probs_targets_CIFARC_single_corruption = {}
        for alpha_level in range(5):

            print(f"Corruption [{file_count}:{len(CORRUPTION_FILES)}] {corruption_file}, level {alpha_level}")
            
            # datasets
            if args.cifar_mode == "CIFAR10":
                test_data_CIFAR10C = data_CIFAR10C.get_cifar10C(data_dir_CIFARC, corruption_file, alpha_level)
                transform = data_CIFAR10.test_transform(args.do_augmentation_test)
                test_dataset_CIFARC = data_CIFAR10C.create_test_dataset_CIFARC(test_data_CIFAR10C, transform)
                num_classes = 10

            if args.cifar_mode == "CIFAR100":
                test_data_CIFAR100C = data_CIFAR100C.get_cifar100C(data_dir_CIFARC, corruption_file, alpha_level)
                transform = data_CIFAR100.test_transform(args.do_augmentation_test)
                test_dataset_CIFARC = data_CIFAR100C.create_test_dataset_CIFARC(test_data_CIFAR100C, transform)
                num_classes = 100

            # loaders
            testLoaderCIFARC = torch.utils.data.DataLoader(test_dataset_CIFARC, batch_size=args.eval_batch, shuffle=False)

            # CIFAR with corruption type "corruption_file" and level "alpha_level"
            softmax_probs, targets = get_softmax_probs(nets, args, testLoaderCIFARC, num_classes = num_classes)
            softmax_probs =  torch.mean(softmax_probs, 0).to(device)
            targets = targets.to(device)

            # add to the dictionary
            softmax_probs_targets = {
                "softmax_probs":softmax_probs,
                "targets":targets
            }
            softmax_probs_targets_CIFARC_single_corruption[alpha_level] = softmax_probs_targets

        softmax_probs_targets_CIFARC[corruption_file] = softmax_probs_targets_CIFARC_single_corruption


    return {
        "softmax_probs_targets_CIFARC":softmax_probs_targets_CIFARC,
    }



def get_softmax_probs(nets, args, testLoader, num_classes):
    """
        Input: nets, args, testLoader, num_classes
        Output: softmax_probs, targets
    """

    with torch.no_grad():

        # device
        device = args.device

        # nets to device and train()
        init_train_nets(nets, device)

        # data sizes
        size_dataset = len(testLoader.dataset)
        size_batch = args.eval_batch

        # total tensor of softmax outputs and targets
        softmax_probs = torch.zeros(len(nets) * args.num_net_passes, size_dataset, num_classes).to(device)
        targets = torch.zeros(size_dataset).to(device)
        index=0
        for batch, (data_batch, targets_batch) in enumerate(testLoader):

            # to device
            data_batch, targets_batch = data_batch.to(device), targets_batch.to(device)
            size_batch = data_batch.shape[0]
            
            # get the outputs
            softmax_probs_batch = torch.zeros(len(nets) * args.num_net_passes, size_batch, num_classes).to(device)
            for i_net in range(len(nets)):
                net = nets[i_net]
                for i_pass in range(args.num_net_passes):
                    softmax_probs_batch[i_net*args.num_net_passes+i_pass, :, :] = F.softmax(net(data_batch), dim = 1)

            softmax_probs[:, index:index+size_batch, :] = softmax_probs_batch
            targets[index:index+size_batch]=targets_batch
            index=index+size_batch

    return softmax_probs, targets.type(torch.LongTensor) 



# load nets
def load_nets(args):

    if args.method != "multiswag":
        nets = []
        model_folder = create_folder(args, args.method)
        for model_id in range(args.num_nets):

            net = create_model.create_resnet20(args)
            model_path = os.path.join(model_folder, f"model_{model_id}.pt")
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
    for model_id in range(args.num_nets):

        model_path = os.path.join(model_folder, f"sum_model_{model_id}.pt")
        sum_state_dict = torch.load(model_path)

        model_path = os.path.join(model_folder, f"sum_sq_model_{model_id}.pt")
        sum_square_state_dict = torch.load(model_path)

        for i_swag in range(args.num_swag_samples):
            
            # swag sample
            swag_sample = sample_swag_model(sum_state_dict, sum_square_state_dict, args.num_swa_models)

            net = create_model.create_resnet20(args)
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
    if args.do_augmentation_train == True:
        cifar_mode = args.cifar_mode + "_aug"
    else:
        cifar_mode = args.cifar_mode

    # create the output folder
    model_folder = os.path.join(args.output_dir, args.train_folder, cifar_mode, method)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    args_file = pathlib.Path(os.path.join(model_folder, "args.json"))
    with open(args_file, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    return model_folder




def init_train_nets(nets, device):
    for net in nets:
        net = net.to(device)
        net.train()