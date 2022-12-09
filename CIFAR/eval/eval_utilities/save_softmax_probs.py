"""
        EVALUATE THE MODEL      
"""

import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
import json

import data.create_datasets as create_datasets
import data.data_OOD as data_OOD
import data.data_CIFAR10 as data_CIFAR10
import data.data_CIFAR10C as data_CIFAR10C
import data.data_CIFAR100 as data_CIFAR100
import models.create_model as create_model
import utilities.serialize as serialize
from utilities.folders import get_model_folder, get_eval_folder


def save_softmax_probs_CIFAR(args):

    print("CIFAR mode: ", args.cifar_mode)
    print("stochastic method: ", args.method)

    # load nets
    nets = load_nets(args)

    # compute and save the softmax probs
    serialize.save(
        ens_softmax_probs(nets, args),
        get_eval_folder(args) / f"softmax_probs_targets_nn{args.num_nets:02d}.dill",
    )

    if args.cifar_mode == "CIFAR10":
        serialize.save(
            ens_CIFAR10C(nets, args),
            get_eval_folder(args) / f"accuracy_loss_CIFAR10C_nn{args.num_nets:02d}.dill",
        )

    with open(get_eval_folder(args) / f"args_nn{args.num_nets:02d}.json", "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))


def load_nets(args):

    # get the model folder
    model_folder = get_model_folder(args)

    # load
    nets = []
    for inet in range(args.num_nets):
        net = create_model.create_resnet20(args)
        net.load_state_dict(torch.load(model_folder / f"model_{inet}.pt"))
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
    testLoaderCIFAR10 = torch.utils.data.DataLoader(
        test_dataset_CIFAR10, batch_size=args.eval_batch, shuffle=False
    )
    testLoaderCIFAR100 = torch.utils.data.DataLoader(
        test_dataset_CIFAR100, batch_size=args.eval_batch, shuffle=False
    )
    testLoaderSVHN = torch.utils.data.DataLoader(
        test_dataset_SVHN, batch_size=args.eval_batch, shuffle=False
    )

    # CIFAR 10
    softmax_probs_CIFAR10, targets_CIFAR10 = get_softmax_probs(
        nets, args, testLoaderCIFAR10, num_classes
    )
    softmax_probs_targets_CIFAR10 = {
        "softmax_probs": softmax_probs_CIFAR10,
        "targets": targets_CIFAR10,
    }

    # CIFAR 100
    softmax_probs_CIFAR100, targets_CIFAR100 = get_softmax_probs(
        nets, args, testLoaderCIFAR100, num_classes
    )
    softmax_probs_targets_CIFAR100 = {
        "softmax_probs": softmax_probs_CIFAR100,
        "targets": targets_CIFAR100,
    }

    # SVHN
    softmax_probs_SVHN, targets_SVHN = get_softmax_probs(nets, args, testLoaderSVHN, num_classes)
    softmax_probs_targets_SVHN = {
        "softmax_probs": softmax_probs_SVHN,
        "targets": targets_SVHN,
    }

    return {
        "CIFAR10": softmax_probs_targets_CIFAR10,
        "CIFAR100": softmax_probs_targets_CIFAR100,
        "SVHN": softmax_probs_targets_SVHN,
    }


# accuracy and loss evaluated on the corrupted CIFAR10 dataset
def ens_CIFAR10C(nets, args):

    """

    EVALUATE THE PERFORMANCE: CORRUPTED CIFAR10

    Input: nets, args
    Output:
    {
        "accuracy_loss":accuracy_loss_CIFAR10C,
    }

    """

    # selected corruptions
    CORRUPTION_FILES = [
        "fog.npy",
        "zoom_blur.npy",
        "speckle_noise.npy",
        "glass_blur.npy",
        "spatter.npy",
        "shot_noise.npy",
        "defocus_blur.npy",
        "elastic_transform.npy",
        "gaussian_blur.npy",
        "frost.npy",
        "saturate.npy",
        "brightness.npy",
        "gaussian_noise.npy",
        "contrast.npy",
        "impulse_noise.npy",
        "pixelate.npy",
    ]

    # device
    device = args.device

    # CIFAR10C data folder
    data_dir_CIFARC = args.data_dir_CIFARC

    # CIFAR10 test dataset transform
    transform = data_CIFAR10.test_transform(args.do_augmentation_test)

    result_accuracy = torch.zeros(len(CORRUPTION_FILES), 5)
    result_loss = torch.zeros(len(CORRUPTION_FILES), 5)
    for file_count, corruption_file in enumerate(CORRUPTION_FILES):
        for alpha_level in range(5):

            # datasets
            test_data_CIFAR10C = data_CIFAR10C.get_cifar10C(
                data_dir_CIFARC, corruption_file, alpha_level
            )
            test_dataset_CIFAR10C = data_CIFAR10C.create_test_dataset_CIFARC(
                test_data_CIFAR10C, transform
            )

            # loaders
            testLoaderCIFAR10C = torch.utils.data.DataLoader(
                test_dataset_CIFAR10C, batch_size=args.eval_batch, shuffle=False
            )

            # CIFAR 10C with corruption type "corruption_file" and level "alpha_level"
            softmax_probs, targets = get_softmax_probs(
                nets, args, testLoaderCIFAR10C, num_classes=10
            )
            softmax_probs = torch.mean(softmax_probs, 0).to(device)
            targets = targets.to(device)

            # test metrics
            test_loss = F.nll_loss(torch.log(softmax_probs), targets, reduction="mean").item()
            test_prediction = softmax_probs.data.max(1, keepdim=True)[1]
            test_correct = test_prediction.eq(targets.data.view_as(test_prediction)).sum().item()

            print(
                "TEST SET ENSEMBLE: Corruption: {}, Level: {},  Avg. predictive loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    corruption_file,
                    alpha_level,
                    test_loss,
                    test_correct,
                    len(testLoaderCIFAR10C.dataset),
                    100.0 * test_correct / len(testLoaderCIFAR10C.dataset),
                )
            )

            result_accuracy[file_count, alpha_level] = test_correct / len(
                testLoaderCIFAR10C.dataset
            )
            result_loss[file_count, alpha_level] = test_loss

    accuracy_loss_CIFAR10C = {
        "accuracy": result_accuracy,
        "loss": result_loss,
        "CORRUPTION_FILES": CORRUPTION_FILES,
    }

    return {
        "accuracy_loss_CIFAR10C": accuracy_loss_CIFAR10C,
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
        softmax_probs = torch.zeros(len(nets) * args.num_net_passes, size_dataset, num_classes).to(
            device
        )
        targets = torch.zeros(size_dataset).to(device)
        index = 0
        for batch, (data_batch, targets_batch) in enumerate(testLoader):

            # to device
            data_batch, targets_batch = data_batch.to(device), targets_batch.to(device)
            size_batch = data_batch.shape[0]

            # get the outputs
            softmax_probs_batch = torch.zeros(
                len(nets) * args.num_net_passes, size_batch, num_classes
            ).to(device)
            for i_net in range(len(nets)):
                net = nets[i_net]
                for i_pass in range(args.num_net_passes):
                    softmax_probs_batch[i_net * args.num_net_passes + i_pass, :, :] = F.softmax(
                        net(data_batch), dim=1
                    )

            softmax_probs[:, index : index + size_batch, :] = softmax_probs_batch
            targets[index : index + size_batch] = targets_batch
            index = index + size_batch

    return softmax_probs, targets.type(torch.LongTensor)


def init_train_nets(nets, device):
    for net in nets:
        net = net.to(device)
        net.train()
