"""

TRAIN ENS CIFAR (DETERMINISTIC, DROPOUT, NP_DROPOUT, DROPCONNECT)

"""

import os
import copy
import torch
import torch.nn.functional as F
import json
import wandb

import data.create_datasets as create_datasets
import models.create_model as create_model
from models.model_utilities import FRN
from utilities.folders import get_model_folder


def train_ens_CIFAR(args, wandb_config):

    """
    TRAIN THE NETWORKS

    Input: args, wandb_config
    Output: None

    (saved to torch.save(net.state_dict(),  get_model_folder(args) / f'model_{inet}.pt') )

    """

    # datasets
    train_dataset, valid_dataset, test_dataset = create_datasets.create_datasets(args)

    # Loaders
    trainLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True
    )
    validLoader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.test_batch, shuffle=False
    )
    testLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch, shuffle=False
    )

    model_folder = get_model_folder(args)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    with open(model_folder / "train_args.json", "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))

    # loop over the nets
    for inet in range(args.num_nets):

        wandb_run = wandb.init(**wandb_config, reinit=True)
        # train quantities
        loss_train = []
        accuracy_train = []

        # resnet20
        net = create_model.create_resnet20(args)

        # device
        device = args.device
        net.to(device)

        # create the optimizer
        learning_rate = args.lr
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

        # best valid loss
        best_valid_loss = None

        # loop over the epochs
        for epoch in range(args.total_epochs):

            # learning rate scheduler
            if epoch == round(args.total_epochs / 2):
                learning_rate /= 10
                print("LEARNING RATE: ", learning_rate)
                optimizer = torch.optim.SGD(
                    net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
                )

            if epoch == round(3 * args.total_epochs / 4):
                learning_rate /= 10
                print("LEARNING RATE: ", learning_rate)
                optimizer = torch.optim.SGD(
                    net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
                )

            # training stage
            train_loss = 0
            train_correct = 0
            net.train()

            for batch, (data, targets) in enumerate(trainLoader):
                # to CUDA
                data, targets = data.to(device), targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward the input and get the loss
                outputs = net(data)
                loss_class = F.nll_loss(F.log_softmax(outputs, dim=1), targets)

                # L2 LOSS
                loss_prior = l2_loss(net, args, train_dataset)

                loss = loss_class + args.lambda_prior * loss_prior
                train_loss += loss.item()

                # back propagation
                loss.backward()
                optimizer.step()

                # save statistics
                train_prediction = outputs.data.max(1, keepdim=True)[1]
                train_correct += train_prediction.eq(targets.data.view_as(train_prediction)).sum()

            # save/print the training statistics
            train_loss /= len(trainLoader)
            loss_train.append(train_loss)
            train_accuracy = 100.0 * train_correct / len(trainLoader.dataset)
            accuracy_train.append(train_accuracy)

            # valid
            valid_loss, valid_accuracy = validate_model(args, net, validLoader)

            if (best_valid_loss == None) or (valid_loss < best_valid_loss):
                best_valid_loss = valid_loss
                best_epoch = epoch

            # print
            if epoch % 10 == 0:
                print(
                    "\nTrain set: Avg. total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch: {}".format(
                        train_loss, train_correct, len(trainLoader.dataset), train_accuracy, epoch
                    )
                )

                print(
                    "Valid set: Avg. predictive loss: {:.4f}, Accuracy: {:.0f}%, Best Epoch: {}\n".format(
                        valid_loss, valid_accuracy, best_epoch
                    )
                )

            wandb.log(
                dict(
                    loss_train=train_loss,
                    validation_loss=valid_loss,
                    acc_train=train_accuracy,
                    acc_validation=valid_accuracy,
                    learning_rate=learning_rate,
                    epoch=epoch,
                )
            )

        # create the output folder
        if args.do_augmentation_train == True:
            cifar_mode = args.cifar_mode + "aug"
        else:
            cifar_mode = args.cifar_mode

        # model_folder = args.output_dir + args.save_folder + '/' + cifar_mode + '/' + args.method
        model_folder = get_model_folder(args)

        # save
        torch.save(net.state_dict(), model_folder / f"model_{inet}.pt")

        last_test_acc, last_test_accuracy = test_model(args, net, testLoader)

        wandb.log(
            dict(
                last_test_acc=last_test_acc,
                last_test_accuracy=last_test_accuracy,
            )
        )
        wandb_run.finish()


def l2_loss(net, args, train_dataset):
    loss_prior = 0
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):

            # L2 factor from the variational inference analysis
            l2_dr = l2_drop_rate(args.method, args.drop_rate_conv)

            # weight
            if layer.weight != None:
                a = l2_dr / (2 * len(train_dataset))
                loss_prior += a * torch.sum(torch.pow(layer.weight, 2))

            # bias
            if layer.bias != None:
                b = l2_dr / (2 * len(train_dataset))
                loss_prior += b * torch.sum(torch.pow(layer.bias, 2))

        if isinstance(layer, FRN):

            # L2 factor from the variational inference analysis (stochastic only for np_dropout)
            if args.method == "np_dropout":
                l2_dr = 0.5
            else:
                l2_dr = 1

            # weight
            if layer.weight != None:
                a = l2_dr / (2 * len(train_dataset))
                loss_prior += a * torch.sum(torch.pow(layer.weight, 2))

            # bias
            if layer.bias != None:
                b = l2_dr / (2 * len(train_dataset))
                loss_prior += b * torch.sum(torch.pow(layer.bias, 2))

            # tau
            if layer.tau != None:
                c = l2_dr / (2 * len(train_dataset))
                loss_prior += c * torch.sum(torch.pow(layer.tau, 2))

        if isinstance(layer, torch.nn.Linear):

            # there is only one linear layer and it is the output layer
            l2_dr = l2_drop_rate(args.method, args.drop_rate_linear)

            # weight
            if layer.weight != None:
                a = l2_dr / (2 * len(train_dataset))
                loss_prior += a * torch.sum(torch.pow(layer.weight, 2))

            # bias
            if layer.bias != None:
                b = l2_dr / (2 * len(train_dataset))
                loss_prior += b * torch.sum(torch.pow(layer.bias, 2))

    return loss_prior


def validate_model(args, net, data_loader):
    valid_loss = 0
    valid_correct = 0
    device = args.device
    with torch.no_grad():
        net.eval()
        for batch, (data, targets) in enumerate(data_loader):

            # to CUDA
            data, targets = data.to(device), targets.to(device)

            # get the outputs
            outputs = net(data)

            valid_loss += F.nll_loss(F.log_softmax(outputs, dim=1), targets).item()
            valid_prediction = outputs.data.max(1, keepdim=True)[1]
            valid_correct += (
                valid_prediction.eq(targets.data.view_as(valid_prediction)).sum().item()
            )

        # the validation statistics
        valid_loss /= len(data_loader)

    valid_accuracy = 100.0 * valid_correct / len(data_loader.dataset)

    return valid_loss, valid_accuracy


def test_model(args, net, data_loader):
    # test (last model)
    test_loss = 0
    test_correct = 0

    with torch.no_grad():

        # device
        device = args.device
        net = net.to(device)
        net.eval()

        for batch, (data, targets) in enumerate(data_loader):

            # to device
            data, targets = data.to(device), targets.to(device)

            # get the outputs
            outputs = net(data)

            test_loss += F.nll_loss(F.log_softmax(outputs, dim=1), targets).item()
            test_prediction = outputs.data.max(1, keepdim=True)[1]
            test_correct += test_prediction.eq(targets.data.view_as(test_prediction)).sum().item()

        # save/print the test statistics
        test_loss /= len(data_loader)

        test_accuracy = 100.0 * test_correct / len(data_loader.dataset)

        print(
            "\nTest set (last): Avg. predictive loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss, test_correct, len(data_loader.dataset), test_accuracy
            )
        )

        return test_loss, test_accuracy


# constant for l2 regularization
def l2_drop_rate(method, drop_rate):

    if method == "regular":
        return 1

    if method == "dropout":
        return 1 - drop_rate

    if method == "np_dropout":
        return 0.5

    if method == "dropconnect":
        return 1 - drop_rate
