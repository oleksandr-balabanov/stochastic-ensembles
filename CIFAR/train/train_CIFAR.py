"""

TRAIN ENS CIFAR (DETERMINISTIC, DROPOUT, NP_DROPOUT, DROPCONNECT, MULTISWA, MULTISWAG)

"""

import os
import pathlib
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

        (saved to os.path.join(model_folder, f'model_{model_id}.pt') )

    """

    # load nets
    nets = load_nets(args)

    # datasets
    train_dataset, valid_dataset, test_dataset = create_datasets.create_datasets(args)

    # Loaders
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    validLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch, shuffle=False)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    model_folder = create_folder(args, args.method)

    # loop over the nets
    for model_id in range(args.num_nets):

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
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov = True)


        # loop over the epochs
        for epoch in range(args.total_epochs):

            # learning rate scheduler 
            if epoch == round(args.total_epochs/3):
                learning_rate /= 10
                print('LEARNING RATE: ', learning_rate)
                optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov = True)

            if epoch == round(2 * args.total_epochs/3):
                learning_rate /= 10 
                print('LEARNING RATE: ', learning_rate)
                optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov = True)

            # train
            train_loss, train_accuracy, train_correct = train_one_epoch(net, train_dataset, trainLoader, optimizer, args)
            loss_train.append(train_loss)
            accuracy_train.append(train_accuracy)

            # valid
            valid_loss, valid_accuracy = validate_model(args, net, validLoader)

            # print
            if epoch % 10 == 0:
                print('\nTrain set: Avg. total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Epoch: {}'.format(
                    train_loss, train_correct, len(trainLoader.dataset),
                    train_accuracy, epoch)
                    )
                
                print('Valid set: Avg. predictive loss: {:.4f}, Accuracy: {:.0f}%, Epoch: {} \n'.format(
                    valid_loss, valid_accuracy, epoch)
                    )

            wandb.log(dict(
                loss_train=train_loss,
                validation_loss=valid_loss,
                acc_train=train_accuracy,
                acc_validation=valid_accuracy,
                learning_rate=learning_rate,
                epoch=epoch
            ))

        # save
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f'model_{model_id}.pt')).absolute()
        )
        torch.save(net.state_dict(), model_path)

        test_loss, test_accuracy = test_model(args, net, testLoader)
        wandb.log(dict(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
        ))
        wandb_run.finish()

        print('Test set: Avg. predictive loss: {:.4f}, Accuracy: {:.0f}%, Epoch: {} \n'.format(
            test_loss, test_accuracy, epoch)
        )


def train_multiswag_CIFAR(args, wandb_config):

    """
        POSTTRAIN THE NETWORKS FOR GETTING MULTISWAG
        
        Input: args, wandb_config
        Output: None

        (saved to os.path.join(model_folder, f'model_{model_id}.pt') )

    """
    

    # load pretrained nets
    nets = load_nets(args)    

    # datasets
    train_dataset, valid_dataset, test_dataset = create_datasets.create_datasets(args)

    # Loaders
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)
    validLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch, shuffle=False)
    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False)

    # loop over the nets
    for model_id in range(args.num_nets):

        wandb_run = wandb.init(**wandb_config, reinit=True)
        # train quantities
        loss_train = []
        accuracy_train = []

        # pretrained resnet20
        net = nets[model_id]

        # device
        device = args.device
        net.to(device)
        
        # init params
        num_swa_models = 1
        sum_state_dict = copy.deepcopy(net.state_dict())
        sum_square_state_dict = copy.deepcopy(net.state_dict())
        layer_names = list(sum_state_dict.keys())
        for layer_name in layer_names:
            sum_square_state_dict[layer_name] = torch.square(sum_state_dict[layer_name])

        # loop over the epochs
        for epoch in range(args.swa_swag_total_epochs):

            # swa-cycle learning rate scheduler
            t = ( epoch % args.swa_swag_cycle_epochs)/args.swa_swag_cycle_epochs
            learning_rate = (1-t)*args.swa_swag_lr1 + t*args.swa_swag_lr2
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov = True)

            # train
            train_loss = 0
            train_correct = 0
            net.train()
            train_loss, train_accuracy, train_correct = train_one_epoch(net, train_dataset, trainLoader, optimizer, args)
            loss_train.append(train_loss)
            accuracy_train.append(train_accuracy)

            # valid
            valid_loss, valid_accuracy = validate_model(args, net, validLoader)

            # print
            if epoch % 10 == 0:
                print('\nTrain set: Avg. total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), SWA Epoch: {}'.format(
                    train_loss, train_correct, len(trainLoader.dataset),
                    train_accuracy, epoch)
                    )
                
                print('Valid set: Avg. predictive loss: {:.4f}, Accuracy: {:.0f}%, SWA Epoch: {} \n'.format(
                    valid_loss, valid_accuracy, epoch)
                    )

            wandb.log(dict(
                loss_train=train_loss,
                validation_loss=valid_loss,
                acc_train=train_accuracy,
                acc_validation=valid_accuracy,
                learning_rate=learning_rate,
                epoch=epoch
            ))

            # update swa weights
            if t == 0.0:
                update_swa_params(net, sum_state_dict,  sum_square_state_dict)
                num_swa_models += 1


        # save swa model
        args.num_swa_models = num_swa_models
        model_folder = create_folder(args, "multiswa")
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"model_{model_id}.pt")).absolute()
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

        # regular model
        test_loss, test_accuracy = test_model(args, net, testLoader)
        wandb.log(dict(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
        ))

        print('Test set (regular): Avg. predictive loss: {:.4f}, Accuracy: {:.0f}%, Epoch: {} \n'.format(
            test_loss, test_accuracy, epoch)
        )

        # swa model
        net.load_state_dict(get_mean_model(sum_state_dict, num_swa_models))
        test_loss, test_accuracy = test_model(args, net, testLoader)
        wandb.log(dict(
            test_loss_swa=test_loss,
            test_accuracy_swa=test_accuracy,
        ))
        wandb_run.finish()

        print('Test set (swa): Avg. predictive loss: {:.4f}, Accuracy: {:.0f}%, Epoch: {} \n'.format(
            test_loss, test_accuracy, epoch)
        )



def train_one_epoch(net, train_dataset, train_loader, optimizer, args):

    # train the net
    train_loss = 0
    train_correct = 0
    net.train()
    device = args.device
    
    for batch, (data, targets) in enumerate(train_loader):
        # to CUDA
        data, targets = data.to(device), targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward the input and get the loss
        outputs = net(data)
        loss_class = F.nll_loss(F.log_softmax(outputs, dim = 1), targets)

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

    # scale
    train_loss /= len(train_loader)
    train_accuracy = 100. * train_correct / len(train_loader.dataset)

    return train_loss, train_accuracy, train_correct


def l2_loss(net, args, train_dataset):
    loss_prior = 0
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Conv2d):

            # L2 factor from the variational inference analysis
            l2_dr = l2_drop_rate(args.method, args.drop_rate_conv)

            # weight
            if layer.weight != None:
                a = l2_dr/(2 * len(train_dataset))
                loss_prior += a*torch.sum(torch.pow(layer.weight, 2))

            # bias
            if layer.bias != None:
                b = l2_dr/(2 * len(train_dataset))
                loss_prior += b*torch.sum(torch.pow(layer.bias, 2))

        if isinstance(layer, FRN):

            # L2 factor from the variational inference analysis (stochastic only for np_dropout)
            if (args.method == "np_dropout"):
                l2_dr = 0.5
            else:
                l2_dr = 1

            # weight
            if layer.weight != None:
                a = l2_dr/(2 * len(train_dataset))
                loss_prior += a*torch.sum(torch.pow(layer.weight, 2))

            # bias
            if layer.bias != None:
                b = l2_dr/(2 * len(train_dataset))
                loss_prior += b*torch.sum(torch.pow(layer.bias, 2))

            # tau
            if layer.tau != None:
                c = l2_dr/(2 * len(train_dataset))
                loss_prior += c*torch.sum(torch.pow(layer.tau, 2))

        if isinstance(layer, torch.nn.Linear):

            # there is only one linear layer and it is the output layer
            l2_dr = l2_drop_rate(args.method, args.drop_rate_linear)

            # weight
            if layer.weight != None:
                a = l2_dr/(2 * len(train_dataset))
                loss_prior += a*torch.sum(torch.pow(layer.weight, 2))

            # bias
            if layer.bias != None:
                b = l2_dr/(2 * len(train_dataset))
                loss_prior += b*torch.sum(torch.pow(layer.bias, 2))

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

            valid_loss += F.nll_loss(F.log_softmax(outputs, dim = 1), targets).item()
            valid_prediction = outputs.data.max(1, keepdim=True)[1]
            valid_correct += valid_prediction.eq(targets.data.view_as(valid_prediction)).sum().item()

        # the validation statistics
        valid_loss /= len(data_loader)

    valid_accuracy = 100. * valid_correct / len(data_loader.dataset)

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

            test_loss += F.nll_loss(F.log_softmax(outputs, dim = 1), targets).item()
            test_prediction = outputs.data.max(1, keepdim=True)[1]
            test_correct += test_prediction.eq(targets.data.view_as(test_prediction)).sum().item()

        # save/print the test statistics
        test_loss /= len(data_loader)

        test_accuracy = 100. * test_correct / len(data_loader.dataset)

        print('\nTest set (last): Avg. predictive loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, test_correct, len(data_loader.dataset),
            test_accuracy))


        return test_loss, test_accuracy

# constant for l2 regularization
def l2_drop_rate(method, drop_rate):

    if (method == "regular" or method == "multiswa" or method == "multiswag"):
        return 1    

    if method == "dropout":  
        return 1 - drop_rate

    if method == "np_dropout":    
        return 0.5

    if method == "dropconnect":  
        return 1 - drop_rate


# load nets
def load_nets(args):

    nets = []
    model_folder = create_folder(args, "regular", save_args = False)
    for model_id in range(args.num_nets):

        net = create_model.create_resnet20(args)
        model_path = str(
            pathlib.Path(os.path.join(model_folder, f"model_{model_id}.pt")).absolute()
        )
        net.load_state_dict(torch.load(model_path))
        net.to(args.device)
        net.train()
        nets.append(net)

    # load the same seed to synchronize the data partitioning (into train and valid) for regular and multiswag networks
    args_file = str(
        pathlib.Path(os.path.join(model_folder, "args.json")).absolute()
    )
    with open(args_file, "r") as f:
        data = json.load(f)

    torch.manual_seed(data["seed"])
    args.seed = data["seed"]
    print('MULTISWA(G) SEED: ', data["seed"])


    return nets


def create_folder(args, method, save_args = True):

    # cifar mode
    if args.do_augmentation_train == True:
        cifar_mode = args.cifar_mode + "_aug"
    else:
        cifar_mode = args.cifar_mode

    # create the output folder
    model_folder = os.path.join(args.output_dir, args.train_folder, cifar_mode, method)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if save_args:
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