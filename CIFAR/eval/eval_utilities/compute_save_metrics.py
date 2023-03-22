"""
        EVALUATE THE MODEL      
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import utilities.serialize as serialize
from utilities.folders import get_model_folder, get_eval_folder



def save_performance_metrics(args):
    """
        EVALUATE THE PERFORMANCE: save and print accuracy, loss, ECE, ODD, entropy, MI
        Input: args
        Output: None

        (saved to get_eval_folder(args) / f"performance_metrics_nn{args.num_nets:02d}.dill")

    """

    # device
    device = args.device


    # load softmax probs and targets
    softmax_probs_dic = serialize.load(get_eval_folder(args) / f"softmax_probs_targets_nn{args.num_nets:02d}.dill")

    softmax_probs_CIFAR10 = softmax_probs_dic["CIFAR10"]["softmax_probs"].to(device)
    softmax_probs_CIFAR100 = softmax_probs_dic["CIFAR100"]["softmax_probs"].to(device)
    softmax_probs_SVHN = softmax_probs_dic["SVHN"]["softmax_probs"].to(device)

    if args.cifar_mode == "CIFAR10":
        softmax_probs = softmax_probs_dic["CIFAR10"]["softmax_probs"].to(device)
        targets = softmax_probs_dic["CIFAR10"]["targets"].to(device)
    else:
        softmax_probs = softmax_probs_dic["CIFAR100"]["softmax_probs"].to(device)
        targets = softmax_probs_dic["CIFAR100"]["targets"].to(device)

    # compute performance metrics
    accuracy_loss_metrics = get_accuracy_loss(softmax_probs, targets, args)
    ODD_metrics = get_OOD_metrics(softmax_probs_CIFAR10, softmax_probs_CIFAR100, softmax_probs_SVHN, args)
    calibration_metrics = get_calibration(softmax_probs, targets, num_bins=10)
    entropy_metrics = get_entropy(softmax_probs, eps = 10**(-10))

    res_dic = {
        "accuracy_loss_metrics": accuracy_loss_metrics,
        "ODD_metrics": ODD_metrics,
        "calibration_metrics":calibration_metrics,
        "entropy_metrics":entropy_metrics,
    }

    # save
    serialize.save(res_dic, get_eval_folder(args) / f"performance_metrics_nn{args.num_nets:02d}.dill")
    with open(get_eval_folder(args) / f"args_nn{args.num_nets:02d}.json", "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))


    # corrupted CIFAR
    softmax_probs_dic_C = serialize.load(get_eval_folder(args) / f"softmax_probs_targets_nn{args.num_nets:02d}_{args.cifar_mode}C.dill")
    serialize.save(get_accuracy_loss_CIFARC(softmax_probs_dic_C, args), get_eval_folder(args) / f"accuracy_loss_{args.cifar_mode}C_nn{args.num_nets:02d}.dill")


def get_accuracy_loss(softmax_probs, targets, args):

    """
        EVALUATE THE PERFORMANCE: get accuracy and loss
        Input: softmax_probs, targets, args
        Output: 
        {
            "accuracy":100. * test_correct / targets.shape[0], 
            "loss": test_loss,
        }

    """

    test_loss = 0
    test_loss = F.nll_loss(torch.log(softmax_probs), targets, reduction='mean').item()
    test_prediction = softmax_probs.data.max(1, keepdim=True)[1]
    test_correct = test_prediction.eq(targets.data.view_as(test_prediction)).sum().item()

    print('TEST SET ENSEMBLE: Avg. predictive loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, targets.shape[0],
        100. * test_correct / targets.shape[0]))

    return {
        "accuracy": test_correct / targets.shape[0], 
        "loss": test_loss,
    }


def get_OOD_metrics(softmax_probs_CIFAR10, softmax_probs_CIFAR100, softmax_probs_SVHN, args):
    """

        EVALUATE THE PERFORMANCE: OOD CIFAR10, CIFAR100, SVHN
        Input: softmax_probs_CIFAR10, softmax_probs_CIFAR100, softmax_max_probs_SVHN, args
        Output:      

        if args.cifar_mode == "CIFAR10":
            {
                "CIFAR10_CIFAR100":result_CIFAR10_CIFAR100,
                "CIFAR10_SVHN":result_CIFAR10_SVHN,
            }
        else:
            {
                "CIFAR100_CIFAR10":result_CIFAR100_CIFAR10,
                "CIFAR100_SVHN":result_CIFAR100_SVHN,
            }

    """

    softmax_max_probs_CIFAR100 = softmax_probs_CIFAR100.data.max(1, keepdim = False)[0]
    softmax_max_probs_CIFAR10 = softmax_probs_CIFAR10.data.max(1, keepdim = False)[0]
    softmax_max_probs_SVHN = softmax_probs_SVHN.data.max(1, keepdim = False)[0]

    # AUC_ROC
    if args.cifar_mode == "CIFAR10":
        # CIFAR10 + CIFAR100
        max_probs_0 = softmax_max_probs_CIFAR100.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR10.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR10_CIFAR100 = compute_AUC_ROC(sorted_classes)
        print('AUC ROC CIFAR10 + CIFAR100: {:.4f}\n'.format(result_CIFAR10_CIFAR100))
        
        # CIFAR10 + SVHN
        max_probs_0 = softmax_max_probs_SVHN.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR10.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR10_SVHN = compute_AUC_ROC(sorted_classes)
        print('AUC ROC CIFAR10 + SVHN: {:.4f}\n'.format(result_CIFAR10_SVHN))
        return {
            "CIFAR10_CIFAR100":result_CIFAR10_CIFAR100,
            "CIFAR10_SVHN":result_CIFAR10_SVHN,
        }

    if args.cifar_mode == "CIFAR100":
        # CIFAR100 + CIFAR10
        max_probs_0 = softmax_max_probs_CIFAR10.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR100.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR100_CIFAR10 = compute_AUC_ROC(sorted_classes)
        print('AUC ROC CIFAR100 + CIFAR10: {:.4f}\n'.format(result_CIFAR100_CIFAR10))
        
        # CIFAR100 + SVHN
        max_probs_0 = softmax_max_probs_SVHN.sort(descending=True)[0]
        max_probs_1 = softmax_max_probs_CIFAR100.sort(descending=True)[0]
        sorted_classes = create_sorted_classes(max_probs_0, max_probs_1)
        result_CIFAR100_SVHN = compute_AUC_ROC(sorted_classes)
        print('AUC ROC CIFAR100 + SVHN: {:.4f}\n'.format(result_CIFAR100_SVHN))
        return {
            "CIFAR100_CIFAR10":result_CIFAR100_CIFAR10,
            "CIFAR100_SVHN":result_CIFAR100_SVHN,
        }


# AUC_ROC
def compute_AUC_ROC(sorted_classes):
    
    # total number of class 1 and class 0
    N = sorted_classes[sorted_classes == True].shape[0]
    M = sorted_classes.shape[0]-N
    
    # initialize
    num_class_1 = 0
    area = 0
    
    # loop over the classes
    for i in range(N + M):
        if sorted_classes[i] == True:
            num_class_1 +=1
        else:
            area += num_class_1
            
    return area/(M * N)


# sort class data by prob values in the given sets of class 0 and class 1 
def create_sorted_classes(max_probs_0, max_probs_1):
    
    # initialize
    id_0 = 0
    id_1 = 0
    sorted_classes = torch.zeros((max_probs_0.shape[0] + max_probs_1.shape[0]), dtype=torch.bool)
    
    # rank the largest elements by class
    while (id_0 < max_probs_0.shape[0]) and (id_1 < max_probs_1.shape[0]):
        if max_probs_0[id_0] > max_probs_1[id_1]:
            sorted_classes[id_0+id_1] = False
            id_0 += 1
        else:
            sorted_classes[id_0+id_1] = True
            id_1 += 1
            
    # fill the rest of classes with 1  
    if id_0 == max_probs_0.shape[0]:
        sorted_classes[id_0 + id_1:] = True
    
    # fill the rest of classes with 0 
    if id_1 == max_probs_1.shape[0]:
        sorted_classes[id_0 + id_1:] = False
             
    return sorted_classes


def get_calibration(softmax_probs, targets, num_bins=10):
    """

        EVALUATE THE PERFORMANCE: Calibration
        
        adapted from https://github.com/google-research/google-research/tree/master/bnn_hmc 
        by Izmailov et al.
        
        Input: softmax_probs, targets, num_bins
        Output: 
        {
            "confidence": bin_confidences,
            "accuracy": bin_accuracies,
            "proportions": bin_proportions,
            "ece": ece
        }

    """

    confidences, predictions = torch.max(softmax_probs, dim=-1)

    num_inputs = confidences.shape[0]
    step = (num_inputs + num_bins - 1) // num_bins
    bins = torch.sort(confidences)[0][::step]
    
    if num_inputs % step != 1:
        bins = torch.cat((bins, torch.max(confidences).unsqueeze(0)))
        
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = (predictions == targets)

    bin_confidences = []
    bin_accuracies = []
    bin_proportions = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_confidences.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_proportions.append(prop_in_bin)
            
    
    bin_confidences, bin_accuracies, bin_proportions = map(
      lambda lst: torch.Tensor(lst),
      (bin_confidences, bin_accuracies, bin_proportions))

    print('ECE: {:.4f}\n'.format(ece))

    return {
      "confidence": bin_confidences,
      "accuracy": bin_accuracies,
      "proportions": bin_proportions,
      "ece": ece
    }



def get_entropy(p_ens, eps = 10**(-10)): 

    """

        EVALUATE THE PERFORMANCE: Entropy and Mutual Information
        
        Input: p_ens, eps = 10**(-10)
        Output: 
        {
            "entropy": entropy,
        }

    """

    eps = 10**(-10) 
    entropy = - torch.sum(p_ens * torch.log(p_ens + eps), dim = -1)

    return  {
        "entropy": entropy, 
        }


# accuracy and loss evaluated on the corrupted CIFAR dataset
def get_accuracy_loss_CIFARC(softmax_probs_dic_C, args):
    
    """

        EVALUATE THE PERFORMANCE: CORRUPTED CIFAR10 or CIFAR100

        Input: nets, args
        Output: 
        {
            "accuracy_loss":accuracy_loss_CIFARC,
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
    softmax_probs_targets_CIFARC = softmax_probs_dic_C["softmax_probs_targets_CIFARC"]
    
    result_accuracy = torch.zeros(len(CORRUPTION_FILES), 5)
    result_loss = torch.zeros(len(CORRUPTION_FILES), 5)

    for file_count, corruption_file in enumerate(CORRUPTION_FILES):
        softmax_probs_targets_CIFARC_single_corruption = softmax_probs_targets_CIFARC[corruption_file]
        for alpha_level in range(5):
            softmax_probs_targets = softmax_probs_targets_CIFARC_single_corruption[alpha_level]
            softmax_probs = softmax_probs_targets["softmax_probs"]
            targets = softmax_probs_targets["targets"]

            softmax_probs = softmax_probs.to(device)
            targets = targets.to(device)

            # test metrics
            test_loss = F.nll_loss(torch.log(softmax_probs), targets, reduction='mean').item()
            test_prediction = softmax_probs.data.max(1, keepdim=True)[1]
            test_correct = test_prediction.eq(targets.data.view_as(test_prediction)).sum().item()


            print('TEST SET ENSEMBLE: Corruption: {}, Level: {},  Avg. predictive loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                corruption_file, alpha_level,
                test_loss, test_correct, targets.shape[0],
                100. * test_correct / targets.shape[0]))

            result_accuracy[file_count, alpha_level] = test_correct / targets.shape[0]
            result_loss[file_count, alpha_level] = test_loss

    accuracy_loss_CIFARC = {
        "accuracy":result_accuracy,
        "loss":result_loss,
        "CORRUPTION_FILES":CORRUPTION_FILES,
    }

    return {
        "accuracy_loss_CIFARC":accuracy_loss_CIFARC,
    }