"""
    PLOT IMGS      
"""

import os
import torch
import torch.nn.functional as F
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from utilities.folders import get_model_folder, get_eval_folder
import utilities.serialize as serialize

# plot and save
def plot_metrics(args):
    """
    EVALUATE THE PERFORMANCE: make plots for a single stochastic method
    Input: args
    Output: None

    (saved to get_eval_folder(args) / ")

    """

    # load softmax probs and targets
    res_dic = serialize.load(
        get_eval_folder(args) / f"performance_metrics_nn{args.num_nets:02d}.dill"
    )

    # compute performance metrics
    calibration_metrics = res_dic["calibration_metrics"]
    entropy_MI_metrics = res_dic["entropy_MI_metrics"]

    # plot and save the calibration curve
    file_name = "calibration_curve"
    plot_single_calibration_curve(calibration_metrics, file_name, args)

    # plot the entropy histogram
    bins = np.arange(0, 2, 2 / 9)
    file_name = "entropy_histogram"
    plot_single_entropy_MI(entropy_MI_metrics["entropy"], bins, file_name, args)

    # plot the mutual information histogram
    bins = np.arange(0, 1, 1 / 9)
    file_name = "mutual_info_histogram"
    plot_single_entropy_MI(entropy_MI_metrics["mutual_info"], bins, file_name, args)

    # plot accuracy and loss estimates associated with CIFAR10C corruptions
    if args.cifar_mode == "CIFAR10":

        # load and shape the data
        accuracy_loss_CIFAR10C = serialize.load(
            get_eval_folder(args) / f"accuracy_loss_CIFAR10C_nn{args.num_nets:02d}.dill"
        )["accuracy_loss_CIFAR10C"]
        accuracy_mean, accuracy_min_max, accuracy_q_min_max = get_accuracy_metrics_CIFAR10C(
            accuracy_loss_CIFAR10C["accuracy"]
        )
        loss_mean, loss_min_max, loss_q_min_max = get_loss_metrics_CIFAR10C(
            accuracy_loss_CIFAR10C["loss"]
        )

        # plot
        file_name = "accuracy_cifar10c"
        plot_single_cifar10c_loss_accuracy(
            accuracy_mean, accuracy_min_max, accuracy_q_min_max, file_name, args
        )

        file_name = "loss_cifar10c"
        plot_single_cifar10c_loss_accuracy(loss_mean, loss_min_max, loss_q_min_max, file_name, args)


# plot and save the calibriation curve
def plot_single_calibration_curve(calibration_metrics, file_name, args):

    fig = plt.figure(figsize=(4, 4))
    spec = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(spec[0, 0])

    label = args.method
    color = "orange"
    marker = "*"

    ax.plot(
        calibration_metrics["confidence"],
        calibration_metrics["accuracy"] - calibration_metrics["confidence"],
        marker=marker,
        color=color,
        markeredgecolor="black",
        markersize=12,
        label=label,
    )

    ax.set_xlabel("confidence", fontsize=14)
    ax.set_ylabel("accuracy - confidence", fontsize=14)

    plt.savefig(
        get_eval_folder(args) / f"img_{file_name}_nn{args.num_nets:02d}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


# plot one entropy or mutual info subplot
def plot_single_entropy_MI(data, bins, file_name, args, xtext=0.28, ytext=0.73):

    fig = plt.figure(figsize=(4, 4))
    spec = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(spec[0, 0])

    msg = args.method
    color = "orange"

    # from torch to numpy
    data = data.data.cpu().numpy()

    # fontsize of the tick labels
    SMALL_SIZE = 12
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)

    h, _, img = ax.hist(data, bins=(bins), edgecolor="black", color=color)

    ax.text(xtext, ytext, msg, transform=ax.transAxes, fontsize=10)
    plt.grid(b=True)

    ax.set_xlabel(file_name, fontsize=14)

    plt.savefig(
        get_eval_folder(args) / f"img_{file_name}_nn{args.num_nets:02d}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


# plot loss CIFAR10C
def plot_single_cifar10c_loss_accuracy(res_mean, res_min_max, q, file_name, args):

    fig = plt.figure(figsize=(4, 4))
    spec = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(spec[0, 0])

    color = "orange"
    label = args.method

    # unpack q
    [q_min, q_max] = q

    # plot errorbars
    xlist = [0.25 + 1 + x for x in range(5)]
    ax.errorbar(
        xlist, res_mean, res_min_max, linestyle="None", color="black", capsize=2, capthick=1, lw=1
    )

    # add rectangle to the plot
    for level in range(5):
        ax.add_patch(
            Rectangle(
                (level + 1, q_min[level]),
                0.5,
                -q_min[level] + q_max[level],
                edgecolor="black",
                facecolor=color,
                fill=True,
                lw=1,
                zorder=2,
            )
        )

    ax.set_xlabel("Corruption Intensity", fontsize=14)
    ax.set_ylabel(file_name, fontsize=14)

    plt.savefig(
        get_eval_folder(args) / f"img_{file_name}_nn{args.num_nets:02d}.pdf",
        format="pdf",
        bbox_inches="tight",
    )


def get_accuracy_metrics_CIFAR10C(accuracy):

    accuracy_mean = accuracy.mean(dim=0)
    accuracy_min = accuracy_mean - accuracy.min(dim=0)[0]
    accuracy_max = accuracy.max(dim=0)[0] - accuracy_mean
    accuracy_min_max = torch.stack((accuracy_min, accuracy_max), dim=0)
    accuracy_q_min_max = [accuracy.sort(dim=0)[0][3, :], accuracy.sort(dim=0)[0][11, :]]

    return accuracy_mean, accuracy_min_max, accuracy_q_min_max


def get_loss_metrics_CIFAR10C(loss):

    loss_mean = loss.mean(dim=0)
    loss_min = loss_mean - loss.min(dim=0)[0]
    loss_max = loss.max(dim=0)[0] - loss_mean
    loss_min_max = torch.stack((loss_min, loss_max), dim=0)
    loss_q_min_max = [loss.sort(dim=0)[0][3, :], loss.sort(dim=0)[0][11, :]]

    return loss_mean, loss_min_max, loss_q_min_max
