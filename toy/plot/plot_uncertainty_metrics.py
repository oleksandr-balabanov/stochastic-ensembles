import os
import pathlib
import torch
import matplotlib.pyplot as plt
from data.datasets import ToyDataset


# save imgs of entropy and mutual info
def compute_and_plot_entropy_mutual_info(args, scaling_factor):

    [softmax_probs, data] = load_softmax_probs(args, scaling_factor)
    [entropy, mutual_info] = get_entropy_MI(args, softmax_probs)

    # plot entropy
    plot_single_entropy_MI(args, entropy, data, "entropy", scaling_factor)

    # plot mutual info
    plot_single_entropy_MI(args, mutual_info, data, "mutual_info", scaling_factor)


# save imgs of entropy and mutual info
def compute_and_save_abs_mean_error_to_HMC(args, scaling_factors):

    error_entropy, error_mutual_info = compute_abs_mean_error_to_HMC(args, scaling_factors)

    # plot entropy abs mean error
    plot_error_vs_scaling_factors(args, error_entropy, "abs_error_entropy", scaling_factors)

    # plot mutual info abs mean error
    plot_error_vs_scaling_factors(args, error_mutual_info, "abs_error_mutual_info", scaling_factors)


# plot error vs scaling factors
def plot_error_vs_scaling_factors(args, errors, file_name, scaling_factors):

    fig = plt.figure(figsize=(4, 4))
    spec = fig.add_gridspec(1, 1)

    # fontsize of the tick labels
    SMALL_SIZE = 14
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)

    ax = fig.add_subplot(spec[0, 0])

    ax.plot(
        scaling_factors,
        errors,
        markeredgecolor="black",
        markersize=14,
        linewidth=3,
        color="orange",
        marker="*",
        label=file_name,
    )
    ax.text(
        0,
        1.1,
        f"{file_name}_{args.method} \n scaling_factors = {scaling_factors}",
        transform=ax.transAxes,
        fontsize=14,
    )

    # save folder
    model_folder = os.path.join(args.output_dir, args.eval_folder, "imgs")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    plt.savefig(
        os.path.join(model_folder, f"img_error_{file_name}_{args.method}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )


# plot data-resolved entropy or MI
def plot_single_entropy_MI(args, metric, data, file_name, scaling_factor):

    x1 = data[:, 0].cpu().numpy()
    x2 = data[:, 1].cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    spec = fig.add_gridspec(1, 1)

    # fontsize of the tick labels
    SMALL_SIZE = 14
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)

    ax = fig.add_subplot(spec[0, 0])
    colors = metric.cpu().numpy()
    im = ax.scatter(x1, x2, c=colors)
    fig.colorbar(im)
    ax.text(
        0,
        1.1,
        f"{file_name}_{args.method} \n scaling_factor = {scaling_factor}",
        transform=ax.transAxes,
        fontsize=14,
    )

    # save folder
    model_folder = os.path.join(args.output_dir, args.eval_folder, "imgs")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    plt.savefig(
        os.path.join(model_folder, f"img_single_{file_name}_{args.method}_sf_{scaling_factor}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )


# compute the errors
def compute_abs_mean_error_to_HMC(args, scaling_factors):

    method = args.method
    eval_folder = args.eval_folder
    error_entropy = []
    error_mutual_info = []
    for scaling_factor in scaling_factors:

        scaling_factor = float(scaling_factor)

        # method metrics
        args.method = method
        args.eval_folder = eval_folder
        [softmax_probs, data] = load_softmax_probs(args, scaling_factor)
        [entropy, mutual_info] = get_entropy_MI(args, softmax_probs)

        # HMC metrics
        args.method = "HMC"
        args.eval_folder = args.eval_folder_HMC
        [softmax_probs, data] = load_softmax_probs(args, scaling_factor)
        [entropy_HMC, mutual_info_HMC] = get_entropy_MI(args, softmax_probs)

        # append abs mean errors to the res lists
        error_entropy.append((entropy - entropy_HMC).abs().mean().item())
        error_mutual_info.append((mutual_info - mutual_info_HMC).abs().mean().item())

    args.method = method
    args.eval_folder = eval_folder

    return error_entropy, error_mutual_info


# entropy and mutual info
def get_entropy_MI(args, softmax_probs, eps=10 ** (-10)):

    # device
    device = args.device

    # load
    eps = 10 ** (-10)

    entropy_mean_mean = -torch.sum(
        torch.mean(softmax_probs, dim=0) * torch.log(torch.mean(softmax_probs, dim=0) + eps), dim=-1
    )
    mean_mean_entropy = -torch.mean(
        torch.sum(softmax_probs * torch.log(softmax_probs + eps), dim=-1), dim=0
    )
    mutual_info = entropy_mean_mean - mean_mean_entropy

    return [entropy_mean_mean, mutual_info]


# load saved softmax probs
def load_softmax_probs(args, scaling_factor):

    scaling_factor = float(scaling_factor)

    # load the datasets
    data_dir = os.path.join(args.output_dir, args.data_folder)
    eval_dataset = ToyDataset(
        data_dir,
        num_data_train=args.num_data_train,
        num_data_eval=args.num_data_eval,
        mode="eval",
        scaling_factor=scaling_factor,
    )

    # load softmax_probs
    model_folder = os.path.join(args.output_dir, args.eval_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if args.method == "HMC":
        softmax_path = str(
            pathlib.Path(
                os.path.join(model_folder, f"softmax_probs_HMC_{eval_dataset.get_identifier()}.pt")
            ).absolute()
        )
    else:
        softmax_path = str(
            pathlib.Path(
                os.path.join(
                    model_folder,
                    f"softmax_probs_{eval_dataset.get_identifier()}_num_pass_{args.num_samples_pass}.pt",
                )
            ).absolute()
        )
    softmax_probs = torch.load(softmax_path)

    # load data
    data = torch.load(os.path.join(model_folder, f"data_{eval_dataset.get_identifier()}.pt"))

    return [softmax_probs, data]
