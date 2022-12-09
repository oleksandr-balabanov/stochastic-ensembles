"""
    FOLDERS FOR SAVING / LOADING MODELS AND RESULTS
"""
import os
from pathlib import Path


def get_eval_folder(args):
    if args.do_augmentation_train == True:
        cifar_mode = args.cifar_mode + "aug"
    else:
        cifar_mode = args.cifar_mode

    eval_folder = Path(args.output_dir) / args.save_folder / cifar_mode / f"eval_{args.method}"
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    return eval_folder


def get_model_folder(args):
    if args.do_augmentation_train == True:
        cifar_mode = args.cifar_mode + "aug"
    else:
        cifar_mode = args.cifar_mode

    model_folder = Path(args.output_dir) / args.save_folder / cifar_mode / f"train_{args.method}"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    return model_folder
