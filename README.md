# Stochastic Ensembles

"Bayesian posterior approximation with stochastic ensembles" by Oleksandr Balabanov, Bernhard Mehlig, and Hampus Linander.

## Basic description

1. Train and evaluate a stochastic ensemble of neural networks. The ensemble types cover Regular (Non-Stochastic), Monte Carlo Dropout, DropConnect, Non-Parametric Dropout. 

2. Two classification tasks are considered: (1) a toy problem solved using a fully-connected network and (2) classification of CIFAR images with ResNet20-FRN model.

3. The considered evaluation metrics: (1) predictive entropy and mutual information for the toy model and (2) test accuracy, loss, ECE, out-of-domain detection, predictive entropy, mutual information, calibration curves for CIFAR. For CIFAR10 we also looked at resilience to distribution shift by evaluating on CIFAR10-C dataset.

## Prerequisites

Python 3 dependencies in `container/requirements.txt`

Singularity and Docker definitions provided.

## Prepare Singularity/Docker images

```
./build_docker.sh
```
or
``` sh
# Might need sudo
./build_singularity
```

## Start Singularity/Docker shell

``` sh
./shell_docker.sh
```
or
``` sh
./shell_singularity.sh
```

## WandB

To enable WandB logging for the training examples:

``` sh
--enable_wandb
```

## Toy model

Start at toy model root
``` sh
cd toy/
```

### Toy Model: Create Dataset

```
python3 run_create_data.py --output_dir ./output --num_data_train 2000  --num_data_eval 2000 --scaling_factors 0 1 2 3 4 5 6 7 8 9
```

### Toy Model: Train

Example 1:
```
  # --method covers HMC, regular, dropout, dropconnect, np_dropout options
  python3 run_train.py --output_dir ./output --method dropout --num_nets 1024  --drop_rate 0.1
```
Example 2:
```
  python3 run_train.py --output_dir ./output --method HMC
```

### Toy Model: Eval

Example 1: Produce plots of entropy and mutual information for nonparametric dropout ensemble
```
python3 run_eval.py --output_dir ./output --method dropout --drop_rate 0.1 --num_samples_ens 1024 --compute_save_softmax_probs True --plot_scaling_factor 1
```
Example 2: Produce plots of entropy and mutual information for HMC
```
python3 run_eval.py --output_dir ./output --method HMC  --num_samples_HMC 1024 --compute_save_softmax_probs True --plot_scaling_factor 1
```
Example 3: Produce mean abs error to HMC plots (beforehand need to produce HMC softmax_probs)
```
python3 run_eval.py --output_dir ./output --method dropout --drop_rate 0.1 --compute_save_softmax_probs False --plot_scaling_factors 0 1 2 3 4 5 6 7 8
```

## CIFAR
Start at CIFAR root

``` sh
cd CIFAR/
```

### CIFAR: Datasets

Download the datasets from

    CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html

    CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html

    SVHN: from torchvision with torchvision.datasets.SVHN(root=SVHN_dir, split = "test", download=True)

    CIFAR10C: https://zenodo.org/record/2535967#.Y5MtEHbMJD8
  
and rearrange them as follows
  
    CIFAR10: PATH_CIFAR / CIFAR / cifar-10-batches-py / (data_batch_{i} and test_batch) with i = 1, ..., 5. 

    CIFAR100: PATH_CIFAR / CIFAR / cifar-100-python / (test and train)

    SVHN: PATH_SVHN / (test_32x32.mat and train_32x32.mat)  

    CIFAR10C: PATH_CIFAR10C / CIFAR-10-C / (20 .numpy files with CIFAR-10-C corruption names)
    
with 

    --data_dir_CIFAR PATH_CIFAR (for run_train.py and run_eval.py)

    --data_dir_SVHN PATH_SVHN (for run_eval.py)

    --data_dir_CIFARC PATH_CIFAR10C (for run_eval.py)
  


### CIFAR: Train
```
# --method covers regular, dropout, dropconnect, np_dropout options
# --cifar_mode = CIFAR10 or CIFAR100
python3 ./run_train.py --output_dir ./output --cifar_mode CIFAR100 --method dropout --num_nets 20  --drop_rate_conv 0.3 --drop_rate_linear 0
```

### CIFAR: Eval
```
python3 ./run_eval.py --output_dir ./output --cifar_mode CIFAR100 --method dropout --num_nets 20  --drop_rate_conv 0.3 --drop_rate_linear 0 --compute_save_softmax_probs True
```




