"""
	EVAL A STOCHASITC ENSEMBLE OF RESNET-20-FRN NETWORKS ON CIFAR10 or CIFAR100
"""

import eval.eval_utilities.compute_save_metrics as compute_save_metrics
import eval.eval_utilities.save_softmax_probs as save_softmax_probs
import eval.eval_utilities.plot_metrics as plot_metrics

def eval_ens_CIFAR(args):

    # save softmax probs
    if args.compute_save_softmax_probs:
        save_softmax_probs.save_softmax_probs_CIFAR(args)
       
    compute_save_metrics.save_performance_metrics(args)
    plot_metrics.plot_metrics(args)
        








 