"""

	CREATE MODEL (TORCH)

"""

import models.dropout as dropout
import models.np_dropout as np_dropout
import models.dropconnect as dropconnect


# create resnet20
def create_resnet20(args):

    if args.cifar_mode == "CIFAR10":
        num_classes = 10
    else:
        num_classes = 100

    if args.method == "regular":
        return dropout.dropout_resnet20(
            num_classes=num_classes,
            drop_rate_conv=0,
            drop_rate_linear=0,
        )

    if args.method == "dropout":
        return dropout.dropout_resnet20(
            num_classes=num_classes,
            drop_rate_conv=args.drop_rate_conv,
            drop_rate_linear=args.drop_rate_linear,
        )

    if args.method == "np_dropout":
        return np_dropout.np_dropout_resnet20(
            num_classes=num_classes,
        )

    if args.method == "dropconnect":
        return dropconnect.dropconnect_resnet20(
            num_classes=num_classes,
            drop_rate_conv=args.drop_rate_conv,
            drop_rate_linear=args.drop_rate_linear,
        )
