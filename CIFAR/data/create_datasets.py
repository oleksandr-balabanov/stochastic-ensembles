import data.data_CIFAR10 as data_CIFAR10
import data.data_CIFAR100 as data_CIFAR100

def create_datasets(args):

    if args.cifar_mode == "CIFAR10":

        # get data
        train_data, test_data = data_CIFAR10.get_cifar10(args.data_dir_CIFAR)

        # datasets
        return data_CIFAR10.create_datasets(
            train_data, 
            test_data, 
            args.num_train,
            do_augmentation_train = args.do_augmentation_train, 
            do_augmentation_test = args.do_augmentation_test,
        )

    if args.cifar_mode == "CIFAR100":

        # get data
        train_data, test_data = data_CIFAR100.get_cifar100(args.data_dir_CIFAR)

        # datasets
        return data_CIFAR100.create_datasets(
            train_data, 
            test_data, 
            args.num_train,
            do_augmentation_train = args.do_augmentation_train, 
            do_augmentation_test = args.do_augmentation_test,
        )