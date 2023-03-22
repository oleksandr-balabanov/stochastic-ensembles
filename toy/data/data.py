"""

	LOAD DATA

"""

import os
import pathlib

def load_train_data(args):
    data_x_path = str(
        pathlib.Path(
            os.path.join(args.data_folder, f"x_train_{args.case}.pt")
        ).absolute()
    )

    data_y_path = str(
        pathlib.Path(
            os.path.join(args.data_folder, f"y_train_{args.case}.pt")
        ).absolute()
    )

    x_train = torch.load(pathlib.Path(os.path.join(args.data_folder, f"x_train_{args.case}.pt")))
    y_train = torch.load(pathlib.Path(os.path.join(args.data_folder, f"y_train_{args.case}.pt")))
    
    return x_train, y_train
