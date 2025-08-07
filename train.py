from args import parse_train_opt
from EDGE import EDGE
import torch


def train(opt):
    model = EDGE(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        is_training=True,
        )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    
    # Print PyTorch GPU diagnostics
    print("---------- PyTorch GPU Diagnostics ----------")
    print(f"Is CUDA available?          : {torch.cuda.is_available()}")
    print(f"Number of GPUs PyTorch sees : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device ID       : {torch.cuda.current_device()}")
        print(f"Current device name     : {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("---------------------------------------------")

    # Print the hyperparameters for logging
    print("-------------- Hyperparameters --------------")
    # vars(opt) converts the argparse.Namespace object to a dictionary
    for k, v in vars(opt).items():
        print(f"{k:<20}: {v}")
    print("----------------------------------------------")

    train(opt)
