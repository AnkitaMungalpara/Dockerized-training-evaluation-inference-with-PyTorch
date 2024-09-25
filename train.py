import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from infer import load_checkpoint
from model import Net


def train_epoch(epoch, args, model, device, train_loader, optimizer):
    """
    Trains a model for single epoch, calculates the loss using negative log-likihood and
    performs back-propagation to update the model weights.

    Args:
        epoch (int): the current epoch number
        args (argparse.Namespace): arguments containing configurations
        model (torch.nn.Module): neural network model to train
        device (torch.device): CPU device to perform computations
        train_loader (torch.utils.data.DataLoader): DataLoader
        optimizer (torch.optim.Optimizer): optimizer for updating model weights
    """

    model.train()


    for batch_id, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target=target.to(device))
        loss.backward()
        optimizer.step()

        if batch_id % args.log_interval == 0:
            print(
                f"{os.getpid()}\t Train epoch: {epoch} [{batch_id * len(data)}/{len(train_loader.dataset)} ({100. * batch_id / len(train_loader):.0f}%)]"
                f"\t Loss: {loss.item():.4f}"
            )
            if args.dry_run:
                break


def train(rank, args, model, dataset, dataloader_kwargs, device):
    """
    Trains a model on MNIST Hogwild dataset. This function sets the random seed, creates a data loader,
    and trains a model for specified number of epochs with stochastic gradient descent (SGD) with momentum.

    Args:
        rank (int): process ID in multiprocessing
        args (argparse.Namespace): arguments containing configurations
        model (torch.nn.Module): neural network model to train
        dataset (torch.utils.data.Dataset): dataset used for training
        dataloader_kwargs (dict): keyword arguments for creating data loader
        device (torch.device): CPU device to perform computations
    """

    torch.manual_seed(args.seed + rank)

    train_loader = DataLoader(dataset=dataset, **dataloader_kwargs)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)


def save_checkpoint(save_dir, model):
    """
    saves model dictionary into specified directory.

    Args:
        save_dir (str): The directory where the model checkpoint will be saved.
        model (torch.nn.Module): PyTorch model whose state dictionary is to be saved.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, "model", "mnist_cnn.pt")
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Model saved at: {checkpoint_path}")


def main():
    """
    main function to run MNIST training script.

    This function defines the argument parser for command-line arguments,
    initializes training environment, and handles training of the MNIST model.
    """

    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs lto train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=5,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=True,
        help="save the trained model or not",
    )

    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    args = parser.parse_args()

    # setting up device to run script on
    device = torch.device("cpu")

    torch.manual_seed(args.seed)
    # create model and setup mp
    model = Net().to(device)
    mp.set_start_method("spawn")
    model.share_memory()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_processes,
        "pin_memory": True,
        "shuffle": True,
    }

    # create mnist train dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data",
        train=True,
        transform=transform,
        download=True,
    )

    # loading model if checkpoint found
    checkpoint_path = os.path.join(args.save_dir, "model", "mnist_cnn.pt")
    if os.path.exists(checkpoint_path):
        load_checkpoint(model=model, checkpoint_path=checkpoint_path)
    else:
        # mnist hogwild training process
        processes = []

        for id in range(args.num_processes):
            p = mp.Process(
                target=train, args=(id, args, model, train_dataset, kwargs, device)
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # save model ckpt
        if args.save_model:
            save_checkpoint(args.save_dir, model)


if __name__ == "__main__":
    main()
