import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Net


def test_epoch(model, data_loader, device):
    """
    evaluates model on the dataset for one epoch, and computes test loss and accuracy.

    Args:
        model (torch.nn.Module): model to be evaluated
        data_loader (torch.utils.data.DataLoader): DataLoader
        device (torch.device): CPU device to perform computation

    Returns:
        dict: dictionary containing the test loss and accuracy.
            - "Test loss" (float): The average negative log-likelihood loss over the dataset
            - "Accuracy" (float): The percentage of correctly predicted samples
    """

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(
                output, target=target.to(device), reduction="sum"
            ).item()
            prediction = output.max(1)[1]
            correct += prediction.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    out = {"Test loss": test_loss, "Accuracy": accuracy}
    print(out)

    return out


def main():
    """
    main function for model evaluation

    This function parses command-line arguments, sets the random seed, configures test data loader,
    evaluates model on test dataset and saves evaluation results to JSON file.
    """

    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # device
    device = torch.device("cpu")

    # create MNIST test dataset and loader
    test_dataset = datasets.MNIST(
        "../data", train=False, transform=transform, download=True
    )

    test_loader = DataLoader(test_dataset, **kwargs)

    # create model and load state dict
    model = Net().to(device)
    model.share_memory()

    # test epoch function call
    eval_results = test_epoch(model, test_loader, device)

    with (Path(args.save_dir) / "model" / "eval_results.json").open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
