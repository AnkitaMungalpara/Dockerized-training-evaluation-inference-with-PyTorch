import os
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import Net
from uuid import uuid4


def infer(model, dataset, save_dir, num_samples=5):
    """
    perform inference on subset of images and save the predictions as images.

    Args:
        model (torch.nn.Module): The model to be used for inference
        dataset (torch.utils.data.Dataset): The dataset from which images are sampled
        save_dir (str or Path): Directory where the results will be saved
        num_samples (int, optional): Number of samples to infer and save. Defaults to 5
    """

    model.eval()
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, actual_label = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))

        # Get predicted label and confidence
        probabilities = F.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True).item()
        confidence = probabilities.max().item()

        img = Image.fromarray(image.squeeze().numpy() * 255).convert("L")

         # Display the predicted image 
        plt.figure(figsize=(9, 9))
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(
            f"Actual: {actual_label} | Predicted: {pred} | (Confidence: {confidence:.2f})"
        )
        plt.show() 
        plt.close()

        # save the image
        img.save(results_dir / f"{pred}.png")


def load_checkpoint(model, checkpoint_path):
    """
    load the model state from checkpoint file.

    Args:
        model (torch.nn.Module): The model instance to load the state into
        checkpoint_path (str): The path to the checkpoint file
    """

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, started training from scratch.")


def main():
    """
    Main function to initialize the model, load the checkpoint, and perform inference on the MNIST dataset.

    """

    save_dir = os.getcwd()
    model_checkpoint_path = os.path.join(save_dir, "model", "mnist_cnn.pt")

    # init model and load checkpoint here
    device = torch.device("cpu")
    model = Net().to(device)

    load_checkpoint(model, model_checkpoint_path)

    # create transforms and test dataset for mnist
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = datasets.MNIST("../data", train=False, transform=transform, download=True)

    infer(model, dataset, save_dir)
    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()
