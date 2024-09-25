# 🐳 MNIST Training, Evaluation, and Inference with Docker Compose ⚙️

This project provides a **Docker Compose** configuration to handle training, evaluation, and inference on the [MNIST Hogwild](https://github.com/pytorch/examples/tree/main/mnist_hogwild) dataset with PyTorch. It uses Docker Compose to orchestrate three services: **train**, **evaluate**, and **infer**.

## Table of Contents

- [Requirements](#requirements)
- [Introduction to Docker and Docker Compose](#introduction-to-docker-and-docker-compose)
- [Docker Compose Services](#docker-compose-services)
   - [Train](#1-train)
   - [Evaluate](#2-evaluate)
   - [Infer](#3-infer)
- [Command-Line Arguments](#command-line-arguments)
- [Docker Compose Configuration](#docker-compose-configuration)
- [Instructions](#instructions)
- [References](#references)

## Requirements 📦

- `torch`
- `torchvision`

You can install the requirements using the following command:
```bash
pip install -r requirements.txt
```

## Introduction to Docker and Docker Compose 🐳

[**Docker**](https://aws.amazon.com/docker/) is an open-source platform that automates the deployment of applications in lightweight, portable containers. Containers allow developers to package an application along with its dependencies, ensuring consistency across environments.

[**Docker Compose**](https://docs.docker.com/compose/) is a tool specifically designed to define and manage multi-container Docker applications. It allows you to describe how different services (e.g., training, evaluation, and inference) in an application interact with each other, making it easier to maintain, scale, and manage. Docker Compose helps in building machine learning solutions in the following ways:


✅ **Simplify Deployment**: 
  - Quickly set up training, evaluation, and inference environments in an isolated, reproducible way.

✅ **Maintain Consistency**: 
  - Avoid compatibility issues by packaging dependencies with the code.

✅ **Streamline Workflow**: 
  - Execute tasks (like training, evaluation, and inference) effortlessly across services.

## Docker Compose Services 🛠️

The Docker Compose configuration file `docker-compose.yaml` defines three services:

### 🔷 train 

- Trains the MNIST model.
- Checks for a checkpoint file in the shared volume. If found, resumes training from that checkpoint.
- Saves the final checkpoint as `mnist_cnn.pt` and exits.

### 🔷 evaluate 

- Checks for the final checkpoint (`mnist_cnn.pt`) in the shared volume.
- Evaluates the model and saves metrics in `eval_results.json`.
- The model code is imported rather than copy-pasted into `eval.py`.

### 🔷 infer 

- Runs inference on sample MNIST images.
- Saves the results (images with predicted numbers) in the `results` folder within the Docker container and exits.

## Command-Line Arguments 🔧

The MNIST training script accepts the following command-line arguments:

| Argument         | Description                                                        | Default   |
|------------------|--------------------------------------------------------------------|-----------|
| `--batch-size`   | Input batch size for training                             | 64        |
| `--epochs`       | Number of epochs to train                                     | 10         |
| `--lr`           | Learning rate                                                 | 0.01      |
| `--momentum`     | SGD momentum                                                   | 0.5       |
| `--seed`         | Random seed                                                   | 1         |
| `--log-interval` | How many batches to wait before logging training status            | 10        |
| `--num-processes`| Number of processes to run script on for distributed processing | 2         |
| `--dry-run`      | Quickly check a single pass without full training                | False     |
| `--save_model`   | Flag to save the trained model                               | True      |
| `--save-dir`     | Directory where the checkpoint will be saved                 | `./`      |

## Docker Compose Configuration 📝

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  train:
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - mnist:/opt/mount
      - ./model:/opt/mount/model
      - ./data:/opt/mount/data

  evaluate:
    build:
      context: .
      dockerfile: Dockerfile.eval
    volumes:
      - mnist:/opt/mount
      - ./model:/opt/mount/model
      - ./data:/opt/mount/data

  infer:
    build:
      context: .
      dockerfile: Dockerfile.infer
    volumes:
      - mnist:/opt/mount
      - ./data:/opt/mount/data

volumes:
  mnist:
```

## Instructions 🚀

1️⃣ **Build Docker Images**:
   ```bash
   docker compose build
   ```
- This command builds the Docker images for each service (train, evaluate, infer). It ensures that the necessary dependencies are installed, and the code is properly packaged.

2️⃣ **Run Services**:
  
  - **Train**:
    
    ```bash
     docker compose run train
     ```
    
      Command that starts the training process. It will look for existing checkpoints in the volume and resume training if any are found.

  - **Evaluate**:
    ```bash
     docker compose run evaluate
    ```
     The above command evaluates the trained model using the saved checkpoint and generates metrics like accuracy and test loss.

  - **Inference**:
     ```bash
     docker compose run infer
     ```
    The inference service runs predictions on a few random MNIST images and saves the output images with predicted labels.

3️⃣ **Verify Results**:

✍️ **Checkpoint File**: 
  
  - Check if `mnist_cnn.pt` is in the `mnist` volume.
     - If found: "Checkpoint file found."
     - If not found: "Checkpoint file not found!" and exit with an error.
   
✍️ **Evaluation Results**: 

  - Verify `eval_results.json` in the `mnist` volume.
     - Example format: `{"Test loss": 0.0890245330810547, "Accuracy": 97.12}`
   
✍️ **Inference Results**: 
  
  - Check the `results` folder in the `mnist` volume for saved images with predicted numbers.

## Results 📊
Here are some sample predicted images generated by the `infer` service:

![Image1](results/1.png)

## References 🔗

- [PyTorch MNIST Hogwild Example](https://github.com/pytorch/examples/tree/main/mnist_hogwild)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

