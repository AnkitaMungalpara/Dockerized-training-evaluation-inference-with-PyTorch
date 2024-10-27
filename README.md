#  MNIST Digit Classification Using Docker Compose: Training, Evaluation, and Inference

This project provides a **Docker Compose** configuration to handle training, evaluation, and inference on the [MNIST Hogwild](https://github.com/pytorch/examples/tree/main/mnist_hogwild) dataset with PyTorch. It uses Docker Compose to orchestrate three services: **train**, **evaluate**, and **infer**.

## Table of Contents

  - [Requirements](#requirements-)
  - [Introduction to Docker and Docker Compose](#introduction-to-docker-and-docker-compose-)
  - [Docker Compose Services](#docker-compose-services-️)
    - [Train](#-train)
    - [Evaluate](#-evaluate)
    - [Infer](#-infer)
  - [Command-Line Arguments](#command-line-arguments-)
  - [Docker Compose Configuration](#docker-compose-configuration-)
  - [Instructions](#instructions-)
- [Results](#results-)
- [References](#references-)

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

<table>
    <thead>
        <tr>
            <th>Argument</th>
            <th>Description</th>
            <th>Default</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>--batch-size</code></td>
            <td>Input batch size for training</td>
            <td>64</td>
        </tr>
        <tr>
            <td><code>--epochs</code></td>
            <td>Number of epochs to train</td>
            <td>10</td>
        </tr>
        <tr>
            <td><code>--lr</code></td>
            <td>Learning rate</td>
            <td>0.01</td>
        </tr>
        <tr>
            <td><code>--momentum</code></td>
            <td>SGD momentum</td>
            <td>0.5</td>
        </tr>
        <tr>
            <td><code>--seed</code></td>
            <td>Random seed</td>
            <td>1</td>
        </tr>
        <tr>
            <td><code>--log-interval</code></td>
            <td>How many batches to wait before logging training status</td>
            <td>10</td>
        </tr>
        <tr>
            <td><code>--num-processes</code></td>
            <td>Number of processes to run script on for distributed processing</td>
            <td>2</td>
        </tr>
        <tr>
            <td><code>--dry-run</code></td>
            <td>Quickly check a single pass without full training</td>
            <td>False</td>
        </tr>
        <tr>
            <td><code>--save_model</code></td>
            <td>Flag to save the trained model</td>
            <td>True</td>
        </tr>
        <tr>
            <td><code>--save-dir</code></td>
            <td>Directory where the checkpoint will be saved</td>
            <td><code>./</code></td>
        </tr>
    </tbody>
</table>


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


<table>
  <tr>
    <td>
      <img src="https://i.imgur.com/fN3jX5S.png" alt="7" width="100%">
      <p>Actual Label: 7 | Predicted: 7 | (Confidence: 1.00)</p>
    </td>
    <td>
      <img src="https://i.imgur.com/mial6h2.png" alt="9" width="100%">
      <p>Actual Label: 9 | Predicted: 9 | (Confidence: 1.00)</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://i.imgur.com/tpuPqwC.png" alt="4" width="100%">
      <p>Actual Label: 4 | Predicted: 4 | (Confidence: 1.00)</p>
    </td>
    <td>
      <img src="https://i.imgur.com/wefu1pT.png" alt="2" width="100%">
      <p>Actual Label: 1 | Predicted: 1 | (Confidence: 1.00)</p>
    </td>
  </tr>
</table>


## References 🔗

- [PyTorch MNIST Hogwild Example](https://github.com/pytorch/examples/tree/main/mnist_hogwild)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

