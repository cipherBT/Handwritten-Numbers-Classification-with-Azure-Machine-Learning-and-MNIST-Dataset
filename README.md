# Handwritten-Numbers-Classification-with-Azure-Machine-Learning-and-MNIST-Dataset

This project demonstrates how to build, train, tune, and deploy a deep learning model for handwritten digit recognition using the MNIST dataset, leveraging Azure Machine Learning. The workflow includes data exploration, model training, hyperparameter tuning, and deploying a real-time inference API.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Steps](#project-steps)
    - [1. Data Exploration](#1-data-exploration)
    - [2. Azure ML Workspace Setup](#2-azure-ml-workspace-setup)
    - [3. Compute Target and Environment Configuration](#3-compute-target-and-environment-configuration)
    - [4. Training the Deep Learning Model](#4-training-the-deep-learning-model)
    - [5. Hyperparameter Tuning (Sweep)](#5-hyperparameter-tuning-sweep)
    - [6. Model Registration](#6-model-registration)
    - [7. Deployment and Real-time Inference](#7-deployment-and-real-time-inference)
    - [8. Testing the Endpoint](#8-testing-the-endpoint)
    - [9. Training Script Overview](#9-training-script-overview)
- [Deep Learning Concepts](#deep-learning-concepts)
- [References](#references)

---

## Overview

The goal is to train a neural network to classify handwritten digits (0-9) from the MNIST dataset using Azure Machine Learning. The project covers the full ML lifecycle from data exploration to deployment of a REST API for inference.

---

## Prerequisites

- An Azure account and access to Azure Machine Learning Studio.
- Python 3.8+ environment, preferably via [Anaconda](https://www.anaconda.com/products/distribution).
- VS Code and Jupyter Notebook.
- [Azure ML Python SDK v2](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Postman](https://www.postman.com/) for API testing.

---

## Project Steps

### 1. Data Exploration

- **Dataset**: Uses the MNIST dataset (available in TensorFlow or via Azure Storage).
- **Explore in Python**: Visualize samples, reshape, normalize, and convert images to JSON for model input. (exploreMNISTDataset.py)

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import json

# Load the hand written Dataset
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()

# Pick a sample to plot
sample = 1200

# Reshape image into a single row
image = X_train[sample]
image_array = image.reshape((1,784))
# Normalize the dataset
image_array_decimal = image_array/255
# Convert to JSON
lists = image_array_decimal.tolist()
json_str = json.dumps(lists)
print(json_str)

# Display the image
plt.imshow(image, cmap='gray')
plt.title(f"Label: {Y_train[sample]}")
plt.axis('off')
plt.show()
```

### 2. Azure ML Workspace Setup

- **Authenticate** and connect using `MLClient` and `DefaultAzureCredential`.
- **Configure** your subscription, resource group, and workspace:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(), "SUBSCRIPTION_ID", "RESOURCE_GROUP", "WORKSPACE_NAME"
)
```
Note: Make sute to login to azure using terminal with 
```bash 
az login
```

### 3. Compute Target and Environment Configuration

- **Create or use an existing GPU compute cluster** for training.
- **Set up environment** using curated AzureML TensorFlow environment.

```python
from azure.ai.ml.entities import AmlCompute

gpu_compute_target = "YOUR_COMPUTE_CLUSTER"
curated_env_name = "AzureML-tensorflow-2.12-cuda11@latest"
```

### 4. Training the Deep Learning Model

- **Prepare the MNIST data path** (Azure Storage).
```python
webpath = "wasbs://..."
```

- **Build command job** to run the training script using specified parameters.

```python
from azure.ai.ml import command, Input
from azure.ai.ml.entities  import UserIdentityConfiguration

job = command(
    inputs=dict(
        data_folder=Input(type="uri_folder", path="web_path"),
        batch_size=64,
        first_layer_neurons=256,
        second_layer_neurons=128,
        learning_rate=0.01,
    ),
    compute=gpu_compute_target,
    environment=curated_env_name,
    code="./",
    command="python deep_learning_training_script.py --data-folder ${{inputs.data_folder}} --batch-size ${{inputs.batch_size}} --first-layer-neurons ${{inputs.first_layer_neurons}} --second-layer-neurons ${{inputs.second_layer_neurons}} --learning-rate ${{inputs.learning_rate}}",
    experiment_name="tf-dnn-image-classify",
    display_name="tensorflow-classify-mnist-digit-images-with-dnn",
)
ml_client.jobs.create_or_update(job)
```

- **Monitor training** in Azure ML Studio.

### 5. Hyperparameter Tuning (Sweep)

- **Configure sweep** to search for best batch size, neuron counts, and learning rate.
- **Use BanditPolicy for early stopping.**
- **Run sweep job** and select the best performing configuration.

```python
from azure.ai.ml.sweep import Choice, LogUniform

job_for_sweep = job(
    batch_size=Choice(values=[32, 64, 128]),
    first_layer_neurons=Choice(values=[16, 64, 128, 256, 512]),
    second_layer_neurons=Choice(values=[16, 64, 256, 512]),
    learning_rate=LogUniform(min_value=-6, max_value=-1),
)

sweep_job = job_for_sweep.sweep(
    compute=gpu_compute_target,
    sampling_algorithm="random",
    primary_metric="validation_acc",
    goal="Maximize",
    max_total_trials=8,
    max_concurrent_trials=4,
    early_termination_policy=BanditPolicy(slack_factor=0.1, evaluation_interval=2),
)
ml_client.create_or_update(sweep_job)
```

### 6. Model Registration

- **Register the best model** from the sweep job for future inference.

```python
from azure.ai.ml.entities import Model

model = Model(
    path="azureml://jobs/{best_run}/outputs/artifacts/paths/outputs/model/",
    name="mnist-handwriting-model",
    description="Model created from hyperparameter sweep.",
    type="custom_model",
)
registered_model = ml_client.models.create_or_update(model=model)
```

### 7. Deployment and Real-time Inference

- **Deploy model** as a real-time endpoint using Azure Container Instance or Kubernetes.
- **Configure web service** input/output for prediction.

### 8. Testing the Endpoint

- **Test using Postman**:
    - Send JSON formatted input to the endpoint.
    - View prediction output.

- **Test using Python requests**:
    - Use the generated code snippet to send requests and parse responses.

---

### 9. Training Script Overview

#### When and How the Training Script Is Used

The file `deep_learning_training_script.py` is the core training script executed by Azure ML during the command job submission. In your notebook, you specify this script to run on the remote compute target with all required arguments:

```python
command="python deep_learning_training_script.py --data-folder ${{inputs.data_folder}} --batch-size ${{inputs.batch_size}} --first-layer-neurons ${{inputs.first_layer_neurons}} --second-layer-neurons ${{inputs.second_layer_neurons}} --learning-rate ${{inputs.learning_rate}}"
```

Note: Make sure it is in the same folder as the notebook in your local computer.

Azure ML takes care of provisioning resources, passing parameters, running the script, tracking metrics, and uploading outputs.

#### What the Script Does

- **Imports dependencies**: TensorFlow, NumPy, MLflow, and utilities for data loading and management.
- **Loads and preprocesses the MNIST dataset**: Reads compressed `.gz` files, normalizes images, and prepares labels.
- **Builds a configurable neural network**: Two hidden layers with ReLU activation, output layer for digit classification.
- **Defines training logic**:
    - One-hot encoding for labels.
    - Loss and accuracy metrics.
    - Training loop: shuffles data, splits into batches, updates weights using gradient descent.
    - Tracks metrics and parameters with MLflow.
    - Periodically saves checkpoints and finally saves the trained model.
- **Supports hyperparameter tuning**: Accepts batch size, neuron counts, learning rate as command-line arguments.
- **Handles resume-from-checkpoint**: Optionally resumes training from a previous checkpoint if specified.

**Summary of Workflow:**

- Model Definition: A neural network with two hidden layers.
- Data Loading: Preprocess and load the MNIST dataset.
- Training Loop: Shuffle and split data into mini-batches, update weights, log metrics, checkpoint.
- Model Saving: Save final trained model to disk.
- Experiment Logging: Use MLflow to track all parameters, metrics, and artifacts.

---

## Deep Learning Concepts

- **Neural Networks**: Structure with input, multiple hidden layers, and output neurons.
- **Activation Functions**: Threshold, sigmoid, rectifier (ReLU), hyperbolic tangent.
- **Gradient Descent**: Optimization algorithm to minimize loss and update weights.
- **Hyperparameter Tuning**: Automated search for the best configuration.
- **Deployment States**: Transitioning, unhealthy, unschedulable, failed, healthy.

---

## References

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Postman API Tool](https://www.postman.com/)
- [Anaconda Distribution](https://www.anaconda.com/products/distribution)
- [VS Code](https://code.visualstudio.com/)
- [YouTube Tutorial](https://youtu.be/N_d5Veid-_o)

---

## Credits

Tutorial and walkthrough based on Deeptech data science and machine learning [YouTube session](https://youtu.be/N_d5Veid-_o) and provided Jupyter Notebook.
