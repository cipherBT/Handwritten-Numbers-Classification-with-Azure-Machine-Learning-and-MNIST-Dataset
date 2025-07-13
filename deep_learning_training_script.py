#!/usr/bin/env python
# coding: utf-8

# ## Training a Deep Learning Model with TensorFlow: Step-by-Step
# In this example, we are training a Neural Network to classify handwritten digits from the MNIST dataset. The main goal is to train the model, evaluate its performance, and track the process using MLflow.

# ### Imports
# We start by importing all the libraries we need for data processing, neural network construction, and experiment tracking.
# - **`numpy`**: Used for numerical computations, particularly for handling arrays and matrices.
# - **`argparse`**: A Python library to handle command-line arguments, allowing users to pass parameters when running the script.
# - **`os`**: Provides functions to interact with the operating system, like handling file paths.
# - **`re`**: For regular expression operations (used here to match and extract numbers from strings).
# - **`tensorflow`**: The core deep learning framework used for building and training models.
# - **`time`**: Used to track the time it takes for training.
# - **`glob`**: For file path operations (to match file patterns).
# - **`mlflow`**: Used for experiment tracking, which logs metrics and parameters.
# - **`gzip`** & **`struct`**: To handle compressed MNIST dataset files in a binary format.
# 

# In[ ]:


# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os
import re
import tensorflow as tf
import time
import glob
import mlflow
import mlflow.tensorflow
# from utils import load_data
from tensorflow.keras import Model, layers
import gzip
import struct


# ## Step 2: Loading Data
# 
# Now we need to load our data, which in this case is the MNIST dataset. It's stored in compressed `.gz` files, and we need to extract that data so we can use it for training.
# 
# ### Data Loading Function
# 
# - **`gzip.open`**: Opens the compressed file.
# - **`struct.unpack`**: Decodes binary data into integers (like the number of images, size of the images).
# - The data is then loaded into NumPy arrays and reshaped.
# - This function either returns the image data or the labels (numbers 0-9), depending on the `label` argument.
# 

# In[ ]:


# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    print("Filename:", filename)
    with gzip.open(filename) as gz:
        struct.unpack("I", gz.read(4))
        n_items = struct.unpack(">I", gz.read(4))
        if not label:
            n_rows = struct.unpack(">I", gz.read(4))[0]
            n_cols = struct.unpack(">I", gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# ## Step 3: One-Hot Encoding
#     When working with classification tasks, the labels need to be one-hot encoded, which means representing each label as a binary vector.
# *For example:*
#     If the label is 3, it gets converted to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (for 10 classes).

# In[ ]:


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]


# ## Step 4: Building the Neural Network
# 
# This is where the magic happens: creating the Neural Network architecture.
# 
# - **Dense Layer**: Fully connected layer with a specified number of neurons.
# - **ReLU Activation**: Introduces non-linearity, helping the model learn complex patterns.
# - **Softmax Activation**: Applied on the output layer during inference (i.e., when we're not training), which converts raw scores into probabilities.
# 
# This model has two hidden layers and an output layer. Each layer is a dense layer.
# 

# In[ ]:


# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First hidden layer.
        self.h1 = layers.Dense(n_h1, activation=tf.nn.relu)
        # Second hidden layer.
        self.h2 = layers.Dense(n_h2, activation=tf.nn.relu)
        self.out = layers.Dense(n_outputs)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)
        if not is_training:
            # Apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# ## Step 5: Loss Function and Accuracy
# 
# Now, we need a way to measure how well our model is doing. We use the **cross-entropy loss** for classification and **accuracy** to measure performance.
# 
# - **`cross_entropy_loss`**: Computes the difference between predicted probabilities and actual labels.
# - **`accuracy`**: Calculates the percentage of correct predictions.
# 

# In[ ]:


def cross_entropy_loss(y, logits):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


# ### Accuracy Metric Function
# 
# This function calculates the accuracy of the model's predictions.
# 
# - **`tf.argmax`**: Finds the index of the maximum value in the predicted vector (i.e., the predicted class).
# - **`tf.equal`**: Compares the predicted class to the true label, returning `True` or `False`.
# - **`tf.reduce_mean`**: Computes the mean accuracy over all the predictions.
# 

# In[ ]:


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# ## Step 6: Training Loop
# 
# Now comes the core of the training process — the training loop. This is where we repeatedly adjust the model’s parameters to minimize the loss.
# 
# Here’s how it works:
# 
# - We use **`GradientTape`** to track the operations that are needed for computing gradients.
# - After calculating the loss, we compute gradients with respect to the model’s parameters.
# - Finally, the optimizer updates the parameters by applying the gradients.
# 

# In[ ]:


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        logits = neural_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(y, logits)

    # Variables to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


print("TensorFlow version:", tf.__version__)


# ## Parse Command Line Arguments
# We use argparse to take input parameters like batch size, learning rate, and whether to resume training from a checkpoint.
# Learning Rate: Controls the size of steps during training.
# Batch Size: Number of training samples used in one update.

# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-folder",
    type=str,
    dest="data_folder",
    default="data",
    help="data folder mounting point",
)
parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=128,
    help="mini batch size for training",
)
parser.add_argument(
    "--first-layer-neurons",
    type=int,
    dest="n_hidden_1",
    default=128,
    help="# of neurons in the first layer",
)
parser.add_argument(
    "--second-layer-neurons",
    type=int,
    dest="n_hidden_2",
    default=128,
    help="# of neurons in the second layer",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    dest="learning_rate",
    default=0.01,
    help="learning rate",
)
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="location of the model or checkpoint files from where to resume the training",
)
args = parser.parse_args()


# ## Setup MLflow Logging
# We initialize MLflow to track our experiment metrics (e.g., accuracy) and automatically log model parameters.
# autolog(): Automatically logs TensorFlow metrics, parameters, and models.
# 

# In[ ]:


mlflow.start_run()
mlflow.tensorflow.autolog()


# ## 5. Preparing Data and Model Initialization
#     Before we start training, we need to:
#     
#     Load the dataset (train and test images/labels).
#     Normalize the pixel values for better model convergence.
#     Initialize the model and optimizer.
#     Resume from a previous checkpoint if specified.

# ### 5.1 Load the Dataset

# In[ ]:


previous_model_location = args.resume_from
# You can also use environment variable to get the model/checkpoint files location
# previous_model_location = os.path.expandvars(os.getenv("AZUREML_DATAREFERENCE_MODEL_LOCATION", None))

data_folder = args.data_folder
print("Data folder:", data_folder)

# load train and test set into numpy arrays
# note we scale the pixel intensity values to 0-1 (by dividing it with 255.0) so the model can converge faster.
X_train = load_data(
    glob.glob(
        os.path.join(data_folder, "**/train-images-idx3-ubyte.gz"), recursive=True
    )[0],
    False,
) / np.float32(255.0)
X_test = load_data(
    glob.glob(
        os.path.join(data_folder, "**/t10k-images-idx3-ubyte.gz"), recursive=True
    )[0],
    False,
) / np.float32(255.0)
y_train = load_data(
    glob.glob(
        os.path.join(data_folder, "**/train-labels-idx1-ubyte.gz"), recursive=True
    )[0],
    True,
).reshape(-1)
y_test = load_data(
    glob.glob(
        os.path.join(data_folder, "**/t10k-labels-idx1-ubyte.gz"), recursive=True
    )[0],
    True,
).reshape(-1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep="\n")

training_set_size = X_train.shape[0]


# ### Step 5.2: Set Model Parameters
# Once the data is loaded, we define key model parameters based on the user's input.

# In[ ]:


n_inputs = 28 * 28
n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_outputs = 10
learning_rate = args.learning_rate
n_epochs = 20
batch_size = args.batch_size


# ### Step 5.3: Initialize the Model
# We now create the neural network model and define an optimizer to update the model weights.

# In[ ]:


# Initialize the neural network model
neural_net = NeuralNet()

# Use Stochastic Gradient Descent (SGD) optimizer
optimizer = tf.optimizers.SGD(learning_rate)


# ### Step 5.4: Resume Training from Checkpoint (Optional)
# If we have a saved checkpoint from a previous training run, we can resume training from that point.

# In[ ]:


# Build neural network model.
neural_net = NeuralNet()

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

if previous_model_location:
    # Restore variables from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)
    checkpoint_file_path = tf.train.latest_checkpoint(previous_model_location)
    checkpoint.restore(checkpoint_file_path)
    checkpoint_filename = os.path.basename(checkpoint_file_path)
    num_found = re.search(r"\d+", checkpoint_filename)
    if num_found:
        start_epoch = int(num_found.group(0))
        print("Resuming from epoch {}".format(str(start_epoch)))


# ## Training and Evaluation
# We now begin the training loop. We perform the following tasks in each epoch:
# 
# Shuffle the training data.
# Split the data into mini-batches.
# Update the model parameters using Gradient Descent.
# Evaluate the model’s performance on both the training and validation sets.

# In[ ]:


start_time = time.perf_counter()
for epoch in range(0, n_epochs):

    # randomly shuffle training set
    indices = np.random.permutation(training_set_size)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # batch index
    b_start = 0
    b_end = b_start + batch_size
    for _ in range(training_set_size // batch_size):
        # get a batch
        X_batch, y_batch = X_train[b_start:b_end], y_train[b_start:b_end]

        # update batch index for the next batch
        b_start = b_start + batch_size
        b_end = min(b_start + batch_size, training_set_size)

        # train
        run_optimization(X_batch, y_batch)

    # evaluate training set
    pred = neural_net(X_batch, is_training=False)
    acc_train = accuracy(pred, y_batch)

    # evaluate validation set
    pred = neural_net(X_test, is_training=False)
    acc_val = accuracy(pred, y_test)

    # log accuracies
    mlflow.log_metric("training_acc", float(acc_train))
    mlflow.log_metric("validation_acc", float(acc_val))
    print(epoch, "-- Training accuracy:", acc_train, "\b Validation accuracy:", acc_val)

    # Save checkpoints in the "./outputs" folder so that they are automatically uploaded into run history.
    checkpoint_dir = "./outputs/"
    checkpoint = tf.train.Checkpoint(model=neural_net, optimizer=optimizer)

    if epoch % 2 == 0:
        checkpoint.save(checkpoint_dir)


# ## Saving the Model
# After training, we save the trained model for later use (e.g., inference).

# In[ ]:


mlflow.log_metric("final_acc", float(acc_val))
os.makedirs("./outputs/model", exist_ok=True)

# files saved in the "./outputs" folder are automatically uploaded into run history
# this is workaround for https://github.com/tensorflow/tensorflow/issues/33913 and will be fixed once we move to >tf2.1
neural_net._set_inputs(X_train)
tf.saved_model.save(neural_net, "./outputs/model/")


# ## End of Training
# Finally, we measure the total training time and log it using MLflow.

# In[ ]:


stop_time = time.perf_counter()
training_time = (stop_time - start_time) * 1000
print("Total time in milliseconds for training: {}".format(str(training_time)))


# ## Summary of the Workflow
# Model Definition: A neural network with two hidden layers.
# Data Loading: Preprocess and load the MNIST dataset.
# Training Loop:
# Shuffle and split data into mini-batches.
# Update model weights using Gradient Descent.
# Log metrics (training and validation accuracy).
# Checkpointing: Save the model periodically to avoid losing progress.
# Model Saving: After training, save the trained model to disk.
# Experiment Logging: Use MLflow to track experiment parameters, metrics, and model artifacts.

# In[ ]:




