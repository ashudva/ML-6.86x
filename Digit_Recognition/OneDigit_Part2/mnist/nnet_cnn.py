#! /usr/bin/env python

from train_utils import batchify_data, run_epoch, train_model, Flatten
from utils import *
import utils
import _pickle as c_pickle
import gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")


def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # We need to reshape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

    # Split into train and validation
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    # Shuffle the data
    permutation = torch.randperm(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    # Model specification
    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        Flatten(),
        nn.Linear(1600, 128),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )
    ##################################

    # Moving model and data to GPU
    if torch.cuda.is_available():
        print("----------------- Using the Device: CUDA -----------------")
        model = model.to('cuda')
    else:
        print("----------------- Using the Device: CPU ----------------- ")
    train_model(train_batches, dev_batches, model, nesterov=True)

    # Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print("Loss on test set:" + str(loss) +
          " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle.
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    main()
