#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from tabulate import tabulate
import sys
sys.path.append("..")
from utils import *
from train_utils import batchify_data, run_epoch, train_model


def main(batch_size=32, lr=1e-1, hidden_size=10, momentum=0):
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(X_train.shape[0])])
    np.random.shuffle(permutation)
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )
    ##################################

    val_acr = train_model(train_batches, dev_batches,
                          model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))

    return batch_size, lr, momentum, hidden_size, val_acr


if __name__ == '__main__':
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility

    print("Performing Grid Search over batch size")
    batch_size = np.linspace(32, 64, num=10, dtype='int')
    parameters = []
    pbr = tqdm(batch_size, desc="Batch size: ",
               total=len(batch_size), leave=True)
    for size in pbr:
        parameters.append(tuple(main(batch_size=size)))
    tabulate(parameters, headers=["batch_size", "lr",
                                  "momentum", "hidden_size", "accuracy"])
