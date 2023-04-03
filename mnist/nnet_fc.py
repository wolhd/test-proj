#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    # batch_size = 64
    
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              # nn.Linear(784, 10),
              nn.Linear(784, 128),
                nn.ReLU(),
               # nn.LeakyReLU(),
              # nn.Linear(10, 10),
              nn.Linear(128, 10),
            )
    lr=0.1
    # lr=0.01
    
    momentum=0
    # momentum=0.9
    
    ##################################

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()

# change    val acc   test acc
# baseline  .932487  .920472
# batch 64  .940020  .931490
# learn .01 .934653  .920673
# momen .9  .908590  .895432
# leakyrel  .931985  .920773

# hidden128 
# baseline  .977440  .976963
# batch 64  .976647  .974659
# learn .01 .955047  .942708
# momen .9  .958723  .955428
# leakyrel  .979445  .976963