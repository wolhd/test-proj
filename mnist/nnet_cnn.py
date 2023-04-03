#! /usr/bin/env python

import _pickle as c_pickle, gzip
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
from train_utils import batchify_data, run_epoch, train_model, Flatten

def main():
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # We need to rehape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

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
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Conv2d(1, 32, (3, 3)),
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              nn.Conv2d(32, 64, (3,3)),
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              nn.Flatten(),
              nn.Linear(64 * 5 * 5, 128),
              nn.Dropout(0.5),
              nn.Linear(128, 10)
            )
    ##################################

    train_model(train_batches, dev_batches, model, nesterov=True)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    main()

###
# 32 kernel 5, stride 1

# 32 - (5-1) -1 / 1 + 1 = (32 - 4 - 1)/1 + 1 = 27/1 + 1 = 28

# 32 - 5 + 1 = 28



# 28 - 2 -1 -1 / 2 + 1 = 26/2 + 1= 14



# 6 , kernel 2, stride 2

# 12, 34, 56  = 3

# 6 - (2-1) -1 / 2 + 1= 4/2 + 1 = 3

# 4, kernel 2, stride 2

# 12 34  = 2

# 4 - (2-1) -1 / 2 + 1 = 2/2 + 1 = 2

# ---

# 28x28

# 28 kernel 3 => 28-3+1 = 26

# conv in=1 out=32  26x26

# max pool 2x2 => chan 32 13x13

# conv in 32 out 64 : 13 kern 3 -> 13-3+1= 11

# max pool 2x2 -> chan 64 5x5
 
# flatten 64x5x5  to 128