import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
#nb_epoch = 40
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

# img width 28, height 42
# conv 3x3:  (28-(3-1) -1 / 1) + 1 = 26
#             (42-(3-1) -1 / 1) + 1 = 40
# pool 2,2
#           26-(2-1) -1 / 2 + 1 = 13
#           40-(2-1) -1 / 2 + 1 = 20
#
# conv 3x3  13-> 13-3+1 = 11
#           20-> 20-3+1 = 18
# pool 2,2  
#           11 -> 5
#           18 -> 9

class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # TODO initialize model layers here
        self.model = nn.Sequential(
              nn.Conv2d(1, 84, (3, 3)),
              nn.LeakyReLU(),
              nn.AvgPool2d((2, 2)),
              nn.Conv2d(84, 128, (3,3)),
              nn.LeakyReLU(),
              nn.MaxPool2d((2, 2)),
              nn.Flatten(),
              nn.Linear(128 * 5 * 9, 128),
              nn.Dropout(0.5)
              )

        self.outL1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.outL2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):

        # TODO use model layers to predict the two digits
        xh = self.model(x)
        out_first_digit = self.outL1(xh)
        out_second_digit = self.outL2(xh)
        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension).to('cuda') # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model, n_epochs=nb_epoch)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()