import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return np.maximum(x, 0)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    res = 1.
    if x <= 0:
        res = 0.
    return res

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        w = self.input_to_hidden_weights
        b = self.biases
        wb = np.append(w, b, axis=1) # 3x2 append 3x1 -> 3x3
        
        x = input_values
        xb = np.append(x, np.matrix([[1]]), axis=0) # 2x1 -> 3x1
        
        z = wb * xb # 3x3 * 3x1 = 3x1
        
        # hidden_layer_weighted_input = # TODO (3 by 1 matrix)
        hidden_layer_weighted_input = z # z1 z2 z3  (3x1)

        reluVec = np.vectorize(rectified_linear_unit) 

        # hidden_layer_activation = # TODO (3 by 1 matrix)
        hidden_layer_activation = reluVec(hidden_layer_weighted_input)

        fz = hidden_layer_activation  # 3x1

        v = self.hidden_to_output_weights # 1x3
        u1 = v * fz  # 1x1
        # output =  # TODO
        output = u1.A1[0]
        
        # activated_output = # TODO
        activated_output = output_layer_activation(output)


        ### Backpropagation ### ----------------------------------

        # Compute gradients
        cost = 1/2 * (y - activated_output)**2
        #print(cost)
        
        # output_layer_error = # TODO
        output_layer_error = (activated_output - y) * output_layer_activation_derivative(output)
        deltaL = output_layer_error 
        
        relu_deriv_vecfn = np.vectorize(rectified_linear_unit_derivative)
        fdz = relu_deriv_vecfn(z) # 3x1
        deltal = np.matrix(v.T.A * fdz.A) * deltaL # 3x1
        
        # hidden_layer_error = # TODO (3 by 1 matrix)
        hidden_layer_error = deltal # 3x1
        #print(hidden_layer_error)
        
        
        # bias_gradients = # TODO
        bias_gradients = hidden_layer_error
        
        # hidden_to_output_weight_gradients = # TODO
        hidden_to_output_weight_gradients = deltaL * fz
        
        # input_to_hidden_weight_gradients = # TODO
        cw1 = deltal * x.item((0,0))
        cw2 = deltal * x.item((1,0))
        input_to_hidden_weight_gradients = np.matrix(np.append(cw1, cw2, axis=1))
        

        # Use gradients to adjust weights and biases using gradient descent
        # self.biases = # TODO
        self.biases = self.biases - bias_gradients * self.learning_rate
        
        # self.input_to_hidden_weights = # TODO
        self.input_to_hidden_weights = self.input_to_hidden_weights - input_to_hidden_weight_gradients * self.learning_rate
        #print(self.input_to_hidden_weights)
        
        # self.hidden_to_output_weights = # TODO
        self.hidden_to_output_weights = self.hidden_to_output_weights - hidden_to_output_weight_gradients.T * self.learning_rate
        #print(self.hidden_to_output_weights)

    def predict(self, x1, x2):

        # input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        # hidden_layer_weighted_input = # TODO
        # hidden_layer_activation = # TODO
        # output = # TODO
        # activated_output = # TODO
        # return activated_output.item()
        
        
        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        w = self.input_to_hidden_weights
        b = self.biases
        wb = np.append(w, b, axis=1) # 3x2 append 3x1 -> 3x3
        
        x = input_values
        xb = np.append(x, np.matrix([[1]]), axis=0) # 2x1 -> 3x1
        
        z = wb * xb # 3x3 * 3x1 = 3x1
        
        # hidden_layer_weighted_input = # TODO (3 by 1 matrix)
        hidden_layer_weighted_input = z # z1 z2 z3  (3x1)

        reluVec = np.vectorize(rectified_linear_unit) 

        # hidden_layer_activation = # TODO (3 by 1 matrix)
        hidden_layer_activation = reluVec(hidden_layer_weighted_input)

        fz = hidden_layer_activation  # 3x1

        v = self.hidden_to_output_weights # 1x3
        u1 = v * fz  # 1x1
        # output =  # TODO
        output = u1.A1[0]
        
        # activated_output = # TODO
        activated_output = output_layer_activation(output)
        return activated_output

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return




x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()

# """
# # Compute gradients 
# output_layer_error = # derivative of loss with respect to the "not-yet-activated output" (1 by 1) 
# hidden_layer_error = # derivative of loss with respect to the the "not-yet-activated hidden layer values" (3 by 1) 
# bias_gradients = # derivative of loss with respect to the biases that contribute to the not-yet-activated hidden layer values (3 by 1) 
# hidden_to_output_weight_gradients = # derivative of loss with respect to the weights that turn hidden activations to not-yet-activated outputs (3 by 1) 
# input_to_hidden_weight_gradients = # derivative of loss with respect to the weights that turn inputs to the not-yet-activated hidden layer values (2 by 3)
# """
