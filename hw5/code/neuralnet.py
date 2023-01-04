'''
neuralnet.py

What you need to do:
- Complete random_init
- Implement softmax, cross_entropy, and d_softmax_cross_entropy functions
- Implement Sigmoid module's forward and backward methods
- Implement Linear module's __init__, forward, backward, and step methods
- Implement train and test functions
- Write code to write predicted labels + error/loss results to output files

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming 
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
'''

import numpy as np
import argparse
import logging
from typing import Callable

import matplotlib.pyplot as plt

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             str, str, str, int, int, int, float]:
    '''
    No need to modify this function!

    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in* (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in* (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    '''
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:] # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:] # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    '''
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    '''
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def random_init(shape):
    '''
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    '''
    M, D = shape
    np.random.seed(M*D) # Don't change this line!
    
    # TODO: create the random matrix here! 
    # Hint: numpy might have some useful function for this
    # W = 
    # raise NotImplementedError
    W = np.random.uniform(-0.1, 0.1, (M, D))
    return W


def zero_init(shape):
    '''
    Do not modify this function.

    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    '''
    return np.zeros(shape = shape)


def softmax(z: np.ndarray) -> np.ndarray:
    '''
    Implement softmax function.
    :param z: input logits of shape (num_classes,)
    :return: softmax output of shape (num_classes,)
    '''
    # TODO: implement this based on your written answers!
    # raise NotImplementedError
    return np.exp(z)/np.sum(np.exp(z))


def cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    '''
    Compute cross entropy loss.
    :param y: label (a number or an array containing a single element)
    :param y_hat: prediction with shape (num_classes,)
    :return: cross entropy loss
    '''
    # TODO: implement this based on your written answers!
    # raise NotImplementedError
    y_onehot = np.zeros(y_hat.shape)
    y_onehot[y] = 1
    return - np.sum(y_onehot * np.log(y_hat)) 


def d_softmax_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    '''
    Compute gradient of loss w.r.t. ** softmax input **.
    Note that here instead of calculating the gradient w.r.t. the softmax 
    probabilities, we are directly computing gradient w.r.t. the softmax input.

    Try deriving the gradient yourself (see Question 1.2(b) on the written), 
    and you'll see why we want to calculate this in a single step.

    :param y: label (a number or an array containing a single element)
    :param y_hat: predicted softmax probability with shape (num_classes,)
    :return: gradient with shape (num_classes,)
    '''
    # TODO: implement this based on your written answers!
    # raise NotImplementedError
    y_onehot = np.zeros(y_hat.shape)
    y_onehot[y] = 1
    return - y_onehot + y_hat

class Sigmoid(object):
    def __init__(self):
        '''
        Initialize state for sigmoid activation layer
        '''
        # Create cache to hold values for backward pass
        self.cache: dict[str, np.ndarray] = dict()

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape (output_size,)
        '''
        # TODO: implement this!
        # raise NotImplementedError
        z = 1/(1+np.exp(-x))
        self.cache['z'] = z
        return 1/(1+np.exp(-x))
    
    def backward(self, dz: np.ndarray) -> np.ndarray:
        '''
        :param dz: partial derivative of loss with respect to output of sigmoid activation
        :return: partial derivative of loss with respect to input of sigmoid activation
        '''
        # TODO: implement this based on your written answers!
        # raise NotImplementedError
        z = self.cache['z'] 
        return dz[1:] * z * (1-z)


# This refers to a function type that takes in a tuple of 2 integers (row, col) 
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[tuple[int, int]], np.ndarray]

class Linear(object):
    def __init__(self, input_size: int, output_size: int, 
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        '''
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        '''
        # TODO: Initialize weight matrix for this layer - since we are 
        # folding the bias into the weight matrix, be careful about the 
        # shape you pass in.
        # self.w =
        # raise NotImplementedError
        self.w = weight_init_fn((output_size, input_size))
        self.w = np.hstack((np.zeros((output_size, 1)), self.w))

        # TODO: Initialize matrix to store gradient with respect to weights
        # self.dw = 
        # raise NotImplementedError
        self.dw = np.zeros(self.w.shape)

        # TODO: Initialize learning rate for SGD
        # self.lr = 
        # raise NotImplementedError
        self.lr = learning_rate

        # Create cache to hold certain values for backward pass
        self.cache: dict[str, np.ndarray] = dict()

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
        :return: output z of linear layer with shape (output_size,)

        HINT: You may want to cache some of the values you compute in this
        function. Inspect your expressions for backprop to see which values
        should be cached.
        '''
        # TODO: implement this based on your written answers!
        # raise NotImplementedError
        x = np.insert(x, 0, 1)
        a =  self.w @ x
        self.cache['x'] = x
        return a

    def backward(self, dz: np.ndarray) -> np.ndarray:
        '''
        :param dz: partial derivative of loss with respect to output z of linear
        :return: dx, partial derivative of loss with respect to input x of linear
        
        Note that this function should set self.dw (gradient of weights with respect to loss)
        but not directly modify self.w; NN.step() is responsible for updating the weights.

        HINT: You may want to use some of the values you previously cached in 
        your forward() method.
        '''
        # TODO: implement this based on your written answers!
        # Hint: when calculating dx, be careful to use the right "version" of
        # the weight matrix!
        # raise NotImplementedError
        self.dw = dz.reshape(-1, 1) @ self.cache['x'].T.reshape(1, -1)
        dx = self.w.T @ dz
        return dx

    def step(self) -> None:
        '''
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        '''
        # TODO: implement this!
        # raise NotImplementedError 
        self.w -= self.lr * self.dw

class NN(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        '''
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        '''
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: initialize modules (see section 9.1.2 of the writeup)
        # Hint: use the classes you've implemented above!
        # self.linear1 = 
        # self.activation = 
        # self.linear2 = 
        # raise NotImplementedError
        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.activation = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Neural network forward computation. 
        Follow the pseudocode!
        :param X: input data point *without the bias folded in*
        :param nn: neural network class
        :return: output prediction with shape (num_classes,). This should be 
                 a valid probability distribution over the classes.
        '''
        # TODO: implement this!
        # raise NotImplementedError
        a = self.linear1.forward(x)
        z = self.activation.forward(a)
        b = self.linear2.forward(z)
        y_hat = softmax(b)
        return y_hat

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        '''
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        :param nn: neural network class
        '''
        # TODO: implement this!
        # raise NotImplementedError
        # g_j = 1
        g_b = d_softmax_cross_entropy(y, y_hat)
        g_z = self.linear2.backward(g_b)
        g_a = self.activation.backward(g_z)
        g_x = self.linear1.backward(g_a)
        return g_x

    def step(self):
        '''
        Apply SGD update to weights.
        '''
        # TODO: implement this!
        # raise NotImplementedError
        self.linear2.step()
        self.linear1.step()
        # print(f'linear1: {self.linear1.w}')
        # print(f'linear2: {self.linear2.w}')

    def print_weights(self) -> None:
        '''
        An example of how to use logging to print out debugging infos.

        Note that we use the debug logging level -- if we use a higher logging
        level, we will log things with the default logging configuration,
        causing potential slowdowns.

        Note that we log NumPy matrices on separate lines -- if we do not do this,
        the arrays will be turned into strings even when our logging is set to
        ignore debug, causing potential massive slowdowns.
        '''
        logging.debug(f"shape of w1: {self.linear1.w.shape}")
        logging.debug(self.linear1.w)
        logging.debug(f"shape of w2: {self.linear2.w.shape}")
        logging.debug(self.linear2.w)


def test(X: np.ndarray, y: np.ndarray, nn: NN) -> tuple[np.ndarray, float]:
    '''
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    '''
    # TODO: implement this!
    # raise NotImplementedError
    labels = []
    error_rate = 0
    for i in range(len(X)):
        y_hat = nn.forward(X[i])
        label = np.argmax(y_hat)
        labels.append(label)
        error_rate += (label!=y[i])
    error_rate = error_rate / len(X)
    return labels, error_rate

def train(X_tr: np.ndarray, y_tr: np.ndarray, 
          X_test: np.ndarray, y_test: np.ndarray, 
          nn: NN, n_epochs: int) -> tuple[list[float], list[float]]:
    '''
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param X_te: train data
    :param y_te: train label
    :param nn: neural network class
    :param n_epochs: number of epochs to train for
    :return:
    train_losses: Training losses *after* each training epoch
    test_losses: Test losses *after* each training epoch
    '''
    # TODO: implement this!
    # Hint: Be sure to shuffle the train data at the start of each epoch
    # using *our provided* shuffle() function.
    # raise NotImplementedError
    train_losses, test_losses = [], []
    for epoch in range(n_epochs):
        X, Y = shuffle(X_tr, y_tr, epoch)
        for i in range(len(X)):
            y_hat = nn.forward(X[i])
            nn.backward(Y[i], y_hat)
            nn.step()
        
        Js = 0
        for i in range(len(X_tr)):
            y_hat= nn.forward(X_tr[i])
            Js += cross_entropy(y_tr[i], y_hat)
        Js = Js/len(X_tr)
    
        Js_test = 0
        for i in range(len(X_test)):
            y_htest = nn.forward(X_test[i])
            Js_test += cross_entropy(y_test[i], y_htest)
        Js_test = Js_test/len(X_test)

        train_losses.append(Js)
        test_losses.append(Js_test)
        
    return train_losses, test_losses

# def plot_nhids(n_hids, X_tr, X_test, y_test, weight_init_fn, lr, n_epochs):
#     train_losses, val_losses = [], []
#     for n_hid in n_hids:
#         nn = NN(X_tr.shape[1], n_hid, 10, weight_init_fn, lr)
#         train_loss, val_loss = train(X_tr, y_tr, X_test, y_test, nn, n_epochs)
#         train_losses.append(train_loss[-1])
#         val_losses.append(val_loss[-1])
    
#     plt.figure(figsize=(10, 8))
#     plt.plot(n_hids, train_losses, '-b')
#     plt.plot(n_hids, val_losses, '-g')
#     plt.xticks(n_hids)
#     plt.xlabel('number of hidden units')
#     plt.ylabel('avg cross-entropy of the final epoch')
#     plt.title('avg train and validation cross-entropy of the final epoch vs number of hidden units')
#     plt.legend(['avg train cross-entropy', 'avg validation cross-entropy'])
#     plt.savefig('2.1.a.png')

def plot_lrs(lrs, X_tr, X_test, y_test, weight_init_fn, lr, n_epochs):
    for lr in lrs:
        nn = NN(X_tr.shape[1], n_hid, 10, weight_init_fn, lr)
        train_losses, test_losses = train(X_tr, y_tr, X_test, y_test, nn, n_epochs)
    
        plt.figure(figsize=(10, 8))
        plt.plot(np.arange(1, n_epochs+1), train_losses, '-b')
        plt.plot(np.arange(1, n_epochs+1), test_losses, '-g')
        plt.xlabel('num_epoch')
        plt.ylabel('avg cross-entropy')
        plt.title(f'avg train and validation cross-entropy vs number of epochs for learning rate {lr}')
        plt.legend(['avg train cross-entropy', 'avg validation cross-entropy'])
        plt.savefig(f'2.2.a_{lr}.png')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate
    # Generally, you can access each argument using the name that was passed 
    # into parser.add_argument() above (see lines 24-44).

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    # Call args2data to get all data + argument values
    # See the docstring of `args2data` for an explanation of 
    # what is being returned.
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)

    # TODO: build model
    #  nn = NN(...)
    # raise NotImplementedError
    if init_flag == 1:
        weight_init_fn = random_init
    else:
        weight_init_fn = zero_init
    
    # nn = NN(X_tr.shape[1], n_hid, 10, weight_init_fn, lr)

    # # train model 
    # # (this line of code is already written for you)
    # train_losses, test_losses = train(X_tr, y_tr, X_test, y_test, nn, n_epochs)

    # # test model and get predicted labels and errors 
    # # (this line of code is written for you)
    # train_labels, train_error_rate = test(X_tr, y_tr, nn)
    # test_labels, test_error_rate = test(X_test, y_test, nn)

    # # Write predicted label and error into file (already implemented for you)
    # # Note that this assumes train_losses and test_losses are lists of floats
    # # containing the per-epoch loss values.
    # with open(out_tr, "w") as f:
    #     for label in train_labels:
    #         f.write(str(label) + "\n")
    # with open(out_te, "w") as f:
    #     for label in test_labels:
    #         f.write(str(label) + "\n")
    # with open(out_metrics, "w") as f:
    #     for i in range(len(train_losses)):
    #         cur_epoch = i + 1
    #         cur_tr_loss = train_losses[i]
    #         cur_te_loss = test_losses[i]
    #         f.write("epoch={} crossentropy(train): {}\n".format(
    #             cur_epoch, cur_tr_loss))
    #         f.write("epoch={} crossentropy(validation): {}\n".format(
    #             cur_epoch, cur_te_loss))
    #     f.write("error(train): {}\n".format(train_error_rate))
    #     f.write("error(validation): {}\n".format(test_error_rate))

    # raise NotImplementedError
    # n_hids = [5, 20, 50, 100, 200]
    # plot_nhids(n_hids, X_tr, X_test, y_test, weight_init_fn, lr, n_epochs)
    # lrs = [0.03, 0.003, 0.0003]
    # plot_lrs(lrs, X_tr, X_test, y_test, weight_init_fn, lr, n_epochs)
    nn = NN(X_tr.shape[1], n_hid, 10, weight_init_fn, 0.0003)
    train_losses, test_losses = train(X_tr, y_tr, X_test, y_test, nn, n_epochs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(1, n_epochs+1), train_losses, '-b')
    plt.plot(np.arange(1, n_epochs+1), test_losses, '-g')
    plt.xlabel('num_epoch')
    plt.ylabel('avg cross-entropy')
    plt.title(f'avg train and validation cross-entropy vs number of epochs for learning rate {lr}')
    plt.legend(['avg train cross-entropy', 'avg validation cross-entropy'])
    plt.savefig(f'2.2.a_{lr}.png')
