import numpy as np
import sys
# import matplotlib.pyplot as plt

def load_data(fpath):
    # data = np.loadtxt(fpath,delimiter='\t')
    data = np.loadtxt(fpath)
    x, y = data[:, 1:], data[:, 0]
    return x, y

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(theta, X, y, num_epoch, learning_rate):
    # TODO: Implement `train` using vectorization
    # add intersect term:
    N = X.shape[0]
    X = np.hstack((X, np.ones((N, 1))))
    for iter in range(num_epoch):
        for i in range(N):
            x = X[i]
            grad = (np.exp(x@theta)/(1+np.exp(x@theta))-y[i]) * x.T
            theta -= learning_rate*grad
    return theta

def predict(theta, X):
    # TODO: Implement `predict` using vectorization
    N = X.shape[0]
    X = np.hstack((X, np.ones((N, 1))))
    y = X @ theta
    probs = np.exp(y)/(1+np.exp(y)) 
    preds = np.ones(N, dtype=np.int8)*(probs >= 0.5)
    return preds


def compute_error(y_pred, y):
    # TODO: Implement `compute_error` using vectorization
    N = y.shape[0]
    return (N-np.count_nonzero(y==y_pred))/N

# def plot_nll(theta, X_train, y_train, X_val, y_val, num_epoch, learning_rate):
#     N_train = X_train.shape[0]
#     N_val = X_val.shape[0]
#     X_train = np.hstack((X_train, np.ones((N_train, 1))))
#     X_val = np.hstack((X_val, np.ones((N_val, 1))))
#     nlls_train, nlls_val = [], []
#     for iter in range(num_epoch):
#         nlls_train.append((-y_train@X_train@theta + np.sum(np.log(1+np.exp(X_train@theta))))/N_train)
#         nlls_val.append((-y_val@X_val@theta + np.sum(np.log(1+np.exp(X_val@theta))))/N_val)
#         for i in range(N_train):
#             x = X_train[i]
#             grad = (np.exp(x@theta)/(1+np.exp(x@theta))-y_train[i]) * x.T
#             theta -= learning_rate*grad
#     nlls_train.append((-y_train@X_train@theta + np.sum(np.log(1+np.exp(X_train@theta))))/N_train)
#     nlls_val.append((-y_val@X_val@theta + np.sum(np.log(1+np.exp(X_val@theta))))/N_val)
    
#     plt.figure(figsize=(10, 8))
#     plt.plot(np.arange(num_epoch+1),  nlls_train, '-b')
#     plt.plot(np.arange(num_epoch+1), nlls_val, '-g')
#     # plt.xticks(np.arange(num_epoch+1))
#     plt.xlabel('num_epoch')
#     plt.ylabel('average negative log-likelihood ')
#     plt.title('average negative log-likelihood for training and validation data after each epoch')
#     plt.legend(['avg neg log-likelihood for train data', 'avg neg log-likelihood for validation data'])
#     plt.savefig('7.1.png')

# def plot_nll_lr(X, y, num_epoch, lr_list):
#     N = X.shape[0]
#     X = np.hstack((X, np.ones((N, 1))))
#     all_nlls = []
#     for lr in lr_list:
#         theta = np.zeros((M+1,))
#         nlls = []
#         for iter in range(num_epoch):
#             nlls.append((-y@X@theta + np.sum(np.log(1+np.exp(X@theta))))/N)
#             for i in range(N):
#                 x = X[i]
#                 grad = (np.exp(x@theta)/(1+np.exp(x@theta))-y[i]) * x.T
#                 theta -= lr*grad
#         nlls.append((-y@X@theta + np.sum(np.log(1+np.exp(X@theta))))/N)
#         all_nlls.append(nlls)
#     all_nlls = np.array(all_nlls)
#     plt.figure(figsize=(10, 8))
#     plt.plot(np.arange(num_epoch+1), all_nlls[0, :], '-b')
#     plt.plot(np.arange(num_epoch+1), all_nlls[1, :], '-g')
#     plt.plot(np.arange(num_epoch+1), all_nlls[2, :], '-k')
#     # plt.xticks(np.arange(num_epoch), np.arange(num_epoch))
#     plt.xlabel('num_epoch')
#     plt.ylabel('average negative log-likelihood ')
#     plt.title('average negative log-likelihood for train data over epochs for different learning rates')
#     plt.legend([f'avg neg log-likelihood for learning rate {lr_list[0]}', f'avg neg log-likelihood for learning rate {lr_list[1]}', 
#     f'avg neg log-likelihood for learning rate {lr_list[2]}'])
#     plt.savefig('7.2.png')

if __name__ == "__main__": 
    train_input = sys.argv[1]
    val_input = sys.argv[2]
    test_input = sys.argv[3]
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    num_epoch = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    
    train_X, train_y = load_data(train_input)
    val_X, val_y = load_data(val_input)
    test_X, test_y = load_data(test_input)
    
    M = train_X.shape[1]
    theta0 = np.zeros((M+1,))
    
    # train
    theta = train(theta0, train_X, train_y, num_epoch, learning_rate)

    # predions and evaluations
    pred_train = predict(theta, train_X)
    file = open(train_out,"w")
    for i in pred_train:
        file.writelines([str(i)+'\n'])
    file.close()

    pred_test = predict(theta, test_X)
    file = open(test_out,"w")
    for i in pred_test:
        file.writelines([str(i)+'\n'])
    file.close()

    err_train = compute_error(pred_train, train_y)
    err_test = compute_error(pred_test, test_y)
    file = open(metrics_out,"w")
    lines = [f'error(train): {"%.6f" %err_train}\n',f'error(test): {"%.6f" %err_test}\n']
    file.writelines(lines)
    file.close()

    # plot_nll(theta0, train_X, train_y, val_X, val_y, num_epoch, learning_rate)
    # lr_list = [0.001, 0.0001, 0.00001] 
    # plot_nll_lr(train_X, train_y, num_epoch, lr_list)
    # print(f'error(train): {"%.6f" %err_train}\n',f'error(test): {"%.6f" %err_test}\n')


